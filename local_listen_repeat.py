from __future__ import annotations
import os, json, math, tempfile, shutil, base64, io
from dataclasses import dataclass
from typing import List, Dict, Any, Callable, Optional, Tuple, Literal
from urllib.parse import urlparse

# ---- audio & ASR deps ----
# pip install faster-whisper pydub jiwer requests openai python-dotenv
from faster_whisper import WhisperModel
from pydub import AudioSegment
import requests
from jiwer import wer
import difflib
import time
from io import BytesIO

from dotenv import load_dotenv
load_dotenv()

# OpenAI client for GPT-4o helpers (now required)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# --------------------- Inputs ---------------------

@dataclass
class ListenRepeatPair:
    """A single (prompt, student) audio pair. Each can be a URL or local path."""
    prompt_audio: str
    student_audio: str


# --------------------- Main class ---------------------

class LocalSpeakingAssessmentReport:
    """
    End-to-end local pipeline (no server needed):

      1) Download (if URL) + transcribe each audio with Whisper (faster-whisper, word timestamps on)
      2) Compute metrics:
         - total duration (MM:SS),
         - speech rate (wpm),
         - repeat accuracy (WER-based) and incorrect_segments,
         - pause statistics (from word timestamps) → levels:
             pause_frequency_level, pause_appropriateness_level
      3) Detect mispronounced words via GPT-4o-audio
      4) Detect grammar errors from the transcript via GPT-4o text
      5) Build JSON in the schema shown in your screenshots and save to disk
    """

    def __init__(
        self,
        whisper_model_size: str = "base",
        device: str = "auto",     # "cuda" or "cpu" or "auto"
        compute_type: str = "auto",
        version: str = "1.0",
        api_key: Optional[str] = None,
        audio_model: str = "gpt-4o-audio-preview",
        text_model: str = "gpt-4o",
        embed_model: str = "text-embedding-3-small",           # <-- for interview relevance
        task: Literal["listen_repeat", "interview"] = "listen_repeat",  # <-- task switch
    ):
        """
        GPT-4o is required. API key is taken from `api_key` or the OPENAI_API_KEY env var.
        """
        self.version = version

        # init Whisper
        self._whisper = WhisperModel(
            whisper_model_size,
            device=self._resolve_device(device),
            compute_type=self._resolve_compute_type(compute_type),
        )

        # OpenAI client (required)
        if OpenAI is None:
            raise RuntimeError("openai package not installed. pip install openai")
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set")
        self._openai = OpenAI(api_key=key)
        self._has_responses = hasattr(self._openai, "responses")
        self._audio_model = audio_model
        self._text_model = text_model
        self._embed_model = embed_model
        self.task = task

        # transcript cache to avoid re-running ASR on same files
        # NOTE: stores (text, duration_seconds); words are recomputed per call when needed
        self._tx_cache: Dict[str, Tuple[str, float]] = {}

    # ----------------- public API -----------------

    def generate_report(self, pairs: List[ListenRepeatPair], out_path: str) -> Dict[str, Any]:
        """
        Process all pairs and write a single report JSON to `out_path`.
        Returns the report as a dict as well.
        """
        temp_dir = tempfile.mkdtemp(prefix="listen_repeat_")
        errors: List[Dict[str, str]] = []

        try:
            # 1) Download + transcribe each pair
            prompt_tx: List[Tuple[str, float]] = []  # (text, dur_unused)
            student_tx: List[Tuple[str, float]] = []  # (text, dur)
            student_durations: List[float] = []

            # aggregated pause stats across pairs (computed per-pair to avoid cross-file gaps)
            total_pauses = 0
            total_long_pauses = 0
            total_gap_sum = 0.0
            total_gap_count = 0

            for p in pairs:
                prompt_path, e1 = self._to_local(p.prompt_audio, temp_dir)
                student_path, e2 = self._to_local(p.student_audio, temp_dir)

                if e1:
                    errors.append(self._err_obj(p.prompt_audio, p.student_audio, "DownloadError", e1))
                if e2:
                    errors.append(self._err_obj(p.prompt_audio, p.student_audio, "DownloadError", e2))

                if e1 or e2:
                    # Spill an empty slot to keep alignment
                    prompt_tx.append(("", 0.0))
                    student_tx.append(("", 0.0))
                    student_durations.append(0.0)
                    continue

                # Transcribe (prompt duration not needed)
                pt_text, _, _ = self._transcribe(prompt_path)
                st_text, st_dur, st_words = self._transcribe(student_path)

                prompt_tx.append((pt_text, 0.0))
                student_tx.append((st_text, st_dur))
                student_durations.append(st_dur)

                # per-pair pause stats (avoid counting a giant gap between files)
                ps = self._pause_stats(st_words, st_dur)
                total_pauses += ps["pauses"]
                total_long_pauses += ps["long_pauses"]
                total_gap_sum += ps["avg_gap"] * ps["gap_count"]
                total_gap_count += ps["gap_count"]

            # 2) Aggregate metrics
            total_secs = sum(student_durations)
            duration_str = self._fmt_mmss(total_secs)

            all_student_text = " ".join(t for t, _ in student_tx).strip()
            total_student_words = len([w for w in all_student_text.split() if w.strip()])
            speech_rate = self._speech_rate(all_student_text, total_secs)

            # Repeat accuracy + incorrect segments (per pair → aggregate)
            acc_scores: List[int] = []
            incorrect_segments: List[str] = []
            for (pt_text, _), (st_text, _) in zip(prompt_tx, student_tx):
                if not pt_text.strip() or not st_text.strip():
                    acc_scores.append(0)
                    continue
                score, segs = self._repeat_accuracy_and_incorrect_segments(pt_text, st_text)
                acc_scores.append(score)
                incorrect_segments.extend(segs)

            repeat_accuracy_score = int(round(sum(acc_scores) / max(len(acc_scores), 1)))
            repeat_accuracy = {"score": repeat_accuracy_score}

            # Aggregate pause statistics across pairs
            pause_stats = {
                "pauses": total_pauses,
                "long_pauses": total_long_pauses,
                "avg_gap": (total_gap_sum / total_gap_count) if total_gap_count else 0.0,
                "pauses_per_min": (total_pauses / (total_secs / 60.0)) if total_secs > 0 else 0.0,
            }
            long_ratio = (pause_stats["long_pauses"] / pause_stats["pauses"]) if pause_stats["pauses"] > 0 else 0.0
            pause_freq_lvl = self._pause_frequency_level(pause_stats["pauses_per_min"])
            pause_app_lvl = self._pause_appropriateness_level(long_ratio)

            # 3) Mispronunciations (GPT-4o audio; required)
            mis_all_for_scoring: List[str] = []
            mispronounced_words: List[str] = []
            for p, (pt_text, _), (st_text, _) in zip(pairs, prompt_tx, student_tx):
                if pt_text.strip() and st_text.strip():
                    try:
                        lst = self._gpt4o_mispronunciations(p.prompt_audio, p.student_audio, pt_text, st_text) or []
                        mis_all_for_scoring.extend(lst)
                        mispronounced_words.extend(lst)
                    except Exception as e:
                        errors.append(self._err_obj(p.prompt_audio, p.student_audio, "GPTAudioError", str(e)))
            # de-dup for display while preserving order
            seen = set()
            mispronounced_words = [w for w in mispronounced_words if not (w in seen or seen.add(w))]

            # Pronunciation score from mispronounced / total words (use raw counts for scoring)
            pron_acc_score = self._pronunciation_score(mis_all_for_scoring, total_student_words)

            # 4) Grammar (GPT-4o text; required)
            grammar_issues: List[Dict[str, Any]] = []
            grammar_score: Optional[int] = None
            if all_student_text:
                try:
                    g = self._gpt4o_grammar(all_student_text) or {}
                    grammar_issues = list(g.get("issues", []))
                    gs = g.get("score", None)
                    if isinstance(gs, (int, float)):
                        grammar_score = max(0, min(100, int(round(gs))))
                    else:
                        grammar_score = None
                except Exception as e:
                    errors.append(self._err_obj("-", "-", "GPTTextError", str(e)))

            # 5) Sections (task-specific branching)
            if self.task == "listen_repeat":
                fluency = self._build_fluency_section(
                    speech_rate=speech_rate,
                    acc=repeat_accuracy_score,
                    pause_freq_lvl=pause_freq_lvl,
                    pause_app_lvl=pause_app_lvl
                )
                pronunciation = self._build_pronunciation_section(
                    mispronounced=mispronounced_words,
                    total_words=total_student_words,
                    accuracy_score=pron_acc_score
                )
                grammar = self._build_grammar_section(
                    acc=repeat_accuracy_score,
                    grammar_score=grammar_score,
                    issues=grammar_issues
                )
                extra_blocks = {}
            else:  # --- interview ---
                # relevance via embeddings (question vs answer)
                rel_label = self._embedding_relevance(prompt_tx, student_tx)
                # discourse via GPT-4o label
                disc_label = self._gpt4o_discourse(all_student_text)
                # fluency with extra word repetition level
                wr_level = self._word_repetition_level(all_student_text)
                fluency = self._build_fluency_section(
                    speech_rate=speech_rate,
                    acc=repeat_accuracy_score,
                    pause_freq_lvl=pause_freq_lvl,
                    pause_app_lvl=pause_app_lvl
                )
                fluency["word_repetition_level"] = wr_level
                pronunciation = self._build_pronunciation_section(
                    mispronounced=mispronounced_words,
                    total_words=total_student_words,
                    accuracy_score=pron_acc_score
                )
                grammar = self._build_interview_grammar(grammar_issues)
                vocab = self._vocabulary_block(all_student_text)
                extra_blocks = {
                    "relevance": {"score": rel_label},
                    "discourse": {"score": disc_label},
                    "vocabulary": vocab,
                }

            # 6) Overall block
            overall_score = self._overall_block(speech_rate, repeat_accuracy_score)

            # 7) generation_failed flag (true only if every pair failed)
            generation_failed = self._all_failed(pairs, errors, prompt_tx, student_tx)

            # 8) Final report
            report = {
                "version": self.version,
                "generation_failed": generation_failed,
                "errors": [e for e in errors if e.get("error_type")],
                "overall_score": overall_score,
                "speech_rate": speech_rate,
                "duration": duration_str,
                "repeat_accuracy": repeat_accuracy,
                "incorrect_segments": incorrect_segments[:50],  # cap if very long
                "mispronounced_words": [{"word": w} for w in mispronounced_words],  # only words
                "fluency": fluency,
                "pronunciation": pronunciation,
                "grammar": grammar,
            }
            report.update(extra_blocks)

            self._save_json(report, out_path)
            return report

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    # ----------------- GPT-4o calls (required) -----------------

    def _gpt4o_mispronunciations(self, prompt_audio: str, student_audio: str,
                             prompt_text: str, student_text: str) -> List[str]:
        p_bytes = self._load_wav_bytes_16k_mono(prompt_audio)
        s_bytes = self._load_wav_bytes_16k_mono(student_audio)

        def audio_part(b: bytes) -> Dict[str, Any]:
            return {
                "type": "input_audio",
                "audio": {"data": base64.b64encode(b).decode("utf-8"), "format": "wav"},
            }

        instruction = (
            'Compare the student’s pronunciation to the reference audio + text. '
            'Return ONLY valid JSON: {"words": ["word1", ...]} with unique, lowercase mispronounced words. '
            'If none, return {"words": []}.'
        )
        system_msg = {"role": "system", "content": "You are a precise ASR/phonetics judge."}
        user_content = [
            {"type": "text", "text": instruction},
            {"type": "text", "text": f"Reference transcript:\n{prompt_text}"},
            audio_part(p_bytes),
            {"type": "text", "text": f"Student transcript:\n{student_text}"},
            audio_part(s_bytes),
        ]

        def _parse_words(raw: str) -> List[str]:
            try:
                return [w.strip().lower() for w in (json.loads(raw).get("words", [])) if str(w).strip()]
            except Exception:
                cleaned = raw.strip().strip("`")
                if cleaned.lower().startswith("json"):
                    cleaned = cleaned[4:].strip()
                return [w.strip().lower() for w in (json.loads(cleaned).get("words", [])) if str(w).strip()]

        # Preferred: Responses API (newer clients)
        if self._has_responses:
            def _call_with_rf():
                return self._openai.responses.create(
                    model=self._audio_model,
                    input=[{"role": "user", "content": user_content}],
                    temperature=0,
                    response_format={"type": "json_object"},
                )
            def _call_without_rf():
                return self._openai.responses.create(
                    model=self._audio_model,
                    input=[{"role": "user", "content": user_content}],
                    temperature=0,
                )
            try:
                resp = self._retry(_call_with_rf)
            except TypeError as e:
                if "response_format" in str(e):
                    resp = self._retry(_call_without_rf)
                else:
                    raise
            raw = getattr(resp, "output_text", "") or (resp.__dict__.get("output_text", "") or "")
            return _parse_words(raw)

        # Fallback: Chat Completions (older clients)
        def _call_chat():
            return self._openai.chat.completions.create(
                model=self._audio_model,
                messages=[system_msg, {"role": "user", "content": user_content}],
                temperature=0,
                response_format={"type": "json_object"},
            )
        resp = self._retry(_call_chat)
        raw = resp.choices[0].message.content.strip() if resp.choices else "{}"
        return _parse_words(raw)

    def _gpt4o_grammar(self, student_transcript: str) -> Dict[str, Any]:
        """
        Call GPT-4o text to return grammar issues and a 0–100 score.
        Forces JSON output and retries on transient errors.
        """
        system_msg = {"role": "system", "content": "You are a strict grammar evaluator."}
        user_msg = {
            "role": "user",
            "content": (
                "Analyze the transcript ONLY for GRAMMAR (agreement, tense, articles, prepositions, word order).\n"
                "STRICTLY IGNORE: mishears, nonsense tokens, misspellings from ASR/pronunciation, and simple word choice.\n"
                "If a token is not a valid English word, IGNORE it and do not flag it.\n"
                "Return ONLY valid JSON with fields:\n"
                "{\n"
                '  "issues": [ {"span": string, "explanation": string, "suggestion": string}... ],\n'
                '  "score": integer 0-100\n'
                "}\n"
                "Keep 'issues' concise and specific to grammar.\n\n"
                f"Transcript:\n{student_transcript}"
            )
        }

        def _call():
            return self._openai.chat.completions.create(
                model=self._text_model,
                messages=[system_msg, user_msg],
                temperature=0,
                response_format={"type": "json_object"}  # <- forces JSON
            )

        resp = self._retry(_call)
        raw = resp.choices[0].message.content.strip() if resp.choices else "{}"

        try:
            data = json.loads(raw)
        except Exception:
            cleaned = raw.strip().strip("`")
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
            data = json.loads(cleaned)

        if not isinstance(data, dict):
            return {"issues": [], "score": 100}

        issues = data.get("issues", [])
        score = data.get("score", 100)
        try:
            score = int(score)
        except Exception:
            score = 100
        score = max(0, min(100, score))

        norm_issues = []
        for it in issues or []:
            if isinstance(it, dict):
                norm_issues.append({
                    "span": str(it.get("span", ""))[:200],
                    "explanation": str(it.get("explanation", ""))[:300],
                    "suggestion": str(it.get("suggestion", ""))[:200],
                })
        return {"issues": norm_issues, "score": score}

    # ----------------- Interview-specific helpers -----------------

    def _embedding_relevance(self, prompt_tx: List[Tuple[str, float]],
                             student_tx: List[Tuple[str, float]]) -> str:
        """
        Compute semantic relevance using embeddings (cosine similarity)
        between each interviewer question (prompt) and its paired answer (student).
        Return the *lowest* (most conservative) CEFR label across pairs.
        """
        pairs = [(q.strip(), a.strip()) for (q,_), (a,_) in zip(prompt_tx, student_tx) if q.strip() and a.strip()]
        if not pairs:
            return "A1"

        # Batch embed all texts in one request where possible
        texts = []
        for q,a in pairs:
            texts.append(q)
            texts.append(a)
        embs = self._openai.embeddings.create(model=self._embed_model, input=texts).data
        vecs: List[List[float]] = [e.embedding for e in embs]

        labels = []
        for i in range(0, len(vecs), 2):
            qv, av = vecs[i], vecs[i+1]
            sim = self._cosine(qv, av)
            labels.append(self._sim_to_cefr(sim))
        # conservative: return the lowest label
        order = ["A1","A2","B1","B2","C1","C2"]
        return min(labels, key=order.index) if labels else "A1"

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        if not a or not b: return 0.0
        dot = sum(x*y for x,y in zip(a,b))
        na = math.sqrt(sum(x*x for x in a))
        nb = math.sqrt(sum(y*y for y in b))
        if na == 0 or nb == 0: return 0.0
        return dot/(na*nb)

    @staticmethod
    def _sim_to_cefr(sim: float) -> str:
        # Heuristic thresholds for embedding cosine sim
        if sim >= 0.82: return "C1"
        if sim >= 0.74: return "B2"
        if sim >= 0.66: return "B1"
        if sim >= 0.58: return "A2"
        return "A1"

    def _tokens(self, text: str) -> List[str]:
        return [t.lower() for t in text.split() if t.strip()]

    def _word_repetition_level(self, text: str) -> str:
        """Map repetition rate to CEFR-ish level (more repetition -> lower level)."""
        toks = [t for t in self._tokens(text) if t.isalpha()]
        if not toks: return "C1"
        from collections import Counter
        cnt = Counter(toks)
        rep_rate = sum((c-1) for c in cnt.values() if c>1) / len(toks)  # repeats / total
        if rep_rate > 0.25: return "A1"
        if rep_rate > 0.18: return "A2"
        if rep_rate > 0.12: return "B1"
        if rep_rate > 0.07: return "B2"
        return "C1"

    def _vocabulary_block(self, text: str) -> Dict[str, str]:
        """Proxies: complexity≈avg word length; diversity≈type/token ratio."""
        toks = [t for t in self._tokens(text) if t.isalpha()]
        if not toks:
            return {"complexity_level": "A1", "diversity_level": "A1",
                    "description": "Not enough data.", "description_cn": "数据不足。"}
        avg_len = sum(len(t) for t in toks)/len(toks)
        ttr = len(set(toks))/len(toks)
        def lvl_len(x):
            if x < 4.2: return "A1"
            if x < 4.8: return "A2"
            if x < 5.4: return "B1"
            if x < 6.0: return "B2"
            return "C1"
        def lvl_ttr(x):
            if x < 0.35: return "A1"
            if x < 0.45: return "A2"
            if x < 0.55: return "B1"
            if x < 0.65: return "B2"
            return "C1"
        return {
            "complexity_level": lvl_len(avg_len),
            "diversity_level": lvl_ttr(ttr),
            "description": "Lexical complexity and diversity estimated from word length and type–token ratio.",
            "description_cn": "基于词长与类型/词汇比估算词汇复杂度与多样性。"
        }

    def _build_interview_grammar(self, gpt_issues: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Convert grammar issues (span/explanation/suggestion)
        into interview schema: errors[{original_sentence, corrected_sentence, fdiff: [...]}]
        """
        errors = []
        for it in gpt_issues or []:
            span = str(it.get("span",""))
            sugg = str(it.get("suggestion",""))
            expl = str(it.get("explanation",""))
            errors.append({
                "original_sentence": span,
                "corrected_sentence": sugg,
                "fdiff": [{
                    "has_error": True,
                    "orig": span,
                    "corr": sugg,
                    "error_type_description": expl[:120],
                    "feedback": expl,
                    "feedback_cn": ""
                }]
            })
        lvl = self._level_from_accuracy(100 - min(50, 10*len(errors)))  # crude fallback
        return {"accuracy_level": lvl, "errors": errors, "description": ""}

    def _gpt4o_discourse(self, answer_text: str) -> str:
        """Rate discourse coherence/organization as CEFR label via GPT-4o."""
        if not answer_text.strip(): return "A1"
        system = {"role":"system","content":"You evaluate discourse coherence and structure."}
        user = {"role":"user","content":(
            "Rate the discourse coherence/organization of the text using CEFR bands (A1..C1). "
            'Return ONLY JSON like {"label":"B1"}.\n\nText:\n'+answer_text)}
        resp = self._retry(lambda: self._openai.chat.completions.create(
            model=self._text_model, messages=[system,user], temperature=0,
            response_format={"type":"json_object"}))
        raw = resp.choices[0].message.content.strip()
        try:
            return json.loads(raw).get("label","A1")
        except Exception:
            return "A1"

    # ----------------- audio, asr, utils -----------------

    @staticmethod
    def _to_local(path_or_url: str, temp_dir: str) -> Tuple[str, Optional[str]]:
        """Download URL to a temp file; if local path, just return it."""
        if not path_or_url:
            return "", "Empty path"
        try:
            parsed = urlparse(path_or_url)
            if parsed.scheme in ("http", "https"):
                fn = os.path.join(temp_dir, os.path.basename(parsed.path) or "audio.wav")
                with requests.get(path_or_url, stream=True, timeout=30) as r:
                    r.raise_for_status()
                    with open(fn, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                return fn, None
            else:
                if not os.path.exists(path_or_url):
                    return "", f"File not found: {path_or_url}"
                return path_or_url, None
        except Exception as e:
            return "", str(e)

    def _transcribe(self, audio_path: str) -> Tuple[str, float, List[Dict[str, float]]]:
        """
        Transcribe an audio file with word timestamps.
        Returns (transcript, duration_seconds, words[{text,start,end}]).

        We cache (text, duration) by path to avoid repeated ASR runs; words are
        recomputed per call if needed.
        """
        cached = self._tx_cache.get(audio_path)
        dur: float
        if cached is not None:
            text, dur = cached
        else:
            # duration
            try:
                dur = AudioSegment.from_file(audio_path).duration_seconds
            except Exception:
                dur = 0.0

        # ASR (always do it when called here, because we need timestamps)
        segments, _ = self._whisper.transcribe(
            audio_path,
            beam_size=5,
            vad_filter=True,
            word_timestamps=True
        )
        words: List[Dict[str, float]] = []
        pieces: List[str] = []
        for seg in segments:
            if seg.text:
                pieces.append(seg.text.strip())
            if getattr(seg, "words", None):
                for w in seg.words:
                    if not w.word or w.start is None or w.end is None:
                        continue
                    words.append({"text": w.word.strip(), "start": float(w.start), "end": float(w.end)})

        text = " ".join(pieces).strip()
        # update cache
        self._tx_cache[audio_path] = (text, float(dur))
        return text, float(dur), words

    # ----------------- metric primitives -----------------

    @staticmethod
    def _speech_rate(transcript: str, total_secs: float) -> int:
        if total_secs <= 0:
            return 0
        words = [w for w in transcript.split() if w.strip()]
        return int(round(len(words) / (total_secs / 60.0)))

    @staticmethod
    def _repeat_accuracy_and_incorrect_segments(prompt_text: str, student_text: str) -> Tuple[int, List[str]]:
        """Accuracy via WER; 'incorrect_segments' = prompt spans that were deleted or replaced."""
        w = wer(prompt_text.lower(), student_text.lower())  # 0..1
        score = max(0, int(round(100 * (1.0 - w))))

        p_tokens = prompt_text.split()
        s_tokens = student_text.split()
        sm = difflib.SequenceMatcher(a=p_tokens, b=s_tokens)
        incorrect: List[str] = []
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag in ("delete", "replace"):  # tokens the student missed/changed
                seg = " ".join(p_tokens[i1:i2]).strip()
                if seg:
                    incorrect.append(seg)
        return score, incorrect

    # ----------------- pause analysis -----------------

    @staticmethod
    def _pause_stats(words: List[Dict[str, float]], total_secs: float) -> Dict[str, float]:
        """
        Compute gaps between consecutive words. Thresholds:
          - minor pause: gap >= 0.30s
          - long pause:  gap >= 1.00s
        Returns counts and rate per minute.
        """
        if not words or total_secs <= 0:
            return {"pauses": 0, "long_pauses": 0, "avg_gap": 0.0, "pauses_per_min": 0.0, "gap_count": 0}

        gaps: List[float] = []
        for i in range(1, len(words)):
            gap = words[i]["start"] - words[i-1]["end"]
            if gap > 0:
                gaps.append(gap)

        pauses = sum(1 for g in gaps if g >= 0.30)
        long_pauses = sum(1 for g in gaps if g >= 1.00)
        avg_gap = (sum(gaps) / len(gaps)) if gaps else 0.0
        pauses_per_min = pauses / (total_secs / 60.0) if total_secs > 0 else 0.0

        return {
            "pauses": pauses,
            "long_pauses": long_pauses,
            "avg_gap": avg_gap,
            "pauses_per_min": pauses_per_min,
            "gap_count": len(gaps),
        }

    @staticmethod
    def _pause_frequency_level(ppm: float) -> str:
        # Tune these cutoffs to your rubric.
        if ppm > 20: return "A1"
        if ppm > 15: return "A2"
        if ppm > 10: return "B1"
        if ppm > 5:  return "B2"
        return "C1"

    @staticmethod
    def _pause_appropriateness_level(long_ratio: float) -> str:
        # long_ratio = long_pauses / pauses; fewer long pauses → more appropriate
        if long_ratio > 0.40: return "A1"
        if long_ratio > 0.30: return "A2"
        if long_ratio > 0.20: return "B1"
        if long_ratio > 0.10: return "B2"
        return "C1"

    @staticmethod
    def _pronunciation_score(mispronounced: List[str], total_words: int) -> int:
        """Simple 0–100: 100 * (1 - mispronounced/total_words)."""
        if total_words <= 0:
            return 100 if not mispronounced else 0
        pct = 1.0 - (len(mispronounced) / float(total_words))
        return max(0, min(100, int(round(100 * pct))))

    @staticmethod
    def _split_wav_bytes(b: bytes, chunk_ms: int = 20000) -> List[bytes]:
        """
        Split a 16kHz mono WAV byte stream into ~chunk_ms segments (default: 20s),
        returning a list of WAV-bytes chunks.
        """
        try:
            seg = AudioSegment.from_file(BytesIO(b), format="wav")
        except Exception:
            return [b]  # if parsing fails, just return original
        chunks = []
        for start in range(0, len(seg), chunk_ms):
            piece = seg[start:start + chunk_ms]
            buf = BytesIO()
            piece.export(buf, format="wav")
            chunks.append(buf.getvalue())
        return chunks

    # ----------------- section builders -----------------

    def _build_fluency_section(self, speech_rate: int, acc: int,
                               pause_freq_lvl: str, pause_app_lvl: str) -> Dict[str, Any]:
        sr_lvl = self._level_from_speech_rate(speech_rate)
        rep_lvl = self._level_from_accuracy(acc)
        return {
            "speech_rate_level": sr_lvl,
            "coherence_level": rep_lvl,
            "pause_frequency_level": pause_freq_lvl,
            "pause_appropriateness_level": pause_app_lvl,
            "repeat_accuracy_level": rep_lvl,
            "description": (
                "Speech is understandable with some pauses; "
                f"rate {speech_rate} wpm; repetition accuracy {acc}%."
            ),
            "description_cn": f"整体可理解；语速约 {speech_rate} 词/分钟；复述准确率约 {acc}% 。"
        }

    def _build_pronunciation_section(self, mispronounced: List[str],
                                     total_words: int,
                                     accuracy_score: int) -> Dict[str, Any]:
        has_err = len(mispronounced) > 0
        base = "A2" if has_err else "B1"
        return {
            "prosody_rhythm_level": base,
            "vowel_fullness_level": base,
            "intonation_level": base,
            "accuracy_score": accuracy_score,
            "description": "Pronunciation is generally intelligible; rhythm and vowel quality can improve.",
            "description_cn": "发音整体清晰；节奏与元音质量仍可提升。"
        }

    def _build_grammar_section(self, acc: int,
                               grammar_score: Optional[int],
                               issues: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        accuracy_level:
            - If GPT grammar_score is available, map that to CEFR-ish level.
            - Else fall back to repeat-accuracy mapping.
        repeat_accuracy_level:
            - Mapped from repeat accuracy as before (schema asks explicitly for this).
        """
        if grammar_score is not None:
            lvl = self._level_from_accuracy(grammar_score)
        else:
            lvl = self._level_from_accuracy(acc)
        return {
            "accuracy_level": lvl,
            "repeat_accuracy_level": self._level_from_accuracy(acc),
            "issues": issues
        }

    def _overall_block(self, speech_rate: int, acc: int) -> Dict[str, str]:
        # Map to a compact overall block per your screenshots.
        toefl_1_6 = self._toefl_band(acc)                    # 1–6
        old_toefl_0_30 = str(int(round(acc * 0.3)))          # rough mapping
        cefr = self._overall_cefr(speech_rate, acc)          # CEFR label
        return {
            "cefr": cefr,
            "toefl_score": str(toefl_1_6),
            "old_toefl_score": old_toefl_0_30
        }

    # ----------------- label helpers -----------------

    @staticmethod
    def _level_from_speech_rate(wpm: int) -> str:
        if wpm < 90: return "A1"
        if wpm < 110: return "A2"
        if wpm < 130: return "B1"
        if wpm < 150: return "B2"
        return "C1"

    @staticmethod
    def _level_from_accuracy(pct: int) -> str:
        if pct < 50: return "A1"
        if pct < 70: return "A2"
        if pct < 85: return "B1"
        if pct < 95: return "B2"
        return "C1"

    @staticmethod
    def _toefl_band(pct: int) -> int:
        if pct < 50: return 1
        if pct < 60: return 2
        if pct < 70: return 3
        if pct < 80: return 4
        if pct < 90: return 5
        return 6

    @staticmethod
    def _overall_cefr(wpm: int, pct: int) -> str:
        levels = ["A1", "A2", "B1", "B2", "C1", "C2"]
        l_sr = levels.index(LocalSpeakingAssessmentReport._level_from_speech_rate(wpm))
        l_acc = levels.index(LocalSpeakingAssessmentReport._level_from_accuracy(pct))
        return levels[min(l_sr, l_acc)]

    # ----------------- misc utils -----------------

    @staticmethod
    def _fmt_mmss(secs: float) -> str:
        secs = int(round(secs))
        m, s = divmod(secs, 60)
        return f"{m:02d}:{s:02d}"

    @staticmethod
    def _err_obj(prompt_url: str, student_url: str, etype: str, msg: str) -> Dict[str, str]:
        return {
            "prompt_audio_url": prompt_url,
            "student_audio_url": student_url,
            "error_type": etype,
            "error_message": msg
        }

    @staticmethod
    def _all_failed(
        pairs: List[ListenRepeatPair],
        errors: List[Dict[str, str]],
        prompt_tx: List[Tuple[str, float]],
        student_tx: List[Tuple[str, float]],
    ) -> bool:
        if not pairs:
            return True
        failed = 0
        for (pt, _), (st, _) in zip(prompt_tx, student_tx):
            if not pt.strip() or not st.strip():
                failed += 1
        return failed == len(pairs)

    @staticmethod
    def _save_json(obj: Dict[str, Any], path: str) -> None:
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            try:
                import torch  # optional
                return "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                return "cpu"
        return device

    @staticmethod
    def _resolve_compute_type(compute_type: str) -> str:
        return compute_type if compute_type != "auto" else "default"

    @staticmethod
    def _guess_audio_format(path_or_url: str) -> str:
        ext = os.path.splitext(urlparse(path_or_url).path)[1].lower().lstrip(".")
        return ext or "wav"

    @staticmethod
    def _maybe_fetch_bytes(path_or_url: str) -> bytes:
        parsed = urlparse(path_or_url)
        if parsed.scheme in ("http", "https"):
            r = requests.get(path_or_url, timeout=30)
            r.raise_for_status()
            return r.content
        with open(path_or_url, "rb") as f:
            return f.read()

    @staticmethod
    def _retry(fn: Callable[[], Any], attempts: int = 3, base_delay: float = 0.75) -> Any:
        last = None
        for i in range(attempts):
            try:
                return fn()
            except Exception as e:
                last = e
                if i < attempts - 1:
                    time.sleep(base_delay * (2 ** i))
        raise last

    @staticmethod
    def _load_wav_bytes_16k_mono(path_or_url: str) -> bytes:
        try:
            parsed = urlparse(path_or_url)
            if parsed.scheme in ("http", "https"):
                r = requests.get(path_or_url, timeout=30)
                r.raise_for_status()
                src = AudioSegment.from_file(BytesIO(r.content))
            else:
                src = AudioSegment.from_file(path_or_url)
            wav16 = src.set_frame_rate(16000).set_channels(1).set_sample_width(2)
            buf = BytesIO()
            wav16.export(buf, format="wav")
            return buf.getvalue()
        except Exception:
            return LocalSpeakingAssessmentReport._maybe_fetch_bytes(path_or_url)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["listen_repeat","interview"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs", nargs="+", help="prompt.wav:student.wav (or URLs)", required=True)
    args = ap.parse_args()

    prs = []
    for p in args.pairs:
        prompt, student = p.split(":", 1)
        prs.append(ListenRepeatPair(prompt, student))

    LocalSpeakingAssessmentReport(task=args.task).generate_report(prs, args.out)
