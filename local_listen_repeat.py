from __future__ import annotations
import os, json, math, tempfile, shutil
from dataclasses import dataclass
from typing import List, Dict, Any, Callable, Optional, Tuple
from urllib.parse import urlparse

# ---- audio & ASR deps ----
# pip install faster-whisper pydub jiwer requests
from faster_whisper import WhisperModel
from pydub import AudioSegment
import requests
from jiwer import wer
import difflib


# --------------------- Inputs ---------------------

@dataclass
class ListenRepeatPair:
    """A single (prompt, student) audio pair. Each can be a URL or local path."""
    prompt_audio: str
    student_audio: str


# --------------------- Main class ---------------------

class LocalListenRepeatReport:
    """
    End-to-end local pipeline (no server needed):

      1) Download (if URL) + transcribe each audio with Whisper (faster-whisper, word timestamps on)
      2) Compute metrics:
         - total duration (MM:SS),
         - speech rate (wpm),
         - repeat accuracy (WER-based) and incorrect_segments,
         - pause statistics (from word timestamps) → levels:
             pause_frequency_level, pause_appropriateness_level
      3) (Optional) detect mispronounced words via a supplied function (e.g., GPT-4o-audio)
      4) Build JSON in the schema shown in your screenshots and save to disk
    """

    def __init__(
        self,
        whisper_model_size: str = "base",
        device: str = "auto",     # "cuda" or "cpu" or "auto"
        compute_type: str = "auto",
        mispronunciation_fn: Optional[
            Callable[[str, str, str, str], List[Dict[str, str]]]
        ] = None,
        version: str = "1.0"
    ):
        """
        mispronunciation_fn signature:
            (prompt_audio_path_or_url, student_audio_path_or_url, prompt_text, student_text)
              -> List[{"word": str, "original_audio_url": str, "corrected_audio_url": str}]
        Provide this to plug in GPT-4o-audio or your own detector. Otherwise leave None.
        """
        self.version = version
        self.mispronunciation_fn = mispronunciation_fn

        # init Whisper
        self._whisper = WhisperModel(
            whisper_model_size,
            device=self._resolve_device(device),
            compute_type=self._resolve_compute_type(compute_type),
        )

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
            prompt_tx: List[Tuple[str, float]] = []                 # (text, dur_unused)
            student_tx: List[Tuple[str, float]] = []                # (text, dur)
            student_words_all: List[Dict[str, float]] = []          # flattened words with {text,start,end}
            student_durations: List[float] = []

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
                student_words_all.extend(st_words)

            # 2) Aggregate metrics
            total_secs = sum(student_durations)
            duration_str = self._fmt_mmss(total_secs)

            all_student_text = " ".join(t for t, _ in student_tx).strip()
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

            # Pause statistics + levels
            pause_stats = self._pause_stats(student_words_all, total_secs)
            long_ratio = (
                (pause_stats["long_pauses"] / pause_stats["pauses"])
                if pause_stats["pauses"] > 0 else 0.0
            )
            pause_freq_lvl = self._pause_frequency_level(pause_stats["pauses_per_min"])
            pause_app_lvl  = self._pause_appropriateness_level(long_ratio)

            # 3) Mispronunciations (optional)
            mispronounced_words: List[Dict[str, str]] = []
            if self.mispronunciation_fn is not None:
                for p, (pt_text, _), (st_text, _) in zip(pairs, prompt_tx, student_tx):
                    if pt_text.strip() and st_text.strip():
                        mispronounced_words.extend(
                            self.mispronunciation_fn(
                                p.prompt_audio, p.student_audio, pt_text, st_text
                            ) or []
                        )

            # 4) Sections
            fluency = self._build_fluency_section(
                speech_rate=speech_rate,
                acc=repeat_accuracy_score,
                pause_freq_lvl=pause_freq_lvl,
                pause_app_lvl=pause_app_lvl
            )
            pronunciation = self._build_pronunciation_section(mispronounced_words)
            grammar = self._build_grammar_section(repeat_accuracy_score)

            # 5) Overall block
            overall_score = self._overall_block(speech_rate, repeat_accuracy_score)

            # 6) generation_failed flag (true only if every pair failed)
            generation_failed = self._all_failed(pairs, errors, prompt_tx, student_tx)

            # 7) Final report (matches the screenshot schema)
            report = {
                "version": self.version,
                "generation_failed": generation_failed,
                "errors": [e for e in errors if e.get("error_type")],
                "overall_score": overall_score,
                "speech_rate": speech_rate,
                "duration": duration_str,
                "repeat_accuracy": repeat_accuracy,
                "incorrect_segments": incorrect_segments[:50],  # cap if very long
                "mispronounced_words": mispronounced_words,
                "fluency": fluency,
                "pronunciation": pronunciation,
                "grammar": grammar,
            }

            self._save_json(report, out_path)
            return report

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    # ----------------- audio, asr, utils -----------------

    def _to_local(self, path_or_url: str, temp_dir: str) -> Tuple[str, Optional[str]]:
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

    def _speech_rate(self, transcript: str, total_secs: float) -> int:
        if total_secs <= 0:
            return 0
        words = [w for w in transcript.split() if w.strip()]
        return int(round(len(words) / (total_secs / 60.0)))

    def _repeat_accuracy_and_incorrect_segments(self, prompt_text: str, student_text: str) -> Tuple[int, List[str]]:
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

    def _pause_stats(self, words: List[Dict[str, float]], total_secs: float) -> Dict[str, float]:
        """
        Compute gaps between consecutive words. Thresholds:
          - minor pause: gap >= 0.30s
          - long pause:  gap >= 1.00s
        Returns counts and rate per minute.
        """
        if not words or total_secs <= 0:
            return {"pauses": 0, "long_pauses": 0, "avg_gap": 0.0, "pauses_per_min": 0.0}

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
        }

    def _pause_frequency_level(self, ppm: float) -> str:
        # Tune these cutoffs to your rubric.
        if ppm > 20: return "A1"
        if ppm > 15: return "A2"
        if ppm > 10: return "B1"
        if ppm > 5:  return "B2"
        return "C1"

    def _pause_appropriateness_level(self, long_ratio: float) -> str:
        # long_ratio = long_pauses / pauses; fewer long pauses → more appropriate
        if long_ratio > 0.40: return "A1"
        if long_ratio > 0.30: return "A2"
        if long_ratio > 0.20: return "B1"
        if long_ratio > 0.10: return "B2"
        return "C1"

    # ----------------- section builders -----------------

    def _build_fluency_section(self, speech_rate: int, acc: int,
                               pause_freq_lvl: str, pause_app_lvl: str) -> Dict[str, Any]:
        sr_lvl = self._level_from_speech_rate(speech_rate)
        rep_lvl = self._level_from_accuracy(acc)  # proxy for coherence; replace if you add a coherence metric
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

    def _build_pronunciation_section(self, mispronounced: List[Dict[str, str]]) -> Dict[str, Any]:
        has_err = len(mispronounced) > 0
        base = "A2" if has_err else "B1"
        return {
            "prosody_rhythm_level": base,
            "vowel_fullness_level": base,
            "intonation_level": base,
            "description": "Pronunciation is generally intelligible; rhythm and vowel quality can improve.",
            "description_cn": "发音整体清晰；节奏与元音质量仍可提升。"
        }

    def _build_grammar_section(self, acc: int) -> Dict[str, Any]:
        lvl = self._level_from_accuracy(acc)
        return {
            "accuracy_level": lvl,
            "repeat_accuracy_level": lvl
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

    def _level_from_speech_rate(self, wpm: int) -> str:
        if wpm < 90: return "A1"
        if wpm < 110: return "A2"
        if wpm < 130: return "B1"
        if wpm < 150: return "B2"
        return "C1"

    def _level_from_accuracy(self, pct: int) -> str:
        if pct < 50: return "A1"
        if pct < 70: return "A2"
        if pct < 85: return "B1"
        if pct < 95: return "B2"
        return "C1"

    def _toefl_band(self, pct: int) -> int:
        # Simple mapping; replace with ETS table if desired.
        if pct < 50: return 1
        if pct < 60: return 2
        if pct < 70: return 3
        if pct < 80: return 4
        if pct < 90: return 5
        return 6

    def _overall_cefr(self, wpm: int, pct: int) -> str:
        # Conservative overall = lower (weaker) of rate-level and accuracy-level.
        levels = ["A1", "A2", "B1", "B2", "C1", "C2"]
        l_sr = levels.index(self._level_from_speech_rate(wpm))
        l_acc = levels.index(self._level_from_accuracy(pct))
        return levels[min(l_sr, l_acc)]

    # ----------------- misc utils -----------------

    def _fmt_mmss(self, secs: float) -> str:
        secs = int(round(secs))
        m, s = divmod(secs, 60)
        return f"{m:02d}:{s:02d}"

    def _err_obj(self, prompt_url: str, student_url: str, etype: str, msg: str) -> Dict[str, str]:
        return {
            "prompt_audio_url": prompt_url,
            "student_audio_url": student_url,
            "error_type": etype,
            "error_message": msg
        }

    def _all_failed(
        self,
        pairs: List[ListenRepeatPair],
        errors: List[Dict[str, str]],
        prompt_tx: List[Tuple[str, float]],
        student_tx: List[Tuple[str, float]],
    ) -> bool:
        if not pairs:
            return True
        # Consider a pair failed if either transcript is empty.
        failed = 0
        for (pt, _), (st, _) in zip(prompt_tx, student_tx):
            if not pt.strip() or not st.strip():
                failed += 1
        return failed == len(pairs)

    def _save_json(self, obj: Dict[str, Any], path: str) -> None:
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    def _resolve_device(self, device: str) -> str:
        if device == "auto":
            try:
                import torch  # optional
                return "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                return "cpu"
        return device

    def _resolve_compute_type(self, compute_type: str) -> str:
        # Let faster-whisper decide unless explicitly set.
        return compute_type if compute_type != "auto" else "default"
