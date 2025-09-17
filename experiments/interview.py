#!/usr/bin/env python3
import os, argparse, json, time, math, statistics as stats
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # pulls OPENAI_API_KEY from .env

# -------- Config --------
INTERVIEWER_SYS = """You are a rigorous interviewer.

- Ask ONE concise question at a time, then wait for the next follow-up.
- Use the user's last answer to craft the next follow-up.
- Drill down: ask for specifics, examples, tradeoffs, metrics, or code when relevant.
- Stay on the given TOPIC. If the user wanders, steer back politely.
- Keep questions short (<= 25 words).
- Do exactly ROUNDS total questions, then end with: [END].
"""

EMBEDDING_MODEL = "text-embedding-3-small"   # cheap + solid for relevance
DEFAULT_CHAT_MODEL = "gpt-4o-mini"           # interviewer + grammar judging


def _attr(obj, name, default=None):
    """Return attribute `name` if present; if obj is a dict, fallback to dict.get."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)

# -------- Helpers --------
def cosine_sim(a: List[float], b: List[float]) -> float:
    dot = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

# ==== TOEFL/CEFR helpers ====
def cefr_label(score_1to6: float) -> str:
    if score_1to6 >= 6.0: return "C2"
    if score_1to6 >= 5.5: return "C1"
    if score_1to6 >= 5.0: return "C1"
    if score_1to6 >= 4.5: return "B2"
    if score_1to6 >= 4.0: return "B2"
    if score_1to6 >= 3.5: return "B1"
    if score_1to6 >= 3.0: return "B1"
    if score_1to6 >= 2.5: return "A2"
    if score_1to6 >= 2.0: return "A2"
    if score_1to6 >= 1.5: return "A1"
    return "A1"

TOEFL_OLD_MAP = {
    6.0: "29–30", 5.5: "27–28", 5.0: "24–26", 4.5: "22–23",
    4.0: "18–21", 3.5: "12–17", 3.0: "6–11", 2.5: "4–5",
    2.0: "3",     1.5: "2",     1.0: "0–1",
}
def old_toefl_band(score_1to6: float) -> str:
    step = round(score_1to6 * 2) / 2.0
    step = max(1.0, min(6.0, step))
    return TOEFL_OLD_MAP.get(step, TOEFL_OLD_MAP[6.0])

# map a 0..1 score to 1..6 (used for interview subscores)
def toefl_1to6_from_unit(x01: float) -> float:
    return round(1.0 + 5.0 * clamp01(x01), 2)

# speech metrics scorers (used for audio report)
def score_speech_rate(wpm: float) -> float:
    if 110 <= wpm <= 160: return 6.0
    if 90  <= wpm < 110 or 160 < wpm <= 180: return 5.0
    if 75  <= wpm <  90 or 180 < wpm <= 200: return 4.0
    if 60  <= wpm <  75 or 200 < wpm <= 220: return 3.5
    if 45  <= wpm <  60: return 3.0
    return 2.5

def score_duration(seconds: float) -> float:
    if seconds >= 120: return 6.0
    if 90 <= seconds < 120: return 5.0
    if 60 <= seconds <  90: return 4.0
    if 40 <= seconds <  60: return 3.5
    if 20 <= seconds <  40: return 3.0
    return 2.5

def score_pause_frequency(pauses_per_min: float) -> float:
    if 5 <= pauses_per_min <= 15: return 6.0
    if 3 <= pauses_per_min < 5 or 15 < pauses_per_min <= 18: return 5.0
    if 2 <= pauses_per_min < 3 or 18 < pauses_per_min <= 22: return 4.0
    if 1 <= pauses_per_min < 2 or 22 < pauses_per_min <= 26: return 3.5
    if pauses_per_min < 1 or 26 < pauses_per_min <= 30: return 3.0
    return 2.5

# grading table builders
def build_interview_grading_table(avgs: dict) -> tuple[list[dict], float]:
    """Use text-only subscores (0..1) from interview averages; map to 1..6."""
    rel6   = toefl_1to6_from_unit(avgs["relevance"])
    gram6  = toefl_1to6_from_unit(avgs["grammar"])
    think6 = toefl_1to6_from_unit(avgs["thinking_time"])
    rows = [
        {"Metric":"Relevance","Raw":f"{avgs['relevance']:.2f}",
         "CEFR":cefr_label(rel6),"New TOEFL (1–6)":rel6,"Old TOEFL (0–30)":old_toefl_band(rel6)},
        {"Metric":"Grammar","Raw":f"{avgs['grammar']:.2f}",
         "CEFR":cefr_label(gram6),"New TOEFL (1–6)":gram6,"Old TOEFL (0–30)":old_toefl_band(gram6)},
        {"Metric":"Thinking time","Raw":f"{avgs['thinking_time']:.2f}",
         "CEFR":cefr_label(think6),"New TOEFL (1–6)":think6,"Old TOEFL (0–30)":old_toefl_band(think6)},
    ]
    overall6 = round((rel6 + gram6 + think6) / 3.0, 2)
    return rows, overall6

def build_audio_grading_table(agg_metrics: dict) -> tuple[list[dict], float]:
    """Use aggregate audio metrics; no repetition F1 available here."""
    sr   = agg_metrics["speech_rate_wpm"]
    dur  = agg_metrics["duration_s"]
    ppm  = agg_metrics["pauses_per_min"]

    sr6   = score_speech_rate(sr)
    dur6  = score_duration(dur)
    ppm6  = score_pause_frequency(ppm)

    rows = [
        {"Metric":"Speech rate","Raw":f"{sr:.0f} wpm",
         "CEFR":cefr_label(sr6),"New TOEFL (1–6)":sr6,"Old TOEFL (0–30)":old_toefl_band(sr6)},
        {"Metric":"Duration","Raw":f"{dur:.0f}s",
         "CEFR":cefr_label(dur6),"New TOEFL (1–6)":dur6,"Old TOEFL (0–30)":old_toefl_band(dur6)},
        {"Metric":"Pause frequency","Raw":f"{ppm:.1f} / min",
         "CEFR":cefr_label(ppm6),"New TOEFL (1–6)":ppm6,"Old TOEFL (0–30)":old_toefl_band(ppm6)},
    ]

    # overall without repetition: heavier weight on speech + pauses
    overall6 = round(0.5*sr6 + 0.3*ppm6 + 0.2*dur6, 2)
    return rows, overall6




#NEW HELPERS
# -------- Audio analysis (Whisper) --------
from pathlib import Path
from statistics import median

WHISPER_MODEL = "whisper-1"   # OpenAI STT

def _linear_band_score(x, ideal_min, ideal_max, hard_min, hard_max) -> float:
    """
    Score in [0..1]. 1.0 inside [ideal_min..ideal_max].
    Linearly decays to 0.0 at hard_min / hard_max, clipped outside.
    """
    if x <= hard_min or x >= hard_max:
        return 0.0
    if x < ideal_min:
        return (x - hard_min) / (ideal_min - hard_min)
    if x > ideal_max:
        return (hard_max - x) / (hard_max - ideal_max)
    return 1.0

def transcribe_and_analyze_audio_folder(
    client: OpenAI,
    audio_dir: str,
    pause_gap_s: float = 0.35,   # gaps >= this are pauses
) -> dict:
    """
    Transcribe all .wav in `audio_dir` and compute per-file and aggregate metrics.
    Returns:
      {
        "files": [...],
        "per_file": [{file, metrics, scores, transcript}, ...],
        "aggregate": {"metrics": {...}, "scores": {...}},
        "transcript": "<all files concatenated>"
      }
    """
    p = Path(audio_dir)
    wavs = sorted(p.glob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"No .wav files found in {audio_dir}")

    all_text = []
    per_file = []

    # --- helpers reused inside ---
    def _linear_band_score(x, ideal_min, ideal_max, hard_min, hard_max) -> float:
        if x <= hard_min or x >= hard_max:
            return 0.0
        if x < ideal_min:
            return (x - hard_min) / (ideal_min - hard_min)
        if x > ideal_max:
            return (hard_max - x) / (hard_max - ideal_max)
        return 1.0

    for wav in wavs:
        with open(wav, "rb") as f:
            # Prefer verbose_json + timestamps
            try:
                tr = client.audio.transcriptions.create(
                    model=WHISPER_MODEL,
                    file=f,
                    response_format="verbose_json",
                    timestamp_granularities=["segment", "word"],
                )
                segments = tr.segments or []
                text = tr.text or ""
            except Exception:
                f.seek(0)
                tr = client.audio.transcriptions.create(model=WHISPER_MODEL, file=f)
                segments = []
                text = tr.text or ""

        all_text.append(text)

        # ----- word count (robust) -----
        words_spoken = 0
        wlist = getattr(tr, "words", None)
        if wlist:
            words_spoken = len(wlist)
        else:
            seg_word_total = 0
            for seg in segments:
                seg_words = _attr(seg, "words", None)
                if seg_words:
                    try:
                        seg_word_total += len(seg_words)
                    except TypeError:
                        seg_word_total += sum(1 for _ in seg_words)
            words_spoken = seg_word_total if seg_word_total else len(text.split())

        # ----- segment timing & pauses -----
        last_end = 0.0
        all_gaps = []
        total_speech_time = 0.0

        for seg in segments:
            start = float(_attr(seg, "start", 0.0))
            end = _attr(seg, "end", None)
            if end is None:
                dur = float(_attr(seg, "duration", 0.0))
                end = float(start + max(0.0, dur))
            else:
                end = float(end)
                dur = max(0.0, end - start)

            total_speech_time += dur
            gap = start - last_end
            if gap >= pause_gap_s:
                all_gaps.append(gap)
            last_end = end

        # file duration = last segment end (or rough fallback)
        if segments:
            duration_s = float(_attr(segments[-1], "end", _attr(segments[-1], "start", 0.0)))
        else:
            duration_s = max(1.0, len(text.split()) / (160 / 60.0))

        # ----- per-file metrics -----
        spoken_minutes = max(1e-9, duration_s / 60.0)
        speech_rate_wpm = words_spoken / spoken_minutes
        pause_count = len(all_gaps)
        pauses_per_min = pause_count / spoken_minutes
        avg_pause = sum(all_gaps) / pause_count if pause_count else 0.0
        med_pause = median(all_gaps) if pause_count else 0.0
        total_pause_time = sum(all_gaps)

        # ----- per-file scores -----
        sr_score = _linear_band_score(speech_rate_wpm, 110, 160, 80, 220)
        ppm_score = _linear_band_score(pauses_per_min, 5, 15, 0, 30)
        ap_score = _linear_band_score(avg_pause, 0.3, 1.2, 0.0, 2.5)
        audio_score = (sr_score + ppm_score + ap_score) / 3.0

        per_file.append({
            "file": str(wav),
            "metrics": {
                "duration_s": round(duration_s, 2),
                "words_spoken": int(words_spoken),
                "speech_rate_wpm": round(speech_rate_wpm, 2),
                "pause_count": pause_count,
                "pauses_per_min": round(pauses_per_min, 2),
                "avg_pause_s": round(avg_pause, 2),
                "median_pause_s": round(med_pause, 2),
                "total_pause_time_s": round(total_pause_time, 2),
            },
            "scores": {
                "speech_rate_score": round(sr_score, 3),
                "pauses_per_min_score": round(ppm_score, 3),
                "avg_pause_score": round(ap_score, 3),
                "audio_score": round(audio_score, 3),
            },
            "transcript": text,
        })

    # ----- aggregate across files -----
    total_duration = sum(f["metrics"]["duration_s"] for f in per_file)
    total_words = sum(f["metrics"]["words_spoken"] for f in per_file)
    spoken_minutes = max(1e-9, total_duration / 60.0)
    agg_wpm = total_words / spoken_minutes

    agg_pause_count = sum(f["metrics"]["pause_count"] for f in per_file)
    agg_pauses_per_min = agg_pause_count / spoken_minutes
    agg_avg_pause = (sum(f["metrics"]["avg_pause_s"] for f in per_file) / len(per_file)) if per_file else 0.0

    agg_sr_score = _linear_band_score(agg_wpm, 110, 160, 80, 220)
    agg_ppm_score = _linear_band_score(agg_pauses_per_min, 5, 15, 0, 30)
    agg_ap_score = _linear_band_score(agg_avg_pause, 0.3, 1.2, 0.0, 2.5)
    agg_audio_score = (agg_sr_score + agg_ppm_score + agg_ap_score) / 3.0

    return {
        "files": [str(w) for w in wavs],
        "per_file": per_file,
        "aggregate": {
            "metrics": {
                "duration_s": round(total_duration, 2),
                "words_spoken": int(total_words),
                "speech_rate_wpm": round(agg_wpm, 2),
                "pause_count": int(agg_pause_count),
                "pauses_per_min": round(agg_pauses_per_min, 2),
                "avg_pause_s": round(agg_avg_pause, 2),
            },
            "scores": {
                "audio_score": round(agg_audio_score, 3),
                "speech_rate_score": round(agg_sr_score, 3),
                "pauses_per_min_score": round(agg_ppm_score, 3),
                "avg_pause_score": round(agg_ap_score, 3),
            },
        },
        "transcript": "\n".join(all_text).strip(),
    }

def print_grading_table(rows: list[dict], overall_1to6: float, title="=== Rubric (TOEFL/CEFR) ==="):
    col_names = ["Metric", "Raw", "CEFR", "New TOEFL (1–6)", "Old TOEFL (0–30)"]
    widths = [18, 18, 6, 17, 18]
    line = " | ".join(n.ljust(w) for n, w in zip(col_names, widths))
    print("\n" + title)
    print(line)
    print("-" * len(line))
    for r in rows:
        print(" | ".join([
            str(r["Metric"]).ljust(widths[0]),
            str(r["Raw"]).ljust(widths[1]),
            str(r["CEFR"]).ljust(widths[2]),
            f"{r['New TOEFL (1–6)']:.2f}".ljust(widths[3]),
            r["Old TOEFL (0–30)"].ljust(widths[4]),
        ]))
    print("-" * len(line))
    print(f"OVERALL (1–6): {overall_1to6:.2f}  | CEFR: {cefr_label(overall_1to6)} | Old TOEFL: {old_toefl_band(overall_1to6)}")



@dataclass
class RoundEval:
    question: str
    answer: str
    relevance_score: float
    thinking_seconds: float
    thinking_time_score: float
    grammar_errors: int
    grammar_score: float
    round_score: float

# -------- OpenAI calls --------
def ask(client: OpenAI, model: str, messages: List[Dict[str, str]], temperature=0.2, max_tokens=300) -> str:
    r = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return r.choices[0].message.content.strip()

def embed(client: OpenAI, text: str) -> List[float]:
    r = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return r.data[0].embedding

def score_grammar(client: OpenAI, judge_model: str, text: str) -> dict:
    """
    Ask the model to be a deterministic grammar counter.
    Returns: {"errors": int, "score": float in [0,1]}
    """
    sys = (
        "You are a strict grammar and spelling judge. "
        "Count only real grammar, spelling, and punctuation mistakes (not style). "
        "Return JSON with keys: errors (int), score (0..1). "
        "Score guideline: 1.0 = 0 errors, 0.8 = 1-2, 0.6 = 3-4, 0.4 = 5-6, 0.2 = 7-9, 0.0 = 10+."
    )
    user = f"Candidate response:\n{text}"
    r = client.chat.completions.create(
        model=judge_model,
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
        ],
    )
    try:
        data = json.loads(r.choices[0].message.content)
        # harden:
        errors = int(data.get("errors", 0))
        score = float(data.get("score", 1.0))
        return {"errors": max(0, errors), "score": clamp01(score)}
    except Exception:
        # fallback: no penalty if parsing fails
        return {"errors": 0, "score": 1.0}

def thinking_time_to_score(seconds: float) -> float:
    """
    Simple mapping:
      <1s        -> 0.4 (too fast / shallow)
      1-5s       -> 1.0 (crisp)
      5-15s      -> 0.9
      15-30s     -> 0.8
      30-60s     -> 0.6
      >60s       -> 0.3
    """
    if seconds < 1: return 0.4
    if seconds <= 5: return 1.0
    if seconds <= 15: return 0.9
    if seconds <= 30: return 0.8
    if seconds <= 60: return 0.6
    return 0.3

# -------- Main loop --------
def run_interview(topic: str, rounds: int, model: str, judge_model: str | None = None):
    judge_model = judge_model or model
    client = OpenAI()

    # seed system + initial instruction
    messages = [
        {"role": "system", "content": INTERVIEWER_SYS.replace("ROUNDS", str(rounds))},
        {"role": "user", "content": f"Topic: {topic}. Start the interview."},
    ]

    evals: List[RoundEval] = []

    for i in range(1, rounds + 1):
        # interviewer asks
        question = ask(client, model, messages)
        print(f"\nQ{i}: {question}")

        # time the human's thinking/typing
        t0 = time.time()
        answer = input("Your answer: ").strip()
        t1 = time.time()
        think_sec = t1 - t0

        # append to conversation
        messages.append({"role": "assistant", "content": question})
        messages.append({"role": "user", "content": answer})

        # --- per-round evaluation ---
        # relevance via embeddings
        try:
            q_vec = embed(client, question)
            a_vec = embed(client, answer)
            cos = cosine_sim(q_vec, a_vec)
            # map cosine [-1,1] -> [0,1]
            relevance = clamp01((cos + 1.0) / 2.0)
        except Exception:
            relevance = 0.0

        # grammar via judge_model
        g = score_grammar(client, judge_model, answer)
        grammar_errors = g["errors"]
        grammar_score = g["score"]

        # thinking time score
        think_score = thinking_time_to_score(think_sec)

        round_score = (relevance + grammar_score + think_score) / 3.0

        ev = RoundEval(
            question=question,
            answer=answer,
            relevance_score=relevance,
            thinking_seconds=think_sec,
            thinking_time_score=think_score,
            grammar_errors=grammar_errors,
            grammar_score=grammar_score,
            round_score=round_score,
        )
        evals.append(ev)

        # show round summary
        print(
            f"  ↳ Relevance: {relevance:.2f} | Grammar: {grammar_score:.2f} (errors: {grammar_errors}) "
            f"| Thinking: {think_sec:.1f}s → {think_score:.2f} | Round score: {round_score:.2f}"
        )

    # interviewer closes
    messages.append({"role": "assistant", "content": "[END] Thanks for the discussion."})
    print("\n[END] Thanks for the discussion.\n")

    # --- final report ---
    rel_avg = stats.mean(e.relevance_score for e in evals)
    gram_avg = stats.mean(e.grammar_score for e in evals)
    think_avg = stats.mean(e.thinking_time_score for e in evals)
    final_score = (rel_avg + gram_avg + think_avg) / 3.0

    print("=== Round-by-round ===")
    for i, e in enumerate(evals, 1):
        print(
            f"R{i}: score {e.round_score:.2f}  "
            f"(rel {e.relevance_score:.2f}, gram {e.grammar_score:.2f}, think {e.thinking_time_score:.2f} [{e.thinking_seconds:.1f}s])"
        )

    print("\n=== Averages ===")
    print(f"Relevance avg:     {rel_avg:.2f}")
    print(f"Grammar avg:       {gram_avg:.2f}")
    print(f"Thinking time avg: {think_avg:.2f}")
    print(f"\nFINAL SCORE:       {final_score:.2f}")

    # optional: save JSON
    out = {
        "topic": topic,
        "rounds": rounds,
        "model": model,
        "judge_model": judge_model,
        "per_round": [asdict(e) for e in evals],
        "averages": {
            "relevance": rel_avg,
            "grammar": gram_avg,
            "thinking_time": think_avg,
            "final_score": final_score,
        },
    }
    # --- attach grading table to interview report (text-only subscores) ---
    rows_intv, overall6_intv = build_interview_grading_table({
        "relevance": rel_avg,
        "grammar": gram_avg,
        "thinking_time": think_avg,
    })
    out["grading"] = {
        "rows": rows_intv,
        "overall_new_toefl_1to6": overall6_intv,
        "overall_cefr": cefr_label(overall6_intv),
        "overall_old_toefl_0to30": old_toefl_band(overall6_intv),
    }

    with open("interview_report.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved: interview_report.json")

    # optional: save JSON
    out = {
        "topic": topic,
        "rounds": rounds,
        "model": model,
        "judge_model": judge_model,
        "per_round": [asdict(e) for e in evals],
        "averages": {
            "relevance": rel_avg,
            "grammar": gram_avg,
            "thinking_time": think_avg,
            "final_score_text_only": final_score,
        },
    }

    # If user provided an audio folder via CLI env/arg, analyze it
    audio_dir = os.environ.get("INTERVIEW_AUDIO_DIR")  # or wire via argparse, see below
    if audio_dir and os.path.isdir(audio_dir):
        print("\nTranscribing + analyzing audio…")
        audio_result = transcribe_and_analyze_audio_folder(client, audio_dir)

        m = audio_result["aggregate"]["metrics"]   # FIXED path
        s = audio_result["aggregate"]["scores"]    # FIXED path

        print("\n=== Speech Metrics ===")
        print(f"Duration: {m['duration_s']} s")
        print(f"Speech rate (spoken): {m['speech_rate_wpm']} wpm")
        print(f"Words spoken: {m['words_spoken']}")

        print("\n=== Pauses ===")
        print(f"Pause count: {m['pause_count']} | Pauses/min: {m['pauses_per_min']}")
        print(f"Avg pause: {m['avg_pause_s']} s | Median pause: {m['median_pause_s']} s")
        # total_pause_time_s is per file; aggregate shows counts + avg pause

        # Merge audio into report and compute overall score
        out["audio"] = audio_result
        final_score_overall = (final_score + s["audio_score"]) / 2.0
        out["averages"]["final_score_audio_only"] = s["audio_score"]
        out["averages"]["final_score_overall"] = final_score_overall
        print(f"\nAUDIO SCORE: {s['audio_score']:.2f}")
        print(f"OVERALL SCORE (text+audio): {final_score_overall:.2f}")

        # --- grading for audio aggregates (and print) ---
        rows_audio, overall6_audio = build_audio_grading_table(m)
        out.setdefault("grading_audio", {})
        out["grading_audio"].update({
            "rows": rows_audio,
            "overall_new_toefl_1to6": overall6_audio,
            "overall_cefr": cefr_label(overall6_audio),
            "overall_old_toefl_0to30": old_toefl_band(overall6_audio),
        })
        print_grading_table(rows_audio, overall6_audio, title="=== Rubric (Audio) ===")


    # grading for audio aggregates
    agg_m = audio_result["aggregate"]["metrics"]
    rows_audio, overall6_audio = build_audio_grading_table(agg_m)
    out.setdefault("grading_audio", {})
    out["grading_audio"].update({
        "rows": rows_audio,
        "overall_new_toefl_1to6": overall6_audio,
        "overall_cefr": cefr_label(overall6_audio),
        "overall_old_toefl_0to30": old_toefl_band(overall6_audio),
    })
  
    with open("interview_report.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved: interview_report.json")


# -------- CLI --------
def main():
    ap = argparse.ArgumentParser(description="AI interviewer + audio analysis")
    ap.add_argument("--topic", required=False, help="Interview topic/focus")
    ap.add_argument("--rounds", type=int, default=0,
                    help="How many interview rounds (0 = skip interview)")
    ap.add_argument("--model", default=DEFAULT_CHAT_MODEL, help="chat model for questions")
    ap.add_argument("--judge-model", default=None, help="model to judge grammar (defaults to --model)")
    ap.add_argument("--audio-dir", default=None, help="Path to folder of .wav files for analysis")
    args = ap.parse_args()

    client = OpenAI()

    # if interview requested
    if args.rounds > 0 and args.topic:
        run_interview(args.topic, args.rounds, args.model, args.judge_model)

    # if audio analysis requested
    if args.audio_dir:
        print("\n[Audio-only analysis]")
        audio_result = transcribe_and_analyze_audio_folder(client, args.audio_dir)

        # per-file
        print("\n=== Per-file scores ===")
        for f in audio_result["per_file"]:
            print(f"- {os.path.basename(f['file'])}: score {f['scores']['audio_score']:.2f}  "
                f"(wpm {f['metrics']['speech_rate_wpm']}, pauses/min {f['metrics']['pauses_per_min']})")

        # aggregate
        agg_m = audio_result["aggregate"]["metrics"]
        agg_s = audio_result["aggregate"]["scores"]

        # --- print rubric to console ---
        rows_audio, overall6_audio = build_audio_grading_table(agg_m)
        print_grading_table(rows_audio, overall6_audio)

        # --- attach grading to audio-only JSON ---
        audio_result["grading"] = {
            "rows": rows_audio,
            "overall_new_toefl_1to6": overall6_audio,
            "overall_cefr": cefr_label(overall6_audio),
            "overall_old_toefl_0to30": old_toefl_band(overall6_audio),
        }


        print("\n=== Aggregate ===")
        print(f"Duration: {agg_m['duration_s']} s | Words: {agg_m['words_spoken']} | WPM: {agg_m['speech_rate_wpm']}")
        print(f"Pauses/min: {agg_m['pauses_per_min']} | Avg pause: {agg_m['avg_pause_s']} s")
        print(f"\nAVERAGE AUDIO SCORE: {agg_s['audio_score']:.2f}")

        # === attach grading to audio-only report ===
        rows_audio, overall6_audio = build_audio_grading_table(agg_m)
        audio_result["grading"] = {
            "rows": rows_audio,
            "overall_new_toefl_1to6": overall6_audio,
            "overall_cefr": cefr_label(overall6_audio),
            "overall_old_toefl_0to30": old_toefl_band(overall6_audio),
        }

        with open("audio_report.json", "w") as f:
            json.dump(audio_result, f, indent=2)
        print("\nSaved: audio_report.json")



if __name__ == "__main__":
    main()

