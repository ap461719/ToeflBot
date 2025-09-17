#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Speech, Fluency & Pronunciation Metrics from Audio + Reference Text

This script computes:
  1) Duration (seconds) and speech rate (words per minute)
  2) Repetition accuracy (precision/recall/F1 vs. the provided reference text)
  3) Pause statistics (#pauses, pauses/min, avg/median pause length, total pause time)
  4) Pronunciation metrics using GPT-4o audio:
       - Mispronunciation accuracy
       - Vowel accuracy
       - Frequency (pitch stats)
       - Amplitude (loudness stats)

Requires:
  pip install openai torch whisperx faster-whisper jiwer librosa soundfile numpy pyloudnorm
"""

import argparse
import base64
import json
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
import soundfile as sf
import librosa

try:
    import pyloudnorm as pyln
except ImportError:
    pyln = None

from dotenv import load_dotenv
load_dotenv()
# ------------------ Utility functions ------------------

def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^\w\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize_words(s: str) -> List[str]:
    s = normalize_text(s)
    return s.split() if s else []

# add near the top
def load_reference_text(s: str) -> str:
    # If it looks like a file on disk, read it; otherwise treat as raw text
    try:
        if os.path.isfile(s):
            with open(s, "r", encoding="utf-8") as f:
                return f.read()
    except Exception:
        pass
    return s


@dataclass
class WordStamp:
    word: str
    start: float
    end: float

@dataclass
class Transcription:
    text: str
    duration_s: float
    segments: List[Tuple[float, float, str]]
    words: Optional[List[WordStamp]] = None

# ------------------ Transcription ------------------

def transcribe_with_whisperx(audio_path: str, device: Optional[str] = None) -> Transcription:
    import torch, whisperx
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    with sf.SoundFile(audio_path) as f:
        duration_s = len(f) / f.samplerate

    #model = whisperx.load_model("small", device)
    model = whisperx.load_model("small", device, compute_type="int8")
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio)

    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    aligned = whisperx.align(result["segments"], model_a, metadata, audio, device)

    segments = [(seg["start"], seg["end"], seg["text"]) for seg in aligned["segments"]]
    words = []
    for seg in aligned["segments"]:
        for w in seg.get("words", []):
            if "word" in w:
                token = re.sub(r"[^\w\s']", "", w["word"].lower()).strip()
                if token:
                    words.append(WordStamp(token, float(w["start"]), float(w["end"])))

    full_text = " ".join([t for _, _, t in segments]).strip()
    return Transcription(full_text, duration_s, segments, words)

# ------------------ Metrics: Fluency ------------------

def compute_duration_and_rate(trans: Transcription, reference_text: str) -> Dict[str, float]:
    spoken_words = tokenize_words(trans.text)
    ref_words = tokenize_words(reference_text)

    minutes = max(trans.duration_s / 60.0, 1e-9)
    return {
        "duration_s": trans.duration_s,
        "words_spoken": len(spoken_words),
        "words_in_reference": len(ref_words),
        "speech_rate_wpm_spoken": len(spoken_words) / minutes if spoken_words else 0.0,
        "speech_rate_wpm_reference": len(ref_words) / minutes if ref_words else 0.0,
    }

def align_and_repetition_metrics(reference_text: str, hypothesis_text: str) -> Dict[str, float]:
    """
    Robust to jiwer schema changes. We compute ref/hyp lengths from text,
    and use jiwer only for insertions/deletions/substitutions/hits/wer.
    """
    from jiwer import compute_measures

    # normalized strings + tokenized lengths
    ref_norm = normalize_text(reference_text)
    hyp_norm = normalize_text(hypothesis_text)
    ref_len = float(len(ref_norm.split()))
    hyp_len = float(len(hyp_norm.split()))

    m = compute_measures(ref_norm, hyp_norm)

    # pull counts with safe defaults
    ins = float(m.get("insertions", 0))
    dels = float(m.get("deletions", 0))
    subs = float(m.get("substitutions", 0))

    # hits may not exist; derive if needed
    hits = m.get("hits")
    if not isinstance(hits, (int, float)):
        hits = max(0.0, ref_len - dels - subs)
    else:
        hits = float(hits)

    # precision/recall/F1
    recall = (hits / ref_len) if ref_len > 0 else 0.0
    precision = (hits / hyp_len) if hyp_len > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    wer = float(m.get("wer", 0.0))

    return {
        "wer": wer,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "insertions": ins,
        "deletions": dels,
        "substitutions": subs,
        "hits": hits,
        "ref_len": ref_len,
        "hyp_len": hyp_len,
    }



def compute_pauses(words: Optional[List[WordStamp]], segments: List[Tuple[float, float, str]], duration_s: float, pause_threshold: float=0.35) -> Dict[str, float]:
    gaps = []
    if words and len(words) >= 2:
        for i in range(1, len(words)):
            gap = max(0.0, words[i].start - words[i-1].end)
            if gap >= pause_threshold: gaps.append(gap)
    else:
        segments = sorted(segments, key=lambda x: x[0])
        for i in range(1, len(segments)):
            gap = max(0.0, segments[i][0] - segments[i-1][1])
            if gap >= max(0.5, pause_threshold): gaps.append(gap)

    minutes = max(duration_s / 60.0, 1e-9)
    return {
        "pause_count": len(gaps),
        "pauses_per_min": len(gaps) / minutes,
        "avg_pause_s": float(np.mean(gaps)) if gaps else 0.0,
        "median_pause_s": float(np.median(gaps)) if gaps else 0.0,
        "total_pause_time_s": float(np.sum(gaps)) if gaps else 0.0,
    }

# ------------------ DSP: Pitch & Loudness ------------------

def analyze_pitch_and_loudness(audio_path: str, sr: int=16000) -> Dict[str, float]:
    y, _ = librosa.load(audio_path, sr=sr, mono=True)

    # Pitch
    f0, _, _ = librosa.pyin(y, fmin=65, fmax=400, sr=sr)
    voiced = f0[~np.isnan(f0)]
    pitch_stats = {
        "pitch_mean_hz": float(np.mean(voiced)) if voiced.size else 0.0,
        "pitch_range_hz": float(np.max(voiced)-np.min(voiced)) if voiced.size else 0.0,
    }

    # Loudness
    rms = librosa.feature.rms(y=y)[0]
    dbfs = 20*np.log10(np.maximum(rms, 1e-9))
    loud_stats = {
        "rms_dbfs_mean": float(np.mean(dbfs)),
        "rms_dbfs_max": float(np.max(dbfs)),
    }
    if pyln:
        meter = pyln.Meter(sr)
        loud_stats["lufs_integrated"] = float(meter.integrated_loudness(y.astype(np.float64)))

    pitch_stats.update(loud_stats)
    return pitch_stats

# ========= Grading helpers =========

# --- CEFR mapping (numeric → label) ---
def cefr_label(score_1to6: float) -> str:
    # Uses common cut points (with half steps)
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

# --- New (1–6) to Old TOEFL (0–30) band text ---
TOEFL_OLD_MAP = {
    6.0: "29–30",
    5.5: "27–28",
    5.0: "24–26",
    4.5: "22–23",
    4.0: "18–21",
    3.5: "12–17",
    3.0: "6–11",
    2.5: "4–5",
    2.0: "3",
    1.5: "2",
    1.0: "0–1",
}
def old_toefl_band(score_1to6: float) -> str:
    # round to nearest 0.5 for lookup
    step = round(score_1to6 * 2) / 2.0
    # clamp into [1.0, 6.0]
    step = max(1.0, min(6.0, step))
    return TOEFL_OLD_MAP.get(step, TOEFL_OLD_MAP[6.0])

# --- Component scorers → New TOEFL (1–6) ---
def score_speech_rate(wpm: float) -> float:
    # Ideal conversational English 110–160
    if 110 <= wpm <= 160: return 6.0
    if 90  <= wpm < 110 or 160 < wpm <= 180: return 5.0
    if 75  <= wpm <  90 or 180 < wpm <= 200: return 4.0
    if 60  <= wpm <  75 or 200 < wpm <= 220: return 3.5
    if 45  <= wpm <  60: return 3.0
    return 2.5

def score_duration(seconds: float) -> float:
    # Aiming for ~90–120s monologue
    if seconds >= 120: return 6.0
    if 90 <= seconds < 120: return 5.0
    if 60 <= seconds <  90: return 4.0
    if 40 <= seconds <  60: return 3.5
    if 20 <= seconds <  40: return 3.0
    return 2.5

def score_repeat_accuracy(f1: float) -> float:
    # f1 in [0,1]; stricter because this is the core of “listen & repeat”
    if f1 >= 0.95: return 6.0
    if f1 >= 0.90: return 5.5
    if f1 >= 0.85: return 5.0
    if f1 >= 0.80: return 4.5
    if f1 >= 0.75: return 4.0
    if f1 >= 0.65: return 3.5
    if f1 >= 0.55: return 3.0
    if f1 >= 0.45: return 2.5
    return 2.0

def score_pause_frequency(pauses_per_min: float) -> float:
    # Best when 5–15 pauses/min, decays outside
    if 5 <= pauses_per_min <= 15: return 6.0
    if 3 <= pauses_per_min < 5 or 15 < pauses_per_min <= 18: return 5.0
    if 2 <= pauses_per_min < 3 or 18 < pauses_per_min <= 22: return 4.0
    if 1 <= pauses_per_min < 2 or 22 < pauses_per_min <= 26: return 3.5
    if pauses_per_min < 1 or 26 < pauses_per_min <= 30: return 3.0
    return 2.5

# Optional pronunciation composite (if you want to use it in an overall score)
def score_pronunciation(mis_acc: float, vowel_acc: float) -> float:
    # Both inputs are 0..1; map their mean to 1..6
    mean = max(0.0, min(1.0, (mis_acc + vowel_acc) / 2.0))
    if mean >= 0.95: return 6.0
    if mean >= 0.90: return 5.5
    if mean >= 0.85: return 5.0
    if mean >= 0.75: return 4.5
    if mean >= 0.65: return 4.0
    if mean >= 0.55: return 3.5
    if mean >= 0.45: return 3.0
    if mean >= 0.35: return 2.5
    return 2.0

# --- Build table rows from your metrics dict ---
def build_grading_table(metrics: dict) -> list[dict]:
    rows = []

    # Speech rate
    sr = metrics["speech_rate_wpm_spoken"]
    sr_score = score_speech_rate(sr)
    rows.append({
        "Metric": "Speech rate",
        "Raw": f"{sr:.0f} wpm",
        "CEFR": cefr_label(sr_score),
        "New TOEFL (1–6)": sr_score,
        "Old TOEFL (0–30)": old_toefl_band(sr_score),
    })

    # Duration
    dur = metrics["duration_s"]
    dur_score = score_duration(dur)
    rows.append({
        "Metric": "Duration",
        "Raw": f"{dur:.0f}s",
        "CEFR": cefr_label(dur_score),
        "New TOEFL (1–6)": dur_score,
        "Old TOEFL (0–30)": old_toefl_band(dur_score),
    })

    # Repeat accuracy (use F1)
    f1 = metrics["rep_f1"]
    rep_score = score_repeat_accuracy(f1)
    rows.append({
        "Metric": "Repeat accuracy",
        "Raw": f"{f1*100:.0f}%",
        "CEFR": cefr_label(rep_score),
        "New TOEFL (1–6)": rep_score,
        "Old TOEFL (0–30)": old_toefl_band(rep_score),
    })

    # Pause frequency
    ppm = metrics["pause_pauses_per_min"]
    pause_score = score_pause_frequency(ppm)
    rows.append({
        "Metric": "Pause frequency",
        "Raw": f"{ppm:.0f} / min",
        "CEFR": cefr_label(pause_score),
        "New TOEFL (1–6)": pause_score,
        "Old TOEFL (0–30)": old_toefl_band(pause_score),
    })

    # Optional: pronunciation row (if you want it shown)
    if "mispronunciation_accuracy" in metrics and "vowel_accuracy" in metrics:
        pr_score = score_pronunciation(metrics["mispronunciation_accuracy"],
                                       metrics["vowel_accuracy"])
        raw_txt = f"mis:{metrics['mispronunciation_accuracy']:.2f}, vowel:{metrics['vowel_accuracy']:.2f}"
        rows.append({
            "Metric": "Pronunciation (model)",
            "Raw": raw_txt,
            "CEFR": cefr_label(pr_score),
            "New TOEFL (1–6)": pr_score,
            "Old TOEFL (0–30)": old_toefl_band(pr_score),
        })

    return rows

# --- Overall score (weighted) ---
def overall_new_toefl_score(rows: list[dict]) -> float:
    """
    Weighted blend of the four main rows:
      repeat 0.4, speech 0.3, pauses 0.2, duration 0.1
    (If pronunciation row exists, you can blend it in too.)
    """
    by_name = {r["Metric"]: r for r in rows}
    rep   = by_name["Repeat accuracy"]["New TOEFL (1–6)"]
    sr    = by_name["Speech rate"]["New TOEFL (1–6)"]
    pause = by_name["Pause frequency"]["New TOEFL (1–6)"]
    dur   = by_name["Duration"]["New TOEFL (1–6)"]
    composite = 0.4*rep + 0.3*sr + 0.2*pause + 0.1*dur
    return round(composite, 2)

def print_grading_table(rows: list[dict], overall: float):
    # neat text table
    col_names = ["Metric", "Raw", "CEFR", "New TOEFL (1–6)", "Old TOEFL (0–30)"]
    widths = [16, 18, 6, 17, 18]
    line = " | ".join(n.ljust(w) for n,w in zip(col_names, widths))
    print("\n" + line)
    print("-"*len(line))
    for r in rows:
        print(" | ".join([
            str(r["Metric"]).ljust(widths[0]),
            str(r["Raw"]).ljust(widths[1]),
            str(r["CEFR"]).ljust(widths[2]),
            f"{r['New TOEFL (1–6)']:.1f}".ljust(widths[3]),
            r["Old TOEFL (0–30)"].ljust(widths[4]),
        ]))
    print("-"*len(line))
    print(f"OVERALL (1–6): {overall:.1f}  | CEFR: {cefr_label(overall)} | Old TOEFL: {old_toefl_band(overall)}")


# ------------------ GPT-4o Audio Evaluation ------------------

def gpt4o_pronunciation_eval(
    reference_text: str,
    audio_path: str,
    feats: Dict[str, float],
    model: str = "gpt-4o-audio-preview-2025-06-03"  
) -> Dict:
    import base64, json, os
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # Read & base64 the audio
    with open(audio_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    # Choose a format the API recognizes
    ext = (os.path.splitext(audio_path)[1].lstrip(".") or "wav").lower()
    if ext == "m4a":
        fmt = "aac"
    elif ext in {"wav", "mp3", "aac", "flac", "ogg", "webm"}:
        fmt = ext
    else:
        fmt = "wav"

    instructions = (
        "You are a careful pronunciation examiner. "
        "Given reference text, the user's spoken audio, and acoustic features, "
        "respond ONLY in compact JSON with keys: "
        "mispronunciation_accuracy (0..1), vowel_accuracy (0..1), "
        "mispronounced_words (list of {word, reason}), notes (short string)."
    )
    user_payload = (
        "REFERENCE TEXT:\n" + reference_text.strip() + "\n\n"
        "ACOUSTIC FEATURES (json):\n" + json.dumps(feats, indent=2)
    )

    def _parse_output_to_json(resp):
        # Collect all text blocks from the response
        chunks = []
        for block in getattr(resp, "output", []) or []:
            for c in getattr(block, "content", []) or []:
                t = getattr(c, "type", None)
                if t in ("output_text", "text"):
                    chunks.append(getattr(c, "text", ""))
        text = "\n".join(chunks).strip()
        return json.loads(text) if text else {}

    # Try audio + text
    try:
        resp = client.responses.create(
            model=model,
            instructions=instructions,
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_payload},
                    {"type": "input_audio", "audio": {"data": b64, "format": fmt}}
                ]
            }],
        )
        data = _parse_output_to_json(resp)
        if not data:
            raise RuntimeError("Empty output_text from model.")
        return data

    except Exception as e_audio:
        # Fallback: text-only judging
        try:
            resp2 = client.responses.create(
                model="gpt-4o-mini",
                instructions=instructions,
                input=[{
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": user_payload + f"\n\nNOTE: Audio unavailable ({e_audio}). Score based on text+features only."}
                    ]
                }],
            )
            data2 = _parse_output_to_json(resp2) or {}
            data2.setdefault("mispronunciation_accuracy", 0.0)
            data2.setdefault("vowel_accuracy", 0.0)
            data2.setdefault("mispronounced_words", [])
            data2["notes"] = (data2.get("notes", "") + " (text-only fallback used)").strip()
            return data2
        except Exception as e_text:
            return {
                "mispronunciation_accuracy": 0.0,
                "vowel_accuracy": 0.0,
                "mispronounced_words": [],
                "notes": f"Audio scoring failed: {e_audio}. Text-only fallback also failed: {e_text}. Returned zeros so the run can complete."
            }




# ------------------ Pipeline ------------------

def run_pipeline(audio: str, text: str, model: str="whisperx", pause_threshold: float=0.35) -> Dict:
    trans = transcribe_with_whisperx(audio)  # only whisperx here for simplicity

    metrics = {}
    metrics.update(compute_duration_and_rate(trans, text))
    metrics.update({"rep_"+k:v for k,v in align_and_repetition_metrics(text, trans.text).items()})
    metrics.update({"pause_"+k:v for k,v in compute_pauses(trans.words, trans.segments, trans.duration_s, pause_threshold).items()})
    feats = analyze_pitch_and_loudness(audio)
    gpt_eval = gpt4o_pronunciation_eval(text, audio, feats)

    metrics.update(feats)
    metrics.update({
        "mispronunciation_accuracy": gpt_eval.get("mispronunciation_accuracy", 0.0),
        "vowel_accuracy": gpt_eval.get("vowel_accuracy", 0.0),
    })
    return metrics, trans, gpt_eval

# ------------------ Main ------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--text", required=True, help="Either the reference text itself OR a path to a .txt file")
    ap.add_argument("--pause-threshold", type=float, default=0.35)
    args = ap.parse_args()

    reference_text = load_reference_text(args.text)

    metrics, trans, gpt_eval = run_pipeline(args.audio, reference_text, pause_threshold=args.pause_threshold)

    out = {
        "audio": args.audio,
        "reference_text": reference_text,   # <- save the actual text, not the path
        "metrics": metrics,
        "gpt_pronunciation_eval": gpt_eval,
        "transcript": trans.text,
    }

    with open("listen_and_repeat_scores.json", "w") as f:
        json.dump(out, f, indent=2)

    print("\nSaved: listen_and_repeat_scores.json")

    print("\n=== Speech Metrics ===")
    print(f"Duration: {metrics['duration_s']:.2f} s")
    print(f"Speech rate (spoken): {metrics['speech_rate_wpm_spoken']:.1f} wpm")
    print(f"Speech rate (reference): {metrics['speech_rate_wpm_reference']:.1f} wpm")
    print(f"Words spoken: {metrics['words_spoken']} | Words in reference: {metrics['words_in_reference']}")

    print("\n=== Repetition Accuracy ===")
    print(f"WER: {metrics['rep_wer']:.3f}")
    print(f"Precision: {metrics['rep_precision']:.3f} | Recall: {metrics['rep_recall']:.3f} | F1: {metrics['rep_f1']:.3f}")

    print("\n=== Pauses ===")
    print(f"Pause count: {metrics['pause_pause_count']} | Pauses/min: {metrics['pause_pauses_per_min']:.2f}")
    print(f"Avg pause: {metrics['pause_avg_pause_s']:.2f} s | Median pause: {metrics['pause_median_pause_s']:.2f} s")
    print(f"Total pause time: {metrics['pause_total_pause_time_s']:.2f} s")

    print("\n=== Pronunciation Metrics (GPT-4o Audio) ===")
    print(f"Mispronunciation accuracy: {metrics['mispronunciation_accuracy']}")
    print(f"Vowel accuracy: {metrics['vowel_accuracy']}")
    print(f"Pitch mean: {metrics['pitch_mean_hz']:.1f} Hz | Range: {metrics['pitch_range_hz']:.1f} Hz")
    print(f"Loudness (RMS mean): {metrics['rms_dbfs_mean']:.1f} dB | Max: {metrics['rms_dbfs_max']:.1f} dB")
    if "lufs_integrated" in metrics:
        print(f"Integrated LUFS: {metrics['lufs_integrated']:.1f}")

    if gpt_eval.get("mispronounced_words"):
        print("Mispronounced words:")
        for item in gpt_eval["mispronounced_words"]:
            print(f"  - {item.get('word')}: {item.get('reason')}")
    if gpt_eval.get("notes"):
        print("Notes:", gpt_eval["notes"])
    
        # === Grading table (New TOEFL 1–6, CEFR, Old TOEFL 0–30) ===
    rows = build_grading_table(metrics)
    overall = overall_new_toefl_score(rows)
    print_grading_table(rows, overall)

    # Also save alongside your existing JSON
    grading_payload = {
        "rows": rows,
        "overall_new_toefl_1to6": overall,
        "overall_cefr": cefr_label(overall),
        "overall_old_toefl_0to30": old_toefl_band(overall),
    }
    with open("grading_table.json", "w") as f:
        json.dump(grading_payload, f, indent=2)
    print("\nSaved: grading_table.json")


if __name__ == "__main__":
    main()
