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
    ap.add_argument("--text", required=True)
    ap.add_argument("--pause-threshold", type=float, default=0.35)
    args = ap.parse_args()

    metrics, trans, gpt_eval = run_pipeline(args.audio, args.text, pause_threshold=args.pause_threshold)

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

if __name__ == "__main__":
    main()
