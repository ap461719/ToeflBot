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
    from jiwer import compute_measures
    ref, hyp = normalize_text(reference_text), normalize_text(hypothesis_text)
    m = compute_measures(ref, hyp)

    r, h = m["reference length"], m["hypothesis length"]
    ins, dels, subs = m["insertions"], m["deletions"], m["substitutions"]
    hits = max(0, r - dels - subs)

    recall = hits / r if r > 0 else 0.0
    precision = hits / h if h > 0 else 0.0
    f1 = (2*precision*recall / (precision+recall)) if (precision+recall) > 0 else 0.0

    return {
        "wer": m["wer"], "precision": precision, "recall": recall, "f1": f1,
        "insertions": ins, "deletions": dels, "substitutions": subs, "hits": hits,
        "ref_len": r, "hyp_len": h,
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

def gpt4o_pronunciation_eval(reference_text: str, audio_path: str, feats: Dict[str, float], model="gpt-4o-mini-transcribe") -> Dict:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    with open(audio_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    ext = os.path.splitext(audio_path)[1].lstrip(".") or "wav"

    system_msg = (
        "You are a pronunciation examiner. Given reference text, audio, and acoustic features, "
        "return JSON with mispronunciation_accuracy, vowel_accuracy, mispronounced_words[], notes."
    )
    user_msg = "REFERENCE:\n" + reference_text + "\n\nFEATURES:\n" + json.dumps(feats, indent=2)

    resp = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": [
                {"type": "input_text", "text": user_msg},
                {"type": "input_audio", "input_audio": {"data": b64, "format": ext}}
            ]}
        ],
        temperature=0.0,
    )
    return json.loads(resp.choices[0].message.content)

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
