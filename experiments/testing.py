
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Speech & Repetition Metrics from Audio + Reference Text

This script computes:
  1) Duration (seconds) and speech rate (words per minute)
  2) Repetition accuracy (precision/recall/F1 vs. the provided reference text)
  3) Pause statistics (#pauses, pauses/min, avg/median pause length)

It prefers word-level timestamps (WhisperX). If unavailable, it falls back
to segment-level timestamps (Whisper or faster-whisper).

Usage:
  python speech_metrics.py --audio demo.wav --text "Hello world ..." \
      --model whisperx --pause-threshold 0.35

Recommended installs:
  pip install torch --index-url https://download.pytorch.org/whl/cu121   # if you have CUDA 12.1
  pip install whisperx jiwer librosa soundfile faster-whisper
"""

import argparse
import math
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

# -------------- Utilities --------------
def normalize_text(s: str) -> str:
    """Lowercase, remove most punctuation, collapse whitespace.
    Keep apostrophes to help with contractions."""
    s = s.lower()
    s = re.sub(r"[^\w\s']", " ", s)   # remove punctuation except apostrophes
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize_words(s: str) -> List[str]:
    s = normalize_text(s)
    if not s:
        return []
    return s.split()

@dataclass
class WordStamp:
    word: str
    start: float
    end: float

@dataclass
class Transcription:
    text: str
    duration_s: float
    segments: List[Tuple[float, float, str]]  # (start, end, text)
    words: Optional[List[WordStamp]] = None

# -------------- Transcription --------------
def transcribe_with_whisperx(audio_path: str, device: Optional[str] = None) -> Transcription:
    """Transcribe & align with WhisperX for word-level timestamps.
    Requires: whisperx, torch, soundfile or torchaudio."""
    import torch
    import whisperx

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load audio and duration
    import soundfile as sf
    with sf.SoundFile(audio_path) as f:
        duration_s = len(f) / f.samplerate

    # Transcribe (ASR)
    model = whisperx.load_model("small", device)
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio)
    # Align to get word timestamps
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    aligned = whisperx.align(result["segments"], model_a, metadata, audio, device)

    segments = [(seg["start"], seg["end"], seg["text"]) for seg in aligned["segments"]]
    words = []
    for seg in aligned["segments"]:
        for w in seg.get("words", []):
            # whisperx may include punctuation tokens; filter to alnum words+apostrophes
            if "word" in w:
                token = re.sub(r"[^\w\s']", "", w["word"].lower()).strip()
                if token:
                    words.append(WordStamp(token, float(w["start"]), float(w["end"])))

    full_text = " ".join([t for _, _, t in segments]).strip()
    return Transcription(text=full_text, duration_s=duration_s, segments=segments, words=words)

def transcribe_with_faster_whisper(audio_path: str, device: Optional[str] = None) -> Transcription:
    """Transcribe with faster-whisper. It can produce word timestamps if enable_word_timestamps=True.
    Requires: faster-whisper, soundfile."""
    from faster_whisper import WhisperModel
    import soundfile as sf

    with sf.SoundFile(audio_path) as f:
        duration_s = len(f) / f.samplerate

    model = WhisperModel("small", device=device or "auto")
    segments, info = model.transcribe(audio_path, word_timestamps=True)

    segs = []
    words: List[WordStamp] = []
    full_text_parts = []
    for seg in segments:
        segs.append((seg.start, seg.end, seg.text))
        full_text_parts.append(seg.text)
        if seg.words:
            for w in seg.words:
                token = re.sub(r"[^\w\s']", "", w.word.lower()).strip()
                if token:
                    words.append(WordStamp(token, float(w.start), float(w.end)))

    full_text = " ".join(full_text_parts).strip()
    return Transcription(text=full_text, duration_s=duration_s, segments=segs, words=words or None)

# -------------- Metrics --------------
def compute_duration_and_rate(trans: Transcription, reference_text: Optional[str]) -> Dict[str, float]:
    # Speech rate can be computed on spoken words (ASR tokens) or reference words.
    # We'll report both.
    spoken_words = tokenize_words(trans.text)
    ref_words = tokenize_words(reference_text or "")

    minutes = max(trans.duration_s / 60.0, 1e-9)
    wpm_spoken = len(spoken_words) / minutes if spoken_words else 0.0
    wpm_reference = len(ref_words) / minutes if ref_words else 0.0
    return {
        "duration_s": trans.duration_s,
        "words_spoken": float(len(spoken_words)),
        "words_in_reference": float(len(ref_words)),
        "speech_rate_wpm_spoken": wpm_spoken,
        "speech_rate_wpm_reference": wpm_reference,
    }

def align_and_repetition_metrics(reference_text: str, hypothesis_text: str) -> Dict[str, float]:
    """Use jiwer to compute alignment-based metrics.
    Our "repetition accuracy" focuses on how much of the reference was covered (recall).
      - recall = 1 - deletion_rate
      - precision ~ 1 - insertion_rate (extra words not in reference)
      - F1 as harmonic mean of precision/recall"""
    from jiwer import compute_measures

    ref = normalize_text(reference_text)
    hyp = normalize_text(hypothesis_text)
    measures = compute_measures(ref, hyp)  # dict with insertions, deletions, substitutions, hits, etc.

    # Words counts
    r = measures["reference length"]
    h = measures["hypothesis length"]
    ins = measures["insertions"]
    dels = measures["deletions"]
    subs = measures["substitutions"]
    hits = measures.get("hits", max(0, r - dels - subs))

    # Define precision/recall/F1 on word boundaries
    recall = (hits / r) if r > 0 else 0.0
    precision = (hits / h) if h > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "wer": measures["wer"],
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "insertions": float(ins),
        "deletions": float(dels),
        "substitutions": float(subs),
        "hits": float(hits),
        "ref_len": float(r),
        "hyp_len": float(h),
    }

def compute_pauses_from_words(words: List[WordStamp], pause_threshold: float = 0.35) -> Dict[str, float]:
    """Count pauses as gaps >= pause_threshold between consecutive words.
    Also compute mean/median pause length and pauses per minute."""
    if not words or len(words) < 2:
        return {
            "pause_count": 0.0,
            "pauses_per_min": 0.0,
            "avg_pause_s": 0.0,
            "median_pause_s": 0.0,
            "total_pause_time_s": 0.0,
        }
    gaps = []
    for i in range(1, len(words)):
        gap = max(0.0, words[i].start - words[i-1].end)
        if gap >= pause_threshold:
            gaps.append(gap)

    if not gaps:
        return {
            "pause_count": 0.0,
            "pauses_per_min": 0.0,
            "avg_pause_s": 0.0,
            "median_pause_s": 0.0,
            "total_pause_time_s": 0.0,
        }

    gaps_sorted = sorted(gaps)
    avg_gap = sum(gaps_sorted) / len(gaps_sorted)
    mid = len(gaps_sorted) // 2
    if len(gaps_sorted) % 2 == 0:
        median_gap = 0.5 * (gaps_sorted[mid-1] + gaps_sorted[mid])
    else:
        median_gap = gaps_sorted[mid]

    total_pause_time = sum(gaps_sorted)
    return {
        "pause_count": float(len(gaps_sorted)),
        "avg_pause_s": float(avg_gap),
        "median_pause_s": float(median_gap),
        "total_pause_time_s": float(total_pause_time),
    }

def compute_pauses_from_segments(segments: List[Tuple[float, float, str]], pause_threshold: float = 0.5) -> Dict[str, float]:
    """Fallback: Use inter-segment gaps as pauses."""
    if not segments or len(segments) < 2:
        return {
            "pause_count": 0.0,
            "pauses_per_min": 0.0,
            "avg_pause_s": 0.0,
            "median_pause_s": 0.0,
            "total_pause_time_s": 0.0,
        }
    # Sort by start time
    segments = sorted(segments, key=lambda x: x[0])
    gaps = []
    for i in range(1, len(segments)):
        prev_end = segments[i-1][1]
        cur_start = segments[i][0]
        gap = max(0.0, cur_start - prev_end)
        if gap >= pause_threshold:
            gaps.append(gap)
    if not gaps:
        return {
            "pause_count": 0.0,
            "pauses_per_min": 0.0,
            "avg_pause_s": 0.0,
            "median_pause_s": 0.0,
            "total_pause_time_s": 0.0,
        }
    gaps_sorted = sorted(gaps)
    avg_gap = sum(gaps_sorted) / len(gaps_sorted)
    mid = len(gaps_sorted) // 2
    if len(gaps_sorted) % 2 == 0:
        median_gap = 0.5 * (gaps_sorted[mid-1] + gaps_sorted[mid])
    else:
        median_gap = gaps_sorted[mid]
    total_pause_time = sum(gaps_sorted)
    return {
        "pause_count": float(len(gaps_sorted)),
        "avg_pause_s": float(avg_gap),
        "median_pause_s": float(median_gap),
        "total_pause_time_s": float(total_pause_time),
    }

# -------------- Main pipeline --------------
def run_pipeline(audio_path: str, reference_text: str, backend: str = "whisperx", pause_threshold: float = 0.35) -> Dict[str, float]:
    if backend.lower() == "whisperx":
        trans = transcribe_with_whisperx(audio_path)
    elif backend.lower() in ("faster-whisper", "faster"):
        trans = transcribe_with_faster_whisper(audio_path)
    else:
        raise ValueError("backend must be one of: whisperx, faster-whisper")

    metrics = {}

    # Duration & speech rate
    dr = compute_duration_and_rate(trans, reference_text)
    metrics.update(dr)

    # Repetition accuracy (alignment-based)
    rep = align_and_repetition_metrics(reference_text, trans.text)
    metrics.update({f"rep_{k}": v for k, v in rep.items()})

    # Pauses
    if trans.words:
        pauses = compute_pauses_from_words(trans.words, pause_threshold=pause_threshold)
    else:
        pauses = compute_pauses_from_segments(trans.segments, pause_threshold=max(0.5, pause_threshold))

    # fill pauses_per_min
    minutes = max(trans.duration_s / 60.0, 1e-9)
    pauses["pauses_per_min"] = (pauses["pause_count"] / minutes) if minutes > 0 else 0.0
    metrics.update({f"pause_{k}": v for k, v in pauses.items()})

    # Convenience: primary headline metrics
    metrics["headline_speech_rate_wpm"] = dr["speech_rate_wpm_spoken"]
    metrics["headline_repetition_recall"] = rep["recall"]
    metrics["headline_pause_count"] = pauses["pause_count"]

    return metrics, trans

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True, help="Path to audio file (wav/m4a/mp3).")
    ap.add_argument("--text", required=True, help="Reference text to be spoken.")
    ap.add_argument("--model", default="whisperx", choices=["whisperx", "faster-whisper"], help="ASR backend.")
    ap.add_argument("--pause-threshold", type=float, default=0.35, help="Seconds between words to count as a pause.")
    args = ap.parse_args()

    metrics, trans = run_pipeline(args.audio, args.text, backend=args.model, pause_threshold=args.pause_threshold)

    # Pretty print results
    print("\n=== Speech Metrics ===")
    print(f"Duration: {metrics['duration_s']:.2f} s")
    print(f"Speech rate (spoken): {metrics['speech_rate_wpm_spoken']:.1f} wpm")
    print(f"Speech rate (reference): {metrics['speech_rate_wpm_reference']:.1f} wpm")
    print(f"Words spoken: {int(metrics['words_spoken'])} | Words in reference: {int(metrics['words_in_reference'])}")

    print("\n=== Repetition Accuracy (vs. reference) ===")
    print(f"WER: {metrics['rep_wer']:.3f}")
    print(f"Precision: {metrics['rep_precision']:.3f} | Recall: {metrics['rep_recall']:.3f} | F1: {metrics['rep_f1']:.3f}")
    print(f"Insertions: {int(metrics['rep_insertions'])} | Deletions: {int(metrics['rep_deletions'])} | Substitutions: {int(metrics['rep_substitutions'])} | Hits: {int(metrics['rep_hits'])}")

    print("\n=== Pauses ===")
    print(f"Pause count: {int(metrics['pause_pause_count'])} | Pauses/min: {metrics['pause_pauses_per_min']:.2f}")
    print(f"Avg pause: {metrics['pause_avg_pause_s']:.2f} s | Median pause: {metrics['pause_median_pause_s']:.2f} s")
    print(f"Total pause time: {metrics['pause_total_pause_time_s']:.2f} s")

    print("\n=== Notes ===")
    if trans.words is None:
        print("* Word-level timestamps not available; pause stats use segment gaps (coarser).")
    else:
        print("* Word-level timestamps used for pause stats.")

if __name__ == "__main__":
    main()
