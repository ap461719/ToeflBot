# Local Speaking Assessment Report

End-to-end **local** speech-analysis pipeline for two tasks:

1) **Listen & Repeat** – compare a student’s utterance to a reference prompt  
2) **Interview** – evaluate a student’s answer to an interviewer’s question

The tool downloads (or opens) audio, transcribes it with **faster-whisper**, and produces JSON reports with fluency, pronunciation, grammar, and task-specific metrics. Mispronunciations and discourse checks leverage **OpenAI GPT-4o**; interview relevance uses **OpenAI embeddings**.

---

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage (CLI)](#usage-cli)
  - [Listen & Repeat](#listen--repeat)
  - [Interview](#interview)
- [Programmatic Use](#programmatic-use)
- [Output (JSON) – What You Get](#output-json--what-you-get)
  - [Common fields](#common-fields)
  - [Interview-only additions](#interview-only-additions)
- [How It Works (brief)](#how-it-works-brief)
- [Configuration Knobs](#configuration-knobs)
- [Troubleshooting](#troubleshooting)
- [Security & Privacy](#security--privacy)
- [Roadmap / Ideas](#roadmap--ideas)
- [License](#license)
- [Quick Copy Commands](#quick-copy-commands)

---

## Features

- Works with **local files or URLs** for both prompt and student audio
- Transcription with **word-level timestamps** (faster-whisper)
- Metrics (per task, aggregated across pairs):
  - **Duration** and **speech rate** (wpm)
  - **Repeat accuracy** (WER) + **incorrect_segments**
  - **Pause stats** → `pause_frequency_level`, `pause_appropriateness_level`
  - **Pronunciation**: mispronounced words via GPT-4o audio + 0–100 accuracy score
  - **Grammar**: issues + score via GPT-4o text
- **Interview-only** extras:
  - **Relevance** (question ↔ answer) via **text embeddings** cosine similarity → CEFR label
  - **Discourse** (coherence/organization) via GPT-4o text → CEFR label
  - **Vocabulary** block (complexity + diversity proxies)
  - **Word repetition level**
  - Grammar formatted into `errors[{original_sentence, corrected_sentence, fdiff[]}]`
- Single class handles both tasks; choose with `--task listen_repeat` or `--task interview`
- JSON written to the path you pass in (`--out`), one file **per task**

---

## Requirements

- **Python** 3.9+ (3.10/3.11 recommended)
- **FFmpeg** installed and on PATH (required by pydub)
- A GPU is optional; CPU works. (For GPU, install CUDA-capable PyTorch; faster-whisper will detect it.)

### Python packages
pip install -U faster-whisper pydub jiwer requests openai python-dotenv

## Installation
- Ensure FFmpeg is installed:
macOS (Homebrew): brew install ffmpeg

## Configuration
- Create a .env file in the project root with your OpenAI API key:
- OPENAI_API_KEY=sk-...
- The script reads this automatically via python-dotenv. You can also pass api_key to the class directly if using the API.

### Usage (CLI)

## Listen & Repeat
- python local_listen_repeat.py \
-  --task listen_repeat \
-  --out out/listen_repeat_report.json \
- --pairs data/p01_prompt.wav:data/p01_student.wav \
-          data/p02_prompt.wav:data/p02_student.wav


## Interview 

- python local_listen_repeat.py \
-  --task interview \
-  --out out/interview_report.json \
-  --pairs https://example.com/q1.wav:https://example.com/a1.wav\
-         https://example.com/q2.wav:https://example.com/a2.wav


- --pairs accepts one or more items, each in the form    prompt:student
- Each side can be a local path or an HTTPS URL
- The output directory is created if it doesn’t exist

## Programmatic Use

from speaking_report import LocalSpeakingAssessmentReport, ListenRepeatPair

# Listen & Repeat
lr_pairs = [
    ListenRepeatPair("data/p01_prompt.wav", "data/p01_student.wav"),
    ListenRepeatPair("data/p02_prompt.wav", "data/p02_student.wav"),
]
lr = LocalSpeakingAssessmentReport(task="listen_repeat")
lr.generate_report(lr_pairs, out_path="out/listen_report.json")

# Interview
int_pairs = [
    ListenRepeatPair("data/q1.wav", "data/a1.wav"),
    ListenRepeatPair("data/q2.wav", "data/a2.wav"),
]
interview = LocalSpeakingAssessmentReport(task="interview")
interview.generate_report(int_pairs, out_path="out/interview_report.json")

{
  "version": "1.0",
  "generation_failed": false,
  "errors": [
    // Any pair-level errors (download, GPT, etc.), if present
  ],
  "overall_score": { "cefr": "B1", "toefl_score": "4", "old_toefl_score": "23" },
  "speech_rate": 123,
  "duration": "02:37",
  "repeat_accuracy": { "score": 76 },

  "incorrect_segments": ["..."],          // prompt spans the student missed/changed
  "mispronounced_words": [{"word": "temperature"}],

  "fluency": {
    "speech_rate_level": "B1",
    "coherence_level": "B1",              // proxy from repeat accuracy
    "pause_frequency_level": "B2",
    "pause_appropriateness_level": "A2",
    "repeat_accuracy_level": "B1",
    "description": "Speech is understandable ...",
    "description_cn": "整体可理解 ..."
  },

  "pronunciation": {
    "prosody_rhythm_level": "B1",
    "vowel_fullness_level": "B1",
    "intonation_level": "B1",
    "accuracy_score": 92,
    "description": "Pronunciation is generally intelligible ...",
    "description_cn": "发音整体清晰 ..."
  },

  "grammar": {
    "accuracy_level": "B1",
    "repeat_accuracy_level": "B1",
    "issues": [
      { "span": "there is many data", "explanation": "Agreement ...", "suggestion": "there are many data" }
    ]
  }
}

## Interview-only additions

{
  "relevance": { "score": "B2" },     // embeddings (question vs answer)
  "discourse": { "score": "B1" },     // GPT-4o coherence label

  "vocabulary": {
    "complexity_level": "B1",
    "diversity_level": "B2",
    "description": "Lexical complexity and diversity ...",
    "description_cn": "..."
  },

  "fluency": { "word_repetition_level": "B2" },  // added to fluency

  "grammar": {
    "accuracy_level": "B1",
    "errors": [
      {
        "original_sentence": "there is many data",
        "corrected_sentence": "there are many data",
        "fdiff": [
          {
            "has_error": true,
            "orig": "there is",
            "corr": "there are",
            "error_type_description": "Subject–verb agreement",
            "feedback": "Use plural verb with 'data'.",
            "feedback_cn": ""
          }
        ]
      }
    ],
    "description": ""
  }
}


## How It Works (brief)

1. **Load audio** (path or URL).
2. **Transcribe** with `faster-whisper` (`word_timestamps=True`).
3. **Compute**
   - **WER** (via `jiwer`) → repeat accuracy; collect `incorrect_segments` via diff.
   - **Pause metrics** from word-time gaps (≥ **0.30 s** pause; ≥ **1.0 s** long pause).
   - **Speech rate** = words per minute (aggregate duration).
4. **Mispronunciations** – GPT-4o **audio** compares prompt audio + text vs. student audio + text and returns unique words.
5. **Grammar** – GPT-4o **text** returns `{"issues": [...], "score": <0–100>}` (JSON enforced).
6. **Interview extras**
   - **Relevance** – embed each `(question, answer)` with `text-embedding-3-small`; cosine similarity → CEFR label (conservative **minimum** across pairs).
   - **Discourse** – GPT-4o text returns a CEFR label from the full answer transcript.
   - **Vocabulary** – proxies: average word length + type/token ratio.
   - **Word repetition** – repeated token rate → CEFR band.
7. **Combine** everything into the final JSON and write to `--out`.
