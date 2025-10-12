
"""
icnale_eval.py
--------------
Run your LocalSpeakingAssessmentReport on ICNALE-style audio and compute
the evaluation metrics used in "An effective automated speaking assessment
approach to mitigating data scarcity and imbalanced distribution" (2024).

Expected CSV input (one row per utterance):
    audio_path,cefr_label,topic

- audio_path: local path or URL to the WAV/MP3
- cefr_label: one of {A1,A2,B1,B2,C1,C2,N} (N = native). Use the subset your data has.
- topic: optional free text prompt (e.g., "smoking in restaurants"). If empty, we pass a dummy prompt.

Usage:
    python icnale_eval.py --csv data/icnale_manifest.csv --out results_icnale.json \
        --task interview \
        --model gpt-4o --audio_model gpt-4o-audio-preview

Notes:
- For ICNALE Spoken Monologues, there is no reference audio to "repeat".
  Use --task=interview so your pipeline skips repeat-comparison logic.
- Ensure OPENAI_API_KEY is set in your environment.
"""

import os, csv, json, argparse
from collections import defaultdict
from typing import Dict, List, Tuple

# import your class from the same directory (or adjust the path accordingly)
from local_listen_repeat_modified import LocalSpeakingAssessmentReport, ListenRepeatPair

# ----------------------- Label mapping -----------------------
# Map CEFR-like labels to ordinal indices for ordinal metrics.
# Adjust the ORDER to match the set/ordering you want to evaluate.
ORDER = ["A1","A2","B1","B2","C1","C2","N"]  # N = native speaker (optional)
INDEX = {lab:i for i,lab in enumerate(ORDER)}

def label_to_index(lab: str) -> int:
    lab = (lab or "").strip().upper()
    if lab not in INDEX:
        raise ValueError(f"Unknown label '{lab}'. Allowed: {list(INDEX)}")
    return INDEX[lab]

def index_to_label(i: int) -> str:
    return ORDER[i]

# ----------------------- Metrics -----------------------------

def accuracy(y_true: List[int], y_pred: List[int]) -> float:
    correct = sum(1 for t,p in zip(y_true, y_pred) if t == p)
    return 100.0 * correct / max(1,len(y_true))

def adjacent_accuracy(y_true: List[int], y_pred: List[int]) -> float:
    # within ±1 label counted as correct
    ok = 0
    for t,p in zip(y_true, y_pred):
        if abs(t - p) <= 1:
            ok += 1
    return 100.0 * ok / max(1,len(y_true))

def macro_accuracy(y_true: List[int], y_pred: List[int]) -> float:
    # mean per-class recall (unweighted)
    from collections import defaultdict
    by_class = defaultdict(lambda: {"tp":0,"tot":0})
    for t,p in zip(y_true, y_pred):
        by_class[t]["tot"] += 1
        if t == p:
            by_class[t]["tp"] += 1
    per = []
    for c,v in by_class.items():
        if v["tot"] > 0:
            per.append(v["tp"]/v["tot"])
    return 100.0 * (sum(per)/max(1,len(per)))

def rmse(y_true: List[int], y_pred: List[int]) -> float:
    # standard root mean squared error on ordinal indices
    import math
    se = [(t-p)**2 for t,p in zip(y_true, y_pred)]
    return math.sqrt(sum(se)/max(1,len(se)))

def macro_rmse(y_true: List[int], y_pred: List[int]) -> float:
    # average RMSE computed PER CLASS then averaged (macro)
    from collections import defaultdict
    err = defaultdict(list)
    for t,p in zip(y_true, y_pred):
        err[t].append((t-p)**2)
    import math
    rms = []
    for c, sq in err.items():
        if sq:
            rms.append(math.sqrt(sum(sq)/len(sq)))
    return sum(rms)/max(1,len(rms))

def pcc(y_true: List[int], y_pred: List[int]) -> float:
    # Pearson correlation coefficient between ordinal indices
    import statistics, math
    if len(y_true) < 2:
        return 0.0
    mean_t = statistics.mean(y_true)
    mean_p = statistics.mean(y_pred)
    num = sum((t-mean_t)*(p-mean_p) for t,p in zip(y_true, y_pred))
    den_t = math.sqrt(sum((t-mean_t)**2 for t in y_true))
    den_p = math.sqrt(sum((p-mean_p)**2 for p in y_pred))
    if den_t == 0 or den_p == 0:
        return 0.0
    return num/(den_t*den_p)

# ----------------------- Runner ------------------------------

def load_manifest(csv_path: str) -> List[Tuple[str,str,str]]:
    rows = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append((row.get("audio_path","").strip(),
                         row.get("cefr_label","").strip().upper(),
                         row.get("topic","").strip()))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Manifest CSV with columns: audio_path,cefr_label,topic")
    ap.add_argument("--out", required=True, help="Where to write JSON results + metrics")
    ap.add_argument("--task", default="interview", choices=["listen_repeat","interview"])
    ap.add_argument("--text_model", default="gpt-4o")
    ap.add_argument("--audio_model", default="gpt-4o-audio-preview")
    ap.add_argument("--embed_model", default="text-embedding-3-small")
    args = ap.parse_args()

    rows = load_manifest(args.csv)
    if not rows:
        raise SystemExit("No rows found in CSV.")

    # Initialize your pipeline
    assessor = LocalSpeakingAssessmentReport(
        task=args.task, text_model=args.text_model, audio_model=args.audio_model, embed_model=args.embed_model
    )

    y_true_idx: List[int] = []
    y_pred_idx: List[int] = []
    per_item: List[Dict] = []

    for i, (audio, lab, topic) in enumerate(rows, 1):
        # ICNALE has only a topic text (no reference audio). We pass the topic as the "prompt".
        # The student audio is the monologue itself.
        if not topic:
            topic = "Please give a one-minute opinion on the assigned topic."
        pair = ListenRepeatPair(prompt_audio=topic, student_audio=audio)  # prompt_audio here is a text placeholder
        # The LocalSpeakingAssessmentReport expects URLs/paths for audio, but our "prompt_audio"
        # is a text string. The 'interview' task ignores the prompt audio content and only uses text,
        # so this is safe. If your class tries to load it as audio, modify speaking_report.py to accept
        # text prompts when task='interview'.

        # Run assessment for this single item
        try:
            report = assessor.generate_report([pair], out_path=os.devnull)
        except Exception as e:
            # If one item fails, record and continue
            per_item.append({"audio_path": audio, "cefr_true": lab, "error": str(e)})
            continue

        pred_lab = (report.get("overall_score", {}) or {}).get("cefr", "A1")
        try:
            y_true_idx.append(label_to_index(lab))
            y_pred_idx.append(label_to_index(pred_lab))
        except Exception as e:
            per_item.append({"audio_path": audio, "cefr_true": lab, "cefr_pred": pred_lab, "error": str(e)})
            continue

        per_item.append({
            "audio_path": audio,
            "topic": topic,
            "cefr_true": lab,
            "cefr_pred": pred_lab,
            "speech_rate": report.get("speech_rate", None),
            "repeat_accuracy": (report.get("repeat_accuracy") or {}).get("score", None),
            "duration": report.get("duration", None),
        })

    # Compute metrics
    metrics = {
        "ACC": accuracy(y_true_idx, y_pred_idx),
        "ADJ": adjacent_accuracy(y_true_idx, y_pred_idx),
        "ACC_MC": macro_accuracy(y_true_idx, y_pred_idx),
        "RMSE": rmse(y_true_idx, y_pred_idx),
        "RMSE_MC": macro_rmse(y_true_idx, y_pred_idx),
        "PCC": pcc(y_true_idx, y_pred_idx),
        "support": len(y_true_idx),
        "label_order": ORDER,
    }

    out = {"items": per_item, "metrics": metrics}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
