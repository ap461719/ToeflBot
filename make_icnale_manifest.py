# make_icnale_manifest.py
import os, csv, re, sys

if len(sys.argv) != 2:
    raise SystemExit("Usage: python make_icnale_manifest.py /path/to/ICNALE_SM_Audio")

ROOT = os.path.abspath(sys.argv[1])  # e.g., /…/ICNALE_SM_Audio

# Map filename token -> topic string used by your pipeline
def infer_topic(name_upper: str) -> str:
    # PTJ = part-time job prompt
    if "PTJ" in name_upper:
        return "part-time employment"
    # SJT/SMK/SR appear in some releases for the smoking prompt
    if any(k in name_upper for k in ("SJT", "SMK", "SR")):
        return "smoking in restaurants"
    return "topic"

# Extract CEFR from tail of filename. Examples:
# ..._B1.mp3, ..._B1_1.mp3, ..._C1_2.wav, ..._N.m4a
LAB_RX = re.compile(r"_(A1|A2|B1|B2|C1|C2|N)(?:_[12])?(?:\.[A-Za-z0-9]+)?$", re.IGNORECASE)

rows = []
for dirpath, _, files in os.walk(ROOT):
    for fn in files:
        if fn.startswith("."):
            continue
        if not fn.lower().endswith((".wav", ".mp3", ".m4a")):
            continue

        up = fn.upper()
        m = LAB_RX.search(up)
        if not m:
            # If any files don’t match, skip them silently (or print a warning)
            # print("Skipping (no label found):", fn)
            continue
        lab = m.group(1).upper()  # CEFR (N = native)
        topic = infer_topic(up)

        rows.append({
            "audio_path": os.path.join(os.path.abspath(dirpath), fn),
            "cefr_label": lab,
            "topic": topic,
        })

# Stable order: by country folder then filename
rows.sort(key=lambda r: (os.path.dirname(r["audio_path"]), os.path.basename(r["audio_path"])))

out_csv = "icnale_manifest.csv"
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["audio_path","cefr_label","topic"])
    w.writeheader()
    w.writerows(rows)

print(f"Wrote {len(rows)} rows to {os.path.abspath(out_csv)}")
