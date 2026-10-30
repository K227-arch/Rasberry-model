"""
Inspect the actual training data used in kathay/runyoro-rutooro-en-parallel
"""
import urllib.request
import json

HF_TOKEN = "os.environ.get("HF_READ_TOKEN","")"
BASE = "https://huggingface.co/datasets/kathay/runyoro-rutooro-en-parallel/resolve/main"

def fetch(path):
    req = urllib.request.Request(f"{BASE}/{path}", headers={"Authorization": f"Bearer {HF_TOKEN}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode()

# ── augmentation report ───────────────────────────────────────────────────────
print("=== augmentation_report.md ===")
print(fetch("pipeline_reports/augmentation_report.md"))

# ── cleaning report ───────────────────────────────────────────────────────────
print("=== cleaning_report.md ===")
print(fetch("pipeline_reports/cleaning_report.md"))

# ── all_pairs.tsv — first 30 lines to see actual source tagging ──────────────
print("=== data/all_pairs.tsv (first 60 lines) ===")
tsv = fetch("data/all_pairs.tsv")
lines = tsv.splitlines()
print(f"Total lines (incl header): {len(lines)}")
for line in lines[:60]:
    print(line)

# ── train split — count and sample ───────────────────────────────────────────
print("\n=== data/train split (first 10 entries) ===")
train = fetch("data/train-00000-of-00001.jsonl")
train_lines = [json.loads(l) for l in train.strip().splitlines()]
print(f"Total train pairs: {len(train_lines)}")
# Show unique source files if tagged
sources = set()
for e in train_lines:
    if "source" in e:
        sources.add(e["source"])
    elif "source_file" in e:
        sources.add(e["source_file"])
if sources:
    print(f"Source files referenced: {sorted(sources)}")
for e in train_lines[:10]:
    print(e)
