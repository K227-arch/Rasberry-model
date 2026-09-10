"""
Stage 4 — Merge new clean pairs with existing HF training data, dedup, then split 85/10/5.
- Downloads existing train/val/test from kathay/runyoro-rutooro-en-parallel
- Deduplicates new pairs against ALL existing pairs (train+val+test)
- Only truly new pairs go into the new splits
- No BT pairs added (per spec)
- Output: pipeline_data/train.jsonl, val.jsonl, test.jsonl
          pipeline_data/stage4_report.txt
"""
import os, csv, json, random, urllib.request

HF_TOKEN = os.environ.get("HF_TOKEN", "")
BASE_HF  = "https://huggingface.co/datasets/kathay/runyoro-rutooro-en-parallel/resolve/main/data"
OUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "pipeline_data")
IN_FILE  = os.path.join(OUT_DIR, "stage3_clean.tsv")
SEED     = 42

def hf_text(url):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {HF_TOKEN}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8")

# ── 1. Load existing dataset pairs ───────────────────────────────────────────
print("Fetching existing dataset from HF...")
existing_pairs = []
existing_keys  = set()

for split in ["train-00000-of-00001.jsonl", "validation-00000-of-00001.jsonl", "test-00000-of-00001.jsonl"]:
    try:
        data = hf_text(f"{BASE_HF}/{split}")
        for line in data.strip().splitlines():
            obj = json.loads(line)
            rny = obj.get("runyoro_rutooro", "").strip()
            eng = obj.get("english", "").strip()
            if rny and eng:
                existing_pairs.append({"runyoro_rutooro": rny, "english": eng})
                existing_keys.add((rny.lower(), eng.lower()))
    except Exception as e:
        print(f"  Warning fetching {split}: {e}")

print(f"Existing pairs loaded: {len(existing_pairs)}")

# ── 2. Load new cleaned pairs ─────────────────────────────────────────────────
new_pairs = []
with open(IN_FILE, encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        rny = row["runyoro_rutooro"].strip()
        eng = row["english"].strip()
        key = (rny.lower(), eng.lower())
        if key not in existing_keys:
            new_pairs.append({"runyoro_rutooro": rny, "english": eng})
            existing_keys.add(key)  # prevent intra-new dupes

print(f"New unique pairs (not in existing data): {len(new_pairs)}")

# ── 3. Split new pairs 85/10/5 ────────────────────────────────────────────────
random.seed(SEED)
random.shuffle(new_pairs)

n = len(new_pairs)
n_val  = max(1, int(n * 0.10))
n_test = max(1, int(n * 0.05))
n_train = n - n_val - n_test

train_new = new_pairs[:n_train]
val_new   = new_pairs[n_train:n_train + n_val]
test_new  = new_pairs[n_train + n_val:]

print(f"New split — train: {len(train_new)}  val: {len(val_new)}  test: {len(test_new)}")

# ── 4. Write combined splits (existing + new) ─────────────────────────────────
# For training we use ONLY new pairs (continue-training from checkpoint)
# so the model doesn't re-learn what it already knows
for split_name, pairs in [("train", train_new), ("val", val_new), ("test", test_new)]:
    out_path = os.path.join(OUT_DIR, f"{split_name}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"  Written {len(pairs)} pairs → {out_path}")

# Also write all_new.jsonl for reference
all_new_path = os.path.join(OUT_DIR, "all_new_pairs.jsonl")
with open(all_new_path, "w", encoding="utf-8") as f:
    for p in new_pairs:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")

report = "\n".join([
    "=== Stage 4 — Merge & Split Report ===",
    f"Existing HF pairs      : {len(existing_pairs)}",
    f"New extracted pairs    : {len(new_pairs)}",
    f"  Train                : {len(train_new)}",
    f"  Val                  : {len(val_new)}",
    f"  Test                 : {len(test_new)}",
    f"BT pairs               : 0 (excluded per spec)",
    f"Split ratio            : 85/10/5  seed={SEED}",
])
print(f"\n{report}")
with open(os.path.join(OUT_DIR, "stage4_report.txt"), "w") as f:
    f.write(report)
