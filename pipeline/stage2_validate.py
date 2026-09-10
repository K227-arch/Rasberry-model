"""
Stage 2 — Validate extracted pairs.
Checks:
  - Min/max token length (2–200 tokens, split by whitespace)
  - Character ratio: min(len(src),len(tgt)) / max(len(src),len(tgt)) >= 0.15
    (looser than v1's 0.4 — these are full sentences with different structure)
  - Max char length multiplier <= 6.0
  - Unicode NFKC normalization
  - Deduplication (exact runyoro+english pair)
  - Skip pairs where runyoro == english (untranslated)
Output:
  pipeline_data/stage2_valid.tsv
  pipeline_data/stage2_rejected.tsv
  pipeline_data/stage2_report.txt
"""
import os, csv, unicodedata

IN_FILE  = os.path.join(os.path.dirname(__file__), "..", "pipeline_data", "stage1_raw.tsv")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "pipeline_data")
VALID_F  = os.path.join(OUT_DIR, "stage2_valid.tsv")
REJECT_F = os.path.join(OUT_DIR, "stage2_rejected.tsv")
REPORT_F = os.path.join(OUT_DIR, "stage2_report.txt")

MIN_TOKENS = 2
MAX_TOKENS = 200
MIN_CHAR_RATIO = 0.15
MAX_CHAR_MULT  = 6.0

def normalize(text):
    return unicodedata.normalize("NFKC", text).strip()

seen = set()
valid = []
rejected = []
reasons = {}

with open(IN_FILE, encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        rny = normalize(row["runyoro_rutooro"])
        eng = normalize(row["english"])
        src = row["source_file"]

        def reject(reason):
            rejected.append({"runyoro_rutooro": rny, "english": eng,
                              "source_file": src, "reason": reason})
            reasons[reason] = reasons.get(reason, 0) + 1

        # Dedup
        key = (rny.lower(), eng.lower())
        if key in seen:
            reject("Duplicate")
            continue
        seen.add(key)

        # Untranslated
        if rny.lower().strip(".,!? ") == eng.lower().strip(".,!? "):
            reject("Untranslated (src==tgt)")
            continue

        # Token length
        rny_toks = len(rny.split())
        eng_toks = len(eng.split())
        if rny_toks < MIN_TOKENS or eng_toks < MIN_TOKENS:
            reject(f"Too short (rny={rny_toks} eng={eng_toks})")
            continue
        if rny_toks > MAX_TOKENS or eng_toks > MAX_TOKENS:
            reject(f"Too long (rny={rny_toks} eng={eng_toks})")
            continue

        # Char ratio
        r_len, e_len = len(rny), len(eng)
        ratio = min(r_len, e_len) / max(r_len, e_len)
        if ratio < MIN_CHAR_RATIO:
            reject(f"Char ratio too low ({ratio:.2f})")
            continue
        mult = max(r_len, e_len) / min(r_len, e_len)
        if mult > MAX_CHAR_MULT:
            reject(f"Char mult too high ({mult:.2f})")
            continue

        valid.append({"runyoro_rutooro": rny, "english": eng, "source_file": src})

# Write valid
with open(VALID_F, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["runyoro_rutooro", "english", "source_file"], delimiter="\t")
    writer.writeheader()
    writer.writerows(valid)

# Write rejected
with open(REJECT_F, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["runyoro_rutooro", "english", "source_file", "reason"], delimiter="\t")
    writer.writeheader()
    writer.writerows(rejected)

# Report
report = [
    "=== Stage 2 — Validation Report ===",
    f"Input pairs  : {len(valid) + len(rejected)}",
    f"Valid pairs  : {len(valid)}",
    f"Rejected     : {len(rejected)}",
    "",
    "Rejection breakdown:",
]
for r, c in sorted(reasons.items(), key=lambda x: -x[1]):
    report.append(f"  {r}: {c}")

report_str = "\n".join(report)
print(report_str)
with open(REPORT_F, "w", encoding="utf-8") as f:
    f.write(report_str)

print(f"\nValid  → {VALID_F}")
print(f"Rejected → {REJECT_F}")
