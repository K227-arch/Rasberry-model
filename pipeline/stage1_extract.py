"""
Stage 1 — Extract parallel pairs from all 18 raw Excel files.
Schema: col[6]=English, col[7]=Runyoro-Rutooro (0-indexed)
Skip rows where either cell is None/empty or col[0] is non-numeric (header row).
Output: data/raw_extracted.tsv  (runyoro TAB english)
"""
import os, csv, openpyxl

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "raw")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "pipeline_data")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FILE = os.path.join(OUT_DIR, "stage1_raw.tsv")

pairs = []
file_counts = {}

for fname in sorted(os.listdir(RAW_DIR)):
    if not fname.endswith(".xlsx"):
        continue
    path = os.path.join(RAW_DIR, fname)
    count = 0
    try:
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        ws = wb.active
        for row in ws.iter_rows(values_only=True):
            # Skip header rows (col[0] is not an integer ID)
            if not row or not isinstance(row[0], (int, float)):
                continue
            # Need at least 8 columns
            if len(row) < 8:
                continue
            eng = row[6]
            rny = row[7]
            # Skip if either is missing
            if not eng or not rny:
                continue
            eng = str(eng).strip()
            rny = str(rny).strip()
            if not eng or not rny:
                continue
            pairs.append((rny, eng, fname))
            count += 1
        wb.close()
        file_counts[fname] = count
    except Exception as e:
        print(f"  ERROR reading {fname}: {e}")
        file_counts[fname] = f"ERROR: {e}"

# Write output
with open(OUT_FILE, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(["runyoro_rutooro", "english", "source_file"])
    for rny, eng, src in pairs:
        writer.writerow([rny, eng, src])

print(f"\n=== Stage 1 — Extraction complete ===")
print(f"Total pairs extracted: {len(pairs)}")
print(f"\nPer-file counts:")
for fname, cnt in sorted(file_counts.items()):
    print(f"  {fname}: {cnt}")
print(f"\nOutput: {OUT_FILE}")
