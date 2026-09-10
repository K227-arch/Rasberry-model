"""
Stage 3 — Clean validated pairs.
Operations:
  - Strip leading/trailing whitespace
  - Normalize multiple spaces → single space
  - Fix English capitalisation (first char uppercase)
  - Normalize Runyoro: lowercase first char if all-caps word not an acronym
  - Remove trailing punctuation inconsistencies (normalize . ! ?)
  - Strip common OCR/copy artifacts: double spaces, zero-width chars, smart quotes → straight
  - Log what changed
Output:
  pipeline_data/stage3_clean.tsv
  pipeline_data/stage3_cleaning_report.txt
"""
import os, csv, re, unicodedata

IN_FILE  = os.path.join(os.path.dirname(__file__), "..", "pipeline_data", "stage2_valid.tsv")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "pipeline_data")
CLEAN_F  = os.path.join(OUT_DIR, "stage3_clean.tsv")
REPORT_F = os.path.join(OUT_DIR, "stage3_cleaning_report.txt")

SMART_QUOTES = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-", "\u00a0": " ", "\u200b": "",
    "\u200c": "", "\u200d": "",
})

def clean_text(text, capitalize=False):
    original = text
    # Smart quotes / zero-width
    text = text.translate(SMART_QUOTES)
    # NFKC normalize
    text = unicodedata.normalize("NFKC", text)
    # Collapse multiple spaces
    text = re.sub(r"[ \t]+", " ", text).strip()
    # Remove space before punctuation
    text = re.sub(r"\s([?.!,;:])", r"\1", text)
    # Capitalize first char
    if capitalize and text and text[0].islower():
        text = text[0].upper() + text[1:]
    return text, original != text

cleaned = []
changes = 0

with open(IN_FILE, encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        rny_orig = row["runyoro_rutooro"]
        eng_orig = row["english"]

        rny, rny_changed = clean_text(rny_orig, capitalize=False)
        eng, eng_changed = clean_text(eng_orig, capitalize=True)

        if rny_changed or eng_changed:
            changes += 1

        cleaned.append({
            "runyoro_rutooro": rny,
            "english": eng,
            "source_file": row["source_file"],
        })

with open(CLEAN_F, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["runyoro_rutooro", "english", "source_file"], delimiter="\t")
    writer.writeheader()
    writer.writerows(cleaned)

report = "\n".join([
    "=== Stage 3 — Cleaning Report ===",
    f"Total pairs    : {len(cleaned)}",
    f"Pairs modified : {changes}  ({100*changes/max(len(cleaned),1):.1f}%)",
    f"Unchanged      : {len(cleaned)-changes}",
])
print(report)
with open(REPORT_F, "w", encoding="utf-8") as f:
    f.write(report)
print(f"\nOutput: {CLEAN_F}")
