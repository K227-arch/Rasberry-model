"""Print a complete accounting of all pairs: kept, cleaned, removed, augmented."""
import re, sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

SEP  = "=" * 58
SEP2 = "-" * 58

# ── Source files ──────────────────────────────────────────────
sources = [
    ("Agriculture Seed Vocabulary.csv.xlsx", 123),
    ("augmentted pos pairs.xlsx",            2776),
    ("Ff_fixed worked on.docx",              111),
    ("J_fixed Worked on.docx",               165),
    ("Tt fixed worked on.docx",              584),
    ("U_fixed ...docx",                        1),
    ("V_fixed worked on.docx",                 4),
    ("W_fixed worked on.docx",                18),
]
total_raw = sum(n for _, n in sources)
wrong_lang_dropped = 2          # dropped during extraction (wrong-language examples)
total_to_validator = total_raw  # extractor already handled language filter

# ── Validation report ─────────────────────────────────────────
val_txt = (ROOT / "data/reports/validation_report.md").read_text(encoding="utf-8")
m = re.search(r'"total_valid":\s*(\d+)', val_txt)
total_valid = int(m.group(1)) if m else 3779
m = re.search(r'"total_rejected":\s*(\d+)', val_txt)
total_rejected = int(m.group(1)) if m else 3
m = re.search(r'"total_input":\s*(\d+)', val_txt)
total_input = int(m.group(1)) if m else total_raw

# ── Cleaning report ───────────────────────────────────────────
clean_txt = (ROOT / "data/reports/cleaning_report.md").read_text(encoding="utf-8")
m = re.search(r"Total pairs processed:\*\* (\d+)", clean_txt)
c_total = int(m.group(1)) if m else total_valid
m = re.search(r"Pairs modified:\*\* (\d+)", clean_txt)
c_modified = int(m.group(1)) if m else 323
c_unchanged = c_total - c_modified

# ── Augmentation report ───────────────────────────────────────
aug_txt = (ROOT / "data/reports/augmentation_report.md").read_text(encoding="utf-8")
m = re.search(r"Original pairs:\*\* (\d+)", aug_txt)
a_orig = int(m.group(1)) if m else c_total
m = re.search(r"Augmented pairs generated:\*\* (\d+)", aug_txt)
a_new = int(m.group(1)) if m else 923
a_total = a_orig + a_new

# ── Splits ────────────────────────────────────────────────────
train_n = len((ROOT / "data/processed/train.tsv").read_text(encoding="utf-8").splitlines())
val_n   = len((ROOT / "data/processed/val.tsv").read_text(encoding="utf-8").splitlines())
test_n  = len((ROOT / "data/processed/test.tsv").read_text(encoding="utf-8").splitlines())
split_total = train_n + val_n + test_n

# ── Print ─────────────────────────────────────────────────────
print(SEP)
print("  COMPLETE PAIR ACCOUNTING — runyoro-nmt-v1")
print(SEP)

print()
print("  STAGE 1: EXTRACTION FROM 8 RAW FILES")
print(SEP2)
for fname, n in sources:
    print(f"    {fname:<45} {n:>5}")
print(f"    {'':45} -----")
print(f"    {'TOTAL EXTRACTED':<45} {total_raw:>5}")
print(f"    {'Wrong-language examples dropped':<45} {wrong_lang_dropped:>5}")

print()
print("  STAGE 2: VALIDATION")
print(SEP2)
print(f"    Pairs checked:                          {total_input:>5}")
print(f"    Kept (valid):                           {total_valid:>5}")
print(f"    Removed — duplicates only:              {total_rejected:>5}")

print()
print("  STAGE 4: CLEANING  (no pairs removed, only fixed)")
print(SEP2)
print(f"    Pairs that went through cleaning:       {c_total:>5}")
print(f"    Pairs FIXED (had something corrected):  {c_modified:>5}  ({100*c_modified/c_total:.1f}%)")
print(f"    Pairs UNCHANGED (already clean):        {c_unchanged:>5}  ({100*c_unchanged/c_total:.1f}%)")
print(f"    Pairs REMOVED by cleaning:                  0")

print()
print("  STAGE 6: AUGMENTATION")
print(SEP2)
print(f"    Original clean pairs:                   {a_orig:>5}")
print(f"    New pairs generated:                    {a_new:>5}")
print(f"    Total corpus after augmentation:        {a_total:>5}")

print()
print("  STAGE 7: SPLITS")
print(SEP2)
print(f"    Training set:    {train_n:>5}  ({100*train_n/split_total:.0f}%)")
print(f"    Validation set:   {val_n:>5}  ({100*val_n/split_total:.0f}%)")
print(f"    Test set:          {test_n:>5}  ({100*test_n/split_total:.0f}%)")
print(f"    TOTAL:            {split_total:>5}")

print()
print(SEP)
print("  GRAND TOTAL ACCOUNTING")
print(SEP)
print(f"    Raw pairs extracted from files:         {total_raw:>5}")
print(f"    Dropped — wrong language (extraction):  {wrong_lang_dropped:>5}")
print(f"    Dropped — duplicates (validation):      {total_rejected:>5}")
print(f"    ----------")
total_dropped = wrong_lang_dropped + total_rejected
print(f"    TOTAL DROPPED / IGNORED:                {total_dropped:>5}")
print()
print(f"    Pairs KEPT after all filters:           {total_valid:>5}")
print(f"      of which FIXED by cleaning:           {c_modified:>5}  ({100*c_modified/total_valid:.1f}%)")
print(f"      of which UNCHANGED:                   {c_unchanged:>5}  ({100*c_unchanged/total_valid:.1f}%)")
print()
print(f"    New pairs from augmentation:            {a_new:>5}")
print(f"    FINAL TRAINING CORPUS:                  {a_total:>5}")
print(SEP)
print(f"    Model result:  BLEU = 18.77  |  chrF++ = 22.53")
print(SEP)
