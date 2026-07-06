#!/usr/bin/env python3
"""
save_augmented_data.py - Re-run data processing from train_fresh.py and save CSVs
"""
import os
import re
import random
import unicodedata
from pathlib import Path

os.environ["USE_TF"] = "0"

ROOT = Path(__file__).parent.parent
RAW_FILE = ROOT.parent / "raw" / "100 sentence pairs 01.xlsx"

import pandas as pd

# Load and clean
df = pd.read_excel(RAW_FILE)
eng_col = "English"
rny_col = "Runyoro-Rutooro (to fill)"

df = df[[eng_col, rny_col]].dropna()
df.columns = ["English", "Runyoro"]

def clean(text):
    text = str(text)
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["English"] = df["English"].apply(clean)
df["Runyoro"] = df["Runyoro"].apply(clean)
df = df[(df["English"].str.len() > 5) & (df["Runyoro"].str.len() > 5)]
df = df.drop_duplicates()

pairs = list(zip(df["Runyoro"].tolist(), df["English"].tolist()))
print(f"Cleaned pairs: {len(pairs)}")

# Augmentation
random.seed(42)
augmented = []
for rny, eng in pairs:
    words_rny = rny.split()
    words_eng = eng.split()

    if len(words_rny) > 4:
        del_rny = " ".join(w for w in words_rny if random.random() > 0.05)
        del_eng = " ".join(w for w in words_eng if random.random() > 0.05)
        if len(del_rny.split()) >= 3 and len(del_eng.split()) >= 3:
            augmented.append((del_rny, del_eng))

    if len(words_rny) > 3:
        idx = random.randint(0, len(words_rny) - 2)
        swapped = words_rny.copy()
        swapped[idx], swapped[idx + 1] = swapped[idx + 1], swapped[idx]
        augmented.append((" ".join(swapped), eng))

print(f"Augmented pairs: {len(augmented)}")

# Save
DATA_DIR = ROOT / "data" / "v3_training"
DATA_DIR.mkdir(parents=True, exist_ok=True)

pd.DataFrame(pairs, columns=["Runyoro", "English"]).to_csv(DATA_DIR / "cleaned_pairs.csv", index=False)
pd.DataFrame(augmented, columns=["Runyoro", "English"]).to_csv(DATA_DIR / "augmented_pairs.csv", index=False)

all_pairs = pairs + augmented
random.shuffle(all_pairs)
pd.DataFrame(all_pairs, columns=["Runyoro", "English"]).to_csv(DATA_DIR / "all_training_pairs.csv", index=False)

print(f"\nSaved to {DATA_DIR}:")
print(f"  cleaned_pairs.csv: {len(pairs)} pairs")
print(f"  augmented_pairs.csv: {len(augmented)} pairs")
print(f"  all_training_pairs.csv: {len(all_pairs)} pairs (combined)")
