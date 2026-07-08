#!/usr/bin/env python3
"""
process_variations2_and_train.py
- Load sentence variations (2) xlsx
- Clean and extract all English-Runyoro pairs
- Combine with existing v3 training data
- Augment
- Train model v4
"""
import os
import re
import sys
import gc
import random
import unicodedata
import logging
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["USE_TF"] = "0"

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(str(ROOT / "training_v4.log"))],
)
logger = logging.getLogger("train_v4")

import pandas as pd
import torch
from datasets import Dataset as HFDataset, concatenate_datasets
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

set_seed(42)

# === STEP 1: Load and parse sentence variations (2) ===
logger.info("=" * 60)
logger.info("STEP 1: Load sentence variations (2)")
logger.info("=" * 60)

RAW_FILE = ROOT.parent / "raw" / "sentence variations (2).xlsx"
if not RAW_FILE.exists():
    # Try alternate location
    RAW_FILE = ROOT.parent / "raw" / "sentence variations 2.xlsx"

df_new = pd.read_excel(RAW_FILE, header=None)
logger.info("Loaded %d rows from %s", len(df_new), RAW_FILE.name)
logger.info("Columns: %s", list(df_new.columns))
logger.info("First row: %s", df_new.iloc[0].tolist())

# The structure based on the document:
# Col 0: row number (100, 101, etc.)
# Col 1: group number (34, 35, etc.)
# Col 2: Original English sentence
# Col 3: Original Tense
# Col 4: Original Runyoro sentence
# Col 5: Variation Tense
# Col 6: Variation English sentence
# Col 7: Variation Runyoro sentence
# Col 8: status (pending)

pairs = []

for idx, row in df_new.iterrows():
    # Extract original pair
    orig_eng = str(row.iloc[2]) if pd.notna(row.iloc[2]) else ""
    orig_rny = str(row.iloc[4]) if pd.notna(row.iloc[4]) else ""
    
    # Extract variation pair
    var_eng = str(row.iloc[6]) if pd.notna(row.iloc[6]) else ""
    var_rny = str(row.iloc[7]) if pd.notna(row.iloc[7]) else ""
    
    if orig_eng.strip() and orig_rny.strip():
        pairs.append((orig_eng.strip(), orig_rny.strip()))
    if var_eng.strip() and var_rny.strip():
        pairs.append((var_eng.strip(), var_rny.strip()))

logger.info("Extracted %d raw pairs from variations (2)", len(pairs))

# === STEP 2: Clean all pairs ===
logger.info("=" * 60)
logger.info("STEP 2: Clean data")
logger.info("=" * 60)

def clean(text):
    text = str(text)
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Remove leading/trailing punctuation artifacts
    text = text.strip("-").strip()
    return text

def is_valid_pair(eng, rny):
    """Filter out bad pairs"""
    if len(eng) < 5 or len(rny) < 5:
        return False
    # Remove if either side is 'nan' or empty
    if eng.lower() == 'nan' or rny.lower() == 'nan':
        return False
    # Remove dictionary-style entries
    if eng.startswith("To ") and len(eng.split()) < 4:
        return False
    # Remove grammar annotations
    if re.search(r'\(v\.\w+\)', eng) or re.search(r'\(v\.\w+\)', rny):
        return False
    return True

cleaned_pairs = []
for eng, rny in pairs:
    eng = clean(eng)
    rny = clean(rny)
    if is_valid_pair(eng, rny):
        cleaned_pairs.append((eng, rny))

# Deduplicate
cleaned_pairs = list(set(cleaned_pairs))
logger.info("After cleaning and dedup: %d pairs", len(cleaned_pairs))

# === STEP 3: Combine with existing v3 data ===
logger.info("=" * 60)
logger.info("STEP 3: Combine with existing training data")
logger.info("=" * 60)

existing_csv = ROOT / "data" / "v3_training" / "cleaned_pairs.csv"
if existing_csv.exists():
    df_existing = pd.read_csv(existing_csv)
    existing_pairs = list(zip(df_existing["Runyoro"].tolist(), df_existing["English"].tolist()))
    logger.info("Loaded %d existing v3 pairs", len(existing_pairs))
else:
    # Load from original xlsx
    orig_file = ROOT.parent / "raw" / "100 sentence pairs 01.xlsx"
    df_orig = pd.read_excel(orig_file)
    df_orig = df_orig[["English", "Runyoro-Rutooro (to fill)"]].dropna()
    df_orig.columns = ["English", "Runyoro"]
    df_orig["English"] = df_orig["English"].apply(clean)
    df_orig["Runyoro"] = df_orig["Runyoro"].apply(clean)
    df_orig = df_orig[(df_orig["English"].str.len() > 5) & (df_orig["Runyoro"].str.len() > 5)]
    existing_pairs = list(zip(df_orig["Runyoro"].tolist(), df_orig["English"].tolist()))
    logger.info("Loaded %d pairs from original xlsx", len(existing_pairs))

# New pairs are (eng, rny) - convert to (rny, eng) to match existing format
new_pairs_formatted = [(rny, eng) for eng, rny in cleaned_pairs]

# Combine
all_pairs = existing_pairs + new_pairs_formatted
all_pairs = list(set(all_pairs))  # Deduplicate
logger.info("Combined: %d existing + %d new = %d total (after dedup)",
            len(existing_pairs), len(new_pairs_formatted), len(all_pairs))

# === STEP 4: Augmentation ===
logger.info("=" * 60)
logger.info("STEP 4: Data Augmentation")
logger.info("=" * 60)

random.seed(42)
augmented = []
for rny, eng in all_pairs:
    words_rny = rny.split()
    words_eng = eng.split()

    # Token deletion (5% chance per word)
    if len(words_rny) > 4:
        del_rny = " ".join(w for w in words_rny if random.random() > 0.05)
        del_eng = " ".join(w for w in words_eng if random.random() > 0.05)
        if len(del_rny.split()) >= 3 and len(del_eng.split()) >= 3:
            augmented.append((del_rny, del_eng))

    # Token swap (swap 2 adjacent words)
    if len(words_rny) > 3:
        idx = random.randint(0, len(words_rny) - 2)
        swapped = words_rny.copy()
        swapped[idx], swapped[idx + 1] = swapped[idx + 1], swapped[idx]
        augmented.append((" ".join(swapped), eng))

logger.info("Augmented pairs: %d", len(augmented))

# Save all data
DATA_DIR = ROOT / "data" / "v4_training"
DATA_DIR.mkdir(parents=True, exist_ok=True)

pd.DataFrame(all_pairs, columns=["Runyoro", "English"]).to_csv(
    DATA_DIR / "cleaned_pairs.csv", index=False)
pd.DataFrame(augmented, columns=["Runyoro", "English"]).to_csv(
    DATA_DIR / "augmented_pairs.csv", index=False)

combined = all_pairs + augmented
random.shuffle(combined)
pd.DataFrame(combined, columns=["Runyoro", "English"]).to_csv(
    DATA_DIR / "all_training_pairs.csv", index=False)

logger.info("Saved to %s:", DATA_DIR)
logger.info("  cleaned_pairs.csv: %d", len(all_pairs))
logger.info("  augmented_pairs.csv: %d", len(augmented))
logger.info("  all_training_pairs.csv: %d", len(combined))

# === STEP 5: Train with NLLB FP32 ===
logger.info("=" * 60)
logger.info("STEP 5: Training with NLLB FP32 (no language codes)")
logger.info("=" * 60)

MODEL_NAME = "facebook/nllb-200-distilled-1.3B"
OUTPUT_DIR = ROOT / "models" / "checkpoints" / "runyoro-nmt-v4"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MAX_LEN = 256

# Clear GPU
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    logger.info("GPUs: %d", torch.cuda.device_count())

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_fn(examples):
    src_enc = tokenizer(examples["src"], max_length=MAX_LEN, truncation=True, padding=False)
    tgt_enc = tokenizer(examples["tgt"], max_length=MAX_LEN, truncation=True, padding=False)
    src_enc["labels"] = tgt_enc["input_ids"]
    return src_enc

def pairs_to_dataset(pairs, desc=""):
    ds = HFDataset.from_dict({"src": [s for s, t in pairs], "tgt": [t for s, t in pairs]})
    return ds.map(tokenize_fn, batched=True, remove_columns=["src", "tgt"], desc=desc)

# Bidirectional training
fwd_ds = pairs_to_dataset(combined, "rny->en")
rev_ds = pairs_to_dataset([(e, r) for r, e in combined], "en->rny")
train_ds = concatenate_datasets([fwd_ds, rev_ds]).shuffle(seed=42)
logger.info("Train dataset: %d samples (bidirectional)", len(train_ds))

# Load model FP32 across both GPUs
logger.info("Loading model: %s (FP32)", MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    device_map="auto",
    max_memory={0: "20GiB", 1: "20GiB"},
)
model.gradient_checkpointing_enable()
logger.info("Model loaded: %.1f M params", sum(p.numel() for p in model.parameters()) / 1e6)

collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8)

training_args = Seq2SeqTrainingArguments(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=20,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    warmup_steps=100,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    fp16=False,
    bf16=False,
    save_strategy="epoch",
    eval_strategy="no",
    logging_steps=10,
    save_total_limit=3,
    predict_with_generate=False,
    dataloader_num_workers=0,
    dataloader_pin_memory=True,
    gradient_checkpointing=True,
    optim="adamw_torch",
    save_safetensors=True,
    save_only_model=True,
    report_to=["none"],
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    processing_class=tokenizer,
    data_collator=collator,
)

logger.info("=" * 60)
logger.info("STARTING TRAINING")
logger.info("  Model: %s (FP32, no language codes)", MODEL_NAME)
logger.info("  Data: %d pairs (%d bidirectional)", len(combined), len(train_ds))
logger.info("  Epochs: 20 | Batch: 4 x 8 = 32 effective")
logger.info("=" * 60)

trainer.train()
logger.info("Training complete!")

model.save_pretrained(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))
logger.info("Model saved to: %s", OUTPUT_DIR)
