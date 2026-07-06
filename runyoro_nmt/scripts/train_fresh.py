#!/usr/bin/env python3
"""
train_fresh.py - Process 100 sentence pairs, clean, augment, and train with NLLB FP32
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
    handlers=[logging.StreamHandler(), logging.FileHandler(str(ROOT / "training_fresh.log"))],
)
logger = logging.getLogger("train_fresh")

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

# === STEP 1: Load and Clean ===
logger.info("=" * 60)
logger.info("STEP 1: Load and Clean Data")
logger.info("=" * 60)

RAW_FILE = ROOT.parent / "raw" / "100 sentence pairs 01.xlsx"
df = pd.read_excel(RAW_FILE)
logger.info("Loaded %d rows from %s", len(df), RAW_FILE.name)

# Extract English and Runyoro columns
eng_col = "English"
rny_col = "Runyoro-Rutooro (to fill)"

# Keep only rows with both English and Runyoro filled
df = df[[eng_col, rny_col]].dropna()
df.columns = ["English", "Runyoro"]

# Clean text
def clean(text):
    text = str(text)
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["English"] = df["English"].apply(clean)
df["Runyoro"] = df["Runyoro"].apply(clean)

# Remove empty/too-short
df = df[(df["English"].str.len() > 5) & (df["Runyoro"].str.len() > 5)]
df = df.drop_duplicates()

logger.info("After cleaning: %d pairs", len(df))
pairs = list(zip(df["Runyoro"].tolist(), df["English"].tolist()))

# === STEP 2: Augmentation ===
logger.info("=" * 60)
logger.info("STEP 2: Data Augmentation")
logger.info("=" * 60)

augmented = []
for rny, eng in pairs:
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

# Save all data to CSV
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

pd.DataFrame(pairs, columns=["Runyoro", "English"]).to_csv(DATA_DIR / "cleaned_pairs.csv", index=False)
pd.DataFrame(augmented, columns=["Runyoro", "English"]).to_csv(DATA_DIR / "augmented_pairs.csv", index=False)
logger.info("Saved cleaned_pairs.csv (%d) and augmented_pairs.csv (%d) to %s", len(pairs), len(augmented), DATA_DIR)

# === STEP 3: Combine all data ===
logger.info("=" * 60)
logger.info("STEP 3: Combine Data")
logger.info("=" * 60)

all_pairs = pairs + augmented
random.shuffle(all_pairs)
logger.info("Total training pairs: %d (original: %d + augmented: %d)", len(all_pairs), len(pairs), len(augmented))

# Save combined
pd.DataFrame(all_pairs, columns=["Runyoro", "English"]).to_csv(DATA_DIR / "all_training_pairs.csv", index=False)
logger.info("Saved all_training_pairs.csv (%d) to %s", len(all_pairs), DATA_DIR)

# === STEP 4: Train with NLLB FP32 ===
logger.info("=" * 60)
logger.info("STEP 4: Training with NLLB FP32 (no language codes)")
logger.info("=" * 60)

MODEL_NAME = "facebook/nllb-200-distilled-1.3B"
OUTPUT_DIR = ROOT / "models" / "checkpoints" / "runyoro-nmt-v3"
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

# Bidirectional
fwd_ds = pairs_to_dataset(all_pairs, "rny->en")
rev_ds = pairs_to_dataset([(e, r) for r, e in all_pairs], "en->rny")
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
logger.info("  Data: %d pairs (%d bidirectional)", len(all_pairs), len(train_ds))
logger.info("  Epochs: 20 | Batch: 4 x 8 = 32 effective")
logger.info("=" * 60)

trainer.train()
logger.info("Training complete!")

model.save_pretrained(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))
logger.info("Model saved to: %s", OUTPUT_DIR)
