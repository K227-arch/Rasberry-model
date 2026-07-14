#!/usr/bin/env python3
"""
train_v6.py — Process sentence pair (4).xlsx + all existing data, train v6 model.
"""
import os, sys, gc, re, random, unicodedata, logging
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["USE_TF"] = "0"

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(), logging.FileHandler(str(ROOT / "training_v6.log"))])
logger = logging.getLogger("train_v6")

import pandas as pd
import torch
from datasets import Dataset as HFDataset, concatenate_datasets
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments, set_seed)

set_seed(42)
random.seed(42)

RAW_DIR = ROOT.parent / "raw"
V5_DATA_DIR = ROOT / "data" / "v5_training"
V6_DATA_DIR = ROOT / "data" / "v6_training"
V6_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = ROOT / "models" / "checkpoints" / "runyoro-nmt-v6"
MODEL_NAME = "facebook/nllb-200-distilled-1.3B"
MAX_LEN = 256

# ===== STEP 1: Process new data =====
logger.info("=" * 60)
logger.info("STEP 1: Process sentence pair (4).xlsx")
logger.info("=" * 60)

def clean(text):
    text = str(text)
    if text.lower() == 'nan' or not text.strip(): return ''
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'^[-–—]+\s*', '', text).strip()
    return text

def valid(eng, rny):
    if len(eng) < 5 or len(rny) < 5: return False
    if eng.lower() == 'nan' or rny.lower() == 'nan': return False
    if re.search(r'\(v\.\w+\)', eng) or re.search(r'\(v\.\w+\)', rny): return False
    return True

df = pd.read_excel(RAW_DIR / "sentence pair (4).xlsx", header=None)
logger.info("Loaded %d rows from sentence pair (4).xlsx", len(df))

new_pairs = []
for _, row in df.iterrows():
    orig_eng = clean(row.iloc[2]) if pd.notna(row.iloc[2]) else ''
    orig_rny = clean(row.iloc[4]) if pd.notna(row.iloc[4]) else ''
    var_eng = clean(row.iloc[6]) if pd.notna(row.iloc[6]) else ''
    var_rny = clean(row.iloc[7]) if pd.notna(row.iloc[7]) else ''
    if orig_eng and orig_rny and valid(orig_eng, orig_rny):
        new_pairs.append((orig_rny, orig_eng))
    if var_eng and var_rny and valid(var_eng, var_rny):
        new_pairs.append((var_rny, var_eng))

new_pairs = list(set(new_pairs))
logger.info("Extracted %d new pairs", len(new_pairs))

# ===== STEP 2: Combine with v5 data =====
logger.info("=" * 60)
logger.info("STEP 2: Combine with existing v5 data")
logger.info("=" * 60)

v5_df = pd.read_csv(V5_DATA_DIR / "cleaned_pairs.csv")
existing = [(str(r["Runyoro"]), str(r["English"])) for _, r in v5_df.iterrows()]
logger.info("Existing v5: %d pairs", len(existing))

all_pairs = existing + new_pairs
seen = set()
unique = []
for r, e in all_pairs:
    key = (r.lower().strip(), e.lower().strip())
    if key not in seen:
        seen.add(key)
        unique.append((r, e))
logger.info("Combined unique: %d pairs", len(unique))

# ===== STEP 3: Augmentation =====
logger.info("=" * 60)
logger.info("STEP 3: Augmentation")
logger.info("=" * 60)

augmented = []
for rny, eng in unique:
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

logger.info("Augmented: %d pairs", len(augmented))

training_pairs = unique + augmented
random.shuffle(training_pairs)

# Save
pd.DataFrame(unique, columns=["Runyoro", "English"]).to_csv(V6_DATA_DIR / "cleaned_pairs.csv", index=False)
pd.DataFrame(augmented, columns=["Runyoro", "English"]).to_csv(V6_DATA_DIR / "augmented_pairs.csv", index=False)
pd.DataFrame(training_pairs, columns=["Runyoro", "English"]).to_csv(V6_DATA_DIR / "all_training_pairs.csv", index=False)
logger.info("Saved: cleaned=%d, augmented=%d, total=%d", len(unique), len(augmented), len(training_pairs))

# ===== STEP 4: Train =====
logger.info("=" * 60)
logger.info("STEP 4: Training v6")
logger.info("=" * 60)

gc.collect()
torch.cuda.empty_cache()
logger.info("GPUs: %d", torch.cuda.device_count())

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_fn(examples):
    src_enc = tokenizer(examples["src"], max_length=MAX_LEN, truncation=True, padding=False)
    tgt_enc = tokenizer(examples["tgt"], max_length=MAX_LEN, truncation=True, padding=False)
    src_enc["labels"] = tgt_enc["input_ids"]
    return src_enc

fwd_ds = HFDataset.from_dict({"src": [s for s, t in training_pairs], "tgt": [t for s, t in training_pairs]})
fwd_ds = fwd_ds.map(tokenize_fn, batched=True, remove_columns=["src", "tgt"], desc="rny->eng")
rev_ds = HFDataset.from_dict({"src": [t for s, t in training_pairs], "tgt": [s for s, t in training_pairs]})
rev_ds = rev_ds.map(tokenize_fn, batched=True, remove_columns=["src", "tgt"], desc="eng->rny")
train_ds = concatenate_datasets([fwd_ds, rev_ds]).shuffle(seed=42)
logger.info("Train dataset: %d samples (bidirectional from %d pairs)", len(train_ds), len(training_pairs))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32,
                                               device_map="auto", max_memory={0: "22GiB", 1: "22GiB"})
model.gradient_checkpointing_enable()
logger.info("Model loaded: %.1fM params", sum(p.numel() for p in model.parameters()) / 1e6)

collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8)

training_args = Seq2SeqTrainingArguments(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=20,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    learning_rate=5e-5,
    warmup_steps=100,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    fp16=False, bf16=False,
    save_strategy="epoch",
    eval_strategy="no",
    logging_steps=10,
    save_total_limit=2,
    predict_with_generate=False,
    dataloader_num_workers=0,
    dataloader_pin_memory=True,
    gradient_checkpointing=True,
    optim="adamw_torch",
    save_safetensors=True,
    save_only_model=True,
    report_to=["none"],
)

trainer = Seq2SeqTrainer(model=model, args=training_args, train_dataset=train_ds,
                         processing_class=tokenizer, data_collator=collator)

logger.info("=" * 60)
logger.info("TRAINING v6")
logger.info("  Clean pairs: %d | Training pairs: %d | Samples: %d", len(unique), len(training_pairs), len(train_ds))
logger.info("  Epochs: 20 | Batch: 2x16=32 effective")
logger.info("=" * 60)

trainer.train()
model.save_pretrained(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))
logger.info("v6 model saved to: %s", OUTPUT_DIR)
logger.info("DONE!")
