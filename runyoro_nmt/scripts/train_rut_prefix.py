#!/usr/bin/env python3
"""
train_rut_prefix.py — Runyoro/Rutooro custom prefix training.

Strategy:
  1. Add `rut_Latn` as a new special token to the NLLB tokenizer.
  2. Fine-tune from runyoro-clean-v4 with forced_bos_token_id:
       EN→RUT: target starts with rut_Latn token  (decoder forced BOS)
       RUT→EN: target starts with eng_Latn token
  3. Model can NEVER fall back to nyk_Latn or other languages —
     forced_bos_token_id at inference locks the decoder language.

Output: runyoro_nmt/models/checkpoints/runyoro-rut-v1

Usage:
    python runyoro_nmt/scripts/train_rut_prefix.py
"""
import gc
import logging
import os
import random
import re
import sys
import unicodedata
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["USE_TF"] = "0"

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(ROOT / "training_rut_prefix.log"), encoding="utf-8"),
    ],
)
logger = logging.getLogger("train_rut_prefix")

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

SEED = 42
MAX_LEN = 256
random.seed(SEED)
set_seed(SEED)

# Start from v4 (our best model so far)
BASE_MODEL = str(ROOT / "models" / "checkpoints" / "runyoro-clean-v4")
OUTPUT_DIR = ROOT / "models" / "checkpoints" / "runyoro-rut-v1"

# Training data — use the best combined set we have
DATA_FILE = ROOT / "data" / "clean_training" / "train.csv"
VAL_FILE  = ROOT / "data" / "clean_training" / "val.csv"

# The new custom token we add for Runyoro-Rutooro
RUT_TOKEN = "rut_Latn"
ENG_TOKEN = "eng_Latn"  # already in NLLB vocab


# ─── Step 1: Extend the tokenizer ────────────────────────────────────────────

def build_extended_tokenizer(base_path: str):
    """
    Load the NLLB tokenizer and add rut_Latn as a new special token
    if it is not already present (or maps to <unk>).
    Returns (tokenizer, rut_token_id, eng_token_id).
    """
    logger.info("Loading tokenizer from: %s", base_path)
    tok = AutoTokenizer.from_pretrained(base_path)

    # Check if rut_Latn is already a real token (not <unk>)
    existing_id = tok.convert_tokens_to_ids(RUT_TOKEN)
    if existing_id == tok.unk_token_id:
        logger.info("rut_Latn not in vocab — adding as new special token")
        tok.add_special_tokens({"additional_special_tokens": [RUT_TOKEN]})
        rut_id = tok.convert_tokens_to_ids(RUT_TOKEN)
        logger.info("rut_Latn added with id=%d  (vocab size now %d)", rut_id, len(tok))
    else:
        rut_id = existing_id
        logger.info("rut_Latn already in vocab with id=%d", rut_id)

    eng_id = tok.convert_tokens_to_ids(ENG_TOKEN)
    logger.info("eng_Latn id=%d", eng_id)

    assert rut_id != tok.unk_token_id, "rut_Latn token was not added correctly"
    assert eng_id != tok.unk_token_id, "eng_Latn not found in vocab"

    return tok, rut_id, eng_id


# ─── Step 2: Load & clean training data ──────────────────────────────────────

def clean_text(t: str) -> str:
    t = str(t).strip()
    if t.lower() in ("nan", ""):
        return ""
    t = unicodedata.normalize("NFC", t)
    t = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"^[-\u2013\u2014]+\s*", "", t).strip()
    return t


def load_pairs(csv_path: Path) -> list:
    df = pd.read_csv(csv_path)
    # Support both column name orderings
    if "Runyoro" in df.columns and "English" in df.columns:
        rny_col, eng_col = "Runyoro", "English"
    else:
        rny_col, eng_col = df.columns[0], df.columns[1]

    pairs = []
    for _, row in df.iterrows():
        rny = clean_text(row[rny_col])
        eng = clean_text(row[eng_col])
        if rny and eng and len(rny) >= 5 and len(eng) >= 5:
            # Skip header-looking rows
            if rny.lower() in ("runyoro", "reference (runyoro-rutooro, original tense)"):
                continue
            if eng.lower() in ("english", "original english"):
                continue
            pairs.append((rny, eng))

    # Deduplicate
    seen = set()
    unique = []
    for pair in pairs:
        key = (pair[0].lower(), pair[1].lower())
        if key not in seen:
            seen.add(key)
            unique.append(pair)
    return unique


# ─── Step 3: Build HF datasets with forced BOS labels ────────────────────────

def make_datasets(train_pairs, val_pairs, tok, rut_id, eng_id):
    """
    For each pair we produce TWO samples:
      - EN→RUT: src=English, tgt=rut_Latn + Runyoro tokens
      - RUT→EN: src=Runyoro, tgt=eng_Latn + English tokens

    The BOS token is prepended to the label sequence so the model
    learns to always start Runyoro output with rut_Latn and
    English output with eng_Latn.
    """

    def encode_pair(src_text, tgt_text, bos_id):
        src_enc = tok(
            src_text,
            max_length=MAX_LEN,
            truncation=True,
            padding=False,
        )
        tgt_enc = tok(
            tgt_text,
            max_length=MAX_LEN - 1,   # leave room for BOS
            truncation=True,
            padding=False,
        )
        # Prepend the forced BOS token to the label sequence
        # The model will learn: given src, first decode token = bos_id, rest = translation
        labels = [bos_id] + tgt_enc["input_ids"]
        # Truncate to MAX_LEN
        labels = labels[:MAX_LEN]
        return {
            "input_ids": src_enc["input_ids"],
            "attention_mask": src_enc["attention_mask"],
            "labels": labels,
        }

    def build_split(pairs):
        en_to_rut = [encode_pair(eng, rny, rut_id) for rny, eng in pairs]
        rut_to_en = [encode_pair(rny, eng, eng_id) for rny, eng in pairs]
        all_samples = en_to_rut + rut_to_en
        random.shuffle(all_samples)

        return HFDataset.from_dict({
            "input_ids":      [s["input_ids"]      for s in all_samples],
            "attention_mask": [s["attention_mask"]  for s in all_samples],
            "labels":         [s["labels"]          for s in all_samples],
        })

    train_ds = build_split(train_pairs)
    val_ds   = build_split(val_pairs)
    logger.info("Train: %d samples | Val: %d samples", len(train_ds), len(val_ds))
    return train_ds, val_ds


# ─── Step 4: Train ───────────────────────────────────────────────────────────

def train(train_ds, val_ds, tok, rut_id, eng_id):
    gc.collect()
    torch.cuda.empty_cache()

    logger.info("Loading model from: %s (float32)", BASE_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)

    # Resize embeddings to accommodate the new rut_Latn token
    model.resize_token_embeddings(len(tok))

    # Initialise the new token embedding as a copy of run_Latn (closest Bantu language)
    # This gives the model a head start instead of random init
    run_id = tok.convert_tokens_to_ids("run_Latn")  # Rundi, id=256146
    if run_id != tok.unk_token_id:
        with torch.no_grad():
            model.model.shared.weight[rut_id] = model.model.shared.weight[run_id].clone()
            # Also initialise the lm_head row
            if model.lm_head.weight.shape[0] > rut_id:
                model.lm_head.weight[rut_id] = model.lm_head.weight[run_id].clone()
        logger.info("Initialised rut_Latn embedding from run_Latn (id=%d)", run_id)

    model.gradient_checkpointing_enable()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    col = DataCollatorForSeq2Seq(tok, model=model, label_pad_token_id=-100, pad_to_multiple_of=8)

    # 5 epochs — we're just teaching the model what rut_Latn means
    # on top of an already well-trained base
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=5,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,   # effective batch = 32
        learning_rate=5e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        fp16=False,
        bf16=False,                      # Always float32 — no bf16
        save_strategy="epoch",
        eval_strategy="epoch",
        logging_steps=10,
        save_total_limit=1,
        load_best_model_at_end=False,    # Do NOT use — incompatible with save_only_model
        predict_with_generate=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        optim="adamw_torch",
        max_grad_norm=1.0,
        save_safetensors=True,
        save_only_model=True,
        report_to=["none"],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tok,
        data_collator=col,
    )

    logger.info("=" * 60)
    logger.info("TRAINING runyoro-rut-v1")
    logger.info("  Base: runyoro-clean-v4 | New token: rut_Latn (id=%d)", rut_id)
    logger.info("  eng_Latn id=%d", eng_id)
    logger.info("  Epochs: 5 | Batch: 32 effective | LR: 5e-5")
    logger.info("  Pairs: %d | Samples: %d", len(train_ds) // 2, len(train_ds))
    logger.info("=" * 60)

    trainer.train()

    logger.info("Saving model + tokenizer to: %s", OUTPUT_DIR)
    model.save_pretrained(str(OUTPUT_DIR))
    tok.save_pretrained(str(OUTPUT_DIR))

    # Save token ID metadata so inference code can read them without hardcoding
    import json
    meta = {
        "rut_token": RUT_TOKEN,
        "rut_token_id": rut_id,
        "eng_token": ENG_TOKEN,
        "eng_token_id": eng_id,
        "base_model": BASE_MODEL,
        "strategy": "forced_bos_token_id",
        "note": "Use forced_bos_token_id=rut_token_id for EN->RUT, eng_token_id for RUT->EN",
    }
    with open(OUTPUT_DIR / "rut_token_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Saved rut_token_meta.json")

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("Runyoro-Rutooro Custom Prefix Training (rut_Latn)")
    logger.info("=" * 60)

    # 1. Build extended tokenizer
    tok, rut_id, eng_id = build_extended_tokenizer(BASE_MODEL)

    # 2. Load data
    logger.info("Loading training data from: %s", DATA_FILE)
    train_pairs = load_pairs(DATA_FILE)
    val_pairs   = load_pairs(VAL_FILE)
    logger.info("  Train pairs: %d | Val pairs: %d", len(train_pairs), len(val_pairs))

    if len(train_pairs) < 50:
        logger.error("Too few training pairs. Aborting.")
        sys.exit(1)

    # 3. Build datasets
    train_ds, val_ds = make_datasets(train_pairs, val_pairs, tok, rut_id, eng_id)

    # 4. Train
    train(train_ds, val_ds, tok, rut_id, eng_id)

    logger.info("=" * 60)
    logger.info("DONE — model saved to: %s", OUTPUT_DIR)
    logger.info("rut_Latn id=%d  |  eng_Latn id=%d", rut_id, eng_id)
    logger.info("At inference:")
    logger.info("  EN→RUT: forced_bos_token_id=%d (%s)", rut_id, RUT_TOKEN)
    logger.info("  RUT→EN: forced_bos_token_id=%d (%s)", eng_id, ENG_TOKEN)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
