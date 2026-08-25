#!/usr/bin/env python3
"""
train_rut_incremental.py — Incremental fine-tune with new raw data.

Reads ALL xlsx files from the raw/ folder, extracts clean sentence pairs,
and incrementally fine-tunes runyoro-rut-v1 → runyoro-rut-v2.
Preserves the rut_Latn forced_bos_token_id mechanism.

Usage:
    python runyoro_nmt/scripts/train_rut_incremental.py
"""
import gc
import json
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
RAW_DIR = ROOT.parent / "raw"
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(ROOT / "training_rut_v2.log"), encoding="utf-8"),
    ],
)
logger = logging.getLogger("train_rut_v2")

import pandas as pd
import torch
from datasets import Dataset as HFDataset
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

BASE_MODEL = str(ROOT / "models" / "checkpoints" / "runyoro-rut-v1")
OUTPUT_DIR = ROOT / "models" / "checkpoints" / "runyoro-rut-v2"


# ── Text cleaning ─────────────────────────────────────────────────────────────

def clean_text(t) -> str:
    t = str(t).strip()
    if t.lower() in ("nan", ""):
        return ""
    t = unicodedata.normalize("NFC", t)
    t = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"^[-\u2013\u2014]+\s*", "", t).strip()
    return t


def is_valid(eng: str, rny: str) -> bool:
    if len(eng) < 5 or len(rny) < 5:
        return False
    if eng.lower() in ("nan", "english", "original english"):
        return False
    if rny.lower() in ("nan", "runyoro", "reference (runyoro-rutooro, original tense)"):
        return False
    # Skip POS-tag style entries
    if re.search(r"\(v\.\w+\)", eng) or re.search(r"\(v\.\w+\)", rny):
        return False
    ratio = max(len(eng), len(rny)) / max(min(len(eng), len(rny)), 1)
    if ratio > 8:
        return False
    return True


# ── Extract pairs from all xlsx files ────────────────────────────────────────

def extract_all_pairs() -> list:
    files = sorted(RAW_DIR.glob("*.xlsx"))
    if not files:
        logger.error("No xlsx files found in: %s", RAW_DIR)
        sys.exit(1)

    all_pairs = []
    for f in files:
        file_pairs = []
        try:
            df = pd.read_excel(f, header=None)
        except Exception as e:
            logger.warning("Could not read %s: %s", f.name, e)
            continue

        if len(df.columns) < 5:
            logger.warning("Skipping %s — too few columns (%d)", f.name, len(df.columns))
            continue

        for _, row in df.iterrows():
            # Columns: 0=id, 1=pair_id, 2=eng_orig, 3=eng_tense, 4=rny_orig, 5=rny_tense, 6=eng_var, 7=rny_var
            # Original pair: col2=English, col4=Runyoro
            eng = clean_text(row.iloc[2]) if pd.notna(row.iloc[2]) else ""
            rny = clean_text(row.iloc[4]) if pd.notna(row.iloc[4]) else ""
            if eng and rny and is_valid(eng, rny):
                file_pairs.append((rny, eng))

            # Variation pair: col6=English variation, col7=Runyoro variation
            if len(df.columns) >= 8:
                eng_v = clean_text(row.iloc[6]) if pd.notna(row.iloc[6]) else ""
                rny_v = clean_text(row.iloc[7]) if pd.notna(row.iloc[7]) else ""
                if eng_v and rny_v and is_valid(eng_v, rny_v):
                    file_pairs.append((rny_v, eng_v))

        logger.info("  %s: %d pairs", f.name, len(file_pairs))
        all_pairs.extend(file_pairs)

    # Deduplicate
    seen = set()
    unique = []
    for pair in all_pairs:
        key = (pair[0].lower().strip(), pair[1].lower().strip())
        if key not in seen:
            seen.add(key)
            unique.append(pair)

    logger.info("Total unique pairs from all files: %d", len(unique))
    return unique


# ── Build HF dataset with forced BOS labels ───────────────────────────────────

def make_dataset(pairs, tok, rut_id, eng_id):
    samples = []
    for rny, eng in pairs:
        # EN → RUT
        src = tok(eng, max_length=MAX_LEN, truncation=True, padding=False)
        tgt = tok(rny, max_length=MAX_LEN - 1, truncation=True, padding=False)
        samples.append({
            "input_ids":      src["input_ids"],
            "attention_mask": src["attention_mask"],
            "labels":         [rut_id] + tgt["input_ids"],
        })
        # RUT → EN
        src = tok(rny, max_length=MAX_LEN, truncation=True, padding=False)
        tgt = tok(eng, max_length=MAX_LEN - 1, truncation=True, padding=False)
        samples.append({
            "input_ids":      src["input_ids"],
            "attention_mask": src["attention_mask"],
            "labels":         [eng_id] + tgt["input_ids"],
        })

    random.shuffle(samples)
    return HFDataset.from_dict({
        "input_ids":      [s["input_ids"]      for s in samples],
        "attention_mask": [s["attention_mask"]  for s in samples],
        "labels":         [s["labels"]          for s in samples],
    })


# ── Train ─────────────────────────────────────────────────────────────────────

def train(train_ds, val_ds, tok, rut_id, eng_id):
    gc.collect()
    torch.cuda.empty_cache()

    logger.info("Loading base model: %s (float32)", BASE_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)
    model.resize_token_embeddings(len(tok))  # already 256205, but safe to call
    model.gradient_checkpointing_enable()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    col = DataCollatorForSeq2Seq(tok, model=model, label_pad_token_id=-100, pad_to_multiple_of=8)

    args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=3,              # 3 epochs — incremental, not from scratch
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,   # effective batch = 32
        learning_rate=2e-5,              # lower LR for incremental fine-tune
        warmup_ratio=0.1,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        fp16=False,
        bf16=False,                      # Always float32
        save_strategy="epoch",
        eval_strategy="epoch",
        logging_steps=10,
        save_total_limit=1,
        load_best_model_at_end=False,
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
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tok,
        data_collator=col,
    )

    logger.info("=" * 60)
    logger.info("TRAINING runyoro-rut-v2 (incremental)")
    logger.info("  Base: runyoro-rut-v1 | rut_Latn id=%d | eng_Latn id=%d", rut_id, eng_id)
    logger.info("  Epochs: 3 | LR: 2e-5 | Batch: 32 effective")
    logger.info("  Train samples: %d | Val samples: %d", len(train_ds), len(val_ds))
    logger.info("=" * 60)

    trainer.train()

    logger.info("Saving model + tokenizer to: %s", OUTPUT_DIR)
    model.save_pretrained(str(OUTPUT_DIR))
    tok.save_pretrained(str(OUTPUT_DIR))

    # Copy token metadata so server picks it up automatically
    meta = {
        "rut_token": "rut_Latn",
        "rut_token_id": rut_id,
        "eng_token": "eng_Latn",
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("Runyoro-Rutooro Incremental Training (rut-v2)")
    logger.info("  Raw dir: %s", RAW_DIR)
    logger.info("  Base:    runyoro-rut-v1")
    logger.info("  Output:  runyoro-rut-v2")
    logger.info("=" * 60)

    # Load tokenizer + token IDs from v1
    logger.info("Loading tokenizer from rut-v1...")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)

    meta_file = Path(BASE_MODEL) / "rut_token_meta.json"
    with open(meta_file) as f:
        meta = json.load(f)
    rut_id = meta["rut_token_id"]
    eng_id = meta["eng_token_id"]
    logger.info("Token IDs — rut_Latn=%d  eng_Latn=%d", rut_id, eng_id)

    # Extract all pairs
    all_pairs = extract_all_pairs()
    if len(all_pairs) < 50:
        logger.error("Too few pairs (%d). Aborting.", len(all_pairs))
        sys.exit(1)

    # Split 90/10
    rng = random.Random(SEED)
    shuffled = all_pairs.copy()
    rng.shuffle(shuffled)
    n_val = max(10, int(len(shuffled) * 0.1))
    val_pairs   = shuffled[:n_val]
    train_pairs = shuffled[n_val:]
    logger.info("Train: %d pairs | Val: %d pairs", len(train_pairs), len(val_pairs))

    # Build datasets
    train_ds = make_dataset(train_pairs, tok, rut_id, eng_id)
    val_ds   = make_dataset(val_pairs,   tok, rut_id, eng_id)

    # Train
    train(train_ds, val_ds, tok, rut_id, eng_id)

    logger.info("=" * 60)
    logger.info("DONE — runyoro-rut-v2 saved to: %s", OUTPUT_DIR)
    logger.info("  rut_Latn id=%d | eng_Latn id=%d", rut_id, eng_id)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
