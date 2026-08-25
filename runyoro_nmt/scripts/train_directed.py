#!/usr/bin/env python3
"""
train_directed.py — Train with explicit direction prefixes.

Every training example is prefixed with >>rny<< or >>eng<< to tell the model
which language to produce. This eliminates direction confusion completely.

Usage:
    python runyoro_nmt/scripts/train_directed.py --epochs 8 --bt-epochs 3
"""
import argparse
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
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["USE_TF"] = "0"

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(ROOT / "training_directed.log"), encoding="utf-8"),
    ],
)
logger = logging.getLogger("train_dir")

import pandas as pd
import torch
from datasets import Dataset as HFDataset, concatenate_datasets
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)

SEED = 42
MAX_LEN = 256
random.seed(SEED)
set_seed(SEED)

RAW_DIR = ROOT.parent / "raw"
CKPT_DIR = ROOT / "models" / "checkpoints"
BASE_MODEL = "facebook/nllb-200-distilled-1.3B"
OUTPUT_DIR = CKPT_DIR / "runyoro-directed-v1"

# Direction prefixes — these tell the model what to output
PREFIX_TO_RNY = ">>rny<< "  # prepend to English input when we want Runyoro output
PREFIX_TO_ENG = ">>eng<< "  # prepend to Runyoro input when we want English output

RAW_FILES = [
    "100 sentence pairs 01.xlsx",
    "sentence variations (2).xlsx",
    "sentence pairs (3).xlsx",
    "sentence pair (4).xlsx",
    "sentence pair (5).xlsx",
    "sentence pairs 6.xlsx",
    "sentence pair 7.xlsx",
    "sentence pairs 8.xlsx",
]


def clean_text(t: str) -> str:
    t = str(t).strip()
    if t.lower() == "nan" or not t:
        return ""
    t = unicodedata.normalize("NFC", t)
    t = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"^[-\u2013\u2014]+\s*", "", t).strip()
    return t


def is_valid_pair(eng: str, rny: str) -> bool:
    if len(eng) < 5 or len(rny) < 5:
        return False
    if eng.lower() == "nan" or rny.lower() == "nan":
        return False
    if re.search(r"\(v\.\w+\)", eng) or re.search(r"\(v\.\w+\)", rny):
        return False
    ratio = max(len(eng), len(rny)) / max(min(len(eng), len(rny)), 1)
    if ratio > 8:
        return False
    return True


def extract_all_pairs() -> list:
    """Extract all sentence pairs from raw xlsx files."""
    all_pairs = []
    for fname in RAW_FILES:
        filepath = RAW_DIR / fname
        if not filepath.exists():
            logger.warning("File not found: %s", filepath)
            continue
        df = pd.read_excel(filepath, header=None)
        pairs_from_file = []
        if len(df.columns) >= 8:
            for _, row in df.iterrows():
                orig_eng = clean_text(row.iloc[2]) if pd.notna(row.iloc[2]) else ""
                orig_rny = clean_text(row.iloc[4]) if pd.notna(row.iloc[4]) else ""
                if "english" in orig_eng.lower() and "runyoro" in orig_rny.lower():
                    continue
                if "original" in orig_eng.lower() and "reference" in orig_rny.lower():
                    continue
                if orig_eng and orig_rny and is_valid_pair(orig_eng, orig_rny):
                    pairs_from_file.append((orig_rny, orig_eng))
                var_eng = clean_text(row.iloc[6]) if pd.notna(row.iloc[6]) else ""
                var_rny = clean_text(row.iloc[7]) if pd.notna(row.iloc[7]) else ""
                if "english" in var_eng.lower() and "runyoro" in var_rny.lower():
                    continue
                if var_eng and var_rny and is_valid_pair(var_eng, var_rny):
                    pairs_from_file.append((var_rny, var_eng))
        logger.info("  %s: %d pairs", fname, len(pairs_from_file))
        all_pairs.extend(pairs_from_file)

    seen = set()
    unique = []
    for rny, eng in all_pairs:
        key = (rny.lower().strip(), eng.lower().strip())
        if key not in seen:
            seen.add(key)
            unique.append((rny, eng))
    logger.info("Total unique pairs: %d (from %d raw)", len(unique), len(all_pairs))
    return unique


def make_directed_dataset(pairs, tokenizer):
    """Create a directed dataset with prefixes.
    
    Each pair (rny, eng) becomes TWO training examples:
    1. Input: ">>rny<< {eng}" -> Target: "{rny}"   (English to Runyoro)
    2. Input: ">>eng<< {rny}" -> Target: "{eng}"   (Runyoro to English)
    """
    sources = []
    targets = []
    for rny, eng in pairs:
        # English -> Runyoro
        sources.append(PREFIX_TO_RNY + eng)
        targets.append(rny)
        # Runyoro -> English
        sources.append(PREFIX_TO_ENG + rny)
        targets.append(eng)

    def tok_fn(ex):
        model_inputs = tokenizer(ex["src"], max_length=MAX_LEN, truncation=True, padding=False)
        tgt = tokenizer(ex["tgt"], max_length=MAX_LEN, truncation=True, padding=False)
        model_inputs["labels"] = tgt["input_ids"]
        return model_inputs

    ds = HFDataset.from_dict({"src": sources, "tgt": targets})
    ds = ds.map(tok_fn, batched=True, remove_columns=["src", "tgt"], desc="Tokenizing")
    return ds.shuffle(seed=SEED)


def split_data(pairs, val_ratio=0.1, test_ratio=0.05):
    rng = random.Random(SEED)
    shuffled = pairs.copy()
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_test = max(1, int(n * test_ratio))
    n_val = max(1, int(n * val_ratio))
    test = shuffled[:n_test]
    val = shuffled[n_test:n_test + n_val]
    train = shuffled[n_test + n_val:]
    return train, val, test


def train_model(train_pairs, val_pairs, model_path, output_dir, epochs, lr):
    gc.collect()
    torch.cuda.empty_cache()

    logger.info("Loading tokenizer from: %s", model_path)
    tok = AutoTokenizer.from_pretrained(model_path)

    train_ds = make_directed_dataset(train_pairs, tok)
    val_ds = make_directed_dataset(val_pairs, tok)
    logger.info("Train: %d samples | Val: %d samples", len(train_ds), len(val_ds))

    n_gpus = torch.cuda.device_count()
    logger.info("GPUs available: %d", n_gpus)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    model.gradient_checkpointing_enable()

    output_dir.mkdir(parents=True, exist_ok=True)
    col = DataCollatorForSeq2Seq(tok, model=model, label_pad_token_id=-100, pad_to_multiple_of=8)

    targs = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=16,    # 16 per GPU × 2 GPUs = 32 effective
        gradient_accumulation_steps=1,
        learning_rate=lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        label_smoothing_factor=0.0,
        fp16=False,
        bf16=True,                         # bf16 on RTX 4090
        save_strategy="epoch",
        eval_strategy="epoch",
        logging_steps=5,                   # log every 5 steps so you see progress
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
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
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tok,
        data_collator=col,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    trainer.train()

    model.save_pretrained(str(output_dir))
    tok.save_pretrained(str(output_dir))
    logger.info("Model saved to: %s", output_dir)

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()
    return str(output_dir)


def back_translate(pairs, model_path, bs=16):
    """Generate BT pairs using direction prefixes."""
    gc.collect()
    torch.cuda.empty_cache()
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    logger.info("Loading model for BT: %s", model_path)
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, torch_dtype=torch.float32).to(device)
    model.eval()

    def translate_batch(texts):
        results = []
        for i in range(0, len(texts), bs):
            batch = texts[i:i + bs]
            enc = tok(batch, return_tensors="pt", max_length=MAX_LEN, truncation=True, padding=True).to(device)
            with torch.no_grad():
                out = model.generate(**enc, num_beams=4, max_length=MAX_LEN)
            decoded = [t.strip() for t in tok.batch_decode(out, skip_special_tokens=True)]
            results.extend(decoded)
        return results

    # Runyoro -> synthetic English (prefix with >>eng<<)
    logger.info("BT: Runyoro -> synthetic English (%d sentences)...", len(pairs))
    rny_texts = [PREFIX_TO_ENG + r for r, e in pairs]
    syn_eng = translate_batch(rny_texts)
    bt_rny_eng = []
    for (rny, _), syn_e in zip(pairs, syn_eng):
        if len(syn_e.strip()) > 5 and syn_e.lower().strip() != rny.lower().strip():
            bt_rny_eng.append((rny, syn_e))

    # English -> synthetic Runyoro (prefix with >>rny<<)
    logger.info("BT: English -> synthetic Runyoro (%d sentences)...", len(pairs))
    eng_texts = [PREFIX_TO_RNY + e for r, e in pairs]
    syn_rny = translate_batch(eng_texts)
    bt_eng_rny = []
    for (_, eng), syn_r in zip(pairs, syn_rny):
        if len(syn_r.strip()) > 5 and syn_r.lower().strip() != eng.lower().strip():
            bt_eng_rny.append((syn_r, eng))

    logger.info("BT results: rny->eng=%d, eng->rny=%d", len(bt_rny_eng), len(bt_eng_rny))

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return bt_rny_eng + bt_eng_rny


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--bt-epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--skip-bt", action="store_true")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("DIRECTED TRAINING — Runyoro NMT with >>rny<< / >>eng<< prefixes")
    logger.info("=" * 70)
    logger.info("  Base model: %s", BASE_MODEL)
    logger.info("  Output: %s", OUTPUT_DIR)
    logger.info("  Epochs: %d base, %d BT | LR: %s", args.epochs, args.bt_epochs, args.lr)
    logger.info("=" * 70)

    # Step 1: Extract
    logger.info("\nSTEP 1: Extracting pairs...")
    all_pairs = extract_all_pairs()

    # Step 2: Split
    logger.info("\nSTEP 2: Splitting...")
    train_pairs, val_pairs, test_pairs = split_data(all_pairs)
    logger.info("  Train: %d | Val: %d | Test: %d", len(train_pairs), len(val_pairs), len(test_pairs))

    # Save splits
    data_dir = ROOT / "data" / "directed_training"
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(train_pairs, columns=["Runyoro", "English"]).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame(val_pairs, columns=["Runyoro", "English"]).to_csv(data_dir / "val.csv", index=False)
    pd.DataFrame(test_pairs, columns=["Runyoro", "English"]).to_csv(data_dir / "test.csv", index=False)

    # Step 3: Base training
    logger.info("\nSTEP 3: Base training with direction prefixes...")
    base_dir = OUTPUT_DIR / "base"
    base_path = train_model(train_pairs, val_pairs, BASE_MODEL, base_dir, args.epochs, args.lr)

    # Step 4: Back-translation
    if not args.skip_bt:
        logger.info("\nSTEP 4: Back-translation...")
        bt_pairs = back_translate(train_pairs, base_path)
        logger.info("  BT pairs: %d", len(bt_pairs))
        pd.DataFrame(bt_pairs, columns=["Runyoro", "English"]).to_csv(data_dir / "bt_pairs.csv", index=False)

        # Step 5: Final training
        logger.info("\nSTEP 5: Final training with BT...")
        final_train = train_pairs + bt_pairs
        random.shuffle(final_train)
        logger.info("  Final: %d pairs (orig %d + BT %d)", len(final_train), len(train_pairs), len(bt_pairs))
        train_model(final_train, val_pairs, base_path, OUTPUT_DIR, args.bt_epochs, 1e-5)
    else:
        import shutil
        if OUTPUT_DIR.exists() and OUTPUT_DIR != base_dir:
            shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
        shutil.copytree(base_dir, OUTPUT_DIR, dirs_exist_ok=True)

    # Cleanup base
    import shutil
    if base_dir.exists() and base_dir != OUTPUT_DIR:
        shutil.rmtree(base_dir)

    # Save metadata
    meta = {
        "model": "runyoro-directed-v1",
        "strategy": "direction_prefixes",
        "prefixes": {"to_runyoro": PREFIX_TO_RNY.strip(), "to_english": PREFIX_TO_ENG.strip()},
        "total_pairs": len(all_pairs),
        "train": len(train_pairs),
        "val": len(val_pairs),
        "test": len(test_pairs),
    }
    json.dump(meta, open(OUTPUT_DIR / "training_metadata.json", "w"), indent=2)

    logger.info("\n" + "=" * 70)
    logger.info("DONE! Model: %s", OUTPUT_DIR)
    logger.info("  Use '>>rny<< {text}' for English->Runyoro")
    logger.info("  Use '>>eng<< {text}' for Runyoro->English")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
