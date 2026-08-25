#!/usr/bin/env python3
"""
train_clean_only.py — Clean training strategy for Runyoro NMT.

Uses ONLY the verified sentence pairs from the raw xlsx files.
No dictionary entries, no augmentation, no noise.
Back-translation only after base model converges.

Usage:
    python runyoro_nmt/scripts/train_clean_only.py
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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Single GPU — dataset is small, avoids DataParallel issues
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["USE_TF"] = "0"

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(ROOT / "training_clean.log"), encoding="utf-8"),
    ],
)
logger = logging.getLogger("train_clean")

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

RAW_DIR = ROOT.parent / "raw"
CKPT_DIR = ROOT / "models" / "checkpoints"
BASE_MODEL = "facebook/nllb-200-distilled-1.3B"
OUTPUT_DIR = CKPT_DIR / "runyoro-clean-v3"

# All raw xlsx files
RAW_FILES = sorted([f.name for f in (ROOT.parent / "raw").glob("*.xlsx")])

def clean_text(t: str) -> str:
    """Normalize and clean a text string."""
    t = str(t).strip()
    if t.lower() == "nan" or not t:
        return ""
    t = unicodedata.normalize("NFC", t)
    t = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"^[-\u2013\u2014]+\s*", "", t).strip()
    return t


def is_valid_pair(eng: str, rny: str) -> bool:
    """Check if a pair is a valid sentence translation."""
    if len(eng) < 5 or len(rny) < 5:
        return False
    if eng.lower() == "nan" or rny.lower() == "nan":
        return False
    # Skip POS tags
    if re.search(r"\(v\.\w+\)", eng) or re.search(r"\(v\.\w+\)", rny):
        return False
    # Skip extreme length mismatches
    ratio = max(len(eng), len(rny)) / max(min(len(eng), len(rny)), 1)
    if ratio > 8:
        return False
    return True


def extract_all_pairs() -> list:
    """Extract all sentence pairs from all 5 raw xlsx files."""
    all_pairs = []

    for fname in RAW_FILES:
        filepath = RAW_DIR / fname
        if not filepath.exists():
            logger.warning("File not found: %s", filepath)
            continue

        df = pd.read_excel(filepath, header=None)
        pairs_from_file = []

        if len(df.columns) >= 8:
            for idx, row in df.iterrows():
                # Skip header rows
                orig_eng = clean_text(row.iloc[2]) if pd.notna(row.iloc[2]) else ""
                orig_rny = clean_text(row.iloc[4]) if pd.notna(row.iloc[4]) else ""

                # Skip if it looks like a header
                if "english" in orig_eng.lower() and "runyoro" in orig_rny.lower():
                    continue
                if "original" in orig_eng.lower() and "reference" in orig_rny.lower():
                    continue

                if orig_eng and orig_rny and is_valid_pair(orig_eng, orig_rny):
                    pairs_from_file.append((orig_rny, orig_eng))

                # Variation pair: col 6 (English variation) + col 7 (Runyoro variation)
                var_eng = clean_text(row.iloc[6]) if pd.notna(row.iloc[6]) else ""
                var_rny = clean_text(row.iloc[7]) if pd.notna(row.iloc[7]) else ""

                if "english" in var_eng.lower() and "runyoro" in var_rny.lower():
                    continue

                if var_eng and var_rny and is_valid_pair(var_eng, var_rny):
                    pairs_from_file.append((var_rny, var_eng))

        logger.info("  %s: %d pairs", fname, len(pairs_from_file))
        all_pairs.extend(pairs_from_file)

    # Deduplicate
    seen = set()
    unique = []
    for rny, eng in all_pairs:
        key = (rny.lower().strip(), eng.lower().strip())
        if key not in seen:
            seen.add(key)
            unique.append((rny, eng))

    logger.info("Total unique pairs: %d (from %d raw)", len(unique), len(all_pairs))
    return unique


def split_data(pairs: list, val_ratio=0.1, test_ratio=0.05):
    """Split into train/val/test."""
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


def train_model(train_pairs, val_pairs, model_path, output_dir, epochs, lr, label_smoothing=0.1):
    """Fine-tune the model with early stopping."""
    gc.collect()
    torch.cuda.empty_cache()

    logger.info("Loading tokenizer from: %s", model_path)
    tok = AutoTokenizer.from_pretrained(model_path)

    def tok_fn(ex):
        # Tokenize source and target separately, assign target as labels.
        # The DataCollatorForSeq2Seq will create decoder_input_ids from labels.
        src = tok(ex["src"], max_length=MAX_LEN, truncation=True, padding=False)
        tgt = tok(ex["tgt"], max_length=MAX_LEN, truncation=True, padding=False)
        src["labels"] = tgt["input_ids"]
        return src

    # Bidirectional training
    def make_dataset(pairs):
        fwd = HFDataset.from_dict({"src": [r for r, e in pairs], "tgt": [e for r, e in pairs]})
        fwd = fwd.map(tok_fn, batched=True, remove_columns=["src", "tgt"], desc="fwd")
        rev = HFDataset.from_dict({"src": [e for r, e in pairs], "tgt": [r for r, e in pairs]})
        rev = rev.map(tok_fn, batched=True, remove_columns=["src", "tgt"], desc="rev")
        return concatenate_datasets([fwd, rev]).shuffle(seed=SEED)

    train_ds = make_dataset(train_pairs)
    val_ds = make_dataset(val_pairs)
    logger.info("Train: %d samples | Val: %d samples", len(train_ds), len(val_ds))

    n_gpus = torch.cuda.device_count()
    logger.info("GPUs available: %d", n_gpus)

    use_bf16 = torch.cuda.is_bf16_supported()
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, torch_dtype=torch.float32)
    model.gradient_checkpointing_enable()

    output_dir.mkdir(parents=True, exist_ok=True)
    col = DataCollatorForSeq2Seq(tok, model=model, label_pad_token_id=-100, pad_to_multiple_of=8)

    targs = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,    # effective batch = 32, more gradient steps
        learning_rate=lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        label_smoothing_factor=0.0,       # Don't use Trainer's label smoother — it pops labels from inputs!
        fp16=False,
        bf16=False,
        save_strategy="epoch",
        eval_strategy="epoch",
        logging_steps=10,
        save_total_limit=2,
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
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tok,
        data_collator=col,
    )
    trainer.train()

    # Save best model
    model.save_pretrained(str(output_dir))
    tok.save_pretrained(str(output_dir))
    logger.info("Model saved to: %s", output_dir)

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()
    return str(output_dir)


def back_translate(pairs, model_path, bs=16):
    """Generate synthetic BT pairs, filtering low-quality outputs."""
    gc.collect()
    torch.cuda.empty_cache()
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    use_bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    logger.info("Loading model for BT: %s", model_path)
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, torch_dtype=dtype).to(device)
    model.eval()

    def translate_batch(texts):
        results = []
        for i in range(0, len(texts), bs):
            batch = texts[i:i + bs]
            enc = tok(batch, return_tensors="pt", max_length=MAX_LEN, truncation=True, padding=True).to(device)
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    num_beams=5,
                    max_length=MAX_LEN,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                )
            decoded = [t.strip() for t in tok.batch_decode(out, skip_special_tokens=True)]
            results.extend(decoded)
        return results

    # Runyoro -> synthetic English
    logger.info("BT: Runyoro -> synthetic English (%d sentences)...", len(pairs))
    rny_texts = [r for r, e in pairs]
    syn_eng = translate_batch(rny_texts)

    # Filter: synthetic output must be different from input and reasonable length
    bt_rny_eng = []
    for rny, syn_e in zip(rny_texts, syn_eng):
        if len(syn_e.strip()) < 5:
            continue
        if syn_e.lower().strip() == rny.lower().strip():
            continue  # Model just echoed back
        ratio = len(syn_e) / max(len(rny), 1)
        if ratio > 5 or ratio < 0.2:
            continue  # Extreme length mismatch
        bt_rny_eng.append((rny, syn_e))

    # English -> synthetic Runyoro
    logger.info("BT: English -> synthetic Runyoro (%d sentences)...", len(pairs))
    eng_texts = [e for r, e in pairs]
    syn_rny = translate_batch(eng_texts)

    bt_eng_rny = []
    for eng, syn_r in zip(eng_texts, syn_rny):
        if len(syn_r.strip()) < 5:
            continue
        if syn_r.lower().strip() == eng.lower().strip():
            continue
        ratio = len(syn_r) / max(len(eng), 1)
        if ratio > 5 or ratio < 0.2:
            continue
        bt_eng_rny.append((syn_r, eng))

    logger.info("BT results: rny->eng=%d, eng->rny=%d (filtered)", len(bt_rny_eng), len(bt_eng_rny))

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return bt_rny_eng + bt_eng_rny


def main():
    parser = argparse.ArgumentParser(description="Clean-only Runyoro NMT training")
    parser.add_argument("--base-epochs", type=int, default=8)
    parser.add_argument("--bt-epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--skip-bt", action="store_true")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("CLEAN TRAINING STRATEGY — Runyoro NMT")
    logger.info("=" * 70)
    logger.info("  Base model: %s", BASE_MODEL)
    logger.info("  Raw data: %s", RAW_DIR)
    logger.info("  Output: %s", OUTPUT_DIR)
    logger.info("  Base epochs: %d | BT epochs: %d | LR: %s", args.base_epochs, args.bt_epochs, args.lr)
    logger.info("  Label smoothing: 0.1")
    logger.info("  Early stopping patience: 3")
    logger.info("=" * 70)

    # ─── STEP 1: Extract clean pairs ───────────────────────────────
    logger.info("\nSTEP 1: Extracting sentence pairs from raw xlsx files...")
    all_pairs = extract_all_pairs()

    if len(all_pairs) < 50:
        logger.error("Too few pairs (%d). Check raw data.", len(all_pairs))
        sys.exit(1)

    # ─── STEP 2: Split ─────────────────────────────────────────────
    logger.info("\nSTEP 2: Splitting data...")
    train_pairs, val_pairs, test_pairs = split_data(all_pairs)
    logger.info("  Train: %d | Val: %d | Test: %d", len(train_pairs), len(val_pairs), len(test_pairs))

    # Save splits
    data_dir = ROOT / "data" / "clean_training"
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(train_pairs, columns=["Runyoro", "English"]).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame(val_pairs, columns=["Runyoro", "English"]).to_csv(data_dir / "val.csv", index=False)
    pd.DataFrame(test_pairs, columns=["Runyoro", "English"]).to_csv(data_dir / "test.csv", index=False)

    # ─── STEP 3: Base training ─────────────────────────────────────
    logger.info("\nSTEP 3: Base training on clean pairs only...")
    base_dir = OUTPUT_DIR / "base"
    base_path = train_model(
        train_pairs, val_pairs,
        BASE_MODEL, base_dir,
        epochs=args.base_epochs, lr=args.lr, label_smoothing=0.1
    )

    # ─── STEP 4: Back-translation ─────────────────────────────────
    if not args.skip_bt:
        logger.info("\nSTEP 4: Back-translation with quality filtering...")
        bt_pairs = back_translate(train_pairs, base_path)
        logger.info("  Filtered BT pairs: %d", len(bt_pairs))

        # Save BT data
        pd.DataFrame(bt_pairs, columns=["Runyoro", "English"]).to_csv(data_dir / "bt_pairs.csv", index=False)

        # ─── STEP 5: Final training with BT ───────────────────────
        logger.info("\nSTEP 5: Final training with original + BT data...")
        final_train = train_pairs + bt_pairs
        random.shuffle(final_train)
        logger.info("  Final training set: %d pairs (orig %d + BT %d)", len(final_train), len(train_pairs), len(bt_pairs))

        final_path = train_model(
            final_train, val_pairs,
            base_path, OUTPUT_DIR,
            epochs=args.bt_epochs, lr=1e-5, label_smoothing=0.1
        )
    else:
        logger.info("\nSkipping BT. Using base model as final.")
        import shutil
        if OUTPUT_DIR.exists() and OUTPUT_DIR != base_dir:
            shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
        shutil.copytree(base_dir, OUTPUT_DIR, dirs_exist_ok=True)

    # ─── Cleanup ──────────────────────────────────────────────────
    import shutil
    if base_dir.exists() and base_dir != OUTPUT_DIR:
        shutil.rmtree(base_dir)
        logger.info("Cleaned up base checkpoint")

    # ─── Save metadata ────────────────────────────────────────────
    meta = {
        "model": "runyoro-clean-v1",
        "strategy": "clean_only_with_bt",
        "base_model": BASE_MODEL,
        "total_clean_pairs": len(all_pairs),
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "test_pairs": len(test_pairs),
        "bt_pairs": len(bt_pairs) if not args.skip_bt else 0,
        "base_epochs": args.base_epochs,
        "bt_epochs": args.bt_epochs,
        "lr": args.lr,
        "label_smoothing": 0.1,
    }
    json.dump(meta, open(OUTPUT_DIR / "training_metadata.json", "w"), indent=2)

    logger.info("\n" + "=" * 70)
    logger.info("DONE!")
    logger.info("  Model: %s", OUTPUT_DIR)
    logger.info("  Clean pairs: %d | BT pairs: %d", len(all_pairs), len(bt_pairs) if not args.skip_bt else 0)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
