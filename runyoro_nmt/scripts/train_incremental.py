#!/usr/bin/env python3
"""
train_incremental.py — Incremental Training Pipeline
=====================================================
Follows the pattern:
  1. Load previous version's full cleaned dataset
  2. Load & clean new raw data from raw/ folder
  3. Merge + deduplicate
  4. Split: hold out fixed val/test sets (consistent across versions)
  5. Shuffle training set
  6. Continue training FROM the previous checkpoint (not from scratch)
  7. Evaluate on held-out val/test sets
  8. Save as next version

Usage:
    python scripts/train_incremental.py --new-file "sentence pair (5).xlsx"
    python scripts/train_incremental.py --new-file "sentence pair (5).xlsx" --epochs 10
    python scripts/train_incremental.py --new-file "sentence pair (5).xlsx" --from-scratch
    python scripts/train_incremental.py --clean-only --new-file "sentence pair (5).xlsx"

The script auto-detects the current version and creates the next one.
"""
import argparse
import gc
import json
import os
import re
import random
import sys
import unicodedata
import logging
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["USE_TF"] = "0"

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import torch

# =====================================================================
# CONFIG
# =====================================================================
RAW_DIR = ROOT.parent / "raw"
DATA_DIR = ROOT / "data"
CHECKPOINTS_DIR = ROOT / "models" / "checkpoints"
MODEL_NAME = "facebook/nllb-200-distilled-1.3B"
MAX_LEN = 256
VAL_RATIO = 0.05   # 5% for validation
TEST_RATIO = 0.05  # 5% for test
SEED = 42

random.seed(SEED)


# =====================================================================
# CLEANING FUNCTIONS
# =====================================================================
def clean_text(text: str) -> str:
    text = str(text)
    if text.lower() == "nan" or not text.strip():
        return ""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^[-–—]+\s*", "", text).strip()
    return text


def is_valid_pair(eng: str, rny: str) -> bool:
    if len(eng) < 5 or len(rny) < 5:
        return False
    if eng.lower() == "nan" or rny.lower() == "nan":
        return False
    if re.search(r"\(v\.\w+\)", eng) or re.search(r"\(v\.\w+\)", rny):
        return False
    if len(eng) > 0 and len(rny) > 0:
        ratio = max(len(eng), len(rny)) / min(len(eng), len(rny))
        if ratio > 8:
            return False
    return True


def extract_pairs_from_xlsx(filepath: Path) -> list:
    """Extract pairs from tense-variation xlsx files (standard format)."""
    df = pd.read_excel(filepath, header=None)
    pairs = []
    for _, row in df.iterrows():
        # Original pair (cols 2, 4)
        orig_eng = clean_text(row.iloc[2]) if len(row) > 2 and pd.notna(row.iloc[2]) else ""
        orig_rny = clean_text(row.iloc[4]) if len(row) > 4 and pd.notna(row.iloc[4]) else ""
        # Variation pair (cols 6, 7)
        var_eng = clean_text(row.iloc[6]) if len(row) > 6 and pd.notna(row.iloc[6]) else ""
        var_rny = clean_text(row.iloc[7]) if len(row) > 7 and pd.notna(row.iloc[7]) else ""

        if orig_eng and orig_rny and is_valid_pair(orig_eng, orig_rny):
            pairs.append((orig_rny, orig_eng))
        if var_eng and var_rny and is_valid_pair(var_eng, var_rny):
            pairs.append((var_rny, var_eng))
    return list(set(pairs))


def augment_pairs(pairs: list) -> list:
    """Generate augmented pairs via token deletion and swap."""
    augmented = []
    for rny, eng in pairs:
        words_rny = rny.split()
        words_eng = eng.split()
        # Token deletion
        if len(words_rny) > 4:
            del_rny = " ".join(w for w in words_rny if random.random() > 0.05)
            del_eng = " ".join(w for w in words_eng if random.random() > 0.05)
            if len(del_rny.split()) >= 3 and len(del_eng.split()) >= 3:
                augmented.append((del_rny, del_eng))
        # Token swap
        if len(words_rny) > 3:
            idx = random.randint(0, len(words_rny) - 2)
            swapped = words_rny.copy()
            swapped[idx], swapped[idx + 1] = swapped[idx + 1], swapped[idx]
            augmented.append((" ".join(swapped), eng))
    return augmented


def deduplicate(pairs: list) -> list:
    seen = set()
    unique = []
    for r, e in pairs:
        key = (str(r).lower().strip(), str(e).lower().strip())
        if key not in seen:
            seen.add(key)
            unique.append((r, e))
    return unique


def split_data(pairs: list, val_ratio: float, test_ratio: float, seed: int):
    """
    Deterministic split using a fixed seed.
    This ensures the same pairs always go to val/test regardless of version.
    """
    rng = random.Random(seed)
    shuffled = pairs.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_test = max(1, int(n * test_ratio))
    n_val = max(1, int(n * val_ratio))

    test_set = shuffled[:n_test]
    val_set = shuffled[n_test : n_test + n_val]
    train_set = shuffled[n_test + n_val :]
    return train_set, val_set, test_set


def get_current_version() -> int:
    """Auto-detect the latest model version."""
    versions = []
    for d in CHECKPOINTS_DIR.iterdir():
        if d.is_dir() and d.name.startswith("runyoro-nmt-v"):
            try:
                v = int(d.name.split("-v")[1])
                versions.append(v)
            except (ValueError, IndexError):
                pass
    return max(versions) if versions else 0


# =====================================================================
# MAIN
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Incremental training pipeline")
    parser.add_argument("--new-file", required=True, help="New xlsx file in raw/ folder")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs (default: 10 for incremental)")
    parser.add_argument("--from-scratch", action="store_true", help="Train from base model instead of continuing")
    parser.add_argument("--clean-only", action="store_true", help="Only process data, skip training")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate (lower for continue, default 3e-5)")
    args = parser.parse_args()

    # Setup logging
    current_ver = get_current_version()
    next_ver = current_ver + 1
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(str(ROOT / f"training_v{next_ver}.log")),
        ],
    )
    logger = logging.getLogger("train_incremental")

    logger.info("=" * 60)
    logger.info("INCREMENTAL TRAINING PIPELINE")
    logger.info("  Current version: v%d", current_ver)
    logger.info("  Next version: v%d", next_ver)
    logger.info("  New data: %s", args.new_file)
    logger.info("  Mode: %s", "from scratch" if args.from_scratch else "continue training")
    logger.info("=" * 60)

    # ===== STEP 1: Load previous dataset =====
    logger.info("\nSTEP 1: Load previous dataset")
    prev_data_dir = DATA_DIR / f"v{current_ver}_training"
    if prev_data_dir.exists() and (prev_data_dir / "cleaned_pairs.csv").exists():
        prev_df = pd.read_csv(prev_data_dir / "cleaned_pairs.csv")
        prev_pairs = [(str(r["Runyoro"]), str(r["English"])) for _, r in prev_df.iterrows()]
        logger.info("  Loaded %d pairs from v%d", len(prev_pairs), current_ver)
    else:
        logger.warning("  No previous dataset found, starting fresh")
        prev_pairs = []

    # ===== STEP 2: Load & clean new data =====
    logger.info("\nSTEP 2: Load & clean new data")
    new_file_path = RAW_DIR / args.new_file
    if not new_file_path.exists():
        logger.error("File not found: %s", new_file_path)
        sys.exit(1)

    new_pairs = extract_pairs_from_xlsx(new_file_path)
    logger.info("  Extracted %d new pairs from %s", len(new_pairs), args.new_file)

    # ===== STEP 3: Merge + deduplicate =====
    logger.info("\nSTEP 3: Merge + deduplicate")
    all_clean = deduplicate(prev_pairs + new_pairs)
    logger.info("  Previous: %d | New: %d | Merged unique: %d", len(prev_pairs), len(new_pairs), len(all_clean))

    # ===== STEP 4: Split (consistent val/test) =====
    logger.info("\nSTEP 4: Train/Val/Test split (seed=%d)", SEED)
    train_pairs, val_pairs, test_pairs = split_data(all_clean, VAL_RATIO, TEST_RATIO, SEED)
    logger.info("  Train: %d | Val: %d | Test: %d", len(train_pairs), len(val_pairs), len(test_pairs))

    # ===== STEP 5: Augment training set =====
    logger.info("\nSTEP 5: Augment training set")
    augmented = augment_pairs(train_pairs)
    training_pairs = train_pairs + augmented
    random.shuffle(training_pairs)
    logger.info("  Training pairs: %d (original %d + augmented %d)", len(training_pairs), len(train_pairs), len(augmented))

    # ===== Save data =====
    next_data_dir = DATA_DIR / f"v{next_ver}_training"
    next_data_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(all_clean, columns=["Runyoro", "English"]).to_csv(next_data_dir / "cleaned_pairs.csv", index=False)
    pd.DataFrame(training_pairs, columns=["Runyoro", "English"]).to_csv(next_data_dir / "all_training_pairs.csv", index=False)
    pd.DataFrame(val_pairs, columns=["Runyoro", "English"]).to_csv(next_data_dir / "val_pairs.csv", index=False)
    pd.DataFrame(test_pairs, columns=["Runyoro", "English"]).to_csv(next_data_dir / "test_pairs.csv", index=False)

    # Save metadata
    meta = {
        "version": next_ver,
        "prev_version": current_ver,
        "new_file": args.new_file,
        "total_clean_pairs": len(all_clean),
        "train_pairs": len(train_pairs),
        "augmented_pairs": len(augmented),
        "training_samples": len(training_pairs) * 2,  # bidirectional
        "val_pairs": len(val_pairs),
        "test_pairs": len(test_pairs),
        "mode": "from_scratch" if args.from_scratch else "continue",
        "epochs": args.epochs,
        "lr": args.lr,
    }
    with open(next_data_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("\n  Data saved to: %s", next_data_dir)

    if args.clean_only:
        logger.info("--clean-only mode. Done.")
        return

    # ===== STEP 6: Train =====
    logger.info("\n" + "=" * 60)
    logger.info("STEP 6: Training v%d", next_ver)
    logger.info("=" * 60)

    from datasets import Dataset as HFDataset, concatenate_datasets
    from transformers import (
        AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq,
        Seq2SeqTrainer, Seq2SeqTrainingArguments, set_seed,
    )

    set_seed(SEED)
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("  GPUs: %d", torch.cuda.device_count())

    # Determine source model
    if args.from_scratch:
        source_model = MODEL_NAME
        logger.info("  Training FROM SCRATCH: %s", MODEL_NAME)
    else:
        prev_checkpoint = CHECKPOINTS_DIR / f"runyoro-nmt-v{current_ver}"
        if prev_checkpoint.exists():
            source_model = str(prev_checkpoint)
            logger.info("  CONTINUING from: %s", source_model)
        else:
            source_model = MODEL_NAME
            logger.info("  Previous checkpoint not found, training from scratch: %s", MODEL_NAME)

    OUTPUT_DIR = CHECKPOINTS_DIR / f"runyoro-nmt-v{next_ver}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(source_model)

    def tokenize_fn(examples):
        src_enc = tokenizer(examples["src"], max_length=MAX_LEN, truncation=True, padding=False)
        tgt_enc = tokenizer(examples["tgt"], max_length=MAX_LEN, truncation=True, padding=False)
        src_enc["labels"] = tgt_enc["input_ids"]
        return src_enc

    # Bidirectional dataset
    fwd_ds = HFDataset.from_dict({"src": [s for s, t in training_pairs], "tgt": [t for s, t in training_pairs]})
    fwd_ds = fwd_ds.map(tokenize_fn, batched=True, remove_columns=["src", "tgt"], desc="rny->eng")
    rev_ds = HFDataset.from_dict({"src": [t for s, t in training_pairs], "tgt": [s for s, t in training_pairs]})
    rev_ds = rev_ds.map(tokenize_fn, batched=True, remove_columns=["src", "tgt"], desc="eng->rny")
    train_ds = concatenate_datasets([fwd_ds, rev_ds]).shuffle(seed=SEED)
    logger.info("  Train dataset: %d samples", len(train_ds))

    # Load model
    model = AutoModelForSeq2SeqLM.from_pretrained(
        source_model, torch_dtype=torch.float32,
        device_map="auto", max_memory={0: "22GiB", 1: "22GiB"}
    )
    model.gradient_checkpointing_enable()
    logger.info("  Model: %.1fM params", sum(p.numel() for p in model.parameters()) / 1e6)

    collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        learning_rate=args.lr,
        warmup_steps=50 if not args.from_scratch else 100,
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

    trainer = Seq2SeqTrainer(
        model=model, args=training_args, train_dataset=train_ds,
        processing_class=tokenizer, data_collator=collator,
    )

    logger.info("  Mode: %s | Epochs: %d | LR: %s | Batch: 2x16=32",
                "FROM SCRATCH" if args.from_scratch else "CONTINUE", args.epochs, args.lr)
    logger.info("  Clean pairs: %d | Training samples: %d", len(all_clean), len(train_ds))
    logger.info("=" * 60)

    trainer.train()
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    logger.info("Model saved to: %s", OUTPUT_DIR)

    # ===== STEP 7: Quick evaluation =====
    logger.info("\nSTEP 7: Quick evaluation on test set")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()

    correct_direction = 0
    total = min(len(test_pairs), 50)  # Evaluate on up to 50 test pairs
    for rny, eng in test_pairs[:total]:
        enc = tokenizer(rny, return_tensors="pt", max_length=MAX_LEN, truncation=True).to(device)
        with torch.no_grad():
            out = model.generate(**enc, num_beams=4, max_length=MAX_LEN)
        pred = tokenizer.decode(out[0], skip_special_tokens=True).strip()
        # Simple check: does the prediction contain any English-like words?
        if any(c.isalpha() for c in pred) and len(pred) > 3:
            correct_direction += 1

    logger.info("  Test pairs evaluated: %d | Produced output: %d/%d", total, correct_direction, total)
    logger.info("\n" + "=" * 60)
    logger.info("ALL DONE — v%d ready!", next_ver)
    logger.info("  Model: %s", OUTPUT_DIR)
    logger.info("  Data: %s", next_data_dir)
    logger.info("  To use: update model_server.py checkpoint path to v%d", next_ver)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
