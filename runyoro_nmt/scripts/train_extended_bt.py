#!/usr/bin/env python3
"""
train_extended_bt.py - Fine-tune the extended tokenizer model with back-translation.

Pipeline:
1. Load all clean + augmented Runyoro-English pairs
2. Back-translate using the extended model to generate synthetic data
3. Combine original + augmented + back-translated data
4. Fine-tune the extended model on the full combined dataset (bidirectional)

Usage:
    python runyoro_nmt/scripts/train_extended_bt.py --epochs 15
"""
import argparse
import gc
import json
import logging
import os
import random
import sys
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
        logging.FileHandler(str(ROOT / "training_extended_bt.log"), encoding="utf-8"),
    ],
)
logger = logging.getLogger("train_ext_bt")

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

DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "models" / "checkpoints"
EXTENDED_MODEL = CKPT_DIR / "runyoro-inc-v4-extended"
OUTPUT_DIR = CKPT_DIR / "runyoro-ext-v1"


def gather_all_pairs():
    """Gather all clean + augmented pairs from all versions."""
    all_pairs = []

    # 1. Cleaned pairs from all versions
    for csv_path in sorted(DATA_DIR.rglob("cleaned_pairs.csv")):
        try:
            df = pd.read_csv(csv_path)
            if "Runyoro" in df.columns and "English" in df.columns:
                pairs = [(str(r), str(e)) for r, e in zip(df["Runyoro"], df["English"])
                         if str(r).strip() and str(e).strip()
                         and str(r).lower() != "nan" and str(e).lower() != "nan"]
                logger.info("  %s: %d clean pairs", csv_path.parent.name, len(pairs))
                all_pairs.extend(pairs)
        except Exception as e:
            logger.warning("  Skipping %s: %s", csv_path, e)

    # 2. Augmented pairs from all versions
    for csv_path in sorted(DATA_DIR.rglob("augmented_pairs.csv")):
        try:
            df = pd.read_csv(csv_path)
            if "Runyoro" in df.columns and "English" in df.columns:
                pairs = [(str(r), str(e)) for r, e in zip(df["Runyoro"], df["English"])
                         if str(r).strip() and str(e).strip()
                         and str(r).lower() != "nan" and str(e).lower() != "nan"]
                logger.info("  %s: %d augmented pairs", csv_path.parent.name, len(pairs))
                all_pairs.extend(pairs)
        except Exception as e:
            logger.warning("  Skipping %s: %s", csv_path, e)

    # 3. Back-translated data from incremental runs
    for csv_path in sorted(DATA_DIR.rglob("back_translated*.csv")):
        try:
            df = pd.read_csv(csv_path)
            if "Runyoro" in df.columns and "English" in df.columns:
                pairs = [(str(r), str(e)) for r, e in zip(df["Runyoro"], df["English"])
                         if str(r).strip() and str(e).strip()
                         and str(r).lower() != "nan" and str(e).lower() != "nan"]
                logger.info("  %s: %d BT pairs", csv_path.parent.name, len(pairs))
                all_pairs.extend(pairs)
        except Exception as e:
            logger.warning("  Skipping %s: %s", csv_path, e)

    # Deduplicate
    seen = set()
    unique = []
    for r, e in all_pairs:
        key = (r.lower().strip(), e.lower().strip())
        if key not in seen:
            seen.add(key)
            unique.append((r, e))

    logger.info("Total unique pairs gathered: %d", len(unique))
    return unique


def back_translate(pairs, model_path, bs=16):
    """Generate synthetic back-translated pairs using the extended model."""
    gc.collect()
    torch.cuda.empty_cache()
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    use_bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    logger.info("Loading model for back-translation: %s", model_path)
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, torch_dtype=dtype).to(device)
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
            if (i // bs) % 5 == 0:
                logger.info("  BT progress: %d / %d", min(i + bs, len(texts)), len(texts))
        return results

    # Runyoro -> synthetic English
    logger.info("Back-translating Runyoro -> synthetic English (%d sentences)...", len(pairs))
    rny_texts = [r for r, e in pairs]
    syn_eng = translate_batch(rny_texts)
    bt_rny_eng = [(r, se) for r, se in zip(rny_texts, syn_eng) if len(se.strip()) > 3]
    logger.info("  Generated %d rny->eng pairs", len(bt_rny_eng))

    # English -> synthetic Runyoro
    logger.info("Back-translating English -> synthetic Runyoro (%d sentences)...", len(pairs))
    eng_texts = [e for r, e in pairs]
    syn_rny = translate_batch(eng_texts)
    bt_eng_rny = [(sr, e) for sr, e in zip(syn_rny, eng_texts) if len(sr.strip()) > 3]
    logger.info("  Generated %d eng->rny pairs", len(bt_eng_rny))

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return bt_rny_eng + bt_eng_rny


def train_model(pairs, model_path, output_dir, epochs, lr):
    """Fine-tune the extended model on all combined data."""
    gc.collect()
    torch.cuda.empty_cache()

    logger.info("Loading tokenizer from: %s", model_path)
    tok = AutoTokenizer.from_pretrained(model_path)

    def tok_fn(ex):
        s = tok(ex["src"], max_length=MAX_LEN, truncation=True, padding=False)
        t = tok(ex["tgt"], max_length=MAX_LEN, truncation=True, padding=False)
        s["labels"] = t["input_ids"]
        return s

    # Bidirectional: both directions
    fwd = HFDataset.from_dict({"src": [r for r, e in pairs], "tgt": [e for r, e in pairs]})
    fwd = fwd.map(tok_fn, batched=True, remove_columns=["src", "tgt"], desc="Forward")
    rev = HFDataset.from_dict({"src": [e for r, e in pairs], "tgt": [r for r, e in pairs]})
    rev = rev.map(tok_fn, batched=True, remove_columns=["src", "tgt"], desc="Reverse")
    ds = concatenate_datasets([fwd, rev]).shuffle(seed=SEED)
    logger.info("Training dataset: %d samples (bidirectional from %d pairs)", len(ds), len(pairs))

    n_gpus = torch.cuda.device_count()
    logger.info("GPUs available: %d", n_gpus)

    use_bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, torch_dtype=dtype)
    model.gradient_checkpointing_enable()

    output_dir.mkdir(parents=True, exist_ok=True)
    col = DataCollatorForSeq2Seq(tok, model=model, label_pad_token_id=-100, pad_to_multiple_of=8)

    # Calculate effective steps: with 40k samples, batch=16, 1 grad_accum => ~2540 steps/epoch
    # 3 epochs = ~7620 steps × 20s = ~42 hours. With 20k data, 3 epochs is plenty.
    targs = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        learning_rate=lr,
        warmup_steps=100,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        fp16=not use_bf16,
        bf16=use_bf16,
        save_strategy="epoch",
        eval_strategy="no",
        logging_steps=50,
        save_total_limit=2,
        predict_with_generate=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        optim="adamw_torch",
        save_safetensors=True,
        save_only_model=True,
        report_to=["none"],
    )

    trainer = Seq2SeqTrainer(
        model=model, args=targs, train_dataset=ds, processing_class=tok, data_collator=col
    )
    trainer.train()

    model.save_pretrained(str(output_dir))
    tok.save_pretrained(str(output_dir))
    logger.info("Model saved to: %s", output_dir)

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()
    return str(output_dir)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune extended tokenizer model with BT")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--bt-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--skip-bt", action="store_true", help="Skip back-translation, use existing data only")
    args = parser.parse_args()

    model_path = str(EXTENDED_MODEL)
    if not EXTENDED_MODEL.exists():
        logger.error("Extended model not found at %s. Run extend_tokenizer.py first.", EXTENDED_MODEL)
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("EXTENDED MODEL FINE-TUNING WITH BACK-TRANSLATION")
    logger.info("=" * 70)
    logger.info("  Model: %s", model_path)
    logger.info("  Output: %s", OUTPUT_DIR)
    logger.info("  Epochs (base): %d | Epochs (BT): %d | LR: %s", args.epochs, args.bt_epochs, args.lr)
    logger.info("=" * 70)

    # Step 1: Gather all data
    logger.info("\nSTEP 1: Gathering all clean + augmented pairs...")
    all_pairs = gather_all_pairs()

    # Step 2: Initial fine-tuning on all existing data
    logger.info("\nSTEP 2: Base fine-tuning on %d pairs...", len(all_pairs))
    base_output = CKPT_DIR / "runyoro-ext-v1-base"
    base_path = train_model(all_pairs, model_path, base_output, args.epochs, args.lr)

    # Step 3: Back-translation
    if not args.skip_bt:
        logger.info("\nSTEP 3: Back-translation...")
        # Use only the unique clean pairs for BT (not augmented duplicates)
        clean_pairs = []
        for csv_path in sorted(DATA_DIR.rglob("cleaned_pairs.csv")):
            try:
                df = pd.read_csv(csv_path)
                if "Runyoro" in df.columns and "English" in df.columns:
                    pairs = [(str(r), str(e)) for r, e in zip(df["Runyoro"], df["English"])
                             if str(r).strip() and str(e).strip()
                             and str(r).lower() != "nan" and str(e).lower() != "nan"]
                    clean_pairs.extend(pairs)
            except Exception:
                pass
        # Deduplicate clean pairs
        seen = set()
        unique_clean = []
        for r, e in clean_pairs:
            key = (r.lower().strip(), e.lower().strip())
            if key not in seen:
                seen.add(key)
                unique_clean.append((r, e))
        logger.info("  Clean pairs for BT: %d", len(unique_clean))

        bt_pairs = back_translate(unique_clean, base_path)
        logger.info("  Total BT pairs: %d", len(bt_pairs))

        # Save BT data
        bt_dir = DATA_DIR / "extended_bt"
        bt_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(bt_pairs, columns=["Runyoro", "English"]).to_csv(
            bt_dir / "back_translated.csv", index=False
        )

        # Step 4: Final training with BT data
        logger.info("\nSTEP 4: Final training with BT data...")
        final_pairs = all_pairs + bt_pairs
        random.shuffle(final_pairs)
        logger.info("  Final dataset: %d pairs (orig %d + BT %d)", len(final_pairs), len(all_pairs), len(bt_pairs))

        pd.DataFrame(final_pairs, columns=["Runyoro", "English"]).to_csv(
            bt_dir / "final_training.csv", index=False
        )

        final_path = train_model(final_pairs, base_path, OUTPUT_DIR, args.bt_epochs, 2e-5)
    else:
        logger.info("\nSkipping BT. Copying base as final.")
        import shutil
        if OUTPUT_DIR.exists():
            shutil.rmtree(OUTPUT_DIR)
        shutil.copytree(base_output, OUTPUT_DIR)
        final_path = str(OUTPUT_DIR)

    # Cleanup base checkpoint
    import shutil
    if base_output.exists():
        shutil.rmtree(base_output)
        logger.info("Cleaned up base checkpoint: %s", base_output)

    # Save metadata
    meta = {
        "model": "runyoro-ext-v1",
        "base": "runyoro-inc-v4-extended",
        "total_pairs": len(all_pairs),
        "bt_pairs": len(bt_pairs) if not args.skip_bt else 0,
        "epochs_base": args.epochs,
        "epochs_bt": args.bt_epochs,
        "lr": args.lr,
    }
    json.dump(meta, open(OUTPUT_DIR / "training_metadata.json", "w"), indent=2)

    logger.info("\n" + "=" * 70)
    logger.info("ALL DONE!")
    logger.info("  Final model: %s", OUTPUT_DIR)
    logger.info("  Total training data: %d pairs", len(all_pairs) + (len(bt_pairs) if not args.skip_bt else 0))
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
