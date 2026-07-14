#!/usr/bin/env python3
"""
train_v5_full.py
================
Full pipeline: Process new raw data + back-translation + train v5 models.

Steps:
  1. Load and clean new raw data: "sentence pairs (3).xlsx"
  2. Combine with all existing v4 training data
  3. Augment (token deletion, swap)
  4. Train bidirectional model v5-base (English↔Runyoro)
  5. Back-translate: generate synthetic pairs for both directions
  6. Train final v5 model with back-translated data included

Output models:
  - models/checkpoints/runyoro-nmt-v5/       (final bidirectional with back-translation)

Run:
    python scripts/train_v5_full.py
    python scripts/train_v5_full.py --clean-only       # Only process data, no training
    python scripts/train_v5_full.py --skip-backtrans   # Skip back-translation step
"""

import argparse
import gc
import os
import re
import sys
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
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(ROOT / "training_v5.log"), encoding="utf-8"),
    ],
)
logger = logging.getLogger("train_v5")

import pandas as pd
import torch

random.seed(42)

# =====================================================================
# PATHS
# =====================================================================
RAW_DIR = ROOT.parent / "raw"
V4_DATA_DIR = ROOT / "data" / "v4_training"
V5_DATA_DIR = ROOT / "data" / "v5_training"
V5_DATA_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR_BASE = ROOT / "models" / "checkpoints" / "runyoro-nmt-v5-base"
OUTPUT_DIR_FINAL = ROOT / "models" / "checkpoints" / "runyoro-nmt-v5"

MODEL_NAME = "facebook/nllb-200-distilled-1.3B"
MAX_LEN = 256


# =====================================================================
# CLEANING FUNCTIONS
# =====================================================================
def clean_text(text: str) -> str:
    """Normalize and clean text."""
    text = str(text)
    if text.lower() == "nan" or not text.strip():
        return ""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^[-–—]+\s*", "", text).strip()
    return text


def is_valid_pair(eng: str, rny: str) -> bool:
    """Filter out bad pairs."""
    if len(eng) < 5 or len(rny) < 5:
        return False
    if eng.lower() == "nan" or rny.lower() == "nan":
        return False
    # Remove dictionary-style entries
    if eng.startswith("To ") and len(eng.split()) < 4:
        return False
    # Remove grammar annotations
    if re.search(r"\(v\.\w+\)", eng) or re.search(r"\(v\.\w+\)", rny):
        return False
    # Length ratio check
    if len(eng) > 0 and len(rny) > 0:
        ratio = max(len(eng), len(rny)) / min(len(eng), len(rny))
        if ratio > 8:
            return False
    return True


# =====================================================================
# STEP 1: Process new raw data — sentence pairs (3).xlsx
# =====================================================================
def process_new_raw_data() -> list:
    """
    Process sentence pairs (3).xlsx — same format as sentence variations (2).xlsx
    Columns: [ID, BaseID, Original English, Original Tense, Original Runyoro,
              Target Tense, Variation English, Variation Runyoro, Status]
    """
    logger.info("=" * 60)
    logger.info("STEP 1: Process new raw data — sentence pairs (3).xlsx")
    logger.info("=" * 60)

    raw_file = RAW_DIR / "sentence pairs (3).xlsx"
    if not raw_file.exists():
        logger.error("File not found: %s", raw_file)
        return []

    df = pd.read_excel(raw_file, header=None)
    logger.info("Loaded %d rows from %s", len(df), raw_file.name)

    pairs = []
    for _, row in df.iterrows():
        # Original pair (cols 2, 4)
        orig_eng = clean_text(row.iloc[2]) if pd.notna(row.iloc[2]) else ""
        orig_rny = clean_text(row.iloc[4]) if pd.notna(row.iloc[4]) else ""

        # Variation pair (cols 6, 7)
        var_eng = clean_text(row.iloc[6]) if pd.notna(row.iloc[6]) else ""
        var_rny = clean_text(row.iloc[7]) if pd.notna(row.iloc[7]) else ""

        if orig_eng and orig_rny and is_valid_pair(orig_eng, orig_rny):
            pairs.append((orig_rny, orig_eng))
        if var_eng and var_rny and is_valid_pair(var_eng, var_rny):
            pairs.append((var_rny, var_eng))

    # Deduplicate
    pairs = list(set(pairs))
    logger.info("Extracted %d clean pairs from sentence pairs (3)", len(pairs))
    return pairs


# =====================================================================
# STEP 2: Load existing v4 training data
# =====================================================================
def load_existing_data() -> list:
    """Load the existing v4 cleaned pairs."""
    logger.info("=" * 60)
    logger.info("STEP 2: Load existing v4 training data")
    logger.info("=" * 60)

    csv_path = V4_DATA_DIR / "cleaned_pairs.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        pairs = [
            (str(r["Runyoro"]).strip(), str(r["English"]).strip())
            for _, r in df.iterrows()
            if pd.notna(r.get("Runyoro")) and pd.notna(r.get("English"))
        ]
        logger.info("Loaded %d existing v4 cleaned pairs", len(pairs))
        return pairs
    else:
        logger.warning("v4 cleaned pairs not found at %s", csv_path)
        return []


# =====================================================================
# STEP 3: Augmentation
# =====================================================================
def augment_pairs(pairs: list) -> list:
    """Generate augmented pairs via token deletion and swap."""
    logger.info("=" * 60)
    logger.info("STEP 3: Data Augmentation")
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

    logger.info("Generated %d augmented pairs", len(augmented))
    return augmented


# =====================================================================
# STEP 4: Training function
# =====================================================================
def train_model(
    pairs: list,
    output_dir: Path,
    model_name: str = MODEL_NAME,
    epochs: int = 20,
    batch_size: int = 4,
    grad_accum: int = 8,
    lr: float = 5e-5,
    desc: str = "v5",
):
    """
    Train bidirectional NLLB model (no language codes).
    Creates both rny→eng and eng→rny directions in one model.
    """
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
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clear GPU
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPUs available: %d", torch.cuda.device_count())

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_fn(examples):
        src_enc = tokenizer(
            examples["src"], max_length=MAX_LEN, truncation=True, padding=False
        )
        tgt_enc = tokenizer(
            examples["tgt"], max_length=MAX_LEN, truncation=True, padding=False
        )
        src_enc["labels"] = tgt_enc["input_ids"]
        return src_enc

    def pairs_to_dataset(pair_list, description=""):
        ds = HFDataset.from_dict(
            {"src": [s for s, t in pair_list], "tgt": [t for s, t in pair_list]}
        )
        return ds.map(
            tokenize_fn, batched=True, remove_columns=["src", "tgt"], desc=description
        )

    # Bidirectional: rny→eng AND eng→rny
    fwd_ds = pairs_to_dataset(pairs, f"{desc} rny->eng")
    rev_ds = pairs_to_dataset([(e, r) for r, e in pairs], f"{desc} eng->rny")
    train_ds = concatenate_datasets([fwd_ds, rev_ds]).shuffle(seed=42)
    logger.info("Training dataset: %d samples (bidirectional from %d pairs)", len(train_ds), len(pairs))

    # Load model
    logger.info("Loading model: %s (FP32)", model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="auto",
        max_memory={0: "20GiB", 1: "20GiB"} if torch.cuda.device_count() >= 2 else None,
    )
    model.gradient_checkpointing_enable()
    logger.info(
        "Model loaded: %.1f M params",
        sum(p.numel() for p in model.parameters()) / 1e6,
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
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
        run_name=f"runyoro-nmt-{desc}",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        processing_class=tokenizer,
        data_collator=collator,
    )

    logger.info("=" * 60)
    logger.info("STARTING TRAINING: %s", desc)
    logger.info("  Model: %s (FP32, no language codes)", model_name)
    logger.info("  Data: %d pairs (%d bidirectional samples)", len(pairs), len(train_ds))
    logger.info("  Epochs: %d | Batch: %d x %d = %d effective", epochs, batch_size, grad_accum, batch_size * grad_accum)
    logger.info("=" * 60)

    trainer.train()
    logger.info("Training complete!")

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info("Model saved to: %s", output_dir)

    # Free memory
    del model, trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return str(output_dir)


# =====================================================================
# STEP 5: Back-translation
# =====================================================================
def back_translate(pairs: list, model_path: str, batch_size: int = 16) -> tuple:
    """
    Generate synthetic data via back-translation in both directions.
    Returns (bt_rny_to_eng, bt_eng_to_rny) — synthetic pairs.
    """
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    logger.info("=" * 60)
    logger.info("STEP 5: Back-Translation")
    logger.info("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading model for back-translation from: %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
    ).to(device)
    model.eval()

    def translate_batch(texts: list, bs: int = batch_size) -> list:
        """Translate texts using the model (no language codes)."""
        translations = []
        for i in range(0, len(texts), bs):
            batch = texts[i : i + bs]
            enc = tokenizer(
                batch,
                return_tensors="pt",
                max_length=MAX_LEN,
                truncation=True,
                padding=True,
            ).to(device)

            with torch.no_grad():
                out = model.generate(
                    **enc,
                    num_beams=4,
                    max_length=MAX_LEN,
                    length_penalty=1.0,
                )

            decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
            translations.extend([t.strip() for t in decoded])

            if (i // bs) % 5 == 0:
                logger.info("  Back-translated %d / %d", min(i + bs, len(texts)), len(texts))

        return translations

    # Direction 1: Runyoro → English (feed Runyoro, get synthetic English)
    runyoro_texts = [r for r, e in pairs]
    logger.info("Back-translating %d Runyoro sentences → synthetic English...", len(runyoro_texts))
    syn_english = translate_batch(runyoro_texts)
    bt_rny_to_eng = [(rny, syn_eng) for rny, syn_eng in zip(runyoro_texts, syn_english) if len(syn_eng.strip()) > 3]
    logger.info("Generated %d synthetic rny→eng pairs", len(bt_rny_to_eng))

    # Direction 2: English → Runyoro (feed English, get synthetic Runyoro)
    english_texts = [e for r, e in pairs]
    logger.info("Back-translating %d English sentences → synthetic Runyoro...", len(english_texts))
    syn_runyoro = translate_batch(english_texts)
    bt_eng_to_rny = [(syn_rny, eng) for syn_rny, eng in zip(syn_runyoro, english_texts) if len(syn_rny.strip()) > 3]
    logger.info("Generated %d synthetic eng→rny pairs", len(bt_eng_to_rny))

    # Free memory
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return bt_rny_to_eng, bt_eng_to_rny


# =====================================================================
# MAIN
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Train runyoro-nmt-v5 with back-translation")
    parser.add_argument("--clean-only", action="store_true", help="Only process data, no training")
    parser.add_argument("--skip-backtrans", action="store_true", help="Skip back-translation step")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device batch size")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("RUNYORO-NMT V5 FULL PIPELINE")
    logger.info("  New data + Back-translation + Bidirectional Training")
    logger.info("=" * 60)

    # --- Step 1: Process new raw data ---
    new_pairs = process_new_raw_data()

    # --- Step 2: Load existing data ---
    existing_pairs = load_existing_data()

    # --- Combine and deduplicate ---
    logger.info("=" * 60)
    logger.info("COMBINING DATA")
    logger.info("=" * 60)

    all_clean_pairs = existing_pairs + new_pairs
    # Deduplicate
    seen = set()
    unique_pairs = []
    for rny, eng in all_clean_pairs:
        key = (rny.lower().strip(), eng.lower().strip())
        if key not in seen:
            seen.add(key)
            unique_pairs.append((rny, eng))

    logger.info("  Existing v4 pairs: %d", len(existing_pairs))
    logger.info("  New sentence pairs (3): %d", len(new_pairs))
    logger.info("  Combined (after dedup): %d", len(unique_pairs))

    # --- Step 3: Augmentation ---
    augmented = augment_pairs(unique_pairs)

    # Combine original + augmented
    training_pairs = unique_pairs + augmented
    random.shuffle(training_pairs)

    # Save processed data
    pd.DataFrame(unique_pairs, columns=["Runyoro", "English"]).to_csv(
        V5_DATA_DIR / "cleaned_pairs.csv", index=False
    )
    pd.DataFrame(augmented, columns=["Runyoro", "English"]).to_csv(
        V5_DATA_DIR / "augmented_pairs.csv", index=False
    )
    pd.DataFrame(training_pairs, columns=["Runyoro", "English"]).to_csv(
        V5_DATA_DIR / "all_training_pairs.csv", index=False
    )

    logger.info("\nSaved to %s:", V5_DATA_DIR)
    logger.info("  cleaned_pairs.csv: %d", len(unique_pairs))
    logger.info("  augmented_pairs.csv: %d", len(augmented))
    logger.info("  all_training_pairs.csv: %d", len(training_pairs))

    if args.clean_only:
        logger.info("--clean-only mode. Done.")
        return

    # --- Step 4: Train base model (v5-base) ---
    logger.info("\n")
    base_model_path = train_model(
        pairs=training_pairs,
        output_dir=OUTPUT_DIR_BASE,
        epochs=args.epochs,
        batch_size=args.batch_size,
        desc="v5-base",
    )

    if args.skip_backtrans:
        # Just copy base as final
        logger.info("--skip-backtrans: Using base model as final v5")
        import shutil
        if OUTPUT_DIR_FINAL.exists():
            shutil.rmtree(OUTPUT_DIR_FINAL)
        shutil.copytree(base_model_path, str(OUTPUT_DIR_FINAL))
        logger.info("Final model at: %s", OUTPUT_DIR_FINAL)
        logger.info("ALL DONE!")
        return

    # --- Step 5: Back-translation ---
    bt_rny_eng, bt_eng_rny = back_translate(unique_pairs, base_model_path)

    # Save back-translated data
    pd.DataFrame(bt_rny_eng, columns=["Runyoro", "English"]).to_csv(
        V5_DATA_DIR / "back_translated_rny_to_eng.csv", index=False
    )
    pd.DataFrame(bt_eng_rny, columns=["Runyoro", "English"]).to_csv(
        V5_DATA_DIR / "back_translated_eng_to_rny.csv", index=False
    )
    logger.info("Saved back-translated data:")
    logger.info("  rny→eng synthetic: %d pairs", len(bt_rny_eng))
    logger.info("  eng→rny synthetic: %d pairs", len(bt_eng_rny))

    # --- Step 6: Train final model with back-translated data ---
    logger.info("\n")
    logger.info("=" * 60)
    logger.info("STEP 6: Train Final Model (v5) with Back-Translation")
    logger.info("=" * 60)

    # Combine: original + augmented + back-translated
    final_pairs = training_pairs + bt_rny_eng + bt_eng_rny
    random.shuffle(final_pairs)

    pd.DataFrame(final_pairs, columns=["Runyoro", "English"]).to_csv(
        V5_DATA_DIR / "final_training_pairs.csv", index=False
    )
    logger.info("Final training set: %d pairs", len(final_pairs))
    logger.info("  Original + augmented: %d", len(training_pairs))
    logger.info("  Back-translated rny→eng: %d", len(bt_rny_eng))
    logger.info("  Back-translated eng→rny: %d", len(bt_eng_rny))

    train_model(
        pairs=final_pairs,
        output_dir=OUTPUT_DIR_FINAL,
        epochs=args.epochs,
        batch_size=args.batch_size,
        desc="v5-final",
    )

    logger.info("\n" + "=" * 60)
    logger.info("ALL DONE!")
    logger.info("  Base model: %s", OUTPUT_DIR_BASE)
    logger.info("  Final model (with back-translation): %s", OUTPUT_DIR_FINAL)
    logger.info("  Training data: %s", V5_DATA_DIR)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
