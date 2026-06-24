#!/usr/bin/env python3
"""
clean_and_train.py
==================
1. Loads ALL available data sources
2. Applies strict filtering rules:
   - Removes pairs where Runyoro side is a single word (dictionary headwords)
   - Removes grammar annotations like (v.i.), Oku-, Adv., etc.
   - Keeps only pairs where BOTH sides are complete text
   - No multiple translations crammed together
   - No dictionary metadata noise
3. Trains the model WITHOUT language codes (no nyk_Latn)
4. Uses both RTX 4090 GPUs

Run:
    python scripts/clean_and_train.py
    python scripts/clean_and_train.py --clean-only   # just clean, skip training
"""

import argparse
import gc
import json
import logging
import os
import re
import sys
import unicodedata
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(ROOT / "training_clean.log"), encoding="utf-8"),
    ],
)
logger = logging.getLogger("clean_and_train")

# =====================================================================
# PATHS
# =====================================================================
RAW_DATA_ROOT = ROOT.parent / "raw data"
SENTENCE_WORD_CLEANED = RAW_DATA_ROOT / "sentence and word cleaned"
NEW2_CLEANED = RAW_DATA_ROOT / "new2 cleaned"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "models" / "checkpoints" / "runyoro-nmt-v2"


# =====================================================================
# FILTERING FUNCTIONS
# =====================================================================

# Grammar/dictionary noise patterns to remove
GRAMMAR_NOISE = re.compile(
    r"""
    \(v\.?[it]\.?\)|        # (v.i.), (v.t.)
    \(n\.?\)|               # (n.)
    \(adj\.?\)|             # (adj.)
    \(adv\.?\)|             # (adv.)
    \bOku-|                 # Oku- prefix annotations
    \bAdv\.,?\s*|           # Adv., 
    \bv\.i\.?\s*|           # v.i.
    \bv\.t\.?\s*|           # v.t.
    \bn\.\s*|               # n. (standalone)
    \badj\.\s*|             # adj.
    \bmet\.,?\s*|           # met., (metaphorical)
    \bcf\.\s*|              # cf.
    \be\.g\.\s*|            # e.g.
    \bi\.e\.\s*|            # i.e.
    \bsyn\.\s*|             # syn.
    \blit\.\s*|             # lit.
    \bpl\.\s*|              # pl.
    \bsg\.\s*               # sg.
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Patterns that indicate a dictionary entry, not a translation
DICTIONARY_PATTERNS = re.compile(
    r"""
    ^To\s+[a-z]|            # Starts with "To verb" (infinitive definition)
    ^\([A-Za-z]+\)\s*To\s|  # (prefix) To verb
    ,\s*v\.[it]\b|          # contains ", v.i" or ", v.t"
    \boku\s*-\s*,|          # oku -, (prefix marker)
    ;\s*[A-Z].*;\s*[A-Z]|   # Multiple sentences with semicolons (multi-definition)
    \bOku\b.*\boku\b.*\boku\b  # Multiple "Oku" in one entry
    """,
    re.VERBOSE,
)

# Semicolons separating multiple unrelated translations
MULTI_TRANSLATION = re.compile(r";\s*[A-Z]")


def normalize_text(text: str) -> str:
    """Normalize and clean text."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    # Remove zero-width characters
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    # Remove grammar annotations
    text = GRAMMAR_NOISE.sub("", text)
    # Remove POS tags like [NOUN], [VERB]
    text = re.sub(r"\[[A-Z_]+\]\s*", "", text)
    # Remove standalone numbering like "1.", "2)", "a)"
    text = re.sub(r"^\d+[.)]\s*", "", text)
    # Remove leading dashes
    text = re.sub(r"^[-–—]+\s*", "", text)
    # Normalize quotes
    text = re.sub(r"[""''‛‟]", "'", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Remove trailing punctuation noise
    text = re.sub(r"\s*[;,]+\s*$", "", text)
    return text


def is_clean_pair(rny: str, eng: str) -> tuple:
    """
    Check if a pair meets our strict quality criteria.
    Returns (is_valid, reason) tuple.
    """
    if not rny or not eng:
        return False, "empty"

    # Rule 1: Both sides must have at least 2 words (no single-word dictionary headwords)
    rny_words = len(rny.split())
    eng_words = len(eng.split())
    if rny_words < 2 and eng_words < 2:
        return False, "single_word_both"

    # Rule 2: No dictionary-style definitions on the English side
    if DICTIONARY_PATTERNS.search(eng):
        return False, "dictionary_definition"

    # Rule 3: No multiple translations crammed together (3+ semicolons)
    if len(re.findall(r";", eng)) >= 3 or len(re.findall(r";", rny)) >= 3:
        return False, "multi_translation"

    # Rule 4: Neither side should be excessively long (likely garbage)
    if len(rny) > 300 or len(eng) > 300:
        return False, "too_long"

    # Rule 5: Neither side should be just numbers/symbols
    if re.fullmatch(r"[\d\s\W]+", rny) or re.fullmatch(r"[\d\s\W]+", eng):
        return False, "symbols_only"

    # Rule 6: Ratio check — one side shouldn't be 10x longer than the other
    if len(rny) > 0 and len(eng) > 0:
        ratio = max(len(rny), len(eng)) / min(len(rny), len(eng))
        if ratio > 8:
            return False, "length_ratio"

    # Rule 7: No remaining grammar annotation markers
    if re.search(r"\([a-z]{1,4}\.\)", eng) or re.search(r"\([a-z]{1,4}\.\)", rny):
        return False, "grammar_annotation"

    return True, "ok"


def load_tsv(path: Path) -> list:
    """Load TSV file into list of (col1, col2) tuples."""
    pairs = []
    if not path.exists():
        return pairs
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2:
                pairs.append((parts[0].strip(), parts[1].strip()))
    return pairs


def filter_pairs(pairs: list, source_name: str) -> list:
    """Apply strict filtering to pairs."""
    clean = []
    reasons = {}

    for rny, eng in pairs:
        rny = normalize_text(rny)
        eng = normalize_text(eng)

        valid, reason = is_clean_pair(rny, eng)
        if valid:
            clean.append((rny, eng))
        else:
            reasons[reason] = reasons.get(reason, 0) + 1

    logger.info("  [%s] %d -> %d clean pairs (removed %d)",
                source_name, len(pairs), len(clean), len(pairs) - len(clean))
    if reasons:
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            logger.info("    - %s: %d removed", reason, count)

    return clean


def deduplicate(pairs: list) -> list:
    """Remove duplicate pairs."""
    seen = set()
    unique = []
    for rny, eng in pairs:
        key = (rny.lower().strip(), eng.lower().strip())
        if key not in seen:
            seen.add(key)
            unique.append((rny, eng))
    return unique


def save_tsv(pairs: list, path: Path):
    """Save pairs to TSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rny, eng in pairs:
            f.write(f"{rny}\t{eng}\n")


# =====================================================================
# MAIN
# =====================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-only", action="store_true")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("CLEAN & TRAIN — Strict Filtering + No Language Codes")
    logger.info("=" * 60)

    # ===== PHASE 1: Load all data sources =====
    logger.info("\n--- Loading all data sources ---")

    all_raw_pairs = []

    # Source 1: sentence and word cleaned (already partially cleaned)
    import pandas as pd
    sw_sentences = []
    sw_words = []
    sw_examples = []
    
    sw_sent_path = SENTENCE_WORD_CLEANED / "sentences_cleaned.xlsx"
    sw_words_path = SENTENCE_WORD_CLEANED / "words_cleaned.xlsx"
    sw_examples_path = SENTENCE_WORD_CLEANED / "word_examples_cleaned.xlsx"
    
    if sw_sent_path.exists():
        df = pd.read_excel(sw_sent_path)
        sw_sentences = [(str(r["Runyoro"]).strip(), str(r["English"]).strip()) for _, r in df.iterrows() if pd.notna(r.get("Runyoro")) and pd.notna(r.get("English"))]
    if sw_words_path.exists():
        df = pd.read_excel(sw_words_path)
        sw_words = [(str(r["Runyoro"]).strip(), str(r["English"]).strip()) for _, r in df.iterrows() if pd.notna(r.get("Runyoro")) and pd.notna(r.get("English"))]
    if sw_examples_path.exists():
        df = pd.read_excel(sw_examples_path)
        sw_examples = [(str(r["Runyoro"]).strip(), str(r["English"]).strip()) for _, r in df.iterrows() if pd.notna(r.get("Runyoro")) and pd.notna(r.get("English"))]
    
    logger.info("  Sentence & Word: %d sentences, %d words, %d examples",
                len(sw_sentences), len(sw_words), len(sw_examples))

    # Source 2: new2 cleaned
    new2_pairs = load_tsv(NEW2_CLEANED / "cleaned_pairs.tsv")
    logger.info("  New2 cleaned: %d pairs", len(new2_pairs))

    # Source 3: existing processed data
    existing_pairs = load_tsv(PROCESSED_DIR / "cleaned_pairs.tsv")
    logger.info("  Existing processed: %d pairs", len(existing_pairs))

    # ===== PHASE 2: Apply strict filtering =====
    logger.info("\n--- Applying strict filtering ---")

    clean_sw_sentences = filter_pairs(sw_sentences, "sentence_word_sentences")
    clean_sw_words = filter_pairs(sw_words, "sentence_word_words")
    clean_sw_examples = filter_pairs(sw_examples, "sentence_word_examples")
    clean_new2 = filter_pairs(new2_pairs, "new2")
    clean_existing = filter_pairs(existing_pairs, "existing_processed")

    # ===== PHASE 3: Combine and deduplicate =====
    logger.info("\n--- Combining and deduplicating ---")

    all_clean = clean_sw_sentences + clean_sw_examples + clean_sw_words + clean_new2 + clean_existing
    logger.info("  Total before dedup: %d", len(all_clean))

    all_clean = deduplicate(all_clean)
    logger.info("  Total after dedup: %d", len(all_clean))

    # Separate sentences (2+ words each side) from word pairs
    sentence_pairs = [(r, e) for r, e in all_clean if len(r.split()) >= 2 and len(e.split()) >= 2]
    word_pairs = [(r, e) for r, e in all_clean if len(r.split()) < 2 or len(e.split()) < 2]
    logger.info("  Sentence pairs: %d", len(sentence_pairs))
    logger.info("  Word pairs: %d", len(word_pairs))

    # ===== PHASE 4: Save cleaned data =====
    logger.info("\n--- Saving cleaned data ---")

    output_data_dir = ROOT / "data" / "clean_v2"
    output_data_dir.mkdir(parents=True, exist_ok=True)

    save_tsv(all_clean, output_data_dir / "all_pairs.tsv")
    save_tsv(sentence_pairs, output_data_dir / "sentences_only.tsv")
    save_tsv(word_pairs, output_data_dir / "words_only.tsv")

    logger.info("  Saved to: %s", output_data_dir)
    logger.info("  - all_pairs.tsv: %d pairs", len(all_clean))
    logger.info("  - sentences_only.tsv: %d pairs", len(sentence_pairs))
    logger.info("  - words_only.tsv: %d pairs", len(word_pairs))

    # Summary
    summary = (
        f"Clean V2 Dataset Summary\n{'=' * 40}\n\n"
        f"Sources:\n"
        f"  Sentence & Word sentences: {len(clean_sw_sentences)}\n"
        f"  Sentence & Word words: {len(clean_sw_words)}\n"
        f"  New2 cleaned: {len(clean_new2)}\n"
        f"  Existing processed: {len(clean_existing)}\n\n"
        f"After dedup: {len(all_clean)} total pairs\n"
        f"  Sentence pairs (2+ words each side): {len(sentence_pairs)}\n"
        f"  Word pairs: {len(word_pairs)}\n"
    )
    (output_data_dir / "summary.txt").write_text(summary, encoding="utf-8")

    if args.clean_only:
        logger.info("--clean-only mode. Done.")
        return

    # ===== PHASE 5: Train without language codes =====
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 5: Training (no language codes, both GPUs)")
    logger.info("=" * 60)

    import torch
    from datasets import concatenate_datasets, Dataset as HFDataset
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        set_seed,
    )
    import evaluate as hf_evaluate

    set_seed(42)

    MODEL_NAME = "facebook/nllb-200-distilled-1.3B"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MAX_LEN = 256

    # Clear GPU memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPUs: %d available", torch.cuda.device_count())

    # Use ALL clean pairs for training (sentences are best, words supplement)
    train_pairs = all_clean
    logger.info("Training on %d pairs (bidirectional = %d samples)", len(train_pairs), len(train_pairs) * 2)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Tokenize WITHOUT language codes
    def tokenize_fn(examples):
        src_enc = tokenizer(
            examples["src"],
            max_length=MAX_LEN,
            truncation=True,
            padding=False,
        )
        tgt_enc = tokenizer(
            examples["tgt"],
            max_length=MAX_LEN,
            truncation=True,
            padding=False,
        )
        src_enc["labels"] = tgt_enc["input_ids"]
        return src_enc

    def pairs_to_dataset(pairs, desc=""):
        ds = HFDataset.from_dict({
            "src": [s for s, t in pairs],
            "tgt": [t for s, t in pairs],
        })
        return ds.map(tokenize_fn, batched=True, remove_columns=["src", "tgt"], desc=desc)

    # Build bidirectional dataset
    fwd_ds = pairs_to_dataset(train_pairs, "rny->en")
    rev_ds = pairs_to_dataset([(e, r) for r, e in train_pairs], "en->rny")
    train_ds = concatenate_datasets([fwd_ds, rev_ds]).shuffle(seed=42)
    logger.info("  Train dataset: %d samples", len(train_ds))

    # Load model across both GPUs
    logger.info("Loading model: %s", MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: "20GiB", 1: "20GiB"},
    )
    model.gradient_checkpointing_enable()
    logger.info("  Model loaded: %.1f M params", sum(p.numel() for p in model.parameters()) / 1e6)
    logger.info("  Device map: %s", getattr(model, 'hf_device_map', 'N/A'))

    # Metrics
    bleu_metric = hf_evaluate.load("sacrebleu")
    chrf_metric = hf_evaluate.load("chrf")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = [[max(t, 0) for t in seq] for seq in preds]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        clean_labels = [[max(l, 0) for l in label] for label in labels]
        decoded_labels = tokenizer.batch_decode(clean_labels, skip_special_tokens=True)
        try:
            bleu = bleu_metric.compute(predictions=decoded_preds, references=[[r] for r in decoded_labels])
            chrf = chrf_metric.compute(predictions=decoded_preds, references=[[r] for r in decoded_labels], word_order=2)
            return {"bleu": round(bleu["score"], 2), "chrf": round(chrf["score"], 2)}
        except Exception as e:
            logger.warning("Metric failed: %s", e)
            return {"bleu": 0.0, "chrf": 0.0}

    collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=15,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=8,  # Effective batch = 64
        learning_rate=5e-5,
        warmup_steps=300,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        bf16=True,
        fp16=False,
        bf16_full_eval=True,
        save_strategy="steps",
        save_steps=500,
        eval_strategy="no",  # Skip eval during training for speed
        logging_steps=20,
        save_total_limit=3,
        predict_with_generate=False,  # Skip generation during training
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        gradient_checkpointing=True,
        optim="adamw_torch",
        save_safetensors=True,
        save_only_model=True,
        report_to=["none"],
        run_name="runyoro-nmt-v2-clean",
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
    logger.info("  Model: %s (NO language codes)", MODEL_NAME)
    logger.info("  Data: %d clean pairs (%d bidirectional samples)", len(train_pairs), len(train_ds))
    logger.info("  Epochs: 15 | Batch: 8 x 8 grad_accum = 64 effective")
    logger.info("  NO src_lang | NO forced_bos_token_id")
    logger.info("=" * 60)

    trainer.train()
    logger.info("Training complete!")

    # Save final model
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    logger.info("Model saved to: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
