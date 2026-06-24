#!/usr/bin/env python3
"""
clean_sentence_word.py
======================
Cleans the 'sentence and word' folder data and outputs to 'sentence and word cleaned'.

Sentences CSV format:
  id, source_sentence, target_translation, target_translation_runyooro,
  target_translation_rutooro, source_language, target_language, domain,
  tense, contributor_id, status, created_at, updated_at, completed_by_id

Words CSV format:
  id, local_word, source_language, target_language, domain, pos,
  english_definition, dialect, local_translation, local_translation_runyooro,
  local_translation_rutooro, example_sentence, example_translation_english,
  related_words, contributor_id, status, created_at, updated_at, completed_by_id

Cleaning rules:
  - Only keep rows with status="Completed"
  - Only keep rows where both source and target are non-empty
  - Remove grammar annotations, extra punctuation noise
  - Normalize whitespace
  - Remove duplicates
  - For sentences: extract (source_sentence, target_translation) pairs
  - For words: extract (local_word/local_translation, english_definition) pairs
    AND (example_sentence, example_translation_english) pairs
"""

import csv
import json
import logging
import os
import re
import unicodedata
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("clean_sentence_word")

# Paths
RAW_ROOT = Path(__file__).parent.parent.parent / "raw data" / "sentence and word"
SENTENCES_DIR = RAW_ROOT / "Sentences 23june2026" / "Sentences 23june2026"
WORDS_DIR = RAW_ROOT / "Words 23june2026" / "Words 23june2026"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "raw data" / "sentence and word cleaned"


def normalize_text(text: str) -> str:
    """Clean and normalize text."""
    if not text:
        return ""
    # Unicode normalize
    text = unicodedata.normalize("NFC", text)
    # Remove zero-width chars
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Remove leading/trailing punctuation noise
    text = re.sub(r"^[,;:\-]+\s*", "", text)
    text = re.sub(r"\s*[,;:\-]+$", "", text)
    # Remove isolated grammar tags like (v.i.), (n.), etc.
    text = re.sub(r"\s*\([a-z.]+\)\s*", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_valid_pair(src: str, tgt: str) -> bool:
    """Check if a pair is valid for training."""
    if not src or not tgt:
        return False
    # Skip if either side is too short (just punctuation/symbols)
    if len(src.strip()) < 2 or len(tgt.strip()) < 2:
        return False
    # Skip if either side is just numbers/symbols
    if re.fullmatch(r"[\d\s\W]+", src) or re.fullmatch(r"[\d\s\W]+", tgt):
        return False
    # Skip if either side is excessively long (likely garbage)
    if len(src) > 500 or len(tgt) > 500:
        return False
    return True


def read_csv_files(directory: Path) -> list:
    """Read all CSV files in a directory."""
    all_rows = []
    csv_files = sorted(directory.glob("*.csv"))
    for csv_file in csv_files:
        try:
            with open(csv_file, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    all_rows.append(row)
        except Exception as e:
            logger.warning("Error reading %s: %s", csv_file.name, e)
    return all_rows


def clean_sentences(rows: list) -> list:
    """
    Extract clean sentence pairs from the sentences data.
    Returns list of (runyoro, english) tuples.
    """
    pairs = []
    skipped_status = 0
    skipped_empty = 0
    skipped_invalid = 0

    for row in rows:
        status = row.get("status", "").strip()
        
        # Only keep completed translations
        if status != "Completed":
            skipped_status += 1
            continue

        source = normalize_text(row.get("source_sentence", ""))
        # Try target_translation first, then runyooro, then rutooro
        target = normalize_text(row.get("target_translation", ""))
        if not target:
            target = normalize_text(row.get("target_translation_runyooro", ""))
        if not target:
            target = normalize_text(row.get("target_translation_rutooro", ""))

        if not source or not target:
            skipped_empty += 1
            continue

        # Determine direction: is source English or Runyoro?
        src_lang = row.get("source_language", "").strip()
        
        if "English" in src_lang:
            eng = source
            rny = target
        else:
            rny = source
            eng = target

        eng = normalize_text(eng)
        rny = normalize_text(rny)

        if not is_valid_pair(rny, eng):
            skipped_invalid += 1
            continue

        pairs.append((rny, eng))

    logger.info("  Sentences: %d valid pairs (skipped: %d status, %d empty, %d invalid)",
                len(pairs), skipped_status, skipped_empty, skipped_invalid)
    return pairs


def clean_words(rows: list) -> list:
    """
    Extract clean word pairs from the words data.
    Returns list of (runyoro, english) tuples.
    
    Extracts TWO types of pairs:
    1. word/local_translation <-> english_definition
    2. example_sentence <-> example_translation_english
    """
    word_pairs = []
    sentence_pairs = []
    skipped_status = 0
    skipped_empty = 0

    for row in rows:
        status = row.get("status", "").strip()
        if status != "Completed":
            skipped_status += 1
            continue

        src_lang = row.get("source_language", "").strip()
        local_word = normalize_text(row.get("local_word", ""))
        eng_def = normalize_text(row.get("english_definition", ""))
        local_translation = normalize_text(row.get("local_translation", ""))
        local_rny = normalize_text(row.get("local_translation_runyooro", ""))
        local_rut = normalize_text(row.get("local_translation_rutooro", ""))
        example_rny = normalize_text(row.get("example_sentence", ""))
        example_eng = normalize_text(row.get("example_translation_english", ""))

        # Determine the Runyoro word
        rny_word = local_translation or local_rny or local_rut

        # If source is Runyoro, the local_word IS the Runyoro word
        if "Runyooro" in src_lang or "Rutooro" in src_lang:
            if local_word and not rny_word:
                rny_word = local_word

        # If source is English, local_word is English
        if "English" in src_lang:
            if not eng_def:
                eng_def = local_word
            # The translation fields contain the Runyoro
            if not rny_word:
                rny_word = local_translation or local_rny or local_rut

        # Pair 1: Word translation
        if rny_word and eng_def and is_valid_pair(rny_word, eng_def):
            word_pairs.append((rny_word, eng_def))
        elif local_word and eng_def and is_valid_pair(local_word, eng_def):
            # Fallback: use local_word even if it might be English
            if "English" not in src_lang:
                word_pairs.append((local_word, eng_def))

        # Pair 2: Example sentences (most valuable for NMT!)
        if example_rny and example_eng and is_valid_pair(example_rny, example_eng):
            sentence_pairs.append((example_rny, example_eng))
        elif not example_eng and example_rny:
            skipped_empty += 1

    logger.info("  Words: %d word pairs, %d example sentence pairs (skipped: %d status, %d empty)",
                len(word_pairs), len(sentence_pairs), skipped_status, skipped_empty)
    return word_pairs, sentence_pairs


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
    """Save pairs to TSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rny, eng in pairs:
            f.write(f"{rny}\t{eng}\n")


def main():
    logger.info("=" * 60)
    logger.info("CLEANING: sentence and word folder")
    logger.info("=" * 60)
    logger.info("Input: %s", RAW_ROOT)
    logger.info("Output: %s", OUTPUT_DIR)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ===== SENTENCES =====
    logger.info("\n--- Processing Sentences ---")
    if SENTENCES_DIR.exists():
        sentence_rows = read_csv_files(SENTENCES_DIR)
        logger.info("  Read %d rows from sentence CSVs", len(sentence_rows))
        sentence_pairs = clean_sentences(sentence_rows)
        sentence_pairs = deduplicate(sentence_pairs)
        logger.info("  After dedup: %d sentence pairs", len(sentence_pairs))
        save_tsv(sentence_pairs, OUTPUT_DIR / "sentences_cleaned.tsv")
        logger.info("  Saved to: sentences_cleaned.tsv")
    else:
        logger.warning("  Sentences directory not found: %s", SENTENCES_DIR)
        sentence_pairs = []

    # ===== WORDS =====
    logger.info("\n--- Processing Words ---")
    if WORDS_DIR.exists():
        word_rows = read_csv_files(WORDS_DIR)
        logger.info("  Read %d rows from word CSVs", len(word_rows))
        word_pairs, word_example_pairs = clean_words(word_rows)
        word_pairs = deduplicate(word_pairs)
        word_example_pairs = deduplicate(word_example_pairs)
        logger.info("  After dedup: %d word pairs, %d example sentence pairs",
                    len(word_pairs), len(word_example_pairs))
        save_tsv(word_pairs, OUTPUT_DIR / "words_cleaned.tsv")
        save_tsv(word_example_pairs, OUTPUT_DIR / "word_examples_cleaned.tsv")
        logger.info("  Saved to: words_cleaned.tsv, word_examples_cleaned.tsv")
    else:
        logger.warning("  Words directory not found: %s", WORDS_DIR)
        word_pairs = []
        word_example_pairs = []

    # ===== COMBINED =====
    all_sentence_pairs = sentence_pairs + word_example_pairs
    all_sentence_pairs = deduplicate(all_sentence_pairs)
    all_word_pairs = deduplicate(word_pairs)

    save_tsv(all_sentence_pairs, OUTPUT_DIR / "all_sentences_cleaned.tsv")
    save_tsv(all_word_pairs, OUTPUT_DIR / "all_words_cleaned.tsv")

    # Combined everything
    all_pairs = all_sentence_pairs + all_word_pairs
    all_pairs = deduplicate(all_pairs)
    save_tsv(all_pairs, OUTPUT_DIR / "all_pairs_cleaned.tsv")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info("  Sentence pairs (from sentence CSVs): %d", len(sentence_pairs))
    logger.info("  Word translations: %d", len(all_word_pairs))
    logger.info("  Example sentences (from word CSVs): %d", len(word_example_pairs))
    logger.info("  Total sentence-level pairs: %d", len(all_sentence_pairs))
    logger.info("  Total all pairs: %d", len(all_pairs))
    logger.info("  Output: %s", OUTPUT_DIR)
    logger.info("=" * 60)

    # Save summary file
    summary = (
        f"Sentence and Word Cleaned — Summary\n"
        f"{'=' * 40}\n\n"
        f"Source: {RAW_ROOT}\n"
        f"Output: {OUTPUT_DIR}\n\n"
        f"Sentence pairs (from sentence CSVs): {len(sentence_pairs)}\n"
        f"Word translations: {len(all_word_pairs)}\n"
        f"Example sentences (from word CSVs): {len(word_example_pairs)}\n"
        f"Total sentence-level pairs: {len(all_sentence_pairs)}\n"
        f"Total all pairs (sentences + words): {len(all_pairs)}\n\n"
        f"Files:\n"
        f"  - sentences_cleaned.tsv: Sentence-to-sentence translations\n"
        f"  - words_cleaned.tsv: Word-to-definition translations\n"
        f"  - word_examples_cleaned.tsv: Example sentence pairs from word entries\n"
        f"  - all_sentences_cleaned.tsv: All sentence-level pairs combined\n"
        f"  - all_words_cleaned.tsv: All word-level pairs\n"
        f"  - all_pairs_cleaned.tsv: Everything combined\n"
    )
    (OUTPUT_DIR / "summary.txt").write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()
