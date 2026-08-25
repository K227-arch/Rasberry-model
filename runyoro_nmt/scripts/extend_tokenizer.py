#!/usr/bin/env python3
"""
extend_tokenizer.py - Train a custom SentencePiece tokenizer on the Runyoro/Rutooro corpus
and extend NLLB's vocabulary with new Runyoro-specific subwords.

Pipeline:
1. Gather all Runyoro text from cleaned pairs, augmented data, glossary, and back-translations
2. Train a SentencePiece unigram model on the Runyoro corpus
3. Filter out tokens already present in NLLB's vocabulary
4. Extend the NLLB tokenizer with new Runyoro-specific tokens
5. Resize the model's embedding layer and initialize new tokens smartly
6. Save the extended tokenizer + model checkpoint ready for further fine-tuning

Usage:
    python runyoro_nmt/scripts/extend_tokenizer.py \
        --model-path runyoro_nmt/models/checkpoints/runyoro-inc-v4 \
        --output-dir runyoro_nmt/models/checkpoints/runyoro-inc-v4-extended \
        --spm-vocab-size 4000 \
        --max-new-tokens 2000
"""
import argparse
import gc
import logging
import os
import re
import sys
import unicodedata
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["USE_TF"] = "0"

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(str(ROOT / "extend_tokenizer.log"))],
)
logger = logging.getLogger("extend_tok")

import pandas as pd
import torch
import sentencepiece as spm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def clean_text(t: str) -> str:
    """Normalize text for corpus building."""
    t = str(t).strip()
    if t.lower() == "nan" or not t:
        return ""
    t = unicodedata.normalize("NFC", t)
    t = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def gather_runyoro_corpus(data_dir: Path, raw_dir: Path) -> list:
    """Collect all Runyoro text from the project data sources."""
    texts = []

    # 1. All cleaned pairs across versions
    for csv_path in sorted(data_dir.rglob("cleaned_pairs.csv")):
        try:
            df = pd.read_csv(csv_path)
            if "Runyoro" in df.columns:
                rny_texts = df["Runyoro"].dropna().apply(clean_text).tolist()
                texts.extend([t for t in rny_texts if len(t) > 3])
                logger.info("  %s: %d Runyoro sentences", csv_path.name, len(rny_texts))
        except Exception as e:
            logger.warning("  Skipping %s: %s", csv_path, e)

    # 2. Augmented pairs
    for csv_path in sorted(data_dir.rglob("augmented_pairs.csv")):
        try:
            df = pd.read_csv(csv_path)
            if "Runyoro" in df.columns:
                rny_texts = df["Runyoro"].dropna().apply(clean_text).tolist()
                texts.extend([t for t in rny_texts if len(t) > 3])
                logger.info("  %s: %d augmented Runyoro", csv_path.name, len(rny_texts))
        except Exception as e:
            logger.warning("  Skipping %s: %s", csv_path, e)

    # 3. Back-translated data (synthetic Runyoro)
    for csv_path in sorted(data_dir.rglob("back_translated*.csv")):
        try:
            df = pd.read_csv(csv_path)
            if "Runyoro" in df.columns:
                rny_texts = df["Runyoro"].dropna().apply(clean_text).tolist()
                texts.extend([t for t in rny_texts if len(t) > 3])
                logger.info("  %s: %d BT Runyoro", csv_path.name, len(rny_texts))
        except Exception as e:
            logger.warning("  Skipping %s: %s", csv_path, e)

    # 4. Glossary terms
    glossary_path = data_dir / "tm" / "glossary.csv"
    if glossary_path.exists():
        try:
            df = pd.read_csv(glossary_path)
            if "runyoro_term" in df.columns:
                terms = df["runyoro_term"].dropna().apply(clean_text).tolist()
                texts.extend([t for t in terms if len(t) > 1])
                logger.info("  glossary.csv: %d terms", len(terms))
        except Exception as e:
            logger.warning("  Skipping glossary: %s", e)

    # 5. Final training data (includes everything combined)
    for csv_path in sorted(data_dir.rglob("final_training*.csv")):
        try:
            df = pd.read_csv(csv_path)
            if "Runyoro" in df.columns:
                rny_texts = df["Runyoro"].dropna().apply(clean_text).tolist()
                texts.extend([t for t in rny_texts if len(t) > 3])
                logger.info("  %s: %d final Runyoro", csv_path.name, len(rny_texts))
        except Exception as e:
            logger.warning("  Skipping %s: %s", csv_path, e)

    # 6. TSV/JSONL processed data
    for tsv_path in sorted(data_dir.glob("processed/*.tsv")):
        try:
            df = pd.read_csv(tsv_path, sep="\t", header=None)
            # Assume first column is source (Runyoro)
            rny_texts = df.iloc[:, 0].dropna().apply(clean_text).tolist()
            texts.extend([t for t in rny_texts if len(t) > 3])
            logger.info("  %s: %d rows", tsv_path.name, len(rny_texts))
        except Exception:
            pass

    # Deduplicate
    texts = list(set(texts))
    logger.info("Total unique Runyoro texts gathered: %d", len(texts))
    return texts


def train_sentencepiece(corpus_texts: list, output_dir: Path, vocab_size: int = 4000) -> Path:
    """Train a SentencePiece unigram model on the Runyoro corpus."""
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_file = output_dir / "runyoro_corpus.txt"

    # Write corpus to file
    with open(corpus_file, "w", encoding="utf-8") as f:
        for text in corpus_texts:
            f.write(text + "\n")

    logger.info("Training SentencePiece on %d texts, vocab_size=%d", len(corpus_texts), vocab_size)

    model_prefix = str(output_dir / "runyoro_spm")
    spm.SentencePieceTrainer.train(
        input=str(corpus_file),
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="unigram",
        character_coverage=0.9995,  # High coverage for Runyoro
        num_threads=os.cpu_count(),
        input_sentence_size=min(len(corpus_texts), 500000),
        shuffle_input_sentence=True,
        # Special tokens — keep consistent with NLLB conventions
        pad_id=3,
        bos_id=-1,  # NLLB doesn't use BOS in SPM
        eos_id=2,
        unk_id=0,
        # Subword regularization for robustness
        split_digits=True,
        byte_fallback=True,
        normalization_rule_name="nmt_nfkc",
    )

    spm_model_path = Path(model_prefix + ".model")
    logger.info("SentencePiece model saved: %s", spm_model_path)
    return spm_model_path


def get_new_tokens(spm_model_path: Path, nllb_tokenizer, max_new_tokens: int = 2000) -> list:
    """
    Extract tokens from the custom SPM that are NOT in NLLB's vocabulary.
    Filter by frequency/usefulness.
    """
    sp = spm.SentencePieceProcessor()
    sp.load(str(spm_model_path))

    # Get all pieces from custom SPM
    custom_pieces = []
    for i in range(sp.get_piece_size()):
        piece = sp.id_to_piece(i)
        score = sp.get_score(i)
        # Skip control tokens and single bytes
        if piece.startswith("<") or piece.startswith("0x") or len(piece) <= 1:
            continue
        custom_pieces.append((piece, score))

    logger.info("Custom SPM vocabulary: %d pieces (excl. control tokens)", len(custom_pieces))

    # Filter: keep only tokens NOT already in NLLB vocab
    nllb_vocab = set(nllb_tokenizer.get_vocab().keys())
    new_tokens = []
    for piece, score in custom_pieces:
        # SentencePiece uses ▁ for word boundary
        if piece not in nllb_vocab:
            new_tokens.append((piece, score))

    # Sort by score (higher = more frequent in corpus)
    new_tokens.sort(key=lambda x: x[1], reverse=True)

    # Take top N
    selected = [tok for tok, _ in new_tokens[:max_new_tokens]]
    logger.info(
        "New tokens not in NLLB: %d total, selecting top %d",
        len(new_tokens),
        len(selected),
    )
    logger.info("Sample new tokens: %s", selected[:20])

    return selected


def extend_model_and_tokenizer(
    model_path: str,
    new_tokens: list,
    output_dir: Path,
):
    """
    Extend the NLLB tokenizer with new tokens and resize model embeddings.
    Initialize new embeddings using the mean of existing embeddings for robustness.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading tokenizer from: %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    original_vocab_size = len(tokenizer)
    logger.info("Original vocab size: %d", original_vocab_size)

    # Add new tokens
    num_added = tokenizer.add_tokens(new_tokens)
    logger.info("Added %d new tokens (some may have been duplicates)", num_added)
    logger.info("New vocab size: %d", len(tokenizer))

    # Load model
    logger.info("Loading model from: %s", model_path)
    use_bf16 = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, torch_dtype=dtype).to(device)

    # Resize embeddings
    old_embeddings = model.get_input_embeddings().weight.data.clone()
    model.resize_token_embeddings(len(tokenizer))

    # Initialize new token embeddings with the mean of existing embeddings
    # This is more stable than random initialization
    with torch.no_grad():
        mean_embedding = old_embeddings.mean(dim=0)
        new_embeds = model.get_input_embeddings().weight.data
        new_embeds[original_vocab_size:] = mean_embedding

        # Also resize and initialize the output embeddings (lm_head)
        if hasattr(model, "lm_head") and model.lm_head is not None:
            # For shared embeddings, resize_token_embeddings handles this
            pass
        # For the final output projection, it's typically tied to input embeddings
        # in NLLB, so resize_token_embeddings already handled it.

    logger.info("Embeddings resized. New tokens initialized with mean embedding.")

    # Save
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info("Extended model + tokenizer saved to: %s", output_dir)

    # Print stats
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("  Original vocab: %d", original_vocab_size)
    logger.info("  New tokens added: %d", num_added)
    logger.info("  Final vocab: %d", len(tokenizer))
    logger.info("  Model saved to: %s", output_dir)
    logger.info("=" * 60)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return str(output_dir)


def main():
    parser = argparse.ArgumentParser(description="Extend NLLB tokenizer with Runyoro vocabulary")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to the trained model checkpoint to extend. If not set, uses the latest inc checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to save the extended model. Defaults to <model-path>-extended.",
    )
    parser.add_argument(
        "--spm-vocab-size",
        type=int,
        default=4000,
        help="Vocabulary size for the custom SentencePiece model (default: 4000)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2000,
        help="Max number of new tokens to add to NLLB (default: 2000)",
    )
    args = parser.parse_args()

    # Resolve model path
    if args.model_path is None:
        # Auto-detect the latest incremental checkpoint
        ckpt_dir = ROOT / "models" / "checkpoints"
        candidates = sorted(ckpt_dir.glob("runyoro-inc-v*"), reverse=True)
        candidates = [c for c in candidates if "base" not in c.name and "extended" not in c.name]
        if not candidates:
            logger.error("No checkpoint found in %s", ckpt_dir)
            sys.exit(1)
        model_path = str(candidates[0])
        logger.info("Auto-detected model: %s", model_path)
    else:
        model_path = args.model_path

    if args.output_dir is None:
        output_dir = Path(model_path + "-extended")
    else:
        output_dir = Path(args.output_dir)

    data_dir = ROOT / "data"
    raw_dir = ROOT.parent / "raw"
    spm_output_dir = ROOT / "models" / "tokenizer"

    logger.info("=" * 60)
    logger.info("EXTENDING NLLB TOKENIZER WITH RUNYORO/RUTOORO VOCABULARY")
    logger.info("=" * 60)
    logger.info("  Model path: %s", model_path)
    logger.info("  Output dir: %s", output_dir)
    logger.info("  SPM vocab size: %d", args.spm_vocab_size)
    logger.info("  Max new tokens: %d", args.max_new_tokens)
    logger.info("=" * 60)

    # Step 1: Gather Runyoro corpus
    logger.info("\nSTEP 1: Gathering Runyoro corpus...")
    corpus = gather_runyoro_corpus(data_dir, raw_dir)

    if len(corpus) < 100:
        logger.error("Corpus too small (%d texts). Need at least 100.", len(corpus))
        sys.exit(1)

    # Step 2: Train SentencePiece
    logger.info("\nSTEP 2: Training SentencePiece model...")
    spm_model_path = train_sentencepiece(corpus, spm_output_dir, args.spm_vocab_size)

    # Step 3: Extract new tokens
    logger.info("\nSTEP 3: Extracting new Runyoro-specific tokens...")
    nllb_tokenizer = AutoTokenizer.from_pretrained(model_path)
    new_tokens = get_new_tokens(spm_model_path, nllb_tokenizer, args.max_new_tokens)

    if len(new_tokens) == 0:
        logger.warning("No new tokens found! NLLB already covers the Runyoro vocabulary well.")
        logger.info("Skipping model extension.")
        return

    # Step 4: Extend model and tokenizer
    logger.info("\nSTEP 4: Extending model and tokenizer...")
    result_path = extend_model_and_tokenizer(model_path, new_tokens, output_dir)

    # Step 5: Quick validation
    logger.info("\nSTEP 5: Validation...")
    test_tokenizer = AutoTokenizer.from_pretrained(result_path)
    test_sentences = [
        "Omukama waitu akaba nagambira abantu be",
        "Ekyanga kya Runyoro-Rutooro nikiri kirungi",
        "Ndozirege ekituru kyomutini nikyoleka",
    ]
    for sent in test_sentences:
        tokens = test_tokenizer.tokenize(sent)
        logger.info("  '%s' → %d tokens: %s", sent[:40], len(tokens), tokens[:10])

    logger.info("\nDONE! Extended model ready for fine-tuning at: %s", result_path)
    logger.info("Next step: Fine-tune the extended model on your Runyoro pairs.")


if __name__ == "__main__":
    main()
