#!/usr/bin/env python3
"""
back_translate.py
=================
Back-translation: generates synthetic parallel data from monolingual Runyoro text.

Usage:
    # From existing parallel data (self-training mode — uses Runyoro side of existing pairs)
    python scripts/back_translate.py --model ./models/checkpoints/runyoro-nmt-v1 --source data/augmented/all_pairs.tsv --output data/augmented/back_translated.tsv

    # From a monolingual text file (one sentence per line)
    python scripts/back_translate.py --model ./models/checkpoints/runyoro-nmt-v1 --source monolingual_runyoro.txt --output data/augmented/back_translated.tsv

    # Only generate for English→Runyoro direction (reverse)
    python scripts/back_translate.py --model ./models/checkpoints/runyoro-nmt-v1 --source data/augmented/all_pairs.tsv --output data/augmented/back_translated.tsv --reverse

Strategy:
    For each Runyoro sentence, use the model to generate an English translation.
    The original (rny, eng) pair plus new (rny, synthetic_eng) are both kept.
    This doubles the effective training data for the rny→en direction.

    With --reverse, also generates Runyoro translations of English sentences,
    creating additional (eng, synthetic_rny) pairs for the en→rny direction.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("back_translate")

NLLB_RNY = "nyk_Latn"
NLLB_ENG = "eng_Latn"


def load_model(model_path: str, device: str = "cuda"):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    logger.info("Loading model from: %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
    ).to(device)

    if tokenizer.convert_tokens_to_ids(NLLB_RNY) == tokenizer.unk_token_id:
        lug_id = tokenizer.convert_tokens_to_ids("lug_Latn")
        tokenizer.add_tokens([NLLB_RNY], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))
        nyk_id = tokenizer.convert_tokens_to_ids(NLLB_RNY)
        with torch.no_grad():
            model.get_input_embeddings().weight[nyk_id] = (
                model.get_input_embeddings().weight[lug_id].clone()
            )
            model.get_output_embeddings().weight[nyk_id] = (
                model.get_output_embeddings().weight[lug_id].clone()
            )
        logger.info("Added nyk_Latn token (id=%d) from lug_Latn embedding", nyk_id)

    model.eval()
    logger.info("Model ready on %s", device)
    return model, tokenizer


def load_source_pairs(path: str):
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2:
                pairs.append((parts[0].strip(), parts[1].strip()))
    logger.info("Loaded %d parallel pairs from %s", len(pairs), path)
    return pairs


def load_monolingual_text(path: str):
    sentences = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sentences.append(line)
    logger.info("Loaded %d monolingual sentences from %s", len(sentences), path)
    return sentences


def translate_batch(model, tokenizer, texts, src_lang, tgt_lang, device, batch_size=16):
    forced_bos_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    translations = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        tokenizer.src_lang = src_lang
        enc = tokenizer(
            batch, return_tensors="pt", max_length=256, truncation=True, padding=True
        ).to(device)

        with torch.no_grad():
            out = model.generate(
                **enc,
                forced_bos_token_id=forced_bos_id,
                num_beams=4,
                max_length=256,
                length_penalty=1.0,
            )

        decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
        translations.extend([t.strip() for t in decoded])

        if (i // batch_size) % 10 == 0:
            logger.info("  Translated %d / %d", i + len(batch), len(texts))

    return translations


def save_pairs_tsv(pairs, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for src, tgt in pairs:
            f.write(f"{src}\t{tgt}\n")
    logger.info("Saved %d pairs to %s", len(pairs), path)


def main():
    parser = argparse.ArgumentParser(description="Back-translation for runyoro-nmt-v1")
    parser.add_argument(
        "--model", default=str(ROOT / "models/checkpoints/runyoro-nmt-v1")
    )
    parser.add_argument(
        "--source",
        required=True,
        help="TSV with pairs, or .txt with one sentence per line",
    )
    parser.add_argument(
        "--output", default=str(ROOT / "data/augmented/back_translated.tsv")
    )
    parser.add_argument(
        "--reverse", action="store_true", help="Also back-translate English→Runyoro"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    model, tokenizer = load_model(args.model, args.device)

    if args.source.endswith(".tsv"):
        pairs = load_source_pairs(args.source)
        runyoro_texts = [r for r, e in pairs]
        logger.info(
            "Using Runyoro side of %d parallel pairs for back-translation",
            len(runyoro_texts),
        )
    else:
        runyoro_texts = load_monolingual_text(args.source)

    logger.info(
        "Back-translating %d Runyoro sentences → English...", len(runyoro_texts)
    )
    syn_english = translate_batch(
        model,
        tokenizer,
        runyoro_texts,
        NLLB_RNY,
        NLLB_ENG,
        args.device,
        args.batch_size,
    )

    bt_pairs = list(zip(runyoro_texts, syn_english))
    save_pairs_tsv(bt_pairs, args.output)
    logger.info("Back-translation complete: %d synthetic pairs", len(bt_pairs))

    if args.reverse and args.source.endswith(".tsv"):
        pairs = load_source_pairs(args.source)
        english_texts = [e for r, e in pairs]
        logger.info(
            "Reverse back-translation: %d English sentences → Runyoro...",
            len(english_texts),
        )
        syn_runyoro = translate_batch(
            model,
            tokenizer,
            english_texts,
            NLLB_ENG,
            NLLB_RNY,
            args.device,
            args.batch_size,
        )
        rev_pairs = list(zip(syn_runyoro, english_texts))
        rev_output = args.output.replace(".tsv", "_reverse.tsv")
        save_pairs_tsv(rev_pairs, rev_output)
        logger.info(
            "Reverse back-translation complete: %d synthetic pairs", len(rev_pairs)
        )

    logger.info("ALL DONE")


if __name__ == "__main__":
    main()
