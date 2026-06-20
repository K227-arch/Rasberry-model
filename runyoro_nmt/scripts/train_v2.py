#!/usr/bin/env python3
"""
train_v2.py  —  runyoro-nmt-v1 (full-data, back-translated, nyk_Latn BOS)
=====================================================================
Trains on ALL 4,520 augmented pairs (no held-out split) plus back-translated
synthetic data. Resumes from the existing checkpoint and uses nyk_Latn as
the native Runyoro BOS token instead of lug_Latn.

Launch:
    python scripts/train_v2.py
    python scripts/train_v2.py --resume-from ./models/checkpoints/runyoro-nmt-v1
"""

import json
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_TOKEN", "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")
HF_TOKEN = os.environ["HF_TOKEN"]

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

log_handlers = [
    logging.StreamHandler(),
    logging.FileHandler(str(ROOT / "training_v2.log"), encoding="utf-8"),
]
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=log_handlers,
)
logger = logging.getLogger("train_v2")

import yaml
import torch
import torch.nn as nn
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

with open(ROOT / "configs" / "config.yaml", encoding="utf-8") as f:
    config = yaml.safe_load(f)

tc = config["training"]
mc = config["model"]
MODEL_NAME = mc["base_model_name"]
OUTPUT_DIR = ROOT / tc["output_dir"]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
NLLB_RNY = "nyk_Latn"
NLLB_ENG = "eng_Latn"
MAX_SRC = mc["max_source_length"]
MAX_TGT = mc["max_target_length"]

set_seed(config["data"].get("seed", 42))


def load_tsv(path):
    pairs = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split("\t", 1)
        if len(parts) == 2:
            pairs.append((parts[0].strip(), parts[1].strip()))
    return pairs


def make_tokenise_fn(src_lang, tgt_lang):
    def _fn(examples):
        tokenizer.src_lang = src_lang
        src_enc = tokenizer(
            examples["src"], max_length=MAX_SRC, truncation=True, padding=False
        )
        tokenizer.src_lang = tgt_lang
        tgt_enc = tokenizer(
            examples["tgt"], max_length=MAX_TGT, truncation=True, padding=False
        )
        tokenizer.src_lang = src_lang
        src_enc["labels"] = tgt_enc["input_ids"]
        return src_enc

    return _fn


def pairs_to_hf(pairs, src_lang, tgt_lang, desc=""):
    raw = HFDataset.from_dict(
        {
            "src": [s for s, t in pairs],
            "tgt": [t for s, t in pairs],
        }
    )
    return raw.map(
        make_tokenise_fn(src_lang, tgt_lang),
        batched=True,
        remove_columns=["src", "tgt"],
        desc=desc or f"{src_lang}->{tgt_lang}",
        load_from_cache_file=False,
    )


logger.info("=" * 60)
logger.info("TRAIN V2 — Full-data training with nyk_Latn BOS")
logger.info("=" * 60)

# ── Load ALL data (4,520 augmented pairs) ─────────────────────
all_pairs = load_tsv(ROOT / "data/augmented/all_pairs.tsv")
logger.info("Loaded %d augmented pairs from all_pairs.tsv", len(all_pairs))

# Also load any back-translated pairs if available
bt_path = ROOT / "data/augmented/back_translated.tsv"
bt_rev_path = ROOT / "data/augmented/back_translated_reverse.tsv"
bt_pairs = []
if bt_path.exists():
    bt_pairs = load_tsv(str(bt_path))
    logger.info("Loaded %d back-translated pairs", len(bt_pairs))
if bt_rev_path.exists():
    bt_rev = load_tsv(str(bt_rev_path))
    bt_pairs.extend(bt_rev)
    logger.info("Loaded %d reverse back-translated pairs", len(bt_rev))

all_pairs = all_pairs + bt_pairs
logger.info("TOTAL training pairs (original + back-translated): %d", len(all_pairs))

# ── Tokenizer ─────────────────────────────────────────────────
logger.info("Loading tokenizer: %s", MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)

# Inject nyk_Latn as a real special token (copy lug_Latn embedding)
if tokenizer.convert_tokens_to_ids(NLLB_RNY) == tokenizer.unk_token_id:
    lug_id = tokenizer.convert_tokens_to_ids("lug_Latn")
    tokenizer.add_tokens([NLLB_RNY], special_tokens=True)
    logger.info("nyk_Latn will be resized after model load (id=%d)", len(tokenizer) - 1)

# ── Build bidirectional datasets ──────────────────────────────
logger.info("Tokenising datasets (bidirectional)...")
train_fwd = pairs_to_hf(all_pairs, NLLB_RNY, NLLB_ENG, "Train rny->en")
train_rev = pairs_to_hf(
    [(t, s) for s, t in all_pairs], NLLB_ENG, NLLB_RNY, "Train en->rny"
)
train_ds = concatenate_datasets([train_fwd, train_rev]).shuffle(seed=42)
logger.info("  Train samples=%d (bidirectional)", len(train_ds))

# ── Model (load from checkpoint) ──────────────────────────────
checkpoint_path = None
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--resume-from", default=None)
args, _ = parser.parse_known_args()
if args.resume_from:
    checkpoint_path = str(ROOT / args.resume_from)
    logger.info("Resuming from checkpoint: %s", checkpoint_path)
elif (OUTPUT_DIR / "model.safetensors").exists():
    checkpoint_path = str(OUTPUT_DIR)
    logger.info("Resuming from existing model: %s", checkpoint_path)
else:
    logger.info("Starting from base model: %s", MODEL_NAME)

if checkpoint_path:
    logger.info("Loading model from: %s", checkpoint_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path, token=HF_TOKEN)
else:
    logger.info("Loading base model: %s", MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, token=HF_TOKEN)

model = model.to(torch.bfloat16)
logger.info("  Parameters: %.1f M", sum(p.numel() for p in model.parameters()) / 1e6)

if torch.cuda.is_available():
    free_mem, total_mem = torch.cuda.mem_get_info()
    logger.info("  GPU memory: %.1f / %.1f GB free", free_mem / 1e9, total_mem / 1e9)

# Resize for nyk_Latn token
model.resize_token_embeddings(len(tokenizer))
nyk_id = tokenizer.convert_tokens_to_ids(NLLB_RNY)
lug_id = tokenizer.convert_tokens_to_ids("lug_Latn")
with torch.no_grad():
    model.get_input_embeddings().weight[nyk_id] = (
        model.get_input_embeddings().weight[lug_id].clone()
    )
    model.get_output_embeddings().weight[nyk_id] = (
        model.get_output_embeddings().weight[lug_id].clone()
    )
logger.info("Copied lug_Latn embedding → nyk_Latn (id=%d)", nyk_id)

ENG_BOS_ID = tokenizer.convert_tokens_to_ids(NLLB_ENG)
RNY_BOS_ID = nyk_id
logger.info(
    "  %s bos_id=%d  |  %s bos_id=%d", NLLB_ENG, ENG_BOS_ID, NLLB_RNY, RNY_BOS_ID
)

# ── Metrics ───────────────────────────────────────────────────
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
    decoded_preds = [p.strip() for p in decoded_preds]
    decoded_labels = [l.strip() for l in decoded_labels]
    try:
        bleu = bleu_metric.compute(
            predictions=decoded_preds, references=[[ref] for ref in decoded_labels]
        )
        chrf = chrf_metric.compute(
            predictions=decoded_preds,
            references=[[ref] for ref in decoded_labels],
            word_order=2,
        )
        return {"bleu": round(bleu["score"], 2), "chrf": round(chrf["score"], 2)}
    except Exception as e:
        logger.warning("Metric computation failed: %s", e)
        return {"bleu": 0.0, "chrf": 0.0}


# Strip any DataParallel wrapper from checkpoint
if hasattr(model, "module"):
    model = model.module

# Enable gradient checkpointing to reduce memory
model.gradient_checkpointing_enable()
logger.info("Gradient checkpointing enabled")

# ── Collator ──────────────────────────────────────────────────
collator = DataCollatorForSeq2Seq(
    tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8
)

# ── Training args ─────────────────────────────────────────────
training_args = Seq2SeqTrainingArguments(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=tc["num_train_epochs"],
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=tc["gradient_accumulation_steps"],
    learning_rate=tc["learning_rate"],
    warmup_steps=tc["warmup_steps"],
    weight_decay=tc["weight_decay"],
    lr_scheduler_type=tc["lr_scheduler_type"],
    bf16=False,
    fp16=False,
    save_strategy=tc["save_strategy"],
    eval_strategy="no",
    logging_steps=tc["logging_steps"],
    save_total_limit=tc["save_total_limit"],
    predict_with_generate=True,
    generation_max_length=tc["generation_max_length"],
    generation_num_beams=tc["generation_num_beams"],
    label_smoothing_factor=tc.get("label_smoothing_factor", 0.1),
    dataloader_num_workers=0,
    dataloader_pin_memory=False,
    report_to=["none"],
    logging_dir=str(ROOT / "experiments" / "logs"),
    run_name="runyoro-nmt-v1-v2",
)

# ── Trainer (no eval during training — use full data) ─────────
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    processing_class=tokenizer,
    data_collator=collator,
)

_orig_prepare = trainer._prepare_inputs


def _safe_prepare_inputs(inputs):
    prepared = _orig_prepare(inputs)
    prepared.pop("decoder_inputs_embeds", None)
    return prepared


trainer._prepare_inputs = _safe_prepare_inputs

# ── Launch ────────────────────────────────────────────────────
logger.info("=" * 60)
logger.info("STARTING train_v2")
logger.info("  Model     : %s", checkpoint_path or MODEL_NAME)
logger.info(
    "  Pairs     : %d (%d bidirectional samples)", len(all_pairs), len(train_ds)
)
logger.info("  Epochs    : %d", tc["num_train_epochs"])
logger.info(
    "  Batch     : %d x %d grad_accum",
    tc["per_device_train_batch_size"],
    tc["gradient_accumulation_steps"],
)
logger.info("  BOS rny   : %s (id=%d)", NLLB_RNY, RNY_BOS_ID)
logger.info("  BOS eng   : %s (id=%d)", NLLB_ENG, ENG_BOS_ID)
logger.info("=" * 60)

trainer.train()

logger.info("Training complete.")

# ── Save ──────────────────────────────────────────────────────
model.save_pretrained(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))
logger.info("Saved to %s", OUTPUT_DIR)

# ── Full-data evaluation ──────────────────────────────────────
logger.info("Full-data evaluation...")
eval_ds = pairs_to_hf(all_pairs, NLLB_RNY, NLLB_ENG, "Eval rny->en")
eval_res = trainer.predict(eval_ds)
metrics = eval_res.metrics
logger.info("FULL EVAL (rny->en, %d pairs): %s", len(all_pairs), metrics)

(ROOT / "data/reports/test_results_v2.json").write_text(
    json.dumps(metrics, indent=2), encoding="utf-8"
)

bleu_score = metrics.get("test_bleu", "N/A")
chrf_score = metrics.get("test_chrf", "N/A")
logger.info(
    "  BLEU=%.2f  chrF++=%.2f",
    bleu_score if isinstance(bleu_score, float) else 0,
    chrf_score if isinstance(chrf_score, float) else 0,
)

# ── Push to Hub ───────────────────────────────────────────────
logger.info("Pushing to kathay/runyoro-nmt-v1 ...")
try:
    from huggingface_hub import HfApi

    model.push_to_hub(
        "kathay/runyoro-nmt-v1",
        token=HF_TOKEN,
        commit_message="runyoro-nmt-v1: v2 full-data + nyk_Latn BOS",
    )
    tokenizer.push_to_hub("kathay/runyoro-nmt-v1", token=HF_TOKEN)
    logger.info("Model live: https://huggingface.co/kathay/runyoro-nmt-v1")
except Exception as e:
    logger.error("Hub push failed: %s", e)

logger.info("ALL DONE")
