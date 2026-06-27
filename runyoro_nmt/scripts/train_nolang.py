#!/usr/bin/env python3
"""
train_nolang.py  —  Runyoro-Rutooro ↔ English NMT (no language codes)
====================================================================
Fine-tunes NLLB-200 WITHOUT forced BOS tokens or language codes.
The model learns target language purely from text, not from lang IDs.

Key differences from train.py:
  - No tokenizer.src_lang set during tokenization
  - No forced_bos_token_id during generation
  - No language code prepended to source or target
  - Model learns direction from the text itself

Launch:
    python scripts/train_nolang.py
"""

import json
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_TOKEN", "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")
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

log_handlers = [
    logging.StreamHandler(),
    logging.FileHandler(str(ROOT / "training_nolang.log"), encoding="utf-8"),
]
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=log_handlers,
)
logger = logging.getLogger("train_nolang")

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

MODEL_NAME = "facebook/nllb-200-distilled-1.3B"
OUTPUT_DIR = ROOT / "models/checkpoints/runyoro-nolang-v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MAX_SRC = 256
MAX_TGT = 256
set_seed(42)


def load_tsv(path):
    pairs = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split("\t", 1)
        if len(parts) == 2:
            pairs.append((parts[0].strip(), parts[1].strip()))
    return pairs


logger.info("=" * 60)
logger.info("TRAIN NOLANG — Runyoro-Rutooro ↔ English (no language codes)")
logger.info("=" * 60)

# Load data
all_pairs = load_tsv(ROOT / "data/augmented/all_pairs.tsv")
for bt_path in [
    ROOT / "data/augmented/back_translated.tsv",
    ROOT / "data/augmented/back_translated_reverse.tsv",
]:
    if bt_path.exists():
        all_pairs += load_tsv(str(bt_path))

logger.info("Loaded %d total pairs", len(all_pairs))

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# Tokenize WITHOUT setting src_lang — no language codes
def tokenize_fn(examples):
    src_enc = tokenizer(
        examples["src"],
        max_length=MAX_SRC,
        truncation=True,
        padding=False,
    )
    tgt_enc = tokenizer(
        examples["tgt"],
        max_length=MAX_TGT,
        truncation=True,
        padding=False,
    )
    src_enc["labels"] = tgt_enc["input_ids"]
    return src_enc


def pairs_to_hf(pairs, desc=""):
    raw = HFDataset.from_dict(
        {
            "src": [s for s, t in pairs],
            "tgt": [t for s, t in pairs],
        }
    )
    return raw.map(
        tokenize_fn,
        batched=True,
        remove_columns=["src", "tgt"],
        desc=desc,
        load_from_cache_file=False,
    )


# Build bidirectional dataset (no lang codes — just text pairs)
fwd = pairs_to_hf(all_pairs, "rny->en")
rev = pairs_to_hf([(t, s) for s, t in all_pairs], "en->rny")
train_ds = concatenate_datasets([fwd, rev]).shuffle(seed=42)
logger.info("  Train samples=%d (bidirectional, no lang codes)", len(train_ds))

# Load model — split across 2x RTX 4090
logger.info("Loading model: %s", MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory={0: "20GiB", 1: "20GiB"},
)
model.gradient_checkpointing_enable()
logger.info("  Parameters: %.1f M", sum(p.numel() for p in model.parameters()) / 1e6)
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
        logger.warning("Metric failed: %s", e)
        return {"bleu": 0.0, "chrf": 0.0}


collator = DataCollatorForSeq2Seq(
    tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8
)

training_args = Seq2SeqTrainingArguments(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=15,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    warmup_steps=500,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    bf16=True,
    fp16=False,
    save_strategy="epoch",
    eval_strategy="no",
    logging_steps=20,
    save_total_limit=3,
    predict_with_generate=True,
    generation_max_length=MAX_TGT,
    generation_num_beams=4,
    dataloader_num_workers=0,
    dataloader_pin_memory=False,
    report_to=["none"],
    logging_dir=str(ROOT / "experiments" / "logs_nolang"),
    run_name="runyoro-nolang-v1",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    processing_class=tokenizer,
    data_collator=collator,
)

# Patch to strip decoder_inputs_embeds
_orig_prepare = trainer._prepare_inputs


def _safe_prepare_inputs(inputs):
    prepared = _orig_prepare(inputs)
    prepared.pop("decoder_inputs_embeds", None)
    return prepared


trainer._prepare_inputs = _safe_prepare_inputs

logger.info("=" * 60)
logger.info("STARTING train_nolang")
logger.info("  Model     : %s", MODEL_NAME)
logger.info("  Pairs     : %d (%d bidirectional)", len(all_pairs), len(train_ds))
logger.info("  Epochs    : 15  |  Batch: 16 x 4 grad_accum")
logger.info("  NO src_lang set  |  NO forced_bos_token_id")
logger.info("  Model learns direction from text alone")
logger.info("=" * 60)

trainer.train()
logger.info("Training complete")

model.save_pretrained(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))
logger.info("Saved to %s", OUTPUT_DIR)

# Full evaluation
logger.info("Evaluating (no forced BOS)...")
eval_ds = pairs_to_hf(all_pairs, "eval")
eval_res = trainer.predict(eval_ds)
metrics = eval_res.metrics
logger.info("FULL EVAL: %s", metrics)
(ROOT / "data/reports/test_results_nolang.json").write_text(
    json.dumps(metrics, indent=2), encoding="utf-8"
)
logger.info("ALL DONE")
