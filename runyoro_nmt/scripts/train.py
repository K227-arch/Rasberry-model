#!/usr/bin/env python3
"""
train.py  —  runyoro-nmt-v1
============================
Fine-tunes NLLB-200-distilled-1.3B using DataParallel across 2x RTX 4090.
Single-process, avoids torch.distributed / TCPStore libuv issue on Windows.

Launch:
    python scripts/train.py
"""

import json, logging, os, sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_TOKEN", "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")
HF_TOKEN = os.environ["HF_TOKEN"]

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

# ── logging ───────────────────────────────────────────────────────────────────
log_handlers = [
    logging.StreamHandler(),
    logging.FileHandler(str(ROOT / "training.log"), encoding="utf-8"),
]
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=log_handlers,
)
logger = logging.getLogger("train")

# ── config ────────────────────────────────────────────────────────────────────
import yaml
with open(ROOT / "configs" / "config.yaml", encoding="utf-8") as f:
    config = yaml.safe_load(f)

tc         = config["training"]
mc         = config["model"]
MODEL_NAME = mc["base_model_name"]
OUTPUT_DIR = ROOT / tc["output_dir"]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
NLLB_RNY   = mc["src_lang_nllb"]   # nyk_Latn
NLLB_ENG   = mc["tgt_lang_nllb"]   # eng_Latn
MAX_SRC    = mc["max_source_length"]
MAX_TGT    = mc["max_target_length"]

# ── torch / transformers ──────────────────────────────────────────────────────
import torch
import torch.nn as nn
from datasets import concatenate_datasets, Dataset as HFDataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
import evaluate as hf_evaluate

logger.info("PyTorch %s | CUDA %s | GPUs: %d",
            torch.__version__, torch.version.cuda, torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    logger.info("  GPU %d: %s  %.1f GB VRAM", i, p.name, p.total_memory / 1024**3)

set_seed(config["data"].get("seed", 42))

# ── data helpers ──────────────────────────────────────────────────────────────
def load_tsv(path):
    pairs = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split("\t", 1)
        if len(parts) == 2:
            pairs.append((parts[0].strip(), parts[1].strip()))
    return pairs

# ── load splits ───────────────────────────────────────────────────────────────
logger.info("Loading data splits...")
train_pairs = load_tsv(ROOT / "data/processed/train.tsv")
val_pairs   = load_tsv(ROOT / "data/processed/val.tsv")
test_pairs  = load_tsv(ROOT / "data/processed/test.tsv")
logger.info("  Train=%d  Val=%d  Test=%d",
            len(train_pairs), len(val_pairs), len(test_pairs))

# ── tokenizer ─────────────────────────────────────────────────────────────────
logger.info("Loading tokenizer: %s", MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)

# Resolve forced BOS token IDs once (verified: eng_Latn=256047, nyk_Latn=3)
ENG_BOS_ID = tokenizer.convert_tokens_to_ids(NLLB_ENG)
RNY_BOS_ID = tokenizer.convert_tokens_to_ids(NLLB_RNY)
logger.info("  %s bos_id=%d  |  %s bos_id=%d", NLLB_ENG, ENG_BOS_ID, NLLB_RNY, RNY_BOS_ID)

# ── tokenise fn ───────────────────────────────────────────────────────────────
# Verified working with transformers 4.57.6 + NLLB tokenizer:
# set src_lang, encode source; then set src_lang=tgt_lang, encode labels.
def make_tokenise_fn(src_lang, tgt_lang):
    def _fn(examples):
        # Source
        tokenizer.src_lang = src_lang
        src_enc = tokenizer(
            examples["src"],
            max_length=MAX_SRC,
            truncation=True,
            padding=False,
        )
        # Target  — temporarily set src_lang to tgt_lang so BOS token is correct
        tokenizer.src_lang = tgt_lang
        tgt_enc = tokenizer(
            examples["tgt"],
            max_length=MAX_TGT,
            truncation=True,
            padding=False,
        )
        tokenizer.src_lang = src_lang  # restore
        src_enc["labels"] = tgt_enc["input_ids"]
        return src_enc
    return _fn

def pairs_to_hf(pairs, src_lang, tgt_lang, desc=""):
    raw = HFDataset.from_dict({
        "src": [s for s, t in pairs],
        "tgt": [t for s, t in pairs],
    })
    return raw.map(
        make_tokenise_fn(src_lang, tgt_lang),
        batched=True,
        remove_columns=["src", "tgt"],
        desc=desc or f"{src_lang}->{tgt_lang}",
        load_from_cache_file=False,
    )

# ── build bidirectional datasets ──────────────────────────────────────────────
logger.info("Tokenising datasets (bidirectional)...")
train_fwd = pairs_to_hf(train_pairs, NLLB_RNY, NLLB_ENG, "Train rny->en")
train_rev = pairs_to_hf(
    [(t, s) for s, t in train_pairs], NLLB_ENG, NLLB_RNY, "Train en->rny"
)
train_ds = concatenate_datasets([train_fwd, train_rev]).shuffle(seed=42)

val_fwd = pairs_to_hf(val_pairs, NLLB_RNY, NLLB_ENG, "Val rny->en")
val_rev = pairs_to_hf(
    [(t, s) for s, t in val_pairs], NLLB_ENG, NLLB_RNY, "Val en->rny"
)
val_ds = concatenate_datasets([val_fwd, val_rev])

logger.info("  Train samples=%d  Val samples=%d", len(train_ds), len(val_ds))

# ── model ─────────────────────────────────────────────────────────────────────
logger.info("Loading model: %s", MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    token=HF_TOKEN,
)
model = model.to(torch.bfloat16)
logger.info("  Parameters: %.1f M  |  dtype: bfloat16",
            sum(p.numel() for p in model.parameters()) / 1e6)
# transformers M2M100Decoder already patched by scripts/patch_transformers.py

# ── metrics ───────────────────────────────────────────────────────────────────
bleu_metric = hf_evaluate.load("sacrebleu")
chrf_metric = hf_evaluate.load("chrf")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    # Replace -100 in preds too (can happen with padding)
    preds  = [[max(t, 0) for t in seq] for seq in preds]
    decoded_preds  = tokenizer.batch_decode(preds,  skip_special_tokens=True)
    # Replace -100 in labels (padding token)
    clean_labels   = [[max(l, 0) for l in label] for label in labels]
    decoded_labels = tokenizer.batch_decode(clean_labels, skip_special_tokens=True)
    # Strip whitespace
    decoded_preds  = [p.strip() for p in decoded_preds]
    decoded_labels = [l.strip() for l in decoded_labels]
    try:
        bleu = bleu_metric.compute(
            predictions=decoded_preds,
            references=[[ref] for ref in decoded_labels],
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

# ── collator ──────────────────────────────────────────────────────────────────
collator = DataCollatorForSeq2Seq(
    tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8
)

# ── MLflow ────────────────────────────────────────────────────────────────────
try:
    import mlflow
    mlflow.set_tracking_uri(str(ROOT / "experiments" / "mlruns"))
    mlflow.set_experiment("runyoro-nmt-v1")
    report_to = ["mlflow"]
    logger.info("MLflow enabled")
except Exception:
    report_to = ["none"]

# ── training args ─────────────────────────────────────────────────────────────
training_args = Seq2SeqTrainingArguments(
    output_dir                  = str(OUTPUT_DIR),
    num_train_epochs            = tc["num_train_epochs"],
    per_device_train_batch_size = tc["per_device_train_batch_size"],
    per_device_eval_batch_size  = tc["per_device_eval_batch_size"],
    gradient_accumulation_steps = tc["gradient_accumulation_steps"],
    learning_rate               = tc["learning_rate"],
    warmup_steps                = tc["warmup_steps"],
    weight_decay                = tc["weight_decay"],
    lr_scheduler_type           = tc["lr_scheduler_type"],
    bf16                        = False,   # model cast to bf16 manually; trainer flag triggers conflicting accelerate hooks
    fp16                        = False,
    save_strategy               = tc["save_strategy"],
    eval_strategy               = tc["evaluation_strategy"],
    load_best_model_at_end      = tc["load_best_model_at_end"],
    metric_for_best_model       = tc["metric_for_best_model"],
    greater_is_better           = tc["greater_is_better"],
    logging_steps               = tc["logging_steps"],
    save_total_limit            = tc["save_total_limit"],
    predict_with_generate       = True,
    generation_max_length       = tc["generation_max_length"],
    generation_num_beams        = tc["generation_num_beams"],
    label_smoothing_factor      = tc.get("label_smoothing_factor", 0.1),
    dataloader_num_workers      = 0,
    dataloader_pin_memory       = False,
    report_to                   = report_to,
    logging_dir                 = str(ROOT / "experiments" / "logs"),
    run_name                    = "runyoro-nmt-v1",
)

# ── trainer ───────────────────────────────────────────────────────────────────
trainer = Seq2SeqTrainer(
    model           = model,
    args            = training_args,
    train_dataset   = train_ds,
    eval_dataset    = val_ds,
    processing_class= tokenizer,
    data_collator   = collator,
    compute_metrics = compute_metrics,
    callbacks       = [EarlyStoppingCallback(early_stopping_patience=3)],
)

# ── Targeted fix for transformers 4.57.x + NLLB conflict ─────────────────────
# The Trainer's accelerate wrapper injects decoder_inputs_embeds while
# prepare_decoder_input_ids_from_labels also provides decoder_input_ids.
# Strip decoder_inputs_embeds from every batch before the forward call.
_orig_prepare = trainer._prepare_inputs

def _safe_prepare_inputs(inputs):
    prepared = _orig_prepare(inputs)
    prepared.pop("decoder_inputs_embeds", None)
    return prepared

trainer._prepare_inputs = _safe_prepare_inputs
logger.info("Trainer._prepare_inputs patched to strip decoder_inputs_embeds")

# ── curriculum stages ─────────────────────────────────────────────────────────
curriculum_cfg = tc.get("curriculum_learning", {})

def get_curriculum_subset(pairs, src_lang, tgt_lang, max_tokens):
    filtered = [(s, t) for s, t in pairs
                if max(len(s.split()), len(t.split())) <= max_tokens]
    if not filtered:
        return train_ds
    fwd = pairs_to_hf(filtered, src_lang, tgt_lang)
    rev = pairs_to_hf([(t, s) for s, t in filtered], tgt_lang, src_lang)
    return concatenate_datasets([fwd, rev]).shuffle(seed=42)

# ── launch ────────────────────────────────────────────────────────────────────
logger.info("=" * 60)
logger.info("STARTING  runyoro-nmt-v1")
logger.info("  Model     : %s", MODEL_NAME)
logger.info("  GPUs      : %d x %s",
            torch.cuda.device_count(), torch.cuda.get_device_name(0))
logger.info("  Epochs    : %d", tc["num_train_epochs"])
logger.info("  Batch     : %d x %d grad_accum",
            tc["per_device_train_batch_size"], tc["gradient_accumulation_steps"])
logger.info("  BF16      : %s", tc.get("bf16", True))
logger.info("  Curriculum: %s", curriculum_cfg.get("enabled", False))
logger.info("=" * 60)

if curriculum_cfg.get("enabled"):
    for i, stage in enumerate(curriculum_cfg["stages"]):
        max_tok, stage_epochs = stage["max_tokens"], stage["epochs"]
        logger.info("Curriculum %d/%d — max_tokens=%d  epochs=%d",
                    i + 1, len(curriculum_cfg["stages"]), max_tok, stage_epochs)
        trainer.train_dataset         = get_curriculum_subset(
            train_pairs, NLLB_RNY, NLLB_ENG, max_tok
        )
        trainer.args.num_train_epochs = stage_epochs
        trainer.train(resume_from_checkpoint=(None if i == 0 else True))
else:
    trainer.train()

logger.info("Training complete.")

# ── save ─────────────────────────────────────────────────────────────────────
save_model = trainer.model
save_model.save_pretrained(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))
logger.info("Saved to %s", OUTPUT_DIR)

# ── test evaluation ───────────────────────────────────────────────────────────
logger.info("Test-set evaluation (rny->en)...")
test_ds = pairs_to_hf(test_pairs, NLLB_RNY, NLLB_ENG, "Test rny->en")
test_res = trainer.predict(test_ds)
metrics  = test_res.metrics
logger.info("TEST: %s", metrics)
(ROOT / "data/reports/test_results.json").write_text(
    json.dumps(metrics, indent=2), encoding="utf-8"
)
bleu_score = metrics.get("test_bleu", "N/A")
chrf_score = metrics.get("test_chrf", "N/A")

# ── model card ────────────────────────────────────────────────────────────────
card = f"""---
language:
  - nyk
  - en
license: apache-2.0
tags:
  - translation
  - runyoro-rutooro
  - english
  - nmt
  - nllb
  - runyoro-nmt-v1
  - low-resource
  - bantu
datasets:
  - kathay/runyoro-rutooro-en-parallel
base_model: {MODEL_NAME}
pipeline_tag: translation
---

# runyoro-nmt-v1

**Bidirectional Runyoro-Rutooro <-> English NMT**

Fine-tuned from `{MODEL_NAME}` on
[kathay/runyoro-rutooro-en-parallel](https://huggingface.co/datasets/kathay/runyoro-rutooro-en-parallel).

## Evaluation (held-out test set, rny->en)

| Metric | Score |
|--------|-------|
| BLEU   | {bleu_score} |
| chrF++ | {chrf_score} |

## Usage

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model     = AutoModelForSeq2SeqLM.from_pretrained("kathay/runyoro-nmt-v1")
tokenizer = AutoTokenizer.from_pretrained("kathay/runyoro-nmt-v1")

# Runyoro-Rutooro -> English
tokenizer.src_lang = "nyk_Latn"
inputs = tokenizer("Oraire ota?", return_tensors="pt")
eng_bos = tokenizer.convert_tokens_to_ids("eng_Latn")
out = model.generate(**inputs, forced_bos_token_id=eng_bos, num_beams=4)
print(tokenizer.decode(out[0], skip_special_tokens=True))
# -> "How are you?"

# English -> Runyoro-Rutooro
tokenizer.src_lang = "eng_Latn"
inputs = tokenizer("How are you?", return_tensors="pt")
rny_bos = tokenizer.convert_tokens_to_ids("nyk_Latn")
out = model.generate(**inputs, forced_bos_token_id=rny_bos, num_beams=4)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

## Training Details

| Parameter | Value |
|-----------|-------|
| Base model | {MODEL_NAME} |
| Epochs | {tc['num_train_epochs']} |
| Batch size | {tc['per_device_train_batch_size']} x {tc['gradient_accumulation_steps']} grad_accum |
| Learning rate | {tc['learning_rate']} |
| BF16 | {tc.get('bf16', True)} |
| Curriculum | {curriculum_cfg.get('enabled', False)} |
| Hardware | 2x NVIDIA RTX 4090 (DataParallel) |
"""
(OUTPUT_DIR / "README.md").write_text(card, encoding="utf-8")

# ── push to Hub ───────────────────────────────────────────────────────────────
logger.info("Pushing to kathay/runyoro-nmt-v1 ...")
try:
    from huggingface_hub import HfApi
    save_model.push_to_hub(
        "kathay/runyoro-nmt-v1", token=HF_TOKEN,
        commit_message="runyoro-nmt-v1: trained weights"
    )
    tokenizer.push_to_hub("kathay/runyoro-nmt-v1", token=HF_TOKEN)
    HfApi(token=HF_TOKEN).upload_file(
        path_or_fileobj=str(OUTPUT_DIR / "README.md"),
        path_in_repo="README.md",
        repo_id="kathay/runyoro-nmt-v1",
        token=HF_TOKEN,
        commit_message="runyoro-nmt-v1: update model card with eval scores",
    )
    logger.info("Model live: https://huggingface.co/kathay/runyoro-nmt-v1")
except Exception as e:
    logger.error("Hub push failed: %s", e)

logger.info("ALL DONE  BLEU=%.2f  chrF++=%.2f", bleu_score, chrf_score)
