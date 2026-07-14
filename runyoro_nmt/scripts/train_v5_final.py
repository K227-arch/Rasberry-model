#!/usr/bin/env python3
"""
train_v5_final.py — Train final v5 model from pre-saved back-translated data.
Resumes from where the full pipeline OOM'd.
"""
import os, sys, gc, logging
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["USE_TF"] = "0"

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(), logging.FileHandler(str(ROOT / "training_v5_final.log"))])
logger = logging.getLogger("train_v5_final")

import pandas as pd
import torch
from datasets import Dataset as HFDataset, concatenate_datasets
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments, set_seed)

set_seed(42)

V5_DATA_DIR = ROOT / "data" / "v5_training"
OUTPUT_DIR = ROOT / "models" / "checkpoints" / "runyoro-nmt-v5"
MODEL_NAME = "facebook/nllb-200-distilled-1.3B"
MAX_LEN = 256

# Load final training pairs
df = pd.read_csv(V5_DATA_DIR / "final_training_pairs.csv")
pairs = list(zip(df["Runyoro"].tolist(), df["English"].tolist()))
logger.info("Loaded %d final training pairs (with back-translation)", len(pairs))

# Clear GPU
gc.collect()
torch.cuda.empty_cache()
logger.info("GPUs: %d", torch.cuda.device_count())

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_fn(examples):
    src_enc = tokenizer(examples["src"], max_length=MAX_LEN, truncation=True, padding=False)
    tgt_enc = tokenizer(examples["tgt"], max_length=MAX_LEN, truncation=True, padding=False)
    src_enc["labels"] = tgt_enc["input_ids"]
    return src_enc

def pairs_to_dataset(pair_list, desc=""):
    ds = HFDataset.from_dict({"src": [s for s, t in pair_list], "tgt": [t for s, t in pair_list]})
    return ds.map(tokenize_fn, batched=True, remove_columns=["src", "tgt"], desc=desc)

# Bidirectional
fwd_ds = pairs_to_dataset(pairs, "v5-final rny->eng")
rev_ds = pairs_to_dataset([(e, r) for r, e in pairs], "v5-final eng->rny")
train_ds = concatenate_datasets([fwd_ds, rev_ds]).shuffle(seed=42)
logger.info("Train dataset: %d samples (bidirectional)", len(train_ds))

# Load model - use batch_size=2 and more grad_accum to avoid OOM
logger.info("Loading model: %s (FP32)", MODEL_NAME)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32,
                                               device_map="auto", max_memory={0: "22GiB", 1: "22GiB"})
model.gradient_checkpointing_enable()
logger.info("Model loaded: %.1fM params", sum(p.numel() for p in model.parameters()) / 1e6)

collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8)

training_args = Seq2SeqTrainingArguments(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=20,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,  # effective batch = 32
    learning_rate=5e-5,
    warmup_steps=100,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    fp16=False, bf16=False,
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
)

trainer = Seq2SeqTrainer(model=model, args=training_args, train_dataset=train_ds,
                         processing_class=tokenizer, data_collator=collator)

logger.info("=" * 60)
logger.info("TRAINING v5-final (with back-translation data)")
logger.info("  Pairs: %d | Samples: %d | Epochs: 20 | Batch: 2x16=32", len(pairs), len(train_ds))
logger.info("=" * 60)

trainer.train()
model.save_pretrained(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))
logger.info("v5-final model saved to: %s", OUTPUT_DIR)
logger.info("DONE!")
