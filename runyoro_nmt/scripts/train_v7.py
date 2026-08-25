#!/usr/bin/env python3
"""
train_v7.py — Continual Learning fine-tune: runyoro-rut-v2 → runyoro-rut-v3
=============================================================================

What this does
--------------
1. EXTRACT   — read all new raw xlsx files (5–13) and pull clean pairs
2. COMBINE   — stack on v6 cleaned pairs, deduplicate
3. AUGMENT   — light token-deletion + swap augmentation
4. EWC PREP  — compute Fisher Information Matrix on the *old* data (v6 pairs)
               to identify which weights matter most for existing knowledge
5. TRAIN     — fine-tune from runyoro-rut-v2 with EWC penalty so the model
               learns new patterns WITHOUT forgetting old ones
6. SAVE      — write model + tokenizer + rut_token_meta.json

Why continual learning instead of training from scratch?
---------------------------------------------------------
Training from scratch each time (v4 → v5 → v6) discards all previously
learned representations. With EWC:
  - Important weights (high Fisher score) are penalised heavily if they drift
  - Less-critical weights adapt freely to new data
  - Catastrophic forgetting is suppressed without needing to keep ALL old data
    in memory (though we keep v6 clean pairs as an anchor set anyway)

Key hyperparameter choices
--------------------------
  - LR = 1e-5   (lower than from-scratch 5e-5 — we're adapting, not re-learning)
  - Epochs = 5  (short — just absorb the new data)
  - EWC λ = 500 (moderate penalty; increase if you see regression on old phrases)
  - float32     (NOT bf16 — bf16 causes multilingual leakage in this model)
  - forced_bos_token_id preserved from rut-v2 tokenizer metadata

Usage:
    python scripts/train_v7.py                # full continual learning run
    python scripts/train_v7.py --clean-only   # data prep only, skip training
    python scripts/train_v7.py --ewc-lambda 1000  # stronger forgetting protection
    python scripts/train_v7.py --epochs 8         # more epochs for larger datasets
"""

import argparse
import gc
import json
import logging
import os
import random
import re
import sys
import unicodedata
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
        logging.FileHandler(str(ROOT / "training_v7.log"), encoding="utf-8"),
    ],
)
logger = logging.getLogger("train_v7")

import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset as HFDataset, concatenate_datasets
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

set_seed(42)
random.seed(42)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW_DIR       = ROOT.parent / "raw"
V6_DATA_DIR   = ROOT / "data" / "v6_training"
V7_DATA_DIR   = ROOT / "data" / "v7_training"
V8_DATA_DIR   = ROOT / "data" / "v8_training"
V7_DATA_DIR.mkdir(parents=True, exist_ok=True)
V8_DATA_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL    = str(ROOT / "models" / "checkpoints" / "runyoro-rut-v3")
OUTPUT_DIR    = ROOT / "models" / "checkpoints" / "runyoro-rut-v4"
MAX_LEN       = 256

# ---------------------------------------------------------------------------
# New raw files to process (everything after sentence pair (4).xlsx)
# ---------------------------------------------------------------------------
NEW_FILES = [
    "sentence pair (5).xlsx",
    "sentence pairs 6.xlsx",
    "sentence pair 7.xlsx",
    "sentence pairs 8.xlsx",
    "sentence pair 9.xlsx",
    "SENTENCE PAIRS 10.xlsx",
    "sentence pair 11.xlsx",
    "sentence pair 12.xlsx",
    "sentence pairs 13 (1).xlsx",
    "senence pair 14.xlsx",
]


# ===========================================================================
# DATA PROCESSING
# ===========================================================================

def clean(text) -> str:
    text = str(text)
    if text.lower() in ("nan", "none") or not text.strip():
        return ""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^[-–—]+\s*", "", text).strip()
    text = re.sub(r"\[[A-Z_]+\]\s*", "", text).strip()   # remove [NOUN] etc.
    text = re.sub(r"^\d+[.)]\s*", "", text).strip()       # remove "1. " numbering
    return text


def valid(eng: str, rny: str) -> bool:
    if len(eng) < 5 or len(rny) < 5:
        return False
    if eng.lower() in ("nan", "none", "english", "original english"):
        return False
    if rny.lower() in ("nan", "none", "runyoro"):
        return False
    if re.search(r"\(v\.[it]\.\)", eng) or re.search(r"\(v\.[it]\.\)", rny):
        return False
    if len(eng) > 0 and len(rny) > 0:
        ratio = max(len(eng), len(rny)) / min(len(eng), len(rny))
        if ratio > 8:
            return False
    if re.fullmatch(r"[\d\s\W]+", eng) or re.fullmatch(r"[\d\s\W]+", rny):
        return False
    return True


def extract_standard(path: Path) -> List[Tuple[str, str]]:
    """Standard 8-column xlsx: col2=eng_orig, col4=rny_orig, col6=eng_var, col7=rny_var."""
    df = pd.read_excel(path, header=None)
    pairs = []
    for _, row in df.iterrows():
        orig_eng = clean(row.iloc[2]) if pd.notna(row.iloc[2]) else ""
        orig_rny = clean(row.iloc[4]) if pd.notna(row.iloc[4]) else ""
        var_eng  = clean(row.iloc[6]) if pd.notna(row.iloc[6]) else ""
        var_rny  = clean(row.iloc[7]) if pd.notna(row.iloc[7]) else ""
        if orig_eng and orig_rny and valid(orig_eng, orig_rny):
            pairs.append((orig_rny, orig_eng))
        if var_eng and var_rny and valid(var_eng, var_rny):
            pairs.append((var_rny, var_eng))
    logger.info("  [%s] %d pairs", path.name, len(pairs))
    return pairs


def extract_file10(path: Path) -> List[Tuple[str, str]]:
    """SENTENCE PAIRS 10 has a header row + Status col (col[8]). Skip pending variants."""
    df = pd.read_excel(path, header=None)
    pairs = []
    skipped = 0
    for _, row in df.iloc[1:].iterrows():          # skip header row
        orig_eng = clean(row.iloc[2]) if pd.notna(row.iloc[2]) else ""
        orig_rny = clean(row.iloc[4]) if pd.notna(row.iloc[4]) else ""
        var_eng  = clean(row.iloc[6]) if pd.notna(row.iloc[6]) else ""
        var_rny  = clean(row.iloc[7]) if pd.notna(row.iloc[7]) else ""
        status   = str(row.iloc[8]).strip().lower() if len(row) > 8 and pd.notna(row.iloc[8]) else "pending"
        if orig_eng and orig_rny and valid(orig_eng, orig_rny):
            pairs.append((orig_rny, orig_eng))
        if status == "pending":
            skipped += 1
        elif var_eng and var_rny and valid(var_eng, var_rny):
            pairs.append((var_rny, var_eng))
    logger.info("  [%s] %d pairs (%d variant rows skipped — pending)", path.name, len(pairs), skipped)
    return pairs


def step1_extract() -> List[Tuple[str, str]]:
    logger.info("=" * 60)
    logger.info("STEP 1: Extract from new raw files")
    logger.info("=" * 60)
    new_pairs: List[Tuple[str, str]] = []
    for filename in NEW_FILES:
        path = RAW_DIR / filename
        if not path.exists():
            logger.warning("  Not found — skipping: %s", filename)
            continue
        if filename.upper().startswith("SENTENCE PAIRS 10"):
            new_pairs.extend(extract_file10(path))
        else:
            new_pairs.extend(extract_standard(path))
    logger.info("Total new pairs extracted: %d", len(new_pairs))
    return new_pairs


def step2_combine(new_pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    logger.info("=" * 60)
    logger.info("STEP 2: Combine with v6 cleaned pairs")
    logger.info("=" * 60)
    v6_csv = V6_DATA_DIR / "cleaned_pairs.csv"
    if v6_csv.exists():
        v6_df    = pd.read_csv(v6_csv)
        existing = [(str(r["Runyoro"]).strip(), str(r["English"]).strip()) for _, r in v6_df.iterrows()]
        logger.info("v6 existing: %d pairs", len(existing))
    else:
        logger.warning("v6 cleaned_pairs.csv not found — new data only")
        existing = []

    all_pairs = existing + new_pairs
    seen: set = set()
    unique: List[Tuple[str, str]] = []
    for rny, eng in all_pairs:
        key = (rny.lower().strip(), eng.lower().strip())
        if key not in seen:
            seen.add(key)
            unique.append((rny, eng))

    logger.info("After dedup: %d unique pairs (+%d new vs v6)", len(unique), len(unique) - len(existing))
    return unique


def step3_save(unique: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    logger.info("=" * 60)
    logger.info("STEP 3: Save clean pairs to v8_training/")
    logger.info("=" * 60)
    training = unique.copy()
    random.shuffle(training)
    pd.DataFrame(unique,   columns=["Runyoro", "English"]).to_csv(V8_DATA_DIR / "cleaned_pairs.csv",      index=False)
    pd.DataFrame(training, columns=["Runyoro", "English"]).to_csv(V8_DATA_DIR / "all_training_pairs.csv", index=False)
    logger.info("Saved: clean=%d  (no augmentation)", len(unique))
    return training


# ===========================================================================
# DATASET BUILDING (with forced BOS labels — same as rut-v1/v2)
# ===========================================================================

def make_hf_dataset(
    pairs: List[Tuple[str, str]],
    tok,
    rut_id: int,
    eng_id: int,
) -> HFDataset:
    """
    Each pair → two samples:
      EN→RUT : input=English,  labels=[rut_id] + runyoro_token_ids
      RUT→EN : input=Runyoro,  labels=[eng_id] + english_token_ids

    The prepended BOS token is what forces the decoder to the correct language
    during training — matching what the model server uses at inference.
    """
    samples = []
    for rny, eng in pairs:
        # EN → RUT
        src = tok(eng, max_length=MAX_LEN,     truncation=True, padding=False)
        tgt = tok(rny, max_length=MAX_LEN - 1, truncation=True, padding=False)
        samples.append({
            "input_ids":      src["input_ids"],
            "attention_mask": src["attention_mask"],
            "labels":         [rut_id] + tgt["input_ids"],
        })
        # RUT → EN
        src = tok(rny, max_length=MAX_LEN,     truncation=True, padding=False)
        tgt = tok(eng, max_length=MAX_LEN - 1, truncation=True, padding=False)
        samples.append({
            "input_ids":      src["input_ids"],
            "attention_mask": src["attention_mask"],
            "labels":         [eng_id] + tgt["input_ids"],
        })

    random.shuffle(samples)
    return HFDataset.from_dict({
        "input_ids":      [s["input_ids"]      for s in samples],
        "attention_mask": [s["attention_mask"]  for s in samples],
        "labels":         [s["labels"]          for s in samples],
    })


# ===========================================================================
# EWC — Elastic Weight Consolidation
# ===========================================================================

class EWC:
    """
    Computes and applies the EWC penalty.

    The Fisher Information Matrix (diagonal approximation) measures how
    sensitive each parameter is to the loss on the *old* data.  Parameters
    with high Fisher values are the ones that encode existing knowledge and
    should not drift far during new-data training.

    Penalty added to training loss:
        L_ewc = (λ/2) * Σ F_i * (θ_i − θ*_i)²

    where:
        F_i   = diagonal Fisher estimate for parameter i
        θ*_i  = optimal parameter values from old training (anchor)
        θ_i   = current parameter values (being updated)
        λ     = ewc_lambda (controls how strongly old knowledge is protected)

    References:
        Kirkpatrick et al., 2017 — "Overcoming catastrophic forgetting in
        neural networks"  https://arxiv.org/abs/1612.00796
    """

    def __init__(self, model: nn.Module, ewc_lambda: float = 500.0):
        self.lambda_  = ewc_lambda
        self._anchors: Dict[str, torch.Tensor] = {}   # θ*
        self._fisher:  Dict[str, torch.Tensor] = {}   # F (diagonal)

    # ------------------------------------------------------------------
    def compute_fisher(
        self,
        model: nn.Module,
        anchor_pairs: List[Tuple[str, str]],
        tok,
        rut_id: int,
        eng_id: int,
        n_samples: int = 200,
        device: str = "cpu",
    ) -> None:
        """
        Estimate diagonal Fisher on up to n_samples old pairs on GPU.
        Uses the log-likelihood gradient squared as the Fisher estimator.
        Temporarily disables gradient checkpointing for correct gradient flow.
        """
        logger.info("Computing EWC Fisher Information on %d anchor samples (GPU)...", min(n_samples, len(anchor_pairs) * 2))

        # Temporarily disable gradient checkpointing — it interferes with Fisher gradients
        gc_was_enabled = getattr(model, "is_gradient_checkpointing", False)
        if gc_was_enabled:
            model.gradient_checkpointing_disable()

        # Save parameter anchors (θ*) and zero Fisher accumulators on CPU to save VRAM
        for name, param in model.named_parameters():
            if param.requires_grad:
                self._anchors[name] = param.data.clone().detach().cpu()
                self._fisher[name]  = torch.zeros_like(param.data, device="cpu")

        sample_pairs = anchor_pairs[:n_samples]
        ds = make_hf_dataset(sample_pairs, tok, rut_id, eng_id)

        # Inputs go to the first device — device_map routes layers internally
        first_device = next(model.parameters()).device

        collator = DataCollatorForSeq2Seq(tok, model=model, label_pad_token_id=-100, pad_to_multiple_of=8)
        loader   = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collator)

        model.train()
        model.zero_grad()

        n_batches = 0
        for batch in loader:
            batch = {k: v.to(first_device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            outputs = model(**batch)
            loss    = outputs.loss
            loss.backward()

            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    # Accumulate squared gradients on CPU to avoid VRAM pressure
                    self._fisher[name] += param.grad.data.detach().cpu().pow(2)

            model.zero_grad()
            n_batches += 1

        for name in self._fisher:
            self._fisher[name] /= max(n_batches, 1)

        model.eval()

        # Re-enable gradient checkpointing for actual training
        if gc_was_enabled:
            model.gradient_checkpointing_enable()

        logger.info("Fisher computed over %d batches on GPU.", n_batches)

    # ------------------------------------------------------------------
    def penalty(self, model: nn.Module) -> torch.Tensor:
        """Return the EWC regularisation term to add to the training loss."""
        if not self._anchors:
            return torch.tensor(0.0)

        # Accumulate on CPU to avoid cross-device tensor arithmetic when
        # device_map="auto" splits the model across multiple GPUs.
        loss = torch.tensor(0.0)
        for name, param in model.named_parameters():
            if name in self._anchors and param.requires_grad:
                # Move anchor and fisher to param's device for the subtraction,
                # then immediately move the scalar result back to CPU.
                p_dev  = param.device
                anchor = self._anchors[name].to(p_dev)
                fisher = self._fisher[name].to(p_dev)
                term   = (fisher * (param - anchor).pow(2)).sum().cpu()
                loss  += term

        # Return on the first parameter's device so the Trainer can add it to
        # the cross-entropy loss without a device error.
        target_device = next(model.parameters()).device
        return ((self.lambda_ / 2) * loss).to(target_device)


# ===========================================================================
# CUSTOM TRAINER WITH EWC LOSS
# ===========================================================================

class EWCTrainer(Seq2SeqTrainer):
    """
    Seq2SeqTrainer subclass that adds the EWC penalty to the cross-entropy loss.
    The EWC object is passed at construction time.
    """

    def __init__(self, ewc: EWC, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ewc = ewc

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        ce_loss = outputs.loss

        ewc_penalty = self._ewc.penalty(model)
        total_loss  = ce_loss + ewc_penalty

        # Log both components periodically
        if self.state.global_step % 50 == 0:
            logger.debug(
                "step=%d  ce_loss=%.4f  ewc_penalty=%.4f  total=%.4f",
                self.state.global_step,
                ce_loss.item(),
                ewc_penalty.item(),
                total_loss.item(),
            )

        return (total_loss, outputs) if return_outputs else total_loss


# ===========================================================================
# TRAINING
# ===========================================================================

def step5_train(
    training_pairs: List[Tuple[str, str]],
    anchor_pairs:   List[Tuple[str, str]],
    ewc_lambda:     float,
    n_epochs:       int,
) -> None:
    logger.info("=" * 60)
    logger.info("STEP 5: Continual Learning — runyoro-rut-v3")
    logger.info("=" * 60)

    gc.collect()
    torch.cuda.empty_cache()
    n_gpus  = torch.cuda.device_count()
    device  = "cuda" if n_gpus > 0 else "cpu"
    logger.info("GPUs available: %d  |  device: %s", n_gpus, device)

    # ── Load tokenizer + token IDs from rut-v2 ───────────────────────────────
    logger.info("Loading tokenizer from: %s", BASE_MODEL)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)

    meta_path = Path(BASE_MODEL) / "rut_token_meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    rut_id = meta["rut_token_id"]   # 256204
    eng_id = meta["eng_token_id"]   # 256047
    logger.info("Token IDs — rut_Latn=%d  eng_Latn=%d", rut_id, eng_id)

    # ── Load model from rut-v2 ────────────────────────────────────────────────
    logger.info("Loading model from: %s  (float32)", BASE_MODEL)
    if n_gpus >= 2:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float32,
            device_map="auto",
            max_memory={i: "22GiB" for i in range(n_gpus)},
        )
    elif n_gpus == 1:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.float32
        ).to(device)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.float32
        )
    model.resize_token_embeddings(len(tok))   # safe no-op if already correct size

    logger.info("Model loaded: %.1fM params", sum(p.numel() for p in model.parameters()) / 1e6)

    # ── Compute EWC Fisher on old (anchor) data ───────────────────────────────
    ewc = EWC(model, ewc_lambda=ewc_lambda)

    # Run Fisher computation on the same GPU model — no CPU copy needed.
    # The penalty() method already handles multi-device by accumulating on CPU
    # and moving the scalar result to the first parameter's device.
    ewc.compute_fisher(
        model        = model,
        anchor_pairs = anchor_pairs,
        tok          = tok,
        rut_id       = rut_id,
        eng_id       = eng_id,
        n_samples    = min(50, len(anchor_pairs)),
        device       = device,
    )

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Build training dataset ────────────────────────────────────────────────
    train_ds = make_hf_dataset(training_pairs, tok, rut_id, eng_id)

    # Validation: 10% of anchor pairs (old data) — checks we haven't forgotten
    n_val     = max(20, int(len(anchor_pairs) * 0.10))
    val_pairs = anchor_pairs[:n_val]
    val_ds    = make_hf_dataset(val_pairs, tok, rut_id, eng_id)

    logger.info(
        "Train samples: %d  |  Val samples (old-data anchor): %d",
        len(train_ds), len(val_ds),
    )

    model.gradient_checkpointing_enable()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    collator = DataCollatorForSeq2Seq(tok, model=model, label_pad_token_id=-100, pad_to_multiple_of=8)

    training_args = Seq2SeqTrainingArguments(
        output_dir            = str(OUTPUT_DIR),
        num_train_epochs      = n_epochs,
        per_device_train_batch_size  = 2,
        per_device_eval_batch_size   = 4,
        gradient_accumulation_steps  = 16,    # effective batch = 32
        learning_rate         = 1e-5,         # low LR — we're adapting, not re-learning
        warmup_ratio          = 0.1,
        weight_decay          = 0.01,
        lr_scheduler_type     = "cosine",
        fp16                  = False,
        bf16                  = False,        # float32 — prevents multilingual leakage
        save_strategy         = "epoch",
        eval_strategy         = "epoch",      # eval on old-data val set each epoch
        logging_steps         = 10,
        save_total_limit      = 2,
        load_best_model_at_end = False,       # incompatible with save_only_model
        predict_with_generate = False,
        dataloader_num_workers= 0,            # Windows: avoid multiprocessing issues
        dataloader_pin_memory = False,
        gradient_checkpointing= True,
        optim                 = "adamw_torch",
        max_grad_norm         = 1.0,
        save_safetensors      = True,
        save_only_model       = True,
        report_to             = ["none"],
        run_name              = "runyoro-rut-v4",
    )

    trainer = EWCTrainer(
        ewc              = ewc,
        model            = model,
        args             = training_args,
        train_dataset    = train_ds,
        eval_dataset     = val_ds,
        processing_class = tok,
        data_collator    = collator,
    )

    logger.info("=" * 60)
    logger.info("CONTINUAL LEARNING: runyoro-rut-v4")
    logger.info("  Base:        runyoro-rut-v3")
    logger.info("  Clean pairs: %d  |  Train samples (bidirectional): %d",
                len(training_pairs), len(train_ds))
    logger.info("  Epochs: %d  |  LR: 1e-5  |  EWC λ: %.0f", n_epochs, ewc_lambda)
    logger.info("  Anchors (old data for EWC Fisher): %d pairs", len(anchor_pairs))
    logger.info("=" * 60)

    # Resume from latest checkpoint if one exists in the output dir
    resume_ckpt = None
    ckpt_dirs = sorted(OUTPUT_DIR.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    if ckpt_dirs:
        resume_ckpt = str(ckpt_dirs[-1])
        logger.info("Resuming from checkpoint: %s", resume_ckpt)
    else:
        logger.info("No checkpoint found — training from scratch.")

    trainer.train(resume_from_checkpoint=resume_ckpt)

    # ── Save model + tokenizer + metadata ────────────────────────────────────
    logger.info("Saving model + tokenizer to: %s", OUTPUT_DIR)
    model.save_pretrained(str(OUTPUT_DIR))
    tok.save_pretrained(str(OUTPUT_DIR))

    # Carry forward the rut_token_meta.json — same token IDs
    new_meta = {
        "rut_token":     "rut_Latn",
        "rut_token_id":  rut_id,
        "eng_token":     "eng_Latn",
        "eng_token_id":  eng_id,
        "base_model":    BASE_MODEL,
        "strategy":      "forced_bos_token_id",
        "note":          "Use forced_bos_token_id=rut_token_id for EN->RUT, eng_token_id for RUT->EN",
    }
    with open(OUTPUT_DIR / "rut_token_meta.json", "w") as f:
        json.dump(new_meta, f, indent=2)

    # Save EWC training summary
    summary = {
        "continual_learning": True,
        "ewc_lambda":         ewc_lambda,
        "base_model":         BASE_MODEL,
        "output_model":       str(OUTPUT_DIR),
        "run_name":           "runyoro-rut-v4",
        "total_clean_pairs":  len(anchor_pairs),
        "total_training_pairs": len(training_pairs),
        "epochs":             n_epochs,
        "learning_rate":      1e-5,
        "precision":          "float32",
    }
    with open(OUTPUT_DIR / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("rut_token_meta.json saved.")
    logger.info("DONE — runyoro-rut-v3 saved to: %s", OUTPUT_DIR)

    del model, trainer, ewc
    gc.collect()
    torch.cuda.empty_cache()


# ===========================================================================
# ENTRY POINT
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Continual learning fine-tune: rut-v2 → rut-v3")
    parser.add_argument("--clean-only",   action="store_true",  help="Data prep only, skip training")
    parser.add_argument("--ewc-lambda",   type=float, default=500.0,
                        help="EWC regularisation strength (default: 500). "
                             "Higher = stronger protection against forgetting old knowledge.")
    parser.add_argument("--epochs",       type=int,   default=5,
                        help="Number of training epochs (default: 5)")
    args = parser.parse_args()

    # ── Data pipeline ─────────────────────────────────────────────────────────
    new_pairs = step1_extract()
    unique    = step2_combine(new_pairs)   # v6 pairs + new pairs, deduped
    training  = step3_save(unique)

    if args.clean_only:
        logger.info("--clean-only: stopping before training.")
        logger.info("  Clean pairs: %d", len(unique))
        return

    # ── Anchor pairs = v7 clean pairs (what rut-v3 was trained on) ───────────
    v7_csv = V7_DATA_DIR / "cleaned_pairs.csv"
    if v7_csv.exists():
        v7_df        = pd.read_csv(v7_csv)
        anchor_pairs = [(str(r["Runyoro"]), str(r["English"])) for _, r in v7_df.iterrows()]
    else:
        logger.warning("v7 cleaned_pairs.csv not found — using all unique pairs as anchors")
        anchor_pairs = unique

    logger.info("Anchor pairs for EWC Fisher: %d", len(anchor_pairs))

    # ── Train ─────────────────────────────────────────────────────────────────
    step5_train(
        training_pairs = training,
        anchor_pairs   = anchor_pairs,
        ewc_lambda     = args.ewc_lambda,
        n_epochs       = args.epochs,
    )


if __name__ == "__main__":
    main()
