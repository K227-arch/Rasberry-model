"""
Stage 5 — Continue-train runyoro-rut-v12 → runyoro-rut-v13
Dual RTX 4090 via device_map auto-sharding (splits model layers across both GPUs)
+ 8-bit AdamW (cuts optimizer VRAM ~4x)
+ gradient checkpointing (cuts activation VRAM ~4x)
+ float32 (bfloat16 causes multilingual leakage on NLLB)

With device_map="auto":
  GPU 0: encoder layers + first half of decoder (~12GB)
  GPU 1: second half of decoder + lm_head (~7GB)
  Optimizer states (8-bit): spread across both GPUs via bitsandbytes
"""
import datetime, json, logging, math, os, pathlib, random, shutil
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import sacrebleu as sb

try:
    import bitsandbytes as bnb
    USE_8BIT_ADAM = True
except ImportError:
    USE_8BIT_ADAM = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train")

# ── Paths ─────────────────────────────────────────────────────────────────────
CKPT_ROOT  = pathlib.Path(r"C:\Users\keith\Desktop\projects\runyoro_nmt\models\checkpoints")
MODEL_PATH = CKPT_ROOT / "runyoro-rut-v12"
OUT_PATH   = CKPT_ROOT / "runyoro-rut-v13"
DATA_DIR   = pathlib.Path(r"C:\Users\keith\Desktop\projects\Rasberry-model\pipeline_data")
HF_REPO    = "kathay/runyoro-nmt"
HF_TOKEN   = os.environ.get("HF_WRITE_TOKEN", "os.environ.get("HF_WRITE_TOKEN","")")

# ── Hyperparams ───────────────────────────────────────────────────────────────
SEED            = 42
MAX_LEN         = 256
EPOCHS          = 8
BATCH_SIZE      = 8           # per device — device_map handles distribution
GRAD_ACCUM      = 8           # effective = 8 × 8 = 64
LR              = 5e-6
WARMUP_STEPS    = 50
WEIGHT_DECAY    = 0.01
LABEL_SMOOTHING = 0.1
EARLY_STOP_PAT  = 2

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ── GPU info ──────────────────────────────────────────────────────────────────
n_gpus = torch.cuda.device_count()
log.info(f"GPUs visible: {n_gpus}")
for i in range(n_gpus):
    p = torch.cuda.get_device_properties(i)
    log.info(f"  GPU {i}: {p.name}  {p.total_memory/1e9:.1f}GB")

# ── Load tokenizer ────────────────────────────────────────────────────────────
log.info(f"Loading tokenizer: {MODEL_PATH}")
tok = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)

# ── Load model with device_map="auto" (splits across both GPUs) ───────────────
log.info(f"Loading model with device_map='auto' — splits across {n_gpus} GPUs")
model = AutoModelForSeq2SeqLM.from_pretrained(
    str(MODEL_PATH),
    torch_dtype   = torch.float32,
    device_map    = "auto",          # auto-shards layers across GPU 0 + GPU 1
    local_files_only = True,
)

# Enable gradient checkpointing — trades compute for ~4x less activation memory
model.gradient_checkpointing_enable()
log.info("Gradient checkpointing enabled")

# device_map requires use_cache=False during training
model.config.use_cache = False

log.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
log.info(f"Device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'N/A'}")

# ── BOS token IDs ─────────────────────────────────────────────────────────────
rut_meta_path = MODEL_PATH / "rut_token_meta.json"
RUT_BOS = ENG_BOS = None
USING_RUT_PREFIX = False

if rut_meta_path.exists():
    m = json.loads(rut_meta_path.read_text())
    RUT_BOS, ENG_BOS = m["rut_token_id"], m["eng_token_id"]
    USING_RUT_PREFIX = True
    log.info(f"rut_token_meta: rut={RUT_BOS} eng={ENG_BOS}")
else:
    ENG_BOS = tok.convert_tokens_to_ids("eng_Latn")
    log.warning("No rut_token_meta.json — no forced BOS")

# ── Dataset ───────────────────────────────────────────────────────────────────
class RunyoroDataset(Dataset):
    def __init__(self, path):
        self.items = []
        pairs = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                o = json.loads(line)
                rny = o.get("runyoro_rutooro", "").strip()
                eng = o.get("english", "").strip()
                if rny and eng:
                    pairs.append((rny, eng))

        for rny, eng in pairs:
            # rny → eng
            tok.src_lang = "nyk_Latn"
            s = tok(rny, max_length=MAX_LEN, truncation=True)
            tok.tgt_lang = "eng_Latn"
            t = tok(text_target=eng, max_length=MAX_LEN, truncation=True)
            lbl = ([ENG_BOS] + t["input_ids"]) if ENG_BOS else t["input_ids"]
            self.items.append((s["input_ids"], s["attention_mask"], lbl))

            # eng → rny
            tok.src_lang = "eng_Latn"
            s2 = tok(eng, max_length=MAX_LEN, truncation=True)
            tok.tgt_lang = "nyk_Latn"
            t2 = tok(text_target=rny, max_length=MAX_LEN, truncation=True)
            lbl2 = ([RUT_BOS] + t2["input_ids"]) if (USING_RUT_PREFIX and RUT_BOS) else t2["input_ids"]
            self.items.append((s2["input_ids"], s2["attention_mask"], lbl2))

        log.info(f"  {os.path.basename(path)}: {len(pairs)} pairs → {len(self.items)} examples")

    def __len__(self):  return len(self.items)
    def __getitem__(self, i): return self.items[i]

def collate(batch):
    pad = tok.pad_token_id
    src_ids, src_mask, lbls = zip(*batch)
    max_s = max(len(x) for x in src_ids)
    max_l = max(len(x) for x in lbls)
    S, M, L = [], [], []
    for s, m, l in zip(src_ids, src_mask, lbls):
        ps, pl = max_s - len(s), max_l - len(l)
        S.append(s + [pad]*ps)
        M.append(m + [0]*ps)
        L.append(l + [-100]*pl)
    # Put tensors on CPU — device_map model handles placement internally
    return (
        torch.tensor(S, dtype=torch.long),
        torch.tensor(M, dtype=torch.long),
        torch.tensor(L, dtype=torch.long),
    )

log.info("Loading datasets...")
train_ds = RunyoroDataset(DATA_DIR / "train.jsonl")
val_ds   = RunyoroDataset(DATA_DIR / "val.jsonl")
test_ds  = RunyoroDataset(DATA_DIR / "test.jsonl")

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate, num_workers=0)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate, num_workers=0)

# ── Optimizer ─────────────────────────────────────────────────────────────────
# With device_map, parameters are on different devices.
# bitsandbytes AdamW8bit handles multi-device parameters correctly.
if USE_8BIT_ADAM:
    optimizer = bnb.optim.AdamW8bit(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    log.info("Using 8-bit AdamW across both GPUs")
else:
    from torch.optim import AdamW
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    log.warning("bitsandbytes not found — using standard AdamW (may OOM)")

steps_per_epoch = math.ceil(len(train_dl) / GRAD_ACCUM)
total_steps     = EPOCHS * steps_per_epoch
scheduler = CosineAnnealingLR(optimizer, T_max=max(total_steps - WARMUP_STEPS, 1), eta_min=LR/10)

def warmup_lr(step):
    return float(step) / max(1, WARMUP_STEPS) if step < WARMUP_STEPS else 1.0
warmup_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lr)

# ── Label smoothing loss ──────────────────────────────────────────────────────
def smooth_loss(logits, labels):
    """Cross-entropy with label smoothing, ignoring -100 padding."""
    B, T, V = logits.shape
    logits_flat  = logits.reshape(-1, V)
    labels_flat  = labels.reshape(-1).to(logits_flat.device)
    mask         = labels_flat != -100
    logits_flat  = logits_flat[mask]
    labels_flat  = labels_flat[mask]
    log_probs    = torch.nn.functional.log_softmax(logits_flat, dim=-1)
    nll          = -log_probs.gather(1, labels_flat.unsqueeze(1)).squeeze(1)
    smooth       = -log_probs.mean(dim=-1)
    return ((1 - LABEL_SMOOTHING) * nll + LABEL_SMOOTHING * smooth).mean()

# ── Eval ──────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_split(dl, label="val"):
    model.eval()
    torch.cuda.empty_cache()
    total_loss, n_batches, samples = 0.0, 0, []

    # With device_map, inputs go to the first device (GPU 0)
    first_device = next(model.parameters()).device

    for src_ids, src_mask, lbls in dl:
        src_ids  = src_ids.to(first_device)
        src_mask = src_mask.to(first_device)
        lbls_dev = lbls.to(first_device)

        out = model(input_ids=src_ids, attention_mask=src_mask, labels=lbls_dev)
        total_loss += out.loss.item()
        n_batches  += 1

        if len(samples) < 4:
            torch.cuda.empty_cache()
            gens = model.generate(
                input_ids=src_ids[:1], attention_mask=src_mask[:1],
                num_beams=1, do_sample=False, max_new_tokens=64,
            )
            samples.append(tok.decode(gens[0], skip_special_tokens=True))

    if samples:
        log.info(f"  [{label}] sample: {samples[0][:80]}")
    model.train()
    return total_loss / max(n_batches, 1)

# ── Training loop ─────────────────────────────────────────────────────────────
log.info("=" * 60)
log.info(f"Training: runyoro-rut-v12 → runyoro-rut-v13")
log.info(f"  GPUs        : {n_gpus} × RTX 4090 (device_map auto-shard)")
log.info(f"  Train       : {len(train_ds)} examples  Val: {len(val_ds)}")
log.info(f"  Epochs      : {EPOCHS}  LR: {LR}  Early-stop: {EARLY_STOP_PAT}")
log.info(f"  Batch       : {BATCH_SIZE} × {GRAD_ACCUM} accum = {BATCH_SIZE*GRAD_ACCUM} effective")
log.info(f"  Optimizer   : {'8-bit AdamW' if USE_8BIT_ADAM else 'AdamW'}")
log.info(f"  Grad ckpt   : enabled")
log.info("=" * 60)

first_device  = next(model.parameters()).device
best_val_loss = float("inf")
no_improve    = 0
global_step   = 0
history       = []

model.train()
for epoch in range(1, EPOCHS + 1):
    epoch_loss = 0.0
    optimizer.zero_grad()

    for step, (src_ids, src_mask, lbls) in enumerate(train_dl, 1):
        src_ids  = src_ids.to(first_device)
        src_mask = src_mask.to(first_device)
        lbls     = lbls.to(first_device)

        out    = model(input_ids=src_ids, attention_mask=src_mask, labels=lbls)
        logits = out.logits
        loss   = smooth_loss(logits, lbls) / GRAD_ACCUM
        loss.backward()
        epoch_loss += loss.item() * GRAD_ACCUM

        if step % GRAD_ACCUM == 0 or step == len(train_dl):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if global_step < WARMUP_STEPS:
                warmup_sched.step()
            else:
                scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % 20 == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                log.info(f"  step {global_step:4d}  loss {loss.item()*GRAD_ACCUM:.4f}  lr {lr_now:.2e}")

    avg_train = epoch_loss / len(train_dl)
    val_loss  = evaluate_split(val_dl, "val")
    log.info(f"Epoch {epoch}/{EPOCHS}  train={avg_train:.4f}  val={val_loss:.4f}")
    history.append({"epoch": epoch, "train_loss": avg_train, "val_loss": val_loss})

    # Save epoch checkpoint
    ckpt_dir = OUT_PATH / f"checkpoint-epoch{epoch}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ckpt_dir))
    tok.save_pretrained(str(ckpt_dir))
    log.info(f"  Saved checkpoint-epoch{epoch}")

    # Best model + early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve    = 0
        OUT_PATH.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(OUT_PATH))
        tok.save_pretrained(str(OUT_PATH))
        log.info(f"  ✓ New best val_loss={val_loss:.4f}")
    else:
        no_improve += 1
        log.info(f"  No improvement ({no_improve}/{EARLY_STOP_PAT})")
        if no_improve >= EARLY_STOP_PAT:
            log.info(f"Early stopping at epoch {epoch}")
            break

# ── Test eval ─────────────────────────────────────────────────────────────────
test_dl  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate, num_workers=0)
test_loss = evaluate_split(test_dl, "test")
log.info(f"Test loss: {test_loss:.4f}")

# ── Save metadata ─────────────────────────────────────────────────────────────
if rut_meta_path.exists():
    shutil.copy(str(rut_meta_path), str(OUT_PATH / "rut_token_meta.json"))

meta = {
    "model":             "runyoro-rut-v13",
    "continued_from":    "runyoro-rut-v12",
    "strategy":          "continue_training_clean_only_no_bt",
    "base_model":        "facebook/nllb-200-distilled-1.3B",
    "new_train_pairs":   len(train_ds) // 2,
    "new_val_pairs":     len(val_ds)   // 2,
    "new_test_pairs":    len(test_ds)  // 2,
    "bt_pairs":          0,
    "best_val_loss":     best_val_loss,
    "test_loss":         test_loss,
    "epochs_run":        len(history),
    "lr":                LR,
    "label_smoothing":   LABEL_SMOOTHING,
    "gpus":              n_gpus,
    "gpu_strategy":      "device_map_auto",
    "gradient_checkpointing": True,
    "timestamp":         datetime.datetime.utcnow().isoformat() + "Z",
    "history":           history,
}
(OUT_PATH / "training_metadata.json").write_text(json.dumps(meta, indent=2))
log.info("Saved training_metadata.json")

# ── Push to HF Hub ────────────────────────────────────────────────────────────
log.info(f"Pushing to {HF_REPO} ...")
from huggingface_hub import HfApi
api = HfApi(token=HF_TOKEN)
api.upload_folder(
    folder_path    = str(OUT_PATH),
    repo_id        = HF_REPO,
    repo_type      = "model",
    commit_message = f"runyoro-rut-v13: dual RTX4090 device_map, sp20 included, val_loss={best_val_loss:.4f}",
    ignore_patterns= ["checkpoint-epoch*/**"],
)
log.info(f"Pushed to {HF_REPO} ✓")
log.info("Training complete.")
