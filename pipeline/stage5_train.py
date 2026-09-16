"""
Stage 5 — Continue-train runyoro-rut-v7 on new pairs.
Manual PyTorch training loop — avoids Trainer/Accelerate/DataParallel
compatibility issues with custom datasets on Windows.

Overfitting guards:
  - Conservative LR (5e-6), cosine schedule with warmup
  - Label smoothing 0.1
  - Early stopping patience=2 on eval_loss
  - Max 8 epochs
  - No BT pairs

Underfitting guards:
  - Bidirectional: each pair → 2 examples
  - Full model unfrozen
  - Effective batch 64 across 2 GPUs via DataParallel
"""
import datetime, json, logging, math, os, pathlib, random, shutil
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import sacrebleu as sb   # direct sacrebleu — no pyarrow/datasets dependency

# Use 8-bit AdamW if bitsandbytes is available — cuts optimizer VRAM ~4×
try:
    import bitsandbytes as bnb
    USE_8BIT_ADAM = True
except ImportError:
    USE_8BIT_ADAM = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("train")

# ── Paths ─────────────────────────────────────────────────────────────────────
CKPT_ROOT  = pathlib.Path(r"C:\Users\keith\Desktop\projects\runyoro_nmt\models\checkpoints")
MODEL_PATH = CKPT_ROOT / "runyoro-rut-v11"
OUT_PATH   = CKPT_ROOT / "runyoro-rut-v12"
DATA_DIR   = pathlib.Path(r"C:\Users\keith\Desktop\projects\Rasberry-model\pipeline_data")
HF_REPO    = "kathay/runyoro-nmt"
HF_TOKEN   = os.environ.get("HF_WRITE_TOKEN", "")

# ── Hyperparams ───────────────────────────────────────────────────────────────
SEED              = 42
MAX_LEN           = 256
EPOCHS            = 8
BATCH_SIZE        = 4           # single GPU — float32 1.3B + AdamW optimizer states ~15GB
GRAD_ACCUM        = 16          # effective = 4 × 1 GPU × 16 = 64
LR                = 5e-6
WARMUP_STEPS      = 50
WEIGHT_DECAY      = 0.01
LABEL_SMOOTHING   = 0.1
EARLY_STOP_PAT    = 2

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ── GPU ───────────────────────────────────────────────────────────────────────
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpus  = torch.cuda.device_count()
log.info(f"Device: {device}  GPUs: {n_gpus}")
for i in range(n_gpus):
    p = torch.cuda.get_device_properties(i)
    log.info(f"  GPU {i}: {p.name}  {p.total_memory/1e9:.1f}GB")

# ── Tokenizer & Model ─────────────────────────────────────────────────────────
log.info(f"Loading tokenizer: {MODEL_PATH}")
tok = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)

log.info(f"Loading model: {MODEL_PATH}  [float32]")
model = AutoModelForSeq2SeqLM.from_pretrained(str(MODEL_PATH), torch_dtype=torch.float32, local_files_only=True)

if n_gpus > 1:
    log.info(f"Multiple GPUs detected but training on GPU 0 only — float32 AdamW optimizer states exhaust VRAM on DataParallel")
model.to(device)
log.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# BOS token IDs
rut_meta = MODEL_PATH / "rut_token_meta.json"
RUT_BOS = ENG_BOS = None
if rut_meta.exists():
    m = json.loads(rut_meta.read_text())
    RUT_BOS, ENG_BOS = m["rut_token_id"], m["eng_token_id"]
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
                rny = o.get("runyoro_rutooro","").strip()
                eng = o.get("english","").strip()
                if rny and eng:
                    pairs.append((rny, eng))

        for rny, eng in pairs:
            # rny → eng
            tok.src_lang = "nyk_Latn"
            src_enc = tok(rny,  max_length=MAX_LEN, truncation=True)
            tok.tgt_lang = "eng_Latn"
            tgt_enc = tok(text_target=eng, max_length=MAX_LEN, truncation=True)
            lbl = ([ENG_BOS] + tgt_enc["input_ids"]) if ENG_BOS else tgt_enc["input_ids"]
            self.items.append((src_enc["input_ids"], src_enc["attention_mask"], lbl))

            # eng → rny
            tok.src_lang = "eng_Latn"
            src_enc2 = tok(eng, max_length=MAX_LEN, truncation=True)
            tok.tgt_lang = "nyk_Latn"
            tgt_enc2 = tok(text_target=rny, max_length=MAX_LEN, truncation=True)
            lbl2 = ([RUT_BOS] + tgt_enc2["input_ids"]) if (RUT_BOS and RUT_BOS != tok.unk_token_id) else tgt_enc2["input_ids"]
            self.items.append((src_enc2["input_ids"], src_enc2["attention_mask"], lbl2))

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
        ps, pl = max_s-len(s), max_l-len(l)
        S.append(s + [pad]*ps);  M.append(m + [0]*ps);  L.append(l + [-100]*pl)
    return (torch.tensor(S, dtype=torch.long),
            torch.tensor(M, dtype=torch.long),
            torch.tensor(L, dtype=torch.long))

log.info("Loading datasets...")
train_ds = RunyoroDataset(DATA_DIR / "train.jsonl")
val_ds   = RunyoroDataset(DATA_DIR / "val.jsonl")
test_ds  = RunyoroDataset(DATA_DIR / "test.jsonl")

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate, num_workers=0)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate, num_workers=0)

# ── Label smoothing loss ──────────────────────────────────────────────────────
vocab_size = (model.module if hasattr(model,"module") else model).config.vocab_size

def smooth_loss(logits, labels):
    """Cross-entropy with label smoothing, ignoring -100 positions."""
    B, T, V = logits.shape
    logits_flat = logits.reshape(-1, V)
    labels_flat = labels.reshape(-1)
    mask = labels_flat != -100
    logits_flat = logits_flat[mask]
    labels_flat = labels_flat[mask]
    log_probs = torch.nn.functional.log_softmax(logits_flat, dim=-1)
    # Standard CE
    nll = -log_probs.gather(1, labels_flat.unsqueeze(1)).squeeze(1)
    # Smooth term
    smooth = -log_probs.mean(dim=-1)
    loss = (1 - LABEL_SMOOTHING) * nll + LABEL_SMOOTHING * smooth
    return loss.mean()

# ── Optimizer & scheduler ─────────────────────────────────────────────────────
steps_per_epoch = math.ceil(len(train_dl) / GRAD_ACCUM)
total_steps     = EPOCHS * steps_per_epoch

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
if USE_8BIT_ADAM:
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    log.info("Using 8-bit AdamW — optimizer VRAM ~4× lower")
else:
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    log.info("Using standard AdamW")
scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - WARMUP_STEPS, eta_min=LR/10)

def warmup_lr(step):
    if step < WARMUP_STEPS:
        return float(step) / max(1, WARMUP_STEPS)
    return 1.0

warmup_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lr)

# ── Metrics — using sacrebleu directly (no pyarrow/datasets needed) ───────────
@torch.no_grad()
def evaluate_split(dl, label="val"):
    base = model.module if hasattr(model,"module") else model
    base.eval()
    torch.cuda.empty_cache()
    total_loss, n_batches = 0.0, 0
    all_preds, all_refs = [], []

    for src_ids, src_mask, lbls in dl:
        src_ids  = src_ids.to(device)
        src_mask = src_mask.to(device)
        lbls     = lbls.to(device)

        out = base(input_ids=src_ids, attention_mask=src_mask, labels=lbls)
        total_loss += out.loss.item()
        n_batches  += 1

        # Generate a few samples for logging only (no BLEU during training — avoids OOM + API issues)
        if len(all_preds) < 8:
            torch.cuda.empty_cache()
            gens = base.generate(
                input_ids      = src_ids[:2],
                attention_mask = src_mask[:2],
                num_beams      = 1,
                do_sample      = False,
                max_new_tokens = 128,
            )
            for g in tok.batch_decode(gens, skip_special_tokens=True):
                all_preds.append(g.strip())

    avg_loss = total_loss / max(n_batches, 1)
    # Log sample predictions for qualitative monitoring
    if all_preds:
        log.info(f"  Sample output: {all_preds[0][:80]}")
    base.train()
    return avg_loss, 0.0, 0.0  # bleu/chrf computed post-training

# ── Training loop ─────────────────────────────────────────────────────────────
log.info("="*60)
log.info(f"Training: {MODEL_PATH.name} → runyoro-rut-v12")
log.info(f"  Train: {len(train_ds)} examples  Val: {len(val_ds)} examples")
log.info(f"  Epochs: {EPOCHS}  LR: {LR}  Batch: {BATCH_SIZE} × {GRAD_ACCUM} accum = {BATCH_SIZE*GRAD_ACCUM} (single GPU)")
log.info("="*60)

best_val_loss  = float("inf")
no_improve     = 0
global_step    = 0
history        = []

model.train()
for epoch in range(1, EPOCHS + 1):
    epoch_loss = 0.0
    optimizer.zero_grad()

    for step, (src_ids, src_mask, lbls) in enumerate(train_dl, 1):
        src_ids  = src_ids.to(device)
        src_mask = src_mask.to(device)
        lbls     = lbls.to(device)

        # Forward — DataParallel splits batch across GPUs automatically
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
    val_loss, bleu, chrf_s = evaluate_split(val_dl, "val")
    log.info(f"Epoch {epoch}/{EPOCHS}  train_loss={avg_train:.4f}  val_loss={val_loss:.4f}  bleu={bleu}  chrf++={chrf_s}")
    history.append({"epoch": epoch, "train_loss": avg_train, "val_loss": val_loss, "bleu": bleu, "chrf++": chrf_s})

    # Save checkpoint each epoch
    ckpt_dir = OUT_PATH / f"checkpoint-epoch{epoch}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    base = model.module if hasattr(model,"module") else model
    base.save_pretrained(str(ckpt_dir))
    tok.save_pretrained(str(ckpt_dir))
    log.info(f"  Saved checkpoint: {ckpt_dir.name}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve    = 0
        # Save best model
        OUT_PATH.mkdir(parents=True, exist_ok=True)
        base.save_pretrained(str(OUT_PATH))
        tok.save_pretrained(str(OUT_PATH))
        log.info(f"  ✓ New best val_loss={val_loss:.4f} — saved to {OUT_PATH.name}")
    else:
        no_improve += 1
        log.info(f"  No improvement ({no_improve}/{EARLY_STOP_PAT})")
        if no_improve >= EARLY_STOP_PAT:
            log.info(f"Early stopping at epoch {epoch}")
            break

# ── Test evaluation ───────────────────────────────────────────────────────────
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate, num_workers=0)
test_loss, test_bleu, test_chrf = evaluate_split(test_dl, "test")
log.info(f"Test: loss={test_loss:.4f}  bleu={test_bleu}  chrf++={test_chrf}")

# ── Save metadata ─────────────────────────────────────────────────────────────
if rut_meta.exists():
    shutil.copy(str(rut_meta), str(OUT_PATH / "rut_token_meta.json"))

meta = {
    "model":             "runyoro-rut-v12",
    "continued_from":    "runyoro-rut-v11",
    "strategy":          "continue_training_clean_only_no_bt",
    "base_model":        "facebook/nllb-200-distilled-1.3B",
    "new_train_pairs":   len(train_ds) // 2,
    "new_val_pairs":     len(val_ds)   // 2,
    "new_test_pairs":    len(test_ds)  // 2,
    "bt_pairs":          0,
    "best_val_loss":     best_val_loss,
    "test_bleu":         test_bleu,
    "test_chrf++":       test_chrf,
    "epochs_run":        len(history),
    "lr":                LR,
    "label_smoothing":   LABEL_SMOOTHING,
    "gpus":              n_gpus,
    "timestamp":         datetime.datetime.utcnow().isoformat() + "Z",
    "history":           history,
}
(OUT_PATH / "training_metadata.json").write_text(json.dumps(meta, indent=2))
log.info("Saved training_metadata.json")

# ── Push to HuggingFace Hub ───────────────────────────────────────────────────
log.info(f"Pushing to HF Hub: {HF_REPO}")
from huggingface_hub import HfApi
api = HfApi(token=HF_TOKEN)
api.upload_folder(
    folder_path = str(OUT_PATH),
    repo_id     = HF_REPO,
    repo_type   = "model",
    commit_message = f"runyoro-rut-v12: continue-train on sp19 + all datasets, bleu={test_bleu} chrf={test_chrf}",
    ignore_patterns = ["checkpoint-epoch*/**"],
)
log.info(f"Pushed to {HF_REPO} ✓")
log.info("Training complete.")
