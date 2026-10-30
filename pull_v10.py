"""
Pull kathay/runyoro-nmt → local runyoro-rut-v10
Then delete local runyoro-rut-v7
"""
import os, pathlib, shutil
from huggingface_hub import snapshot_download

HF_TOKEN = "os.environ.get("HF_READ_TOKEN","")"
CKPT     = pathlib.Path(r"C:\Users\keith\Desktop\projects\runyoro_nmt\models\checkpoints")
V10_DEST = CKPT / "runyoro-rut-v10"
V7_PATH  = CKPT / "runyoro-rut-v7"
V8_PATH  = CKPT / "runyoro-rut-v8"

# ── 1. Pull kathay/runyoro-nmt as v10 ─────────────────────────────────────────
print(f"Pulling kathay/runyoro-nmt → {V10_DEST}")
print("(skipping intermediate checkpoints — root weights only)\n")

path = snapshot_download(
    repo_id         = "kathay/runyoro-nmt",
    local_dir       = str(V10_DEST),
    token           = HF_TOKEN,
    ignore_patterns = ["checkpoint-*/model*.safetensors",
                       "checkpoint-*/training_args.bin"],
)

print(f"\nDownload complete.")
for root, dirs, files in os.walk(path):
    rel = os.path.relpath(root, path)
    if rel.startswith("checkpoint-") and "00001" in rel:
        continue   # skip printing shard paths inside checkpoints
    for f in files:
        fp  = os.path.join(root, f)
        mb  = os.path.getsize(fp) / 1e6
        print(f"  {os.path.relpath(fp, path)}  ({mb:.1f} MB)")

# ── 2. Delete v7 ──────────────────────────────────────────────────────────────
for old in [V7_PATH]:
    if old.exists():
        print(f"\nDeleting {old} ...")
        shutil.rmtree(old)
        print(f"  Deleted.")
    else:
        print(f"\n{old} — not found (already gone)")

print("\nDone. runyoro-rut-v10 is the active checkpoint.")
