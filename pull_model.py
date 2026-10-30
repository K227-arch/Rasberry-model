"""
Pull kathay/runyoro-nmt (root final weights only) from HuggingFace
into runyoro_nmt/models/checkpoints/runyoro-rut-v7
"""
import os
from huggingface_hub import snapshot_download

HF_TOKEN = "os.environ.get("HF_READ_TOKEN","")"
REPO_ID  = "kathay/runyoro-nmt"

# Destination: one level up from this workspace, matching model_server.py's expected path
BASE     = os.path.dirname(os.path.abspath(__file__))
DEST     = os.path.join(BASE, "..", "runyoro_nmt", "models", "checkpoints", "runyoro-rut-v7")

print(f"Downloading {REPO_ID} → {os.path.abspath(DEST)}")
print("(Root model files only — skipping training checkpoints)\n")

path = snapshot_download(
    repo_id       = REPO_ID,
    local_dir     = DEST,
    token         = HF_TOKEN,
    # Skip the 5 intermediate training checkpoints — only want the final root weights
    ignore_patterns = [
        "checkpoint-15/*",
        "checkpoint-298/*",
        "checkpoint-384/*",
        "checkpoint-447/*",
        "checkpoint-576/*",
    ],
)

print(f"\nDownload complete: {path}")
print("Files downloaded:")
for root, dirs, files in os.walk(path):
    dirs[:] = [d for d in dirs if not d.startswith("checkpoint-")]
    for f in files:
        fpath = os.path.join(root, f)
        size_mb = os.path.getsize(fpath) / 1_048_576
        print(f"  {os.path.relpath(fpath, path)}  ({size_mb:.1f} MB)")
