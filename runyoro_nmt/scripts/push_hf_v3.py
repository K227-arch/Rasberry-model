#!/usr/bin/env python3
"""
push_hf_v3.py - Push runyoro-nmt-v3 to HuggingFace (final model only, no checkpoints)
"""
import os
os.environ["USE_TF"] = "0"

from pathlib import Path
from huggingface_hub import HfApi, create_repo

HF_TOKEN = "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"
REPO_ID = "kathay/runyoro-nmt-v3"
MODEL_DIR = Path(__file__).parent.parent / "models" / "checkpoints" / "runyoro-nmt-v3"

print(f"Model dir: {MODEL_DIR}")
print(f"Repo: {REPO_ID}")

# Create repo (or get existing)
api = HfApi(token=HF_TOKEN)
try:
    create_repo(REPO_ID, token=HF_TOKEN, exist_ok=True)
    print(f"Repo ready: {REPO_ID}")
except Exception as e:
    print(f"Repo creation note: {e}")

# Upload ONLY the final model files (ignore checkpoint-* folders)
print("Uploading final model (ignoring checkpoint-* folders)...")
api.upload_folder(
    folder_path=str(MODEL_DIR),
    repo_id=REPO_ID,
    token=HF_TOKEN,
    ignore_patterns=["checkpoint-*", "checkpoint-*/**"],
    commit_message="Upload runyoro-nmt-v3 (FP32, 292 pairs, 20 epochs, no lang codes)",
)
print(f"\nDone! Model uploaded to: https://huggingface.co/{REPO_ID}")
