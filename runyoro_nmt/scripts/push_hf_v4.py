#!/usr/bin/env python3
"""push runyoro-nmt-v4 to HuggingFace (final model only, no checkpoints)"""
import os
os.environ["USE_TF"] = "0"
from pathlib import Path
from huggingface_hub import HfApi, create_repo

HF_TOKEN = "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"
REPO_ID = "kathay/runyoro-nmt-v4"
MODEL_DIR = Path(__file__).parent.parent / "models" / "checkpoints" / "runyoro-nmt-v4"

api = HfApi(token=HF_TOKEN)
create_repo(REPO_ID, token=HF_TOKEN, exist_ok=True)
print(f"Uploading {MODEL_DIR} -> {REPO_ID} ...")

api.upload_folder(
    folder_path=str(MODEL_DIR),
    repo_id=REPO_ID,
    token=HF_TOKEN,
    ignore_patterns=["checkpoint-*", "checkpoint-*/**"],
    commit_message="Upload runyoro-nmt-v4 (FP32, 679 pairs, 20 epochs, no lang codes, loss=0.063)",
)
print(f"Done! https://huggingface.co/{REPO_ID}")
