"""
Restore the runyoro-nmt-v1 checkpoint from Hugging Face Hub
(the good BLEU=18.77 version), overwriting the degraded local checkpoint.
"""
import os, shutil
from pathlib import Path
from huggingface_hub import snapshot_download

HF_TOKEN   = "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"
MODEL_ID   = "kathay/runyoro-nmt-v1"
LOCAL_DIR  = Path(__file__).parent.parent / "models" / "checkpoints" / "runyoro-nmt-v1"

print(f"Downloading {MODEL_ID} -> {LOCAL_DIR}")
snapshot_download(
    repo_id=MODEL_ID,
    token=HF_TOKEN,
    local_dir=str(LOCAL_DIR),
    ignore_patterns=["*.msgpack", "flax_model*", "tf_model*"],
)
print(f"Restored: {LOCAL_DIR}")
