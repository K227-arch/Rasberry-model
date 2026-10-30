"""
Create a fresh clean Space: keithtwesigye/runyoro-nmt-space
Uploads only: app.py, Dockerfile, requirements.txt, README.md
Points to kathay/runyoro-nmt (v13)
"""
import json, urllib.request
from huggingface_hub import HfApi

WRITE     = "os.environ.get("HF_WRITE_TOKEN","")"
NEW_SPACE = "keithtwesigye/runyoro-nmt-space"
HF_DIR    = r"C:\Users\keith\Desktop\projects\Rasberry-model\hf-space"

api = HfApi(token=WRITE)

# 1. Create the space
print(f"Creating {NEW_SPACE} ...")
try:
    api.create_repo(
        repo_id   = NEW_SPACE,
        repo_type = "space",
        space_sdk = "docker",
        private   = False,
        exist_ok  = True,
    )
    print("  Created.")
except Exception as e:
    print(f"  Error: {e}")
    exit(1)

# 2. Upload the 4 files — nothing else, no env files, no secrets collision
import os
files = ["app.py", "Dockerfile", "requirements.txt", "README.md"]
for fname in files:
    fpath = os.path.join(HF_DIR, fname)
    if not os.path.isfile(fpath):
        print(f"  Skipping {fname} (not found)")
        continue
    print(f"  Uploading {fname} ...")
    api.upload_file(
        path_or_fileobj = fpath,
        path_in_repo    = fname,
        repo_id         = NEW_SPACE,
        repo_type       = "space",
        commit_message  = f"deploy: {fname} (runyoro-rut-v13)",
    )
    print(f"    OK")

print(f"\nSpace URL : https://huggingface.co/spaces/{NEW_SPACE}")
print(f"API URL   : https://keithtwesigye-runyoro-nmt-space.hf.space")
print(f"Model     : kathay/runyoro-nmt (runyoro-rut-v13)")
