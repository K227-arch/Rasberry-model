"""
Update keithtwesigye/runyoro-translator-api Space to serve kathay/runyoro-nmt (v12)
then restart it.
"""
import json, urllib.request
from huggingface_hub import HfApi

READ  = "os.environ.get("HF_READ_TOKEN","")"
WRITE = "os.environ.get("HF_WRITE_TOKEN","")"
SPACE = "keithtwesigye/runyoro-translator-api"

api = HfApi(token=WRITE)

# Fetch current app.py
req = urllib.request.Request(
    f"https://huggingface.co/spaces/{SPACE}/resolve/main/app.py",
    headers={"Authorization": f"Bearer {READ}"}
)
with urllib.request.urlopen(req, timeout=15) as r:
    content = r.read().decode("utf-8")

# Patch
patched = content.replace(
    'os.getenv("MODEL_ID", "keithtwesigye/runyoro-nmt")',
    'os.getenv("MODEL_ID", "kathay/runyoro-nmt")'
).replace(
    "runyoro-rut-v4", "runyoro-rut-v12"
).replace(
    "Loads keithtwesigye/runyoro-nmt from the Hub.",
    "Loads kathay/runyoro-nmt (runyoro-rut-v12) from the Hub."
)

if "kathay/runyoro-nmt" in patched:
    print("Patch applied — uploading app.py ...")
    api.upload_file(
        path_or_fileobj = patched.encode("utf-8"),
        path_in_repo    = "app.py",
        repo_id         = SPACE,
        repo_type       = "space",
        commit_message  = "fix: MODEL_ID → kathay/runyoro-nmt (runyoro-rut-v12)",
    )
    print("Uploaded.")
else:
    print("Patch failed — MODEL_ID lines found:")
    for l in content.splitlines():
        if "MODEL_ID" in l:
            print(" ", l)

# Restart
print("Restarting space ...")
req2 = urllib.request.Request(
    f"https://huggingface.co/api/spaces/{SPACE}/restart",
    data=b"{}",
    headers={"Authorization": f"Bearer {WRITE}", "Content-Type": "application/json"},
    method="POST"
)
try:
    with urllib.request.urlopen(req2, timeout=30) as r:
        result = json.loads(r.read())
        print("Stage:", result.get("stage"))
except urllib.error.HTTPError as e:
    print(f"Restart HTTP {e.code}:", e.read().decode())
