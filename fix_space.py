"""
Check and delete all Space variables/secrets via HF API,
then restart.
"""
import json, urllib.request
from huggingface_hub import HfApi

WRITE = "os.environ.get("HF_WRITE_TOKEN","")"
READ  = "os.environ.get("HF_READ_TOKEN","")"
SPACE = "keithtwesigye/runyoro-translator-api"

api = HfApi(token=WRITE)

# 1. Get all variables defined in Space settings
req = urllib.request.Request(
    f"https://huggingface.co/api/spaces/{SPACE}/variables",
    headers={"Authorization": f"Bearer {WRITE}"}
)
try:
    with urllib.request.urlopen(req, timeout=15) as r:
        variables = json.loads(r.read())
    print("Space variables:", variables)
except Exception as e:
    print("Variables fetch error:", e)

# 2. Delete each variable
if isinstance(variables, dict):
    for key in variables:
        print(f"Deleting variable: {key}")
        del_req = urllib.request.Request(
            f"https://huggingface.co/api/spaces/{SPACE}/variables",
            data=json.dumps({"key": key}).encode(),
            headers={"Authorization": f"Bearer {WRITE}", "Content-Type": "application/json"},
            method="DELETE"
        )
        try:
            with urllib.request.urlopen(del_req, timeout=10) as r:
                print(f"  Deleted: {r.status}")
        except Exception as e:
            print(f"  Error: {e}")
elif isinstance(variables, list):
    for v in variables:
        key = v.get("key") if isinstance(v, dict) else v
        print(f"Deleting variable: {key}")
        del_req = urllib.request.Request(
            f"https://huggingface.co/api/spaces/{SPACE}/variables",
            data=json.dumps({"key": key}).encode(),
            headers={"Authorization": f"Bearer {WRITE}", "Content-Type": "application/json"},
            method="DELETE"
        )
        try:
            with urllib.request.urlopen(del_req, timeout=10) as r:
                print(f"  Deleted: {r.status}")
        except Exception as e:
            print(f"  Error: {e}")

# 3. Restart
print("\nRestarting ...")
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
        print("Error:", result.get("errorMessage", "none"))
except urllib.error.HTTPError as e:
    print(f"HTTP {e.code}:", e.read().decode())
