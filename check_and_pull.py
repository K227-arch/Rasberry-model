"""
List all models in kathay HF account, then download the latest one.
"""
import json
import os
import urllib.request

HF_TOKEN = "os.environ.get("HF_READ_TOKEN","")"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

def hf_get(url):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())

# ── 1. List all models ────────────────────────────────────────────────────────
models = hf_get("https://huggingface.co/api/models?author=kathay&limit=100")
print(f"\n=== kathay HF models ({len(models)} total) ===")
for m in sorted(models, key=lambda x: x.get("createdAt", "")):
    print(f"  {m['id']}  |  created: {m.get('createdAt','')}  |  downloads: {m.get('downloads',0)}")

# ── 2. Identify the newest model ──────────────────────────────────────────────
newest = sorted(models, key=lambda x: x.get("createdAt", ""), reverse=True)[0]
repo_id = newest["id"]
print(f"\nNewest model: {repo_id}")

# ── 3. List files in the repo ─────────────────────────────────────────────────
repo_info = hf_get(f"https://huggingface.co/api/models/{repo_id}")
siblings = repo_info.get("siblings", [])
print(f"Files in repo ({len(siblings)}):")
for f in siblings:
    size = f.get("size", "?")
    print(f"  {f['rfilename']}  ({size} bytes)" if size != "?" else f"  {f['rfilename']}")

# print full repo info for debugging
import pprint
print("\nFull repo metadata:")
pprint.pprint({k: v for k, v in repo_info.items() if k not in ("siblings",)})
