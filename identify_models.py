"""
Identify runyoro-rut-v7, runyoro-rut-v4, and runyoro-nmt
across both HF accounts (kathay + keithtwesigye) and local checkpoints.
"""
import urllib.request, json, os

TOKENS = {
    "kathay":          "os.environ.get("HF_TOKEN","")",
    "keithtwesigye":   "os.environ.get("HF_TOKEN","")",
}

def hf_get(url, token):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read())

def hf_text(url, token):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return r.read().decode()

def get_meta(repo_id, token):
    try:
        return json.loads(hf_text(f"https://huggingface.co/{repo_id}/resolve/main/training_metadata.json", token))
    except:
        return {}

# ── 1. Check both HF accounts ─────────────────────────────────────────────────
token = TOKENS["kathay"]  # read token works for both public repos
for account in ["kathay", "keithtwesigye"]:
    models = hf_get(f"https://huggingface.co/api/models?author={account}&limit=100", token)
    print(f"\n{'='*60}")
    print(f"HF ACCOUNT: {account}  ({len(models)} model(s))")
    print(f"{'='*60}")
    for m in sorted(models, key=lambda x: x.get("createdAt","")):
        repo_id = m["id"]
        meta = get_meta(repo_id, token)
        info = hf_get(f"https://huggingface.co/api/models/{repo_id}", token)
        siblings = info.get("siblings", [])
        ckpts = sorted(set(
            s["rfilename"].split("/")[0]
            for s in siblings if s["rfilename"].startswith("checkpoint-")
        ))
        params = info.get("safetensors", {}).get("total", "?")
        storage_gb = info.get("usedStorage", 0) / 1e9

        print(f"\n  {repo_id}")
        print(f"    Created      : {m.get('createdAt','?')}")
        print(f"    Last modified: {m.get('lastModified', info.get('lastModified','?'))}")
        print(f"    Parameters   : {params:,}" if isinstance(params, int) else f"    Parameters   : {params}")
        print(f"    Storage      : {storage_gb:.2f} GB")
        print(f"    Checkpoints  : {ckpts if ckpts else 'none'}")
        if meta:
            print(f"    Internal name: {meta.get('model','?')}")
            print(f"    Strategy     : {meta.get('strategy','?')}")
            print(f"    Base model   : {meta.get('base_model','?')}")
            clean = (meta.get('train_pairs',0) or 0) + (meta.get('val_pairs',0) or 0) + (meta.get('test_pairs',0) or 0)
            bt    = meta.get('bt_pairs', 0) or 0
            print(f"    Clean pairs  : {clean}  (train={meta.get('train_pairs')} val={meta.get('val_pairs')} test={meta.get('test_pairs')})")
            print(f"    BT pairs     : {bt}")
            print(f"    Total pairs  : {clean + bt}")
            print(f"    Epochs       : base={meta.get('base_epochs','?')}  bt={meta.get('bt_epochs','?')}")
            print(f"    LR           : {meta.get('lr','?')}")
        else:
            print(f"    training_metadata.json: not found")

# ── 2. Local checkpoints ──────────────────────────────────────────────────────
CKPT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "runyoro_nmt", "models", "checkpoints"))
print(f"\n{'='*60}")
print(f"LOCAL CHECKPOINTS  ({CKPT_ROOT})")
print(f"{'='*60}")

if not os.path.isdir(CKPT_ROOT):
    print("  [directory not found]")
else:
    dirs = sorted(d for d in os.listdir(CKPT_ROOT) if os.path.isdir(os.path.join(CKPT_ROOT, d)))
    print(f"  Found {len(dirs)} checkpoint(s): {dirs}")
    for ckpt in dirs:
        path = os.path.join(CKPT_ROOT, ckpt)
        weight_files = [f for f in os.listdir(path) if f.endswith(".safetensors") and "index" not in f]
        size_gb = sum(os.path.getsize(os.path.join(path,f)) for f in weight_files) / 1e9
        meta_path = os.path.join(path, "training_metadata.json")
        rut_path  = os.path.join(path, "rut_token_meta.json")
        meta = json.load(open(meta_path)) if os.path.isfile(meta_path) else {}
        rut  = json.load(open(rut_path))  if os.path.isfile(rut_path)  else {}

        print(f"\n  [{ckpt}]")
        print(f"    Size         : {size_gb:.2f} GB  ({len(weight_files)} shard(s))")
        print(f"    rut_token_meta: {'YES — rut_id=' + str(rut.get('rut_token_id')) + ' eng_id=' + str(rut.get('eng_token_id')) if rut else 'NO (no forced BOS)'}")
        if meta:
            print(f"    Internal name: {meta.get('model','?')}")
            print(f"    Strategy     : {meta.get('strategy','?')}")
            clean = (meta.get('train_pairs',0) or 0) + (meta.get('val_pairs',0) or 0) + (meta.get('test_pairs',0) or 0)
            bt    = meta.get('bt_pairs', 0) or 0
            print(f"    Clean pairs  : {clean}  BT pairs: {bt}  Total: {clean+bt}")
            print(f"    Epochs       : base={meta.get('base_epochs','?')}  bt={meta.get('bt_epochs','?')}")
            print(f"    LR           : {meta.get('lr','?')}")
