"""
List all models + datasets in kathay HF account with full metadata,
and determine which checkpoint was last trained.
"""
import urllib.request, json

HF_TOKEN = "os.environ.get("HF_READ_TOKEN","")"

def hf_get(url):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {HF_TOKEN}"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read())

def hf_text(url):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {HF_TOKEN}"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return r.read().decode()

# ── Models ────────────────────────────────────────────────────────────────────
models = hf_get("https://huggingface.co/api/models?author=kathay&limit=100")
print(f"{'='*60}")
print(f"KATHAY HF ACCOUNT — {len(models)} model(s)")
print(f"{'='*60}")

for m in sorted(models, key=lambda x: x.get("lastModified", ""), reverse=True):
    repo_id = m["id"]
    print(f"\n  Repo       : {repo_id}")
    print(f"  Created    : {m.get('createdAt','?')}")
    print(f"  Modified   : {m.get('lastModified','?')}")
    print(f"  Downloads  : {m.get('downloads',0)}")
    print(f"  Parameters : {m.get('safetensors',{}).get('total','?'):,}" if m.get('safetensors') else "  Parameters : ?")
    print(f"  Storage    : {m.get('usedStorage',0)/1e9:.2f} GB")

    # Get full repo info for checkpoint listing
    info = hf_get(f"https://huggingface.co/api/models/{repo_id}")
    siblings = info.get("siblings", [])

    # Find all checkpoint directories
    ckpt_dirs = sorted(set(
        s["rfilename"].split("/")[0]
        for s in siblings
        if s["rfilename"].startswith("checkpoint-")
    ))
    print(f"  Checkpoints: {ckpt_dirs if ckpt_dirs else 'none (root weights only)'}")

    # Fetch trainer_state.json from each checkpoint
    for ckpt in ckpt_dirs:
        try:
            state = json.loads(hf_text(
                f"https://huggingface.co/datasets/kathay/runyoro-rutooro-en-parallel/resolve/main/pipeline_reports/pipeline_report.md"
            ))
        except:
            pass
        try:
            state_url = f"https://huggingface.co/{repo_id}/resolve/main/{ckpt}/trainer_state.json"
            state = json.loads(hf_text(state_url))
            epoch   = state.get("epoch", "?")
            best    = state.get("best_metric", "?")
            logs    = state.get("log_history", [])
            last_eval = next((e for e in reversed(logs) if "eval_loss" in e), {})
            print(f"    [{ckpt}] epoch={epoch}  best_metric={best}  "
                  f"eval_loss={last_eval.get('eval_loss','?')}  "
                  f"eval_bleu={last_eval.get('eval_bleu','?')}")
        except Exception as e:
            print(f"    [{ckpt}] trainer_state.json unavailable: {e}")

    # Root trainer_state (final model)
    try:
        state = json.loads(hf_text(f"https://huggingface.co/{repo_id}/resolve/main/trainer_state.json"))
        epoch   = state.get("epoch", "?")
        best    = state.get("best_metric", "?")
        logs    = state.get("log_history", [])
        last_eval = next((e for e in reversed(logs) if "eval_loss" in e), {})
        last_log  = logs[-1] if logs else {}
        print(f"\n  ROOT (final weights):")
        print(f"    Epoch      : {epoch}")
        print(f"    Best metric: {best}")
        print(f"    Last eval  : loss={last_eval.get('eval_loss','?')}  bleu={last_eval.get('eval_bleu','?')}")
        print(f"    Last log   : {last_log}")
    except Exception as e:
        print(f"\n  ROOT trainer_state.json: not present ({e})")

    # training_metadata.json
    try:
        meta = json.loads(hf_text(f"https://huggingface.co/{repo_id}/resolve/main/training_metadata.json"))
        print(f"\n  Training metadata:")
        for k, v in meta.items():
            print(f"    {k}: {v}")
    except:
        pass

# ── Datasets ──────────────────────────────────────────────────────────────────
datasets = hf_get("https://huggingface.co/api/datasets?author=kathay&limit=100")
print(f"\n{'='*60}")
print(f"KATHAY HF ACCOUNT — {len(datasets)} dataset(s)")
print(f"{'='*60}")
for d in datasets:
    print(f"  {d['id']}  |  modified: {d.get('lastModified','?')}  |  downloads: {d.get('downloads',0)}")
