"""
Report status of all local model checkpoints + kathay HF repo models.
"""
import json
import os
import urllib.request

HF_TOKEN = "os.environ.get("HF_READ_TOKEN","")"
CKPT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "runyoro_nmt", "models", "checkpoints"))

# ── 1. Scan local checkpoints ─────────────────────────────────────────────────
print("=" * 65)
print("LOCAL CHECKPOINTS")
print(f"Path: {CKPT_ROOT}")
print("=" * 65)

if not os.path.isdir(CKPT_ROOT):
    print("  [directory not found]")
else:
    dirs = sorted([d for d in os.listdir(CKPT_ROOT) if os.path.isdir(os.path.join(CKPT_ROOT, d))])
    if not dirs:
        print("  [no checkpoint directories found]")
    for ckpt in dirs:
        ckpt_path = os.path.join(CKPT_ROOT, ckpt)
        files = os.listdir(ckpt_path)

        # Size of model weights
        weight_files = [f for f in files if f.endswith(".safetensors") and "index" not in f]
        total_mb = sum(os.path.getsize(os.path.join(ckpt_path, f)) for f in weight_files) / 1_048_576

        # Read training_metadata.json if present
        meta_path = os.path.join(ckpt_path, "training_metadata.json")
        meta = {}
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)

        # Read trainer_state.json if present (has epoch/steps/eval loss)
        state_path = os.path.join(ckpt_path, "trainer_state.json")
        state = {}
        if os.path.isfile(state_path):
            with open(state_path) as f:
                state = json.load(f)

        # Read config.json
        config_path = os.path.join(ckpt_path, "config.json")
        config = {}
        if os.path.isfile(config_path):
            with open(config_path) as f:
                config = json.load(f)

        # rut_token_meta.json
        rut_meta_path = os.path.join(ckpt_path, "rut_token_meta.json")
        rut_meta = {}
        if os.path.isfile(rut_meta_path):
            with open(rut_meta_path) as f:
                rut_meta = json.load(f)

        print(f"\n  [{ckpt}]")
        print(f"    Weight files : {len(weight_files)} shard(s),  {total_mb:.0f} MB total")
        print(f"    Has config   : {'yes' if config else 'no'}")
        print(f"    rut_token_meta: {'yes — rut_id=' + str(rut_meta.get('rut_token_id')) + ' eng_id=' + str(rut_meta.get('eng_token_id')) if rut_meta else 'no (clean-v4 / no forced BOS)'}")

        if meta:
            train_p  = meta.get("train_pairs", "?")
            val_p    = meta.get("val_pairs", "?")
            test_p   = meta.get("test_pairs", "?")
            bt_p     = meta.get("bt_pairs", 0)
            total_p  = (train_p if isinstance(train_p, int) else 0) + \
                       (val_p   if isinstance(val_p,   int) else 0) + \
                       (test_p  if isinstance(test_p,  int) else 0)
            print(f"    Strategy     : {meta.get('strategy', '?')}")
            print(f"    Base model   : {meta.get('base_model', '?')}")
            print(f"    Pairs        : train={train_p}  val={val_p}  test={test_p}  total_clean={total_p}  bt={bt_p}")
            print(f"    Epochs       : base={meta.get('base_epochs','?')}  bt={meta.get('bt_epochs','?')}")
            print(f"    LR           : {meta.get('lr','?')}")
        else:
            print(f"    training_metadata.json: not found")

        if state:
            best = state.get("best_metric", "?")
            best_step = state.get("best_model_checkpoint", "?")
            epoch = state.get("epoch", "?")
            log = state.get("log_history", [])
            last_eval = next((e for e in reversed(log) if "eval_loss" in e), {})
            print(f"    Trainer epoch: {epoch}")
            print(f"    Best metric  : {best}  @ {os.path.basename(str(best_step))}")
            if last_eval:
                print(f"    Last eval    : loss={last_eval.get('eval_loss','?'):.4f}  bleu={last_eval.get('eval_bleu','?')}")

# ── 2. HF repo model card / README for training info ─────────────────────────
print("\n")
print("=" * 65)
print("KATHAY HF REPO — kathay/runyoro-nmt")
print("=" * 65)

req = urllib.request.Request(
    "https://huggingface.co/kathay/runyoro-nmt/resolve/main/training_metadata.json",
    headers={"Authorization": f"Bearer {HF_TOKEN}"}
)
try:
    with urllib.request.urlopen(req, timeout=15) as r:
        hf_meta = json.loads(r.read())
    print(f"  Strategy  : {hf_meta.get('strategy','?')}")
    print(f"  Base model: {hf_meta.get('base_model','?')}")
    train_p = hf_meta.get('train_pairs', '?')
    val_p   = hf_meta.get('val_pairs', '?')
    test_p  = hf_meta.get('test_pairs', '?')
    bt_p    = hf_meta.get('bt_pairs', 0)
    total_p = (train_p if isinstance(train_p, int) else 0) + \
              (val_p   if isinstance(val_p,   int) else 0) + \
              (test_p  if isinstance(test_p,  int) else 0)
    print(f"  Pairs     : train={train_p}  val={val_p}  test={test_p}  total_clean={total_p}  bt={bt_p}")
    print(f"  Epochs    : base={hf_meta.get('base_epochs','?')}  bt={hf_meta.get('bt_epochs','?')}")
    print(f"  LR        : {hf_meta.get('lr','?')}")
except Exception as e:
    print(f"  Could not fetch training_metadata.json: {e}")

# Also count models in account
req2 = urllib.request.Request(
    "https://huggingface.co/api/models?author=kathay&limit=100",
    headers={"Authorization": f"Bearer {HF_TOKEN}"}
)
with urllib.request.urlopen(req2, timeout=15) as r:
    models = json.loads(r.read())
print(f"\n  Total models in kathay account: {len(models)}")
for m in models:
    print(f"    - {m['id']}  (last modified: {m.get('lastModified','')})")
