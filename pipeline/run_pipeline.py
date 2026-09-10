"""
Master pipeline runner — executes all stages in order.
Run from the project root:  python pipeline/run_pipeline.py
"""
import os, sys, subprocess, time

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
STAGES = [
    ("Stage 1 — Extract",          "stage1_extract.py"),
    ("Stage 2 — Validate",         "stage2_validate.py"),
    ("Stage 3 — Clean",            "stage3_clean.py"),
    ("Stage 4 — Merge & Split",    "stage4_merge_split.py"),
    ("Stage 5 — Continue-Train",   "stage5_train.py"),
]

def run_stage(name, script):
    path = os.path.join(PIPELINE_DIR, script)
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    start = time.time()
    result = subprocess.run([sys.executable, path], check=False)
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"\n[FAILED] {name} exited with code {result.returncode}")
        sys.exit(result.returncode)
    print(f"\n[OK] {name} completed in {elapsed:.1f}s")

if __name__ == "__main__":
    # Allow running specific stage: python run_pipeline.py 3
    start_from = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    for i, (name, script) in enumerate(STAGES, start=1):
        if i >= start_from:
            run_stage(name, script)
    print(f"\n{'='*60}")
    print("  All pipeline stages complete.")
    print(f"{'='*60}")
