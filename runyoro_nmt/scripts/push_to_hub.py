#!/usr/bin/env python3
"""
push_to_hub.py
==============
Pushes all runyoro-nmt-v1 artifacts to the kathay Hugging Face Hub:

  1.  Creates repos (model, dataset, space) if they don't exist
  2.  Uploads the parallel dataset
  3.  Uploads linguistic resources (TMX, TBX, glossary)
  4.  Uploads the Gradio Space app + requirements
  5.  Uploads a model README / model card (placeholder until training completes)
  6.  Prints all live URLs
"""

import os
import sys
import json
import csv
from pathlib import Path

HF_TOKEN = os.environ.get("HF_TOKEN", "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")

MODEL_ID   = "kathay/runyoro-nmt-v1"
DATASET_ID = "kathay/runyoro-rutooro-en-parallel"
SPACE_ID   = "kathay/runyoro-translator"

ROOT = Path(__file__).parent.parent

# ── helpers ──────────────────────────────────────────────────────────────────

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

# ── 1. login / API ────────────────────────────────────────────────────────────
section("1. Authenticating")
from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationDelete
api = HfApi(token=HF_TOKEN)
me = api.whoami()
print(f"  Logged in as: {me['name']}")

# ── 2. Create repos ──────────────────────────────────────────────────────────
section("2. Creating / verifying repos")

for repo_id, repo_type, sdk in [
    (MODEL_ID,   "model",   None),
    (DATASET_ID, "dataset", None),
    (SPACE_ID,   "space",   "gradio"),
]:
    try:
        kwargs = dict(repo_id=repo_id, repo_type=repo_type, token=HF_TOKEN, exist_ok=True)
        if sdk:
            kwargs["space_sdk"] = sdk
        api.create_repo(**kwargs)
        print(f"  OK  {repo_type}: {repo_id}")
    except Exception as e:
        print(f"  WARN  {repo_id}: {e}")

# ── 3. Upload dataset ─────────────────────────────────────────────────────────
section("3. Uploading parallel dataset")

train_tsv = ROOT / "data" / "processed" / "train.tsv"
val_tsv   = ROOT / "data" / "processed" / "val.tsv"
test_tsv  = ROOT / "data" / "processed" / "test.tsv"
all_tsv   = ROOT / "data" / "augmented" / "all_pairs.tsv"

def load_tsv(path):
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2:
                pairs.append((parts[0], parts[1]))
    return pairs

# Build a dataset card
dataset_card = """---
language:
  - nyk
  - en
license: cc-by-4.0
task_categories:
  - translation
pretty_name: Runyoro-Rutooro / English Parallel Corpus
tags:
  - runyoro-rutooro
  - english
  - parallel-corpus
  - low-resource
  - bantu
  - runyoro-nmt-v1
---

# Runyoro-Rutooro / English Parallel Corpus

Bidirectional parallel sentence pairs for Runyoro-Rutooro (a Bantu language spoken
in western Uganda) and English, used to train **runyoro-nmt-v1**.

## Data Sources
- Agricultural vocabulary (.xlsx)
- Augmented POS-tagged pairs (.xlsx)
- Alphabetical vocabulary documents (.docx) — letters F, J, T, U, V, W

## Processing
Extraction → validation → alignment check → cleaning → deduplication →
normalisation → augmentation (2×)

## Splits
| Split | Pairs |
|-------|-------|
| train | ~632  |
| validation | ~74 |
| test | ~38 |

## Language Notes
- **Runyoro-Rutooro**: Bantu (Niger-Congo), spoken in Bunyoro-Kitara & Tooro kingdoms
- **ISO 639-3**: nyk (Nyankore-Kiga, closest match)
- **Script**: Latin

## Related Model
[kathay/runyoro-nmt-v1](https://huggingface.co/kathay/runyoro-nmt-v1)
"""

# Write dataset card
(ROOT / "data" / "processed" / "README.md").write_text(dataset_card, encoding="utf-8")

# Convert TSVs to JSONL for easy HF dataset loading
def tsv_to_jsonl(tsv_path, jsonl_path):
    pairs = load_tsv(tsv_path)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rny, eng in pairs:
            f.write(json.dumps({"runyoro_rutooro": rny, "english": eng}, ensure_ascii=False) + "\n")
    return len(pairs)

ops = []

for split_name, tsv_path in [("train", train_tsv), ("validation", val_tsv), ("test", test_tsv)]:
    if tsv_path.exists():
        jsonl_path = tsv_path.with_suffix(".jsonl")
        n = tsv_to_jsonl(tsv_path, jsonl_path)
        print(f"  {split_name}: {n} pairs -> {jsonl_path.name}")
        ops.append(CommitOperationAdd(
            path_in_repo=f"data/{split_name}-00000-of-00001.jsonl",
            path_or_fileobj=str(jsonl_path),
        ))

# Also upload the raw cleaned TSV for full access
if all_tsv.exists():
    ops.append(CommitOperationAdd(
        path_in_repo="data/all_pairs.tsv",
        path_or_fileobj=str(all_tsv),
    ))

# Dataset card
ops.append(CommitOperationAdd(
    path_in_repo="README.md",
    path_or_fileobj=str(ROOT / "data" / "processed" / "README.md"),
))

# Dataset loading script
dataset_script = '''import datasets

_URLS = {
    "train": "data/train-00000-of-00001.jsonl",
    "validation": "data/validation-00000-of-00001.jsonl",
    "test": "data/test-00000-of-00001.jsonl",
}

class RunyoroEnglish(datasets.GeneratorBasedBuilder):
    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features({
                "runyoro_rutooro": datasets.Value("string"),
                "english": datasets.Value("string"),
            })
        )
    def _split_generators(self, dl_manager):
        downloaded = dl_manager.download(_URLS)
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded["validation"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": downloaded["test"]}),
        ]
    def _generate_examples(self, filepath):
        import json
        with open(filepath, encoding="utf-8") as f:
            for i, line in enumerate(f):
                yield i, json.loads(line)
'''
script_path = ROOT / "data" / "processed" / "runyoro_rutooro_en.py"
script_path.write_text(dataset_script, encoding="utf-8")
ops.append(CommitOperationAdd(
    path_in_repo="runyoro_rutooro_en.py",
    path_or_fileobj=str(script_path),
))

try:
    api.create_commit(
        repo_id=DATASET_ID,
        repo_type="dataset",
        operations=ops,
        commit_message="runyoro-nmt-v1: upload parallel corpus (train/val/test splits)",
        token=HF_TOKEN,
    )
    print(f"\n  Dataset uploaded: https://huggingface.co/datasets/{DATASET_ID}")
except Exception as e:
    print(f"  Dataset upload error: {e}")

# ── 4. Upload linguistic resources to dataset repo ────────────────────────────
section("4. Uploading linguistic resources")

resource_ops = []
tm_dir = ROOT / "data" / "tm"

for fname in ["runyoro_en.tmx", "runyoro_en.tbx", "glossary.csv", "glossary.json", "named_entities.json"]:
    fpath = tm_dir / fname
    if fpath.exists():
        resource_ops.append(CommitOperationAdd(
            path_in_repo=f"linguistic_resources/{fname}",
            path_or_fileobj=str(fpath),
        ))
        print(f"  Queued: {fname}")

if resource_ops:
    try:
        api.create_commit(
            repo_id=DATASET_ID,
            repo_type="dataset",
            operations=resource_ops,
            commit_message="runyoro-nmt-v1: add TMX, TBX, glossary, named entities",
            token=HF_TOKEN,
        )
        print("  Linguistic resources uploaded")
    except Exception as e:
        print(f"  Resource upload error: {e}")

# ── 5. Upload pipeline reports to dataset repo ────────────────────────────────
section("5. Uploading pipeline reports")

report_ops = []
reports_dir = ROOT / "data" / "reports"

for fname in ["validation_report.md", "cleaning_report.md", "augmentation_report.md", "pipeline_report.md"]:
    fpath = reports_dir / fname
    if fpath.exists():
        report_ops.append(CommitOperationAdd(
            path_in_repo=f"pipeline_reports/{fname}",
            path_or_fileobj=str(fpath),
        ))
        print(f"  Queued: {fname}")

if report_ops:
    try:
        api.create_commit(
            repo_id=DATASET_ID,
            repo_type="dataset",
            operations=report_ops,
            commit_message="runyoro-nmt-v1: add pipeline reports",
            token=HF_TOKEN,
        )
        print("  Reports uploaded")
    except Exception as e:
        print(f"  Reports upload error: {e}")

# ── 6. Upload model card (placeholder) ───────────────────────────────────────
section("6. Uploading model card to model repo")

model_card = f"""---
language:
  - nyk
  - en
license: apache-2.0
tags:
  - translation
  - runyoro-rutooro
  - english
  - nmt
  - nllb
  - runyoro-nmt-v1
  - low-resource
  - bantu
datasets:
  - {DATASET_ID}
base_model: facebook/nllb-200-distilled-600M
pipeline_tag: translation
---

# runyoro-nmt-v1

**Bidirectional Runyoro-Rutooro <-> English Neural Machine Translation**

> This is a **new, separate fine-tune** (`runyoro-nmt-v1`) distinct from any
> previously trained NLLB checkpoints.

## Model Description

Fine-tuned from `facebook/nllb-200-distilled-600M` for bidirectional translation
between **Runyoro-Rutooro** (a Bantu language spoken in western Uganda —
Bunyoro-Kitara & Tooro kingdoms) and **English**.

## Live Demo

Try it at the [Hugging Face Space]({f'https://huggingface.co/spaces/{SPACE_ID}'}).

## Training Data

- **Dataset:** [{DATASET_ID}](https://huggingface.co/datasets/{DATASET_ID})
- **Domains:** Agriculture, general vocabulary, greetings, idioms
- **Pipeline:** extraction -> validation -> alignment -> cleaning ->
  deduplication -> normalisation -> augmentation (2x)
- **Total training pairs:** ~632 (+ ~316 augmented)

## Training Procedure

| Parameter | Value |
|-----------|-------|
| Base model | facebook/nllb-200-distilled-600M |
| Epochs | 15 |
| Batch size | 16 (+ grad accum 4) |
| Learning rate | 5e-5 |
| Scheduler | Cosine |
| Label smoothing | 0.1 |
| Beam size (inference) | 4 |
| Curriculum learning | 3 stages (30/80/200 tokens) |
| Contrastive loss | NT-Xent on encoder embeddings |
| Domain weighting | Agriculture x1.5 |

## Usage

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("{MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained("{MODEL_ID}")

# Runyoro-Rutooro -> English
tokenizer.src_lang = "nyk_Latn"
inputs = tokenizer("Oraire ota?", return_tensors="pt")
translated = model.generate(
    **inputs,
    forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"],
    num_beams=4,
)
print(tokenizer.batch_decode(translated, skip_special_tokens=True))
# -> ["How are you?"]

# English -> Runyoro-Rutooro
tokenizer.src_lang = "eng_Latn"
inputs = tokenizer("How are you?", return_tensors="pt")
translated = model.generate(
    **inputs,
    forced_bos_token_id=tokenizer.lang_code_to_id["nyk_Latn"],
    num_beams=4,
)
print(tokenizer.batch_decode(translated, skip_special_tokens=True))
```

## Evaluation Metrics

*(Will be updated after training completes)*

| Metric | Rutooro->EN | EN->Rutooro |
|--------|------------|------------|
| BLEU | - | - |
| chrF++ | - | - |
| COMET | - | - |
| BERTScore F1 | - | - |

## Linguistic Notes

- **Language family:** Bantu (Niger-Congo)
- **ISO 639-3:** nyk (Nyankore-Kiga — closest match in NLLB-200)
- **Script:** Latin
- **Morphology:** Agglutinative — noun classes, verb affixes
- **Speakers:** ~3 million (Banyoro + Batoro peoples, western Uganda)

## Limitations

- Low-resource language — quality reflects available training data
- Dialectal variation (Runyoro vs Rutooro) may affect some outputs
- Domain coverage limited to training data domains

## Citation

```bibtex
@misc{{runyoro-nmt-v1,
  author = {{kathay}},
  title = {{Runyoro-Rutooro / English NMT (runyoro-nmt-v1)}},
  year = {{2025}},
  publisher = {{Hugging Face}},
  howpublished = {{\\url{{https://huggingface.co/{MODEL_ID}}}}}
}}
```
"""

card_path = ROOT / "docs" / "MODEL_CARD.md"
card_path.write_text(model_card, encoding="utf-8")

try:
    api.upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=MODEL_ID,
        repo_type="model",
        token=HF_TOKEN,
        commit_message="runyoro-nmt-v1: add model card",
    )
    print(f"  Model card uploaded: https://huggingface.co/{MODEL_ID}")
except Exception as e:
    print(f"  Model card upload error: {e}")

# ── 7. Upload Gradio Space ────────────────────────────────────────────────────
section("7. Deploying Gradio Space")

space_app = ROOT / "ui" / "gradio_app.py"
space_reqs = ROOT / "ui" / "requirements.txt"

# Update app to reference the correct model
app_content = space_app.read_text(encoding="utf-8")
app_content = app_content.replace(
    'MODEL_ID = os.environ.get("MODEL_ID", "kathay/runyoro-nmt-v1")',
    f'MODEL_ID = os.environ.get("MODEL_ID", "{MODEL_ID}")',
)
space_app.write_text(app_content, encoding="utf-8")

space_ops = [
    CommitOperationAdd(path_in_repo="app.py",           path_or_fileobj=str(space_app)),
    CommitOperationAdd(path_in_repo="requirements.txt", path_or_fileobj=str(space_reqs)),
]

# Space README
space_readme = f"""---
title: Runyoro-Rutooro English Translator
emoji: \U0001f30d
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: "4.37.2"
app_file: app.py
pinned: true
license: apache-2.0
---

# AI Stick — Runyoro-Rutooro \u2194 English Translator

Bidirectional neural machine translation for **Runyoro-Rutooro** (western Uganda)
and **English**, powered by **runyoro-nmt-v1** fine-tuned on NLLB-200.

Model: [{MODEL_ID}](https://huggingface.co/{MODEL_ID})
Dataset: [{DATASET_ID}](https://huggingface.co/datasets/{DATASET_ID})
"""
space_readme_path = ROOT / "ui" / "SPACE_README.md"
space_readme_path.write_text(space_readme, encoding="utf-8")
space_ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(space_readme_path)))

try:
    api.create_commit(
        repo_id=SPACE_ID,
        repo_type="space",
        operations=space_ops,
        commit_message="runyoro-nmt-v1: deploy Gradio translator demo",
        token=HF_TOKEN,
    )
    print(f"  Space deployed: https://huggingface.co/spaces/{SPACE_ID}")
except Exception as e:
    print(f"  Space deploy error: {e}")

# ── 8. Summary ────────────────────────────────────────────────────────────────
section("DONE - All Hub resources")
print(f"""
  Model card:  https://huggingface.co/{MODEL_ID}
  Dataset:     https://huggingface.co/datasets/{DATASET_ID}
  Demo Space:  https://huggingface.co/spaces/{SPACE_ID}

  Next step: run training then call trainer.push_to_hub() to upload weights.
  Command:
    python scripts/run_pipeline.py --config configs/config.yaml --skip-hub
""")
