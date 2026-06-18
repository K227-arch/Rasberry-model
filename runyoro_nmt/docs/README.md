# runyoro-nmt-v1

**Production-grade bidirectional Runyoro-Rutooro ↔ English Neural Machine Translation**

> ⚠️ This is a **new, separate model** (`runyoro-nmt-v1`) — not the same as any
> previously trained NLLB checkpoints in this workspace.

---

## Overview

A fully automated pipeline that extracts, validates, cleans, aligns, augments,
and uses raw Runyoro-Rutooro / English parallel data to fine-tune a production
NMT model based on NLLB-200.

**Language pair:** Runyoro-Rutooro (nyk) ↔ English (en)

Runyoro-Rutooro is a Bantu language spoken by the Banyoro and Batoro peoples
of western Uganda (Bunyoro-Kitara Kingdom and Tooro Kingdom).

---

## Project Structure

```
runyoro_nmt/
├── configs/
│   └── config.yaml              ← Master configuration
├── data/
│   ├── raw/                     ← Symlink / copy of raw data/
│   ├── processed/               ← Cleaned TSV splits
│   ├── augmented/               ← Augmented data
│   ├── tm/                      ← TMX, TBX, glossaries
│   ├── glossary/
│   └── reports/                 ← All pipeline reports
├── models/
│   ├── checkpoints/             ← Training checkpoints
│   └── exported/                ← INT8, ONNX, CT2
├── src/
│   ├── data_pipeline/           ← Extract → validate → clean → augment
│   ├── training/                ← Fine-tuning, curriculum, contrastive
│   ├── evaluation/              ← BLEU, chrF++, COMET, error analysis
│   ├── inference/               ← Translator, RAG, quantizer
│   └── utils/                   ← Reports, splits, hub push
├── scripts/
│   └── run_pipeline.py          ← Master orchestrator
├── ui/
│   ├── index.html               ← Dashboard
│   ├── translator.html          ← Translation interface
│   ├── reports.html             ← Pipeline reports
│   └── gradio_app.py            ← HF Space demo
├── experiments/                 ← MLflow runs
├── requirements.txt
└── docs/
    └── README.md
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download NLTK data (for augmentation)
```python
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
```

### 3. Run data pipeline only
```bash
cd runyoro_nmt
python scripts/run_pipeline.py --config configs/config.yaml --data-only
```

### 4. Run full training pipeline
```bash
python scripts/run_pipeline.py --config configs/config.yaml
```

### 5. Run evaluation only (requires trained model)
```bash
python scripts/run_pipeline.py --config configs/config.yaml --skip-training
```

### 6. Push to Hugging Face Hub
```bash
export HF_TOKEN=your_token_here
python scripts/run_pipeline.py --config configs/config.yaml --skip-training
```

### 7. Start the inference API
```bash
uvicorn src.inference.translator:app --reload --port 8000
```

### 8. Open the UI
Open `ui/translator.html` in a browser, or visit the
[HF Space demo](https://huggingface.co/spaces/kathay/runyoro-translator).

---

## Pipeline Stages

| # | Stage | Output |
|---|-------|--------|
| 1 | Data Extraction | Raw (rny, en) pairs from .xlsx/.docx files |
| 2 | Validation & QA | Filtered pairs + validation report |
| 3 | Alignment Check | Alignment scores + suspicious pair flags |
| 4 | Cleaning | Normalised pairs + cleaning report |
| 5 | Linguistic Resources | TMX, TBX, glossary CSV/JSON, NE registry |
| 6 | Augmentation | 2× more pairs + augmentation report |
| 7 | Data Split | train.tsv / val.tsv / test.tsv |
| 8 | Training | Fine-tuned NLLB-200 model (runyoro-nmt-v1) |
| 9 | Evaluation | BLEU, chrF++, COMET, TER, BERTScore |
| 10 | Error Analysis | Categorised errors with examples |
| 11 | Quantization | INT8, ONNX, CTranslate2 exports |
| 12 | Hub Push | kathay/runyoro-nmt-v1 + Space |
| 13 | Final Report | HTML + Markdown pipeline report |

---

## Model Architecture

- **Base:** `facebook/nllb-200-distilled-600M`
- **Fine-tuning:** Seq2Seq with label smoothing (0.1)
- **Curriculum learning:** 3 stages (30→80→200 tokens)
- **Contrastive learning:** NT-Xent on encoder embeddings
- **Domain adaptation:** Agriculture domain ×1.5 weight
- **Bidirectional:** Trains rny→en AND en→rny simultaneously

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| BLEU | N-gram precision (sacreBLEU) |
| chrF++ | Character F-score with word order |
| COMET | Neural quality estimation |
| TER | Translation Edit Rate |
| BERTScore | Semantic similarity |
| Back-translation | Roundtrip consistency |

---

## Hugging Face Resources

| Resource | URL |
|----------|-----|
| Model | https://huggingface.co/kathay/runyoro-nmt-v1 |
| Dataset | https://huggingface.co/datasets/kathay/runyoro-rutooro-en-parallel |
| Demo Space | https://huggingface.co/spaces/kathay/runyoro-translator |

---

## Runyoro-Rutooro Linguistic Notes

- **Language family:** Bantu (Niger-Congo)
- **Script:** Latin
- **Noun classes:** 15 noun classes with prefixes (eki-, ebi-, oku-, obu-, aba-, omu-...)
- **Morphology:** Agglutinative — verb roots take multiple affixes
- **Tones:** Not marked in standard orthography
- **ISO 639-3:** nyk (Nyankore-Kiga, closest match in NLLB)

---

## Citation

```bibtex
@misc{runyoro-nmt-v1,
  author = {kathay},
  title = {Runyoro-Rutooro ↔ English NMT System (runyoro-nmt-v1)},
  year = {2025},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/kathay/runyoro-nmt-v1}}
}
```
