# AI Stick — Runyoro-Rutooro Language Portal

A production-grade Runyoro-Rutooro ↔ English language technology platform built for the AI Stick portal.

---

## Projects

### `stitch_ai_stick_language_portal/`
The UI design system and existing portal screens (home, translator, AI chat, document editor).

### `runyoro_nmt/` ← **NEW**
Full production NMT pipeline — `runyoro-nmt-v1`.  
**Separate from any pre-existing NLLB fine-tunes.**

See [`runyoro_nmt/docs/README.md`](runyoro_nmt/docs/README.md) for full documentation.

**Quick start:**
```bash
cd runyoro_nmt
pip install -r requirements.txt
python scripts/run_pipeline.py --config configs/config.yaml --data-only
```

**Open the UI:** `runyoro_nmt/ui/index.html`

---

## Raw Data

Located in `raw data/`:
- `Agriculture Seed Vocabulary.csv.xlsx`
- `augmentted pos pairs.xlsx`
- `Ff_fixed worked on.docx`, `J_fixed Worked on.docx`, `Tt fixed worked on.docx`
- `U_fixed ...docx`, `V_fixed worked on.docx`, `W_fixed worked on.docx`

---

## Hugging Face Resources

| Resource | URL |
|----------|-----|
| Model | https://huggingface.co/kathay/runyoro-nmt-v1 |
| Dataset | https://huggingface.co/datasets/kathay/runyoro-rutooro-en-parallel |
| Demo | https://huggingface.co/spaces/kathay/runyoro-translator |
