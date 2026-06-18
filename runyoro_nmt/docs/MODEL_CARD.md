---
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
  - kathay/runyoro-rutooro-en-parallel
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

Try it at the [Hugging Face Space](https://huggingface.co/spaces/kathay/runyoro-translator).

## Training Data

- **Dataset:** [kathay/runyoro-rutooro-en-parallel](https://huggingface.co/datasets/kathay/runyoro-rutooro-en-parallel)
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

model = AutoModelForSeq2SeqLM.from_pretrained("kathay/runyoro-nmt-v1")
tokenizer = AutoTokenizer.from_pretrained("kathay/runyoro-nmt-v1")

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
@misc{runyoro-nmt-v1,
  author = {kathay},
  title = {Runyoro-Rutooro / English NMT (runyoro-nmt-v1)},
  year = {2025},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/kathay/runyoro-nmt-v1}}
}
```
