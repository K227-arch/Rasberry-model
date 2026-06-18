---
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
