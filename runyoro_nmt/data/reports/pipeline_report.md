# runyoro-nmt-v1 — Pipeline Report

**Generated:** 2026-06-19T10:58:15.665731Z

---
## Data Extraction

**Total raw pairs extracted:** 3782
**Source directory:** `..\raw data`

## Data Pipeline Statistics

| Metric | Value |
|--------|-------|
| total_input | 3782 |
| total_valid | 3774 |
| total_rejected | 8 |
| rejection_breakdown | {'Misalignment warning': 133, 'Duplicate': 8} |
| total_issues_logged | 3061 |

## Cleaning

**Pairs after cleaning:** 3774

## Linguistic Resources

- `tmx`: `data\tm\runyoro_en.tmx`
- `glossary_csv`: `data\tm\glossary.csv`
- `glossary_json`: `data\tm\glossary.json`
- `tbx`: `data\tm\runyoro_en.tbx`
- `named_entities`: `data\tm\named_entities.json`

## Data Splits

| Split | Count |
|-------|-------|
| Train | 3842 |
| Val | 452 |
| Test | 226 |