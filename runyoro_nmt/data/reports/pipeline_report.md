# runyoro-nmt-v1 — Pipeline Report

**Generated:** 2026-06-18T00:55:33.282555Z

---
## Data Extraction

**Total raw pairs extracted:** 3782
**Source directory:** `..\raw data`

## Data Pipeline Statistics

| Metric | Value |
|--------|-------|
| total_input | 3782 |
| total_valid | 3779 |
| total_rejected | 3 |
| rejection_breakdown | {'Misalignment warning': 140, 'Duplicate': 3} |
| total_issues_logged | 403 |

## Cleaning

**Pairs after cleaning:** 3779

## Linguistic Resources

- `tmx`: `data\tm\runyoro_en.tmx`
- `glossary_csv`: `data\tm\glossary.csv`
- `glossary_json`: `data\tm\glossary.json`
- `tbx`: `data\tm\runyoro_en.tbx`
- `named_entities`: `data\tm\named_entities.json`

## Data Splits

| Split | Count |
|-------|-------|
| Train | 3996 |
| Val | 470 |
| Test | 236 |