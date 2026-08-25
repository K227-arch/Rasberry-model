---
title: Runyoro NMT Translation API
emoji: 🌍
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Runyoro-NMT Translation API

Bidirectional Runyoro-Rutooro ↔ English Neural Machine Translation.

## API Usage

```bash
curl -X POST https://kathay-runyoro-nmt-api.hf.space/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Uganda exports coffee to many countries", "direction": "English to Runyoro"}'
```

## Model

Uses `kathay/runyoro-nmt` with direction prefixes (`>>rny<<` / `>>eng<<`).
