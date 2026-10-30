---
title: Runyoro NMT API
emoji: 🌍
colorFrom: yellow
colorTo: green
sdk: docker
app_port: 7860
pinned: false
models:
  - kathay/runyoro-nmt
---

# Runyoro-Rutooro ↔ English Translation API

FastAPI backend for the AI Stick portal. Serves bidirectional Runyoro-Rutooro ↔ English neural machine translation using `kathay/runyoro-nmt` (runyoro-rut-v11 — NLLB-200-distilled-1.3B, continual learning from v10 on 1,445 new sentence pairs).

## Endpoints

- `GET /health` — health check
- `POST /translate` — translate text

## Request format

```json
{
  "text": "Hello",
  "direction": "English → Runyoro"
}
```

## Response

```json
{
  "translation": "osiibwe",
  "direction": "English → Runyoro"
}
```
