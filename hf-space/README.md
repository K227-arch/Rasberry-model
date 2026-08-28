---
title: Runyoro NMT API
emoji: 🌍
colorFrom: yellow
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Runyoro-Rutooro ↔ English Translation API

FastAPI backend for the AI Stick portal. Serves bidirectional Runyoro-Rutooro ↔ English neural machine translation using `keithtwesigye/runyoro-nmt` (NLLB-200-distilled-1.3B fine-tuned with continual learning).

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
