"""
AI Stick — NLLB Model Server
=============================
FastAPI server that loads the fine-tuned NLLB-200 model
and serves translations to the Next.js frontend.

Usage:
    python model_server.py
    # Server starts at http://127.0.0.1:8000

Key fix: nyk_Latn (Runyoro) = <unk> (id=3) in NLLB vocab.
Use lug_Latn (Luganda, id=256110) as the Runyoro generation BOS.
"""

import os
import re
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model_server")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(
    BASE_DIR, "..", "runyoro_nmt", "models", "checkpoints", "runyoro-nmt-v1"
)
HF_MODEL_ID = "kathay/runyoro-nmt-v1"
MODEL_PATH = CHECKPOINT_DIR if os.path.isdir(CHECKPOINT_DIR) else HF_MODEL_ID

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = None
model = None

NLLB_RNY     = "nyk_Latn"   # source lang token when encoding Runyoro input
NLLB_ENG     = "eng_Latn"   # source/target lang token for English
NLLB_RNY_BOS = "nyk_Latn"   # BOS for Runyoro generation — nyk_Latn is now a real
                              # resized token (id=256204) in the fine-tuned checkpoint
NLLB_ENG_BOS = "eng_Latn"   # BOS token for English generation (id=256047)

POS_TAG_RE = re.compile(r"\[[A-Z_]+\]\s*")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, model
    logger.info("Loading model from: %s on %s", MODEL_PATH, device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    rny_id  = tokenizer.convert_tokens_to_ids(NLLB_RNY)
    eng_id  = tokenizer.convert_tokens_to_ids(NLLB_ENG_BOS)
    lug_id  = tokenizer.convert_tokens_to_ids(NLLB_RNY_BOS)
    logger.info("Token IDs — nyk_Latn=%d  eng_Latn=%d  lug_Latn=%d",
                rny_id, eng_id, lug_id)
    logger.info("Model ready — %s", MODEL_PATH)
    yield


app = FastAPI(title="AI Stick — NLLB Model Server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def clean_translation(text: str, tgt_bos: str) -> str:
    """Strip POS tags, leading hyphens, fix English capitalisation."""
    text = text.strip()
    text = POS_TAG_RE.sub("", text).strip()
    text = re.sub(r"^-\s*", "", text).strip()
    if tgt_bos == NLLB_ENG_BOS and text and text[0].islower():
        text = text[0].upper() + text[1:]
    return text


class TranslateRequest(BaseModel):
    text: str
    direction: str   # "Runyoro → English"  or  "English → Runyoro"


class TranslateResponse(BaseModel):
    translation: str
    direction: str


@app.get("/")
async def root():
    return {
        "name": "AI Stick — NLLB Model Server",
        "model": str(MODEL_PATH),
        "device": device,
        "loaded": model is not None,
        "endpoints": {
            "GET /health": "Server health check",
            "POST /translate": "Translate text",
        },
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": device,
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
        "bleu": 18.77,
        "chrf": 22.53,
    }


@app.post("/translate", response_model=TranslateResponse)
async def translate(req: TranslateRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    if model is None:
        raise HTTPException(status_code=503, detail="Model still loading")

    direction = req.direction.strip()
    is_rny_to_en = "Runyoro" in direction and direction.index("Runyoro") < direction.index("English") if "English" in direction else "Runyoro" in direction

    if is_rny_to_en:
        src_lang = NLLB_RNY
        tgt_bos  = NLLB_ENG_BOS   # eng_Latn  id=256047
    else:
        src_lang = NLLB_ENG
        tgt_bos  = NLLB_RNY_BOS   # lug_Latn  id=256110

    tokenizer.src_lang = src_lang
    forced_bos_id = tokenizer.convert_tokens_to_ids(tgt_bos)

    enc = tokenizer(
        req.text,
        return_tensors="pt",
        max_length=256,
        truncation=True,
    ).to(device)

    with torch.no_grad():
        out = model.generate(
            **enc,
            forced_bos_token_id=forced_bos_id,
            num_beams=4,
            max_length=256,
            length_penalty=1.0,
        )

    raw = tokenizer.decode(out[0], skip_special_tokens=True)
    translation = clean_translation(raw, tgt_bos)

    return TranslateResponse(translation=translation, direction=direction)


if __name__ == "__main__":
    port = int(os.environ.get("MODEL_PORT", 8000))
    uvicorn.run("model_server:app", host="127.0.0.1", port=port, reload=False)
