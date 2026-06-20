"""
AI Stick — NLLB Model Server
=============================
FastAPI server that loads the fine-tuned NLLB-200 model
and serves translations to the Next.js frontend.

Usage:
    python model_server.py
    # Server starts at http://127.0.0.1:8000
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

NLLB_RNY = "nyk_Latn"
NLLB_ENG = "eng_Latn"

POS_TAG_RE = re.compile(r"\[[A-Z_]+\]\s*")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, model
    logger.info("Loading model from: %s on %s", MODEL_PATH, device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
    ).to(device)

    # NLLB does not natively support nyk_Latn (Runyoro) — it maps to <unk> (id=3).
    # Add it as a real special token and copy the closest Bantu language embedding
    # so the model can use nyk_Latn as its own forced BOS token.
    if tokenizer.convert_tokens_to_ids(NLLB_RNY) == tokenizer.unk_token_id:
        lug_id = tokenizer.convert_tokens_to_ids("lug_Latn")
        tokenizer.add_tokens([NLLB_RNY], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))
        nyk_id = tokenizer.convert_tokens_to_ids(NLLB_RNY)
        with torch.no_grad():
            model.get_input_embeddings().weight[nyk_id] = (
                model.get_input_embeddings().weight[lug_id].clone()
            )
            model.get_output_embeddings().weight[nyk_id] = (
                model.get_output_embeddings().weight[lug_id].clone()
            )
        logger.info("Added nyk_Latn token (id=%d) with embedding from lug_Latn", nyk_id)

    model.eval()
    logger.info("Model ready — %s", MODEL_PATH)
    yield


app = FastAPI(title="AI Stick — NLLB Model Server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def clean_translation(text: str, tgt_lang: str) -> str:
    text = text.strip()
    text = POS_TAG_RE.sub("", text).strip()
    text = re.sub(r"^-\s*", "", text).strip()
    if tgt_lang == NLLB_ENG and text and text[0].islower():
        text = text[0].upper() + text[1:]
    return text


class TranslateRequest(BaseModel):
    text: str
    direction: str


class TranslateResponse(BaseModel):
    translation: str


@app.get("/")
async def root():
    return {
        "name": "AI Stick — NLLB Model Server",
        "model": str(MODEL_PATH),
        "device": device,
        "loaded": model is not None,
        "endpoints": {
            "GET /health": "Server health check",
            "POST /translate": "Translate text (body: {text, direction})",
        },
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": device,
        "model_loaded": model is not None,
    }


@app.post("/translate", response_model=TranslateResponse)
async def translate(req: TranslateRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    if model is None:
        raise HTTPException(status_code=503, detail="Model still loading")

    src_lang = NLLB_RNY if req.direction.startswith("Runyoro") else NLLB_ENG
    tgt_lang = NLLB_ENG if req.direction.startswith("Runyoro") else NLLB_RNY

    tokenizer.src_lang = src_lang
    forced_bos_id = tokenizer.convert_tokens_to_ids(tgt_lang)

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

    translation = tokenizer.decode(out[0], skip_special_tokens=True)
    translation = clean_translation(translation, tgt_lang)

    return TranslateResponse(translation=translation)


if __name__ == "__main__":
    port = int(os.environ.get("MODEL_PORT", 8000))
    uvicorn.run("model_server:app", host="127.0.0.1", port=port, reload=False)
