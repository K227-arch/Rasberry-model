"""
AI Stick — NLLB Model Server (runyoro-nmt-v1)
==============================================
FastAPI server using the fine-tuned NLLB-200 model (BLEU=18.77).
Uses forced BOS tokens: nyk_Latn for Runyoro, eng_Latn for English.
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
    BASE_DIR, "..", "runyoro_nmt", "models", "checkpoints", "runyoro-nmt-v2"
)
# Fallback to v1 if v2 doesn't exist
if not os.path.isdir(CHECKPOINT_DIR):
    CHECKPOINT_DIR = os.path.join(
        BASE_DIR, "..", "runyoro_nmt", "models", "checkpoints", "runyoro-nmt-v1"
    )
MODEL_PATH = CHECKPOINT_DIR if os.path.isdir(CHECKPOINT_DIR) else "kathay/runyoro-nmt-v1"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = None
model = None

NLLB_RNY = "nyk_Latn"
NLLB_ENG = "eng_Latn"
NLLB_RNY_BOS = "nyk_Latn"
NLLB_ENG_BOS = "eng_Latn"

POS_TAG_RE = re.compile(r"\[[A-Z_]+\]\s*")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, model
    logger.info("Loading NLLB model from: %s on %s", MODEL_PATH, device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    rny_id = tokenizer.convert_tokens_to_ids(NLLB_RNY)
    eng_id = tokenizer.convert_tokens_to_ids(NLLB_ENG_BOS)
    logger.info("Token IDs — nyk_Latn=%d  eng_Latn=%d", rny_id, eng_id)
    logger.info("NLLB model ready — %s", MODEL_PATH)
    yield


app = FastAPI(title="AI Stick — NLLB Model Server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def clean_translation(text: str, tgt_bos: str) -> str:
    text = text.strip()
    text = POS_TAG_RE.sub("", text).strip()
    text = re.sub(r"^-\s*", "", text).strip()
    if tgt_bos == NLLB_ENG_BOS and text and text[0].islower():
        text = text[0].upper() + text[1:]
    return text


class TranslateRequest(BaseModel):
    text: str
    direction: str


class TranslateResponse(BaseModel):
    translation: str
    direction: str


@app.get("/")
async def root():
    return {
        "name": "AI Stick — NLLB Model Server",
        "model": "runyoro-nmt-v1 (NLLB-200 distilled)",
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
        "model": "runyoro-nmt-v1",
        "bleu": 18.77,
    }


@app.post("/translate", response_model=TranslateResponse)
async def translate(req: TranslateRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    if model is None:
        raise HTTPException(status_code=503, detail="Model still loading")

    direction = req.direction.strip()
    is_rny_to_en = "Runyoro" in direction and (
        "English" not in direction
        or direction.index("Runyoro") < direction.index("English")
    )

    if is_rny_to_en:
        src_lang = NLLB_RNY
        tgt_bos = NLLB_ENG_BOS
    else:
        src_lang = NLLB_ENG
        tgt_bos = NLLB_RNY_BOS

    # v2 model: NO language codes, NO forced_bos_token_id
    # The model learned direction from text patterns alone
    enc = tokenizer(
        req.text,
        return_tensors="pt",
        max_length=256,
        truncation=True,
    ).to(device)

    with torch.no_grad():
        out = model.generate(
            **enc,
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
