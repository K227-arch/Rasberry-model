"""
AI Stick — NLLB Model Server (runyoro-nmt-v6)
==============================================
FastAPI server using the fine-tuned NLLB-200 model.
No language codes — model learns direction from text patterns.
v6: trained on 499 clean pairs (+ augmentation), 20 epochs, bidirectional.

Camera Lens OCR: EasyOCR + OpenCV for text detection, NLLB for translation.
"""
import os
import re
import base64
import logging
from contextlib import asynccontextmanager
from typing import List, Optional

import numpy as np
import cv2
import easyocr
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
    BASE_DIR, "..", "runyoro_nmt", "models", "checkpoints", "runyoro-nmt-v6"
)
# Fallback chain: v6 -> v5 -> v4 -> HuggingFace
if not os.path.isdir(CHECKPOINT_DIR):
    CHECKPOINT_DIR = os.path.join(
        BASE_DIR, "..", "runyoro_nmt", "models", "checkpoints", "runyoro-nmt-v5"
    )
if not os.path.isdir(CHECKPOINT_DIR):
    CHECKPOINT_DIR = os.path.join(
        BASE_DIR, "..", "runyoro_nmt", "models", "checkpoints", "runyoro-nmt-v4"
    )
MODEL_PATH = CHECKPOINT_DIR if os.path.isdir(CHECKPOINT_DIR) else "kathay/runyoro-nmt-v5"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = None
model = None

NLLB_RNY = "nyk_Latn"
NLLB_ENG = "eng_Latn"
NLLB_RNY_BOS = "nyk_Latn"
NLLB_ENG_BOS = "eng_Latn"

POS_TAG_RE = re.compile(r"\[[A-Z_]+\]\s*")

# EasyOCR reader (loaded at startup)
ocr_reader: Optional[easyocr.Reader] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, model, ocr_reader
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

    # Initialize EasyOCR reader (English; works for Runyoro latin script too)
    logger.info("Loading EasyOCR reader...")
    ocr_reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
    logger.info("EasyOCR ready")
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
        "model": "runyoro-nmt-v5 (NLLB-200 distilled, FP32, no lang codes)",
        "device": device,
        "loaded": model is not None,
        "ocr_loaded": ocr_reader is not None,
        "endpoints": {
            "GET /health": "Server health check",
            "POST /translate": "Translate text",
            "POST /ocr": "OCR + translate from camera frame (base64 image)",
        },
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": device,
        "model_loaded": model is not None,
        "model": "runyoro-nmt-v5",
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


# ──────────────────────────────────────────────────────────────
# Camera Lens OCR Endpoint
# ──────────────────────────────────────────────────────────────

class OCRRequest(BaseModel):
    image: str  # base64 encoded image (data URI or raw base64)
    direction: str  # "English → Runyoro" or "Runyoro → English"


class TextBlock(BaseModel):
    text: str
    translation: str
    bbox: List[List[int]]  # [[x0,y0],[x1,y1],[x2,y2],[x3,y3]]
    confidence: float


class OCRResponse(BaseModel):
    blocks: List[TextBlock]
    image_width: int
    image_height: int


def preprocess_frame(img: np.ndarray) -> np.ndarray:
    """OpenCV preprocessing to improve OCR accuracy."""
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Denoise
    denoised = cv2.fastNlMeansDenoising(enhanced, None, h=10, templateWindowSize=7, searchWindowSize=21)

    # Sharpen
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(denoised, -1, kernel)

    return sharpened


def translate_text_sync(text: str, direction: str) -> str:
    """Synchronous translation using the loaded NLLB model."""
    if not text.strip() or model is None:
        return text

    is_rny_to_en = "Runyoro" in direction and (
        "English" not in direction
        or direction.index("Runyoro") < direction.index("English")
    )
    tgt_bos = NLLB_ENG_BOS if is_rny_to_en else NLLB_RNY_BOS

    enc = tokenizer(
        text,
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
    return clean_translation(raw, tgt_bos)

====
@app.post("/ocr", response_model=OCRResponse)
async def ocr_translate(req: OCRRequest):
    """
    Accepts a base64-encoded camera frame, runs OpenCV preprocessing,
    EasyOCR text detection, and NLLB translation. Returns detected text
    blocks with bounding boxes and translations for overlay rendering.
    """
    if ocr_reader is None:
        raise HTTPException(status_code=503, detail="OCR engine still loading")
    if model is None:
        raise HTTPException(status_code=503, detail="Translation model still loading")

    # Decode base64 image
    try:
        image_data = req.image
        # Strip data URI prefix if present
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]
        img_bytes = base64.b64decode(image_data)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")

    img_height, img_width = img.shape[:2]

    # OpenCV preprocessing for better OCR
    processed = preprocess_frame(img)

    # EasyOCR text detection
    results = ocr_reader.readtext(processed, paragraph=False)

    # Filter low-confidence results and short text
    filtered = [
        (bbox, text, conf)
        for bbox, text, conf in results
        if conf > 0.3 and len(text.strip()) > 1
    ]

    # Translate each detected text block
    blocks: List[TextBlock] = []
    for bbox, text, conf in filtered:
        translation = translate_text_sync(text.strip(), req.direction)
        # bbox from EasyOCR is [[x0,y0],[x1,y1],[x2,y2],[x3,y3]]
        blocks.append(TextBlock(
            text=text.strip(),
            translation=translation,
            bbox=[[int(p[0]), int(p[1])] for p in bbox],
            confidence=round(conf, 3),
        ))

    return OCRResponse(blocks=blocks, image_width=img_width, image_height=img_height)


if __name__ == "__main__":
    port = int(os.environ.get("MODEL_PORT", 8000))
    uvicorn.run("model_server:app", host="127.0.0.1", port=port, reload=False)
