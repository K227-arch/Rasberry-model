"""
AI Stick — NLLB Model Server (runyoro-rut-v1)
=============================================
Uses a custom rut_Latn token added to the NLLB tokenizer.
At generation time we set forced_bos_token_id so the decoder
is physically constrained to start in the correct language —
the model cannot fall back to nyk_Latn, English, or anything else.

  EN → Runyoro : forced_bos_token_id = rut_token_id
  Runyoro → EN : forced_bos_token_id = eng_token_id

Always float32 — bfloat16 causes multilingual leakage.
"""
import json
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

# ── Model path: prefer latest rut model, fall back to clean-v4 ───────────────
_CKPT = os.path.join(BASE_DIR, "..", "runyoro_nmt", "models", "checkpoints")
_RUT_V6   = os.path.join(_CKPT, "runyoro-rut-v6")
_RUT_V5   = os.path.join(_CKPT, "runyoro-rut-v5")
_RUT_V4   = os.path.join(_CKPT, "runyoro-rut-v4")
_RUT_V3   = os.path.join(_CKPT, "runyoro-rut-v3")
_RUT_V2   = os.path.join(_CKPT, "runyoro-rut-v2")
_RUT_V1   = os.path.join(_CKPT, "runyoro-rut-v1")
_CLEAN_V4 = os.path.join(_CKPT, "runyoro-clean-v4")

if os.path.isdir(_RUT_V6):
    MODEL_PATH = _RUT_V6
elif os.path.isdir(_RUT_V5):
    MODEL_PATH = _RUT_V5
elif os.path.isdir(_RUT_V4):
    MODEL_PATH = _RUT_V4
elif os.path.isdir(_RUT_V3):
    MODEL_PATH = _RUT_V3
elif os.path.isdir(_RUT_V2):
    MODEL_PATH = _RUT_V2
elif os.path.isdir(_RUT_V1):
    MODEL_PATH = _RUT_V1
else:
    MODEL_PATH = _CLEAN_V4

TORCH_DTYPE = torch.float32   # Never bf16 — causes multilingual leakage
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = None
model = None
RUT_TOKEN_ID: Optional[int] = None   # forced BOS for Runyoro output
ENG_TOKEN_ID: Optional[int] = None   # forced BOS for English output
USING_RUT_PREFIX: bool = False        # True when rut-v1 model is loaded

POS_TAG_RE = re.compile(r"\[[A-Z_]+\]\s*")
ocr_reader: Optional[easyocr.Reader] = None

# ── Phrase dictionary — always-correct fast-path ──────────────────────────────
PHRASE_DICT_EN_TO_RNY: dict[str, str] = {
    "good morning": "oraire ota",
    "good evening": "osibye ota",
    "good night": "oire ota",
    "how are you": "muli kurungi",
    "i am fine": "ndi kurungi",
    "i am fine thank you": "ndi kurungi webale",
    "thank you": "webale",
    "thank you very much": "webale nyo",
    "you are welcome": "kaikuru",
    "yes": "yego",
    "no": "nedda",
    "please": "bakwana",
    "sorry": "mbabarira",
    "i love you": "nkukunda",
    "what is your name": "eizina ryawe ni irihe",
    "my name is": "eizina ryange ni",
    "where are you going": "oli hahi",
    "come here": "iza hano",
    "sit down": "tuura wansi",
    "stand up": "simama",
    "i don't understand": "sisobola kuhangana",
    "i understand": "nsobola kuhangana",
    "speak slowly": "yogera mpola mpola",
    "what time is it": "esaawa ni ngahi",
    "goodbye": "gire munonga",
    "see you later": "turabonana",
    "help me": "nkwasire",
    "water": "amazi",
    "food": "ebyokulya",
    "i am hungry": "nsiima orara",
    "i am thirsty": "nsiima enywa",
    "where is the hospital": "eki'sitera kiri hahi",
    "i am sick": "ndi murwaire",
    "call the police": "ita amapolisi",
    "hello": "osiibwe",
    "i am happy": "ndi omusanyufu",
    "i don't know": "siizi",
    "come back": "garuka",
    "i am tired": "ndi munangifu",
    "give me water": "mpa amaizi",
    "i am going to school": "nyija okugenda isomero",
    "open the door": "gunjura omulyango",
    "please help me": "nkusaba omponye",
    "we are friends": "turi bagenzi",
    "how much": "esente zingahi",
    "where do you live": "oya hahi",
    "i live here": "ntura hano",
}

PHRASE_DICT_RNY_TO_EN: dict[str, str] = {v: k for k, v in PHRASE_DICT_EN_TO_RNY.items()}


def phrase_lookup(text: str, is_rny_to_en: bool) -> Optional[str]:
    key = text.lower().strip().rstrip(".,!?;:")
    lookup = PHRASE_DICT_RNY_TO_EN if is_rny_to_en else PHRASE_DICT_EN_TO_RNY
    return lookup.get(key)


# ── Load token IDs ────────────────────────────────────────────────────────────

def load_token_ids(tok, model_path: str) -> tuple[int, int, bool]:
    """
    Read rut_Latn / eng_Latn token IDs.
    If rut_token_meta.json exists (rut-v1 model), read from there.
    Otherwise fall back to eng_Latn id for both (clean-v4 mode).
    Returns (rut_id, eng_id, using_rut_prefix).
    """
    meta_file = os.path.join(model_path, "rut_token_meta.json")
    if os.path.isfile(meta_file):
        with open(meta_file) as f:
            meta = json.load(f)
        rut_id = meta["rut_token_id"]
        eng_id = meta["eng_token_id"]
        logger.info("Loaded token IDs from rut_token_meta.json: rut=%d eng=%d", rut_id, eng_id)
        return rut_id, eng_id, True

    # Fallback: rut-v1 not trained yet, use clean-v4 without forced BOS
    eng_id = tok.convert_tokens_to_ids("eng_Latn")
    logger.warning(
        "rut_token_meta.json not found — running WITHOUT forced_bos_token_id. "
        "Translations may be in wrong language for ambiguous inputs. "
        "Run train_rut_prefix.py to fix this permanently."
    )
    return None, eng_id, False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, model, RUT_TOKEN_ID, ENG_TOKEN_ID, USING_RUT_PREFIX, ocr_reader

    logger.info("Loading model from: %s  device=%s  dtype=float32", MODEL_PATH, device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_PATH, torch_dtype=TORCH_DTYPE
    ).to(device)
    model.eval()

    RUT_TOKEN_ID, ENG_TOKEN_ID, USING_RUT_PREFIX = load_token_ids(tokenizer, MODEL_PATH)

    if USING_RUT_PREFIX:
        logger.info(
            "rut_Latn prefix active — forced_bos_token_id will lock decoder language. "
            "EN→RUT bos=%d  RUT→EN bos=%d", RUT_TOKEN_ID, ENG_TOKEN_ID
        )
    else:
        logger.info("Running in clean-v4 mode (no forced BOS). Phrase dict active for common phrases.")

    logger.info("Loading EasyOCR...")
    ocr_reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
    logger.info("Ready.")
    yield


app = FastAPI(title="AI Stick — NLLB Model Server", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def clean_translation(text: str, capitalize: bool = False) -> str:
    text = text.strip()
    text = re.sub(r"^>>rny<<\s*", "", text).strip()
    text = re.sub(r"^>>eng<<\s*", "", text).strip()
    text = re.sub(r"^rut_Latn\s*", "", text).strip()
    text = re.sub(r"^eng_Latn\s*", "", text).strip()
    text = POS_TAG_RE.sub("", text).strip()
    text = re.sub(r"^[-–—]\s*", "", text).strip()
    text = text.split("\n")[0].strip()
    text = re.split(r"\s[-–—]\s", text)[0].strip()
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    if len(sentences) > 1:
        text = sentences[0].strip()
    if capitalize and text and text[0].islower():
        text = text[0].upper() + text[1:]
    return text


def is_echo(source: str, output: str) -> bool:
    """True only when output is exactly the same text as input."""
    return (
        source.lower().strip().rstrip(".,!?;:")
        == output.lower().strip().rstrip(".,!?;:")
    )


def run_model(enc: dict, bos_id: Optional[int], capitalize: bool) -> str:
    """
    Run beam search. If bos_id is set, the decoder is forced to emit it
    as the very first token — locking the output language.
    """
    input_len = enc["input_ids"].shape[1]
    max_out_len = min(256, max(input_len * 3, 20))

    generate_kwargs = dict(
        **enc,
        num_beams=5,
        max_length=max_out_len,
        length_penalty=1.0,
        repetition_penalty=1.3,
        no_repeat_ngram_size=3,
    )
    if bos_id is not None:
        generate_kwargs["forced_bos_token_id"] = bos_id

    with torch.no_grad():
        out = model.generate(**generate_kwargs)

    raw = tokenizer.decode(out[0], skip_special_tokens=True)
    return clean_translation(raw, capitalize=capitalize)


# ── Routes ────────────────────────────────────────────────────────────────────

class TranslateRequest(BaseModel):
    text: str
    direction: str


class TranslateResponse(BaseModel):
    translation: str
    direction: str


@app.get("/")
async def root():
    return {
        "model": os.path.basename(MODEL_PATH),
        "device": device,
        "rut_prefix_active": USING_RUT_PREFIX,
        "rut_token_id": RUT_TOKEN_ID,
        "eng_token_id": ENG_TOKEN_ID,
        "loaded": model is not None,
    }


@app.get("/health")
async def health():
    return {"status": "ok", "model": os.path.basename(MODEL_PATH), "device": device}


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

    # ── 1. Phrase dictionary fast-path ────────────────────────────────────────
    hit = phrase_lookup(req.text, is_rny_to_en)
    if hit:
        return TranslateResponse(translation=hit, direction=direction)

    # ── 2. Encode input ───────────────────────────────────────────────────────
    enc = tokenizer(
        req.text, return_tensors="pt", max_length=256, truncation=True
    ).to(device)

    # ── 3. Choose forced BOS token ────────────────────────────────────────────
    if USING_RUT_PREFIX:
        # rut-v1: hard-lock the decoder language
        bos_id = ENG_TOKEN_ID if is_rny_to_en else RUT_TOKEN_ID
    else:
        # clean-v4 fallback: no forced BOS (best effort)
        bos_id = None

    capitalize = is_rny_to_en
    translation = run_model(enc, bos_id, capitalize)

    # ── 4. Echo guard (only meaningful in clean-v4 fallback mode) ─────────────
    if not USING_RUT_PREFIX and is_echo(req.text, translation):
        logger.warning("Echo detected for '%s', retrying with sampling", req.text)
        with torch.no_grad():
            out2 = model.generate(
                **enc,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                num_beams=1,
                max_length=min(256, max(enc["input_ids"].shape[1] * 3, 20)),
                repetition_penalty=1.3,
                forced_bos_token_id=None,
            )
        raw2 = tokenizer.decode(out2[0], skip_special_tokens=True)
        t2 = clean_translation(raw2, capitalize=capitalize)
        if not is_echo(req.text, t2):
            translation = t2

    return TranslateResponse(translation=translation, direction=direction)


# ── OCR endpoint ──────────────────────────────────────────────────────────────

class OCRRequest(BaseModel):
    image: str
    direction: str


class TextBlock(BaseModel):
    text: str
    translation: str
    bbox: List[List[int]]
    confidence: float


class OCRResponse(BaseModel):
    blocks: List[TextBlock]
    image_width: int
    image_height: int


def preprocess_frame(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.fastNlMeansDenoising(enhanced, None, h=10, templateWindowSize=7, searchWindowSize=21)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(denoised, -1, kernel)


def translate_text_sync(text: str, direction: str) -> str:
    if not text.strip() or model is None:
        return text
    is_rny_to_en = "Runyoro" in direction and (
        "English" not in direction
        or direction.index("Runyoro") < direction.index("English")
    )
    hit = phrase_lookup(text, is_rny_to_en)
    if hit:
        return hit
    enc = tokenizer(text, return_tensors="pt", max_length=256, truncation=True).to(device)
    bos_id = (ENG_TOKEN_ID if is_rny_to_en else RUT_TOKEN_ID) if USING_RUT_PREFIX else None
    return run_model(enc, bos_id, capitalize=is_rny_to_en)


@app.post("/ocr", response_model=OCRResponse)
async def ocr_translate(req: OCRRequest):
    if ocr_reader is None:
        raise HTTPException(status_code=503, detail="OCR engine still loading")
    if model is None:
        raise HTTPException(status_code=503, detail="Model still loading")

    try:
        img_data = req.image.split(",", 1)[1] if "," in req.image else req.image
        img = cv2.imdecode(np.frombuffer(base64.b64decode(img_data), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    h, w = img.shape[:2]
    results = ocr_reader.readtext(preprocess_frame(img), paragraph=False)
    blocks = [
        TextBlock(
            text=txt.strip(),
            translation=translate_text_sync(txt.strip(), req.direction),
            bbox=[[int(p[0]), int(p[1])] for p in bbox],
            confidence=round(conf, 3),
        )
        for bbox, txt, conf in results
        if conf > 0.3 and len(txt.strip()) > 1
    ]
    return OCRResponse(blocks=blocks, image_width=w, image_height=h)


if __name__ == "__main__":
    port = int(os.environ.get("MODEL_PORT", 8000))
    uvicorn.run("model_server:app", host="127.0.0.1", port=port, reload=False)
