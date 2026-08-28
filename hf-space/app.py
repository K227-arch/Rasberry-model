"""
Runyoro-NMT Translation API — HuggingFace Space (runyoro-rut-v4)
=================================================================
Loads keithtwesigye/runyoro-nmt from the Hub.
Uses forced_bos_token_id (rut_Latn / eng_Latn) to lock decoder language.
Phrase dictionary fast-path for common phrases.
"""
import json
import os
import re
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_ID = os.getenv("MODEL_ID", "keithtwesigye/runyoro-nmt")

# ---------------------------------------------------------------------------
# Phrase dictionary fast-path
# ---------------------------------------------------------------------------
PHRASE_DICT_EN_TO_RNY: dict = {
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
    "i am sick": "ndi murwaire",
    "call the police": "ita amapolisi",
    "hello": "osiibwe",
    "i am happy": "ndi omusanyufu",
    "i don't know": "siizi",
    "come back": "garuka",
    "i am tired": "ndi munangifu",
    "give me water": "mpa amaizi",
    "open the door": "gunjura omulyango",
    "please help me": "nkusaba omponye",
    "we are friends": "turi bagenzi",
    "how much": "esente zingahi",
    "where do you live": "oya hahi",
    "i live here": "ntura hano",
}
PHRASE_DICT_RNY_TO_EN: dict = {v: k for k, v in PHRASE_DICT_EN_TO_RNY.items()}


def phrase_lookup(text: str, is_rny_to_en: bool) -> Optional[str]:
    key = text.lower().strip().rstrip(".,!?;:")
    return (PHRASE_DICT_RNY_TO_EN if is_rny_to_en else PHRASE_DICT_EN_TO_RNY).get(key)


# ---------------------------------------------------------------------------
# Global model state
# ---------------------------------------------------------------------------
tokenizer = None
model = None
RUT_TOKEN_ID: Optional[int] = None
ENG_TOKEN_ID: Optional[int] = None
USING_RUT_PREFIX: bool = False
POS_TAG_RE = re.compile(r"\[[A-Z_]+\]\s*")


# ---------------------------------------------------------------------------
# Lifespan — load model once at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, model, RUT_TOKEN_ID, ENG_TOKEN_ID, USING_RUT_PREFIX

    print(f"Loading model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model.eval()

    # Load rut_token_meta.json — present in rut-v1+ models
    # The file is uploaded alongside the model weights in the HF repo
    _meta_paths = ["/app/rut_token_meta.json", "rut_token_meta.json"]
    for _p in _meta_paths:
        if os.path.isfile(_p):
            with open(_p) as f:
                meta = json.load(f)
            RUT_TOKEN_ID = meta["rut_token_id"]
            ENG_TOKEN_ID = meta["eng_token_id"]
            USING_RUT_PREFIX = True
            print(f"rut_Latn prefix active: rut={RUT_TOKEN_ID}  eng={ENG_TOKEN_ID}")
            break

    if not USING_RUT_PREFIX:
        # Try loading token IDs directly from tokenizer
        _rut = tokenizer.convert_tokens_to_ids("rut_Latn")
        _eng = tokenizer.convert_tokens_to_ids("eng_Latn")
        if _rut != tokenizer.unk_token_id:
            RUT_TOKEN_ID = _rut
            ENG_TOKEN_ID = _eng
            USING_RUT_PREFIX = True
            print(f"rut_Latn found in tokenizer: rut={RUT_TOKEN_ID}  eng={ENG_TOKEN_ID}")
        else:
            ENG_TOKEN_ID = _eng
            print("No rut_Latn token — running without forced BOS (phrase dict active)")

    print("Ready.")
    yield


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Runyoro-NMT API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def clean_translation(text: str, capitalize: bool = False) -> str:
    text = text.strip()
    text = re.sub(r"^>>rny<<\s*|^>>eng<<\s*|^rut_Latn\s*|^eng_Latn\s*", "", text).strip()
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
    return (
        source.lower().strip().rstrip(".,!?;:")
        == output.lower().strip().rstrip(".,!?;:")
    )


def run_model(enc: dict, bos_id: Optional[int], capitalize: bool) -> str:
    input_len = enc["input_ids"].shape[1]
    max_out_len = min(256, max(input_len * 3, 20))
    gen_kwargs = dict(
        **enc,
        num_beams=5,
        max_length=max_out_len,
        length_penalty=1.0,
        repetition_penalty=1.3,
        no_repeat_ngram_size=3,
    )
    if bos_id is not None:
        gen_kwargs["forced_bos_token_id"] = bos_id
    with torch.no_grad():
        out = model.generate(**gen_kwargs)
    return clean_translation(tokenizer.decode(out[0], skip_special_tokens=True), capitalize=capitalize)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
class TranslateRequest(BaseModel):
    text: str
    direction: str


class TranslateResponse(BaseModel):
    translation: str
    direction: str


@app.get("/")
def root():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "rut_prefix_active": USING_RUT_PREFIX,
        "rut_token_id": RUT_TOKEN_ID,
        "eng_token_id": ENG_TOKEN_ID,
        "loaded": model is not None,
    }


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "loaded": model is not None}


@app.post("/translate", response_model=TranslateResponse)
def translate(req: TranslateRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    if model is None:
        raise HTTPException(status_code=503, detail="Model still loading")

    direction = req.direction.strip()
    is_rny_to_en = "Runyoro" in direction and (
        "English" not in direction
        or direction.index("Runyoro") < direction.index("English")
    )

    # Phrase dict fast-path
    hit = phrase_lookup(req.text, is_rny_to_en)
    if hit:
        return TranslateResponse(translation=hit, direction=direction)

    enc = tokenizer(req.text, return_tensors="pt", max_length=256, truncation=True)

    if USING_RUT_PREFIX:
        bos_id = ENG_TOKEN_ID if is_rny_to_en else RUT_TOKEN_ID
    else:
        bos_id = None

    translation = run_model(enc, bos_id, capitalize=is_rny_to_en)

    # Echo guard for fallback mode
    if not USING_RUT_PREFIX and is_echo(req.text, translation):
        with torch.no_grad():
            out2 = model.generate(
                **enc,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                num_beams=1,
                max_length=min(256, max(enc["input_ids"].shape[1] * 3, 20)),
                repetition_penalty=1.3,
            )
        t2 = clean_translation(tokenizer.decode(out2[0], skip_special_tokens=True), capitalize=is_rny_to_en)
        if not is_echo(req.text, t2):
            translation = t2

    return TranslateResponse(translation=translation, direction=direction)
