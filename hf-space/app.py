"""
Runyoro-NMT Translation API — HuggingFace Space
Uses custom rut_Latn forced_bos_token_id to lock decoder language.
No prefixes in the input text — direction is controlled purely via
forced_bos_token_id at generation time.
"""
import json
import os
import re
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

MODEL_ID = "kathay/runyoro-nmt"

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
}
PHRASE_DICT_RNY_TO_EN: dict = {v: k for k, v in PHRASE_DICT_EN_TO_RNY.items()}


def phrase_lookup(text: str, is_rny_to_en: bool) -> Optional[str]:
    key = text.lower().strip().rstrip(".,!?;:")
    return (PHRASE_DICT_RNY_TO_EN if is_rny_to_en else PHRASE_DICT_EN_TO_RNY).get(key)


print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
model.eval()
print("Model loaded!")

# Load custom token IDs if available (rut-v1 model)
RUT_TOKEN_ID: Optional[int] = None
ENG_TOKEN_ID: Optional[int] = None
USING_RUT_PREFIX = False

_meta_candidates = ["/app/rut_token_meta.json", "rut_token_meta.json"]
for _f in _meta_candidates:
    if os.path.isfile(_f):
        _meta = json.load(open(_f))
        RUT_TOKEN_ID = _meta["rut_token_id"]
        ENG_TOKEN_ID = _meta["eng_token_id"]
        USING_RUT_PREFIX = True
        print(f"rut_Latn prefix active: rut={RUT_TOKEN_ID}, eng={ENG_TOKEN_ID}")
        break

if not USING_RUT_PREFIX:
    ENG_TOKEN_ID = tokenizer.convert_tokens_to_ids("eng_Latn")
    print("No rut_token_meta.json — running without forced BOS (phrase dict active)")

POS_TAG_RE = re.compile(r"\[[A-Z_]+\]\s*")

app = FastAPI(title="Runyoro-NMT API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class TranslateRequest(BaseModel):
    text: str
    direction: str


class TranslateResponse(BaseModel):
    translation: str
    direction: str


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


@app.get("/")
def root():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "rut_prefix_active": USING_RUT_PREFIX,
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/translate", response_model=TranslateResponse)
def translate(req: TranslateRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

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

    if USING_RUT_PREFIX:
        gen_kwargs["forced_bos_token_id"] = ENG_TOKEN_ID if is_rny_to_en else RUT_TOKEN_ID
    # else: no forced BOS — clean-v4 model, phrase dict covers common cases

    with torch.no_grad():
        out = model.generate(**gen_kwargs)

    raw = tokenizer.decode(out[0], skip_special_tokens=True)
    translation = clean_translation(raw, capitalize=is_rny_to_en)

    # Echo guard (only for clean-v4 fallback)
    if not USING_RUT_PREFIX and is_echo(req.text, translation):
        with torch.no_grad():
            out2 = model.generate(
                **enc,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                num_beams=1,
                max_length=max_out_len,
                repetition_penalty=1.3,
            )
        t2 = clean_translation(tokenizer.decode(out2[0], skip_special_tokens=True), capitalize=is_rny_to_en)
        if not is_echo(req.text, t2):
            translation = t2

    return TranslateResponse(translation=translation, direction=direction)
