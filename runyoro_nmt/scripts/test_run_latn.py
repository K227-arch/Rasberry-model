"""Test run_Latn (Rundi) as BOS for Runyoro output vs lug_Latn."""
import torch, re, os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from pathlib import Path

MODEL = Path(__file__).parent.parent / "models/checkpoints/runyoro-nmt-v1"
tok   = AutoTokenizer.from_pretrained(str(MODEL))
model = AutoModelForSeq2SeqLM.from_pretrained(str(MODEL)).cuda().eval()
POS   = re.compile(r"\[[A-Z_]+\]\s*")

for code in ["run_Latn", "lug_Latn", "eng_Latn", "nyk_Latn"]:
    print(f"  {code} -> id={tok.convert_tokens_to_ids(code)}")

def translate(text, src, tgt_bos):
    tok.src_lang = src
    bos = tok.convert_tokens_to_ids(tgt_bos)
    enc = tok(text, return_tensors="pt", max_length=256, truncation=True).to("cuda")
    with torch.no_grad():
        out = model.generate(**enc, forced_bos_token_id=bos, num_beams=4, max_length=256)
    r = tok.decode(out[0], skip_special_tokens=True).strip()
    return POS.sub("", re.sub(r"^-\s*", "", r)).strip()

en_tests = [
    "How are you?",
    "Thank you very much.",
    "people",
    "cassava",
    "cows",
    "We need to plant seeds before the rains.",
    "The sun is rising over the mountains.",
]

print()
print("=" * 65)
print("  English -> Runyoro comparison: run_Latn vs lug_Latn")
print("=" * 65)
print(f"  {'English':<35} {'run_Latn':<25} {'lug_Latn'}")
print("-" * 65)
for text in en_tests:
    run = translate(text, "eng_Latn", "run_Latn")
    lug = translate(text, "eng_Latn", "lug_Latn")
    print(f"  {text:<35} {run:<25} {lug}")

print()
print("  Runyoro -> English")
print("-" * 65)
for text in ["Oraire ota?", "abantu", "ente", "muhogo", "Webale muno."]:
    eng = translate(text, "nyk_Latn", "eng_Latn")
    print(f"  {text:<35} -> {eng}")
