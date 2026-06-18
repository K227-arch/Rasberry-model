"""Diagnose exactly what the tokenised dataset looks like."""
import os, sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, "src")
from pathlib import Path
from transformers import AutoTokenizer
from datasets import Dataset as HFDataset

HF_TOKEN = "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"
tok = AutoTokenizer.from_pretrained(
    "facebook/nllb-200-distilled-1.3B", token=HF_TOKEN
)

def load_tsv(path):
    pairs = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split("\t", 1)
        if len(parts) == 2:
            pairs.append((parts[0].strip(), parts[1].strip()))
    return pairs

pairs = load_tsv("data/processed/train.tsv")[:4]
print(f"Sample pairs: {pairs[:2]}")

# Test the tokenise function exactly as in train.py
def tokenise(examples, src_lang, tgt_lang):
    tok.src_lang = src_lang
    src_enc = tok(
        examples["src"],
        max_length=256,
        truncation=True,
        padding=False,
    )
    tok.src_lang = tgt_lang
    tgt_enc = tok(
        examples["tgt"],
        max_length=256,
        truncation=True,
        padding=False,
    )
    tok.src_lang = src_lang
    src_enc["labels"] = tgt_enc["input_ids"]
    return src_enc

raw = HFDataset.from_dict({
    "src": [s for s, t in pairs],
    "tgt": [t for s, t in pairs],
})
print("Raw columns:", raw.column_names)

result = raw.map(
    lambda ex: tokenise(ex, "nyk_Latn", "eng_Latn"),
    batched=True,
    remove_columns=["src", "tgt"],
    load_from_cache_file=False,
)
print("Tokenised columns:", result.column_names)
print("First row keys:", list(result[0].keys()))
print("input_ids sample:", result[0]["input_ids"][:5])
print("labels sample:   ", result[0]["labels"][:5])
print("All rows have input_ids:", all("input_ids" in result[i] for i in range(len(result))))
print("DIAGNOSIS OK")
