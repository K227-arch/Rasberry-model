"""Quick test to find the correct tokenisation API for this transformers version."""
import os, sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, "src")

from transformers import AutoTokenizer
import transformers
print("transformers:", transformers.__version__)

tok = AutoTokenizer.from_pretrained(
    "facebook/nllb-200-distilled-1.3B",
    token="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",
)

src, tgt = "Oraire ota?", "How are you?"

# Method 1: set src_lang, encode source; set src_lang=tgt_lang, encode target
tok.src_lang = "nyk_Latn"
src_enc = tok([src], max_length=64, truncation=True, padding=False)
tok.src_lang = "eng_Latn"
tgt_enc = tok([tgt], max_length=64, truncation=True, padding=False)
print("Method1 src:", src_enc["input_ids"])
print("Method1 tgt:", tgt_enc["input_ids"])
print("Method1 OK")

# Method 2: forced_bos approach – verify token id
bos_id = tok.lang_code_to_id.get("eng_Latn", None)
print("eng_Latn token id:", bos_id)
bos_id2 = tok.lang_code_to_id.get("nyk_Latn", None)
print("nyk_Latn token id:", bos_id2)
