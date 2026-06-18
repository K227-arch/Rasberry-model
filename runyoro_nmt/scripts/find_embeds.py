"""
Minimal test: manually run one training step to see exactly what keys
are passed to model(**inputs) during Trainer's training_step.
"""
import os, sys, torch
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, "src")

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq
from datasets import Dataset as HFDataset

HF_TOKEN = "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"
tok   = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B", token=HF_TOKEN)
model = AutoModelForSeq2SeqLM.from_pretrained(
    "facebook/nllb-200-distilled-1.3B", token=HF_TOKEN
).to(torch.bfloat16).cuda()
model.train()

pairs = [("Oraire ota?", "How are you?"), ("Webale muno", "Thank you very much")]

def tokenise(src_text, tgt_text):
    tok.src_lang = "nyk_Latn"
    src = tok(src_text, max_length=64, truncation=True, padding=False)
    tok.src_lang = "eng_Latn"
    tgt = tok(tgt_text, max_length=64, truncation=True, padding=False)
    tok.src_lang = "nyk_Latn"
    src["labels"] = tgt["input_ids"]
    return src

samples = [tokenise(s, t) for s, t in pairs]
collator = DataCollatorForSeq2Seq(tok, model=model, label_pad_token_id=-100, pad_to_multiple_of=8)
batch = collator(samples)
print("Batch keys:", list(batch.keys()))
batch_gpu = {k: v.cuda() for k, v in batch.items()}

# Now simulate what Trainer does: prepare_decoder_input_ids_from_labels
labels = batch_gpu["labels"].clone()
labels[labels == -100] = tok.pad_token_id
decoder_input_ids = model.prepare_decoder_input_ids_from_labels(labels=labels)
batch_gpu["decoder_input_ids"] = decoder_input_ids
print("Keys after adding decoder_input_ids:", list(batch_gpu.keys()))

# Run forward
with torch.no_grad():
    out = model(**batch_gpu)
print("Forward OK — loss:", out.loss.item())
