"""
Hugging Face Space — Gradio Demo
=================================
Interactive demo for runyoro-nmt-v1 bidirectional translation.
Deployed to: https://huggingface.co/spaces/kathay/runyoro-translator
"""

import os
import re
import logging

import gradio as gr  # type: ignore
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gradio_app")

MODEL_ID = os.environ.get("MODEL_ID", "./models/checkpoints/runyoro-nmt-v1")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logger.info("Loading model: %s on %s", MODEL_ID, DEVICE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID).to(DEVICE)
model.eval()
logger.info("Model ready")

NLLB_RNY = "nyk_Latn"
NLLB_ENG = "eng_Latn"

# Regex to strip POS tags that leaked from training data
# e.g. [GENERAL_NOUN], [COMMON_ANIMALS_NOUN], [MATHEMATICS_NOUN] etc.
POS_TAG_RE = re.compile(r"\[[A-Z_]+\]\s*")


def _clean_translation(text: str, tgt_lang: str) -> str:
    """Post-process model output: strip POS tags, fix capitalisation."""
    text = text.strip()
    # Strip leaked POS tags like [GENERAL_NOUN], [GENERAL_VERB] etc.
    text = POS_TAG_RE.sub("", text).strip()
    # Strip leading hyphens sometimes prepended by the model
    text = re.sub(r"^-\s*", "", text).strip()
    # Capitalise English output
    if tgt_lang == NLLB_ENG and text and text[0].islower():
        text = text[0].upper() + text[1:]
    return text

EXAMPLES = [
    ["Runyoro → English", "Oraire ota?", ""],
    ["Runyoro → English", "Webale muno kunkonyera.", ""],
    ["Runyoro → English", "Eizooba nirirasa ha nsozi.", ""],
    ["English → Runyoro", "How are you?", ""],
    ["English → Runyoro", "Thank you very much for helping me.", ""],
    ["English → Runyoro", "The sun is rising over the mountains.", ""],
    ["English → Runyoro", "We need to plant the seeds before the rains.", ""],
]


def translate(text: str, direction: str) -> str:
    if not text or not text.strip():
        return "Please enter some text to translate."

    src_lang = NLLB_RNY if "Runyoro" in direction else NLLB_ENG
    tgt_lang = NLLB_ENG if "Runyoro" in direction else NLLB_RNY

    tokenizer.src_lang = src_lang
    forced_bos_id = tokenizer.convert_tokens_to_ids(tgt_lang)

    enc = tokenizer(text, return_tensors="pt", max_length=256, truncation=True).to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **enc,
            forced_bos_token_id=forced_bos_id,
            num_beams=4,
            max_length=256,
            length_penalty=1.0,
        )
    translation = tokenizer.decode(out[0], skip_special_tokens=True)
    translation = _clean_translation(translation, tgt_lang)
    return translation


# -------------------------------------------------------------------------
# Gradio interface (matches AI Stick design language)
# -------------------------------------------------------------------------
CSS = """
:root {
    --primary: #070235;
    --secondary: #006a61;
    --background: #f7f9fb;
    --surface: #ffffff;
    --secondary-container: #86f2e4;
}
body { font-family: Inter, sans-serif; background: var(--background); }
.gr-button-primary { background: var(--secondary) !important; color: white !important; border-radius: 9999px !important; }
.gr-button { border-radius: 9999px !important; }
h1 { color: var(--primary); font-weight: 700; }
.gradio-container { max-width: 900px; margin: 0 auto; }
"""

with gr.Blocks(css=CSS, title="AI Stick — Runyoro-Rutooro ↔ English Translator") as demo:
    gr.HTML("""
        <div style="text-align:center; padding: 24px 0 8px;">
          <h1 style="font-size:32px; color:#070235; margin:0;">AI Stick</h1>
          <p style="color:#006a61; font-size:18px; margin:4px 0 0;">
            Runyoro-Rutooro &harr; English Neural Machine Translation
          </p>
          <p style="color:#787680; font-size:13px;">
            Model: <code>runyoro-nmt-v1</code> — fine-tuned on NLLB-200 &nbsp;|&nbsp;
            <a href="https://huggingface.co/kathay/runyoro-nmt-v1" target="_blank">Model Card</a>
          </p>
        </div>
    """)

    with gr.Row():
        direction = gr.Radio(
            choices=["Runyoro → English", "English → Runyoro"],
            value="Runyoro → English",
            label="Translation Direction",
        )

    with gr.Row(equal_height=True):
        with gr.Column():
            source = gr.Textbox(
                label="Source Text",
                placeholder="Enter text here...",
                lines=6,
                max_lines=12,
            )
            btn = gr.Button("Translate", variant="primary", size="lg")

        with gr.Column():
            output = gr.Textbox(
                label="Translation",
                placeholder="Translation will appear here...",
                lines=6,
                max_lines=12,
                interactive=False,
            )

    with gr.Row():
        gr.Examples(
            examples=[[e[0], e[1]] for e in EXAMPLES],
            inputs=[direction, source],
            label="Try these examples",
        )

    gr.HTML("""
        <div style="text-align:center; padding:16px; color:#787680; font-size:13px;">
          Runyoro-Rutooro is a Bantu language spoken in western Uganda (Bunyoro-Kitara & Tooro kingdoms).<br>
          This model is part of the <strong>AI Stick</strong> language preservation initiative.
        </div>
    """)

    btn.click(fn=translate, inputs=[source, direction], outputs=output)
    source.submit(fn=translate, inputs=[source, direction], outputs=output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
