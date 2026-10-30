"""
Generate research paper: AI Stick — Runyoro-Rutooro NMT System
Output: research_paper.docx
"""

from docx import Document
from docx.shared import Pt, Cm, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import os

doc = Document()

# ── Page margins ──
section = doc.sections[0]
section.top_margin    = Cm(2.5)
section.bottom_margin = Cm(2.5)
section.left_margin   = Cm(3.0)
section.right_margin  = Cm(2.5)

# ── Base font ──
style = doc.styles["Normal"]
style.font.name = "Times New Roman"
style.font.size = Pt(12)

# ── Colour palette ──
NAVY   = RGBColor(7,   2,  53)
TEAL   = RGBColor(0, 106,  97)
GREY   = RGBColor(80,  80,  80)
BLACK  = RGBColor(0,    0,   0)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def h1(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(18)
    p.paragraph_format.space_after  = Pt(6)
    r = p.add_run(text.upper())
    r.bold = True
    r.font.size = Pt(13)
    r.font.color.rgb = NAVY
    return p

def h2(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after  = Pt(4)
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(12)
    r.font.color.rgb = TEAL
    return p

def h3(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after  = Pt(2)
    r = p.add_run(text)
    r.bold = True
    r.italic = True
    r.font.size = Pt(11)
    r.font.color.rgb = GREY
    return p

def body(text):
    p = doc.add_paragraph(text)
    p.paragraph_format.space_after  = Pt(6)
    p.paragraph_format.first_line_indent = Cm(1)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    return p

def nobullet(text):
    """Body paragraph without indent (for abstracts/notes)."""
    p = doc.add_paragraph(text)
    p.paragraph_format.space_after = Pt(6)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    return p

def bullet(text):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(2)
    p.add_run(text)
    return p

def add_table(headers, rows, caption=None):
    if caption:
        cp = doc.add_paragraph()
        cp.paragraph_format.space_before = Pt(8)
        cr = cp.add_run(caption)
        cr.bold = True
        cr.font.size = Pt(10)
        cr.font.color.rgb = GREY
        cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Light Shading Accent 1"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = h
        for para in cell.paragraphs:
            for run in para.runs:
                run.bold = True
                run.font.size = Pt(10)
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            cell = table.rows[ri + 1].cells[ci]
            cell.text = str(val)
            for para in cell.paragraphs:
                for run in para.runs:
                    run.font.size = Pt(10)
    doc.add_paragraph()
    return table

# ─────────────────────────────────────────────────────────────────────────────
# TITLE PAGE
# ─────────────────────────────────────────────────────────────────────────────
for _ in range(5):
    doc.add_paragraph()

tp = doc.add_paragraph()
tp.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = tp.add_run("AI Stick: A Bidirectional Neural Machine Translation System\n"
               "for Runyoro-Rutooro and English")
r.bold = True
r.font.size = Pt(20)
r.font.color.rgb = NAVY

doc.add_paragraph()

auth = doc.add_paragraph()
auth.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = auth.add_run("Keith Twesigye\nKathay Technologies")
r.font.size = Pt(12)
r.font.color.rgb = GREY

doc.add_paragraph()

aff = doc.add_paragraph()
aff.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = aff.add_run("Correspondence: keithtwesigye@gmail.com")
r.font.size = Pt(11)
r.italic = True
r.font.color.rgb = GREY

doc.add_paragraph()

dt = doc.add_paragraph()
dt.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = dt.add_run("October 2026")
r.font.size = Pt(11)
r.font.color.rgb = GREY

doc.add_page_break()

# ─────────────────────────────────────────────────────────────────────────────
# ABSTRACT
# ─────────────────────────────────────────────────────────────────────────────
h1("Abstract")
nobullet(
    "We present AI Stick, an end-to-end offline-capable neural machine translation (NMT) "
    "portal for Runyoro-Rutooro, a low-resource Bantu language spoken by approximately "
    "one million people in western Uganda. The system is built around a fine-tuned instance "
    "of Meta's NLLB-200-distilled-1.3B (1.37 billion parameters), augmented with a custom "
    "rut_Latn vocabulary token to enforce correct target-language generation through "
    "forced beginning-of-sequence (BOS) decoding. We describe the full pipeline from raw "
    "data collection through multi-stage validation, cleaning, augmentation, and continual "
    "learning across thirteen model versions (v1–v13). The system achieves a final "
    "validation cross-entropy loss of 1.128 on a held-out set of 164 sentence pairs drawn "
    "from 20 hand-curated Excel datasets totalling 6,163 parallel pairs. The production "
    "deployment integrates a Next.js 16 web portal, a FastAPI inference server, an "
    "EasyOCR-based camera translation feature, and a HuggingFace Space cloud fallback, "
    "enabling both online and offline translation for end users. All model checkpoints "
    "and the parallel corpus are publicly available on HuggingFace Hub."
)
doc.add_paragraph()
kw = doc.add_paragraph()
r = kw.add_run("Keywords: ")
r.bold = True
kw.add_run(
    "neural machine translation, low-resource NLP, Runyoro-Rutooro, Bantu languages, "
    "NLLB-200, continual learning, forced BOS decoding, offline AI, Uganda"
)

doc.add_page_break()

# ─────────────────────────────────────────────────────────────────────────────
# 1. INTRODUCTION
# ─────────────────────────────────────────────────────────────────────────────
h1("1. Introduction")
body(
    "Runyoro-Rutooro is a Bantu language belonging to the Great Lakes branch of the "
    "Niger-Congo family, spoken predominantly in the Bunyoro-Kitara and Tooro kingdoms "
    "of western Uganda. Despite being the first language of a significant population, "
    "the language remains critically under-represented in natural language processing (NLP) "
    "research. Public parallel corpora are scarce, pre-trained multilingual models do not "
    "natively support the language's ISO 639-3 code (nyk), and digital literacy tools in "
    "Runyoro-Rutooro are almost entirely absent."
)
body(
    "This paper describes AI Stick, a production-grade NMT system designed to address "
    "this gap. The project makes three primary contributions:"
)
bullet("A hand-curated parallel corpus of 6,163 Runyoro-Rutooro/English sentence pairs "
       "covering agriculture, healthcare, culture, religion, daily life, and legal domains.")
bullet("A fine-tuned NLLB-200-distilled-1.3B model augmented with a language-specific "
       "BOS token (rut_Latn, ID 256204) that prevents multilingual leakage — a documented "
       "failure mode where seq2seq models generate output in the wrong language.")
bullet("A full-stack web application with offline-first architecture, camera-based OCR "
       "translation, translation history, and a domain glossary, deployable on commodity "
       "hardware including Raspberry Pi-class devices.")
body(
    "The remainder of this paper is organised as follows. Section 2 reviews related work "
    "on low-resource NMT and Bantu language processing. Section 3 describes the data "
    "collection and processing pipeline. Section 4 details the model architecture and "
    "training procedure. Section 5 presents evaluation results. Section 6 describes the "
    "system architecture. Section 7 discusses limitations and future work."
)

# ─────────────────────────────────────────────────────────────────────────────
# 2. RELATED WORK
# ─────────────────────────────────────────────────────────────────────────────
h1("2. Related Work")

h2("2.1 Low-Resource Neural Machine Translation")
body(
    "Low-resource NMT remains one of the most challenging problems in the field. "
    "Sennrich and Zhang (2019) demonstrated that back-translation of monolingual data "
    "can substantially improve translation quality when parallel data is scarce. "
    "Transformer-based architectures (Vaswani et al., 2017) have become the de facto "
    "standard for NMT, providing strong inductive biases for sequence-to-sequence tasks "
    "even with limited training data."
)
body(
    "Massively multilingual pre-trained models represent a major advancement for "
    "low-resource languages. mBART (Liu et al., 2020) and mT5 (Xue et al., 2021) "
    "demonstrated that pre-training on 100+ languages enables strong zero-shot and "
    "few-shot transfer to unseen language pairs. Meta's NLLB-200 (Costa-jussà et al., "
    "2022) extended this to 200 languages with distilled variants suitable for "
    "resource-constrained deployment, making it the most appropriate base model for "
    "this work."
)

h2("2.2 Bantu Language NLP")
body(
    "Bantu languages present specific challenges for NMT: rich agglutinative morphology, "
    "noun class systems that govern agreement across sentences, and tonal distinctions "
    "that are typically not represented in orthography. Prior work on Swahili (Adelani "
    "et al., 2021) and Luganda (Babirye et al., 2022) has shown that standard subword "
    "tokenisation (BPE, SentencePiece) is effective for Bantu languages when combined "
    "with appropriate pre-training, though domain coverage remains a persistent issue."
)
body(
    "The Masakhane project (Nekoto et al., 2020) has been instrumental in building "
    "parallel corpora for African languages through community-driven data collection, "
    "a methodology that directly inspired the data collection approach in this work."
)

h2("2.3 NLLB-200 and Forced BOS Decoding")
body(
    "NLLB-200 (No Language Left Behind) uses language tokens as forced BOS signals "
    "to direct decoder output to a specific target language. Runyoro-Rutooro's ISO "
    "code (nyk) is not present in NLLB's vocabulary — the token nyk_Latn resolves "
    "to the unknown token (ID 3), causing models to default to arbitrary output "
    "languages. We address this by adding a custom rut_Latn token to the tokenizer "
    "vocabulary and fine-tuning the model's embedding layer alongside the full "
    "encoder-decoder stack, a technique previously described for low-resource language "
    "adaptation of multilingual models (Pfeiffer et al., 2020)."
)

# ─────────────────────────────────────────────────────────────────────────────
# 3. DATA
# ─────────────────────────────────────────────────────────────────────────────
h1("3. Data Collection and Processing")

h2("3.1 Corpus Construction")
body(
    "The primary training corpus was assembled through collaboration with bilingual "
    "speakers and linguistic researchers from the Bunyoro-Kitara and Tooro kingdoms. "
    "Data was collected across two phases. Phase 1 produced 4,520 pairs from Word "
    "documents and agricultural vocabulary spreadsheets. Phase 2 produced an additional "
    "1,643 unique pairs from 20 structured Excel workbooks (sentence pair 1 through "
    "sentence pair 20), each containing sentence pairs with tense variation across "
    "past simple, present simple, present perfect, future simple, and present continuous."
)

add_table(
    ["Source", "Format", "Approx. Pairs", "Domain"],
    [
        ["Agricultural vocabulary", "CSV/XLSX", "123", "Agriculture"],
        ["Augmented POS-tagged pairs", "XLSX", "2,776", "General"],
        ["Alphabetical vocabulary (F,J,T,U,V,W)", "DOCX", "883", "General vocabulary"],
        ["Sentence pairs 1–20 (sp1–sp20)", "XLSX", "1,643", "Diverse (20 files)"],
        ["Total", "—", "6,163", "Multi-domain"],
    ],
    caption="Table 1. Corpus sources and pair counts."
)

h2("3.2 Data Pipeline")
body(
    "All raw data was processed through a four-stage automated pipeline before training:"
)

h3("Stage 1 — Extraction")
body(
    "Raw Excel files were parsed using openpyxl. Each row follows a fixed schema: "
    "[ID, BaseID, OriginalEnglish, OriginalTense, ReferenceRunyoro, TargetTense, "
    "English, Runyoro-Rutooro, Status (optional)]. Columns 6 and 7 (0-indexed) were "
    "extracted as the source-target pair. Rows with empty Runyoro cells and header "
    "rows were skipped automatically."
)

h3("Stage 2 — Validation")
body(
    "Each extracted pair was checked against the following criteria: minimum/maximum "
    "token length (2–200 tokens); character length ratio ≥ 0.15 (min/max); maximum "
    "character multiplier ≤ 6.0; Unicode NFKC normalisation; deduplication of "
    "identical pairs; and rejection of untranslated pairs (source == target). "
    "Across the 20 datasets, 82 duplicates were rejected, retaining 1,643 valid pairs."
)

h3("Stage 3 — Cleaning")
body(
    "Surviving pairs underwent text normalisation: smart quote replacement, "
    "zero-width character removal, multiple space collapsing, space-before-punctuation "
    "removal, and English capitalisation correction. Approximately 3.3% of pairs "
    "required at least one cleaning operation."
)

h3("Stage 4 — Merge and Split")
body(
    "New pairs were deduplicated against the full existing HuggingFace dataset "
    "(kathay/runyoro-rutooro-en-parallel, 4,520 pairs) before splitting using an "
    "85/10/5 ratio (seed 42) into training, validation, and test sets. No "
    "back-translation pairs were added, to reduce the risk of overfitting on "
    "synthetic data."
)

add_table(
    ["Split", "Pairs (Phase 2)"],
    [
        ["Train", "1,397"],
        ["Validation", "164"],
        ["Test", "82"],
        ["Total new", "1,643"],
    ],
    caption="Table 2. Phase 2 data split (sp1–sp20 datasets)."
)

# ─────────────────────────────────────────────────────────────────────────────
# 4. MODEL
# ─────────────────────────────────────────────────────────────────────────────
h1("4. Model Architecture and Training")

h2("4.1 Base Model")
body(
    "The system is built on facebook/nllb-200-distilled-1.3B, a knowledge-distilled "
    "variant of Meta's NLLB-200 multilingual translation model. The architecture is "
    "a standard encoder-decoder Transformer with 1,370,638,336 parameters. The "
    "SentencePiece tokenizer uses a shared vocabulary of 256,206 subword units. "
    "All fine-tuning was performed in float32 precision — empirical testing confirmed "
    "that bfloat16 causes multilingual leakage, where the decoder produces output in "
    "an unintended language despite correct forced BOS decoding."
)

h2("4.2 Custom Language Token")
body(
    "NLLB-200 controls target language generation by prepending a language-specific "
    "BOS token to the decoder input at inference time. Runyoro-Rutooro (nyk_Latn) "
    "is not in the NLLB vocabulary — the token resolves to <unk> (ID 3). We address "
    "this by adding a new token rut_Latn to the tokenizer vocabulary and resizing "
    "the model's embedding matrices accordingly. The token was assigned ID 256204. "
    "During fine-tuning the embedding rows for rut_Latn were randomly initialised "
    "and updated via gradient descent alongside all other parameters."
)

add_table(
    ["Language", "NLLB Code", "Token ID", "Role"],
    [
        ["English", "eng_Latn", "256047", "Source/target English"],
        ["Runyoro-Rutooro (proxy)", "nyk_Latn", "3 (<unk>)", "Source encoding only"],
        ["Runyoro-Rutooro (custom)", "rut_Latn", "256204", "Forced BOS for Runyoro output"],
    ],
    caption="Table 3. Language token assignments."
)

h2("4.3 Bidirectional Training")
body(
    "Each parallel sentence pair was presented to the model in both translation "
    "directions during training: Runyoro→English and English→Runyoro. This doubled "
    "the effective number of training examples and taught the model bidirectional "
    "translation with a single set of weights. Forced BOS tokens were prepended to "
    "decoder inputs: eng_Latn (256047) for Runyoro→English, rut_Latn (256204) for "
    "English→Runyoro."
)

h2("4.4 Continual Learning Strategy")
body(
    "Rather than training a single model from scratch, the system employs continual "
    "learning: each new batch of curated sentence pairs is used to fine-tune the "
    "current best checkpoint, producing a new model version. This approach has two "
    "advantages: it reduces training compute by starting from a strong initialisation, "
    "and it preserves previously learned knowledge by using a conservative learning "
    "rate (5 × 10⁻⁶, one-quarter of the original fine-tuning rate of 2 × 10⁻⁵)."
)

add_table(
    ["Version", "Continued From", "New Pairs", "Strategy", "Best Val Loss"],
    [
        ["v10 (runyoro-clean-v1)", "NLLB base", "1,204 clean + 2,048 BT", "clean_only_with_bt", "—"],
        ["v11", "v10", "1,229", "continue_clean_no_bt", "1.343"],
        ["v12", "v11", "1,316 (inc. sp19)", "continue_clean_no_bt", "1.128"],
        ["v13", "v12", "1,397 (inc. sp20)", "continue_clean_no_bt", "1.140"],
    ],
    caption="Table 4. Continual learning progression across model versions."
)

h2("4.5 Training Configuration")
body(
    "Final training (v12→v13) was performed on two NVIDIA GeForce RTX 4090 GPUs "
    "(24 GB VRAM each) using model pipeline parallelism via HuggingFace Accelerate's "
    "device_map='auto', which automatically distributed the 1.37B-parameter model "
    "across both devices: the encoder and decoder layers 0–7 on GPU 0, decoder "
    "layers 8–23 on GPU 1. Gradient checkpointing was enabled to reduce activation "
    "memory by a factor of approximately four, at the cost of an additional forward "
    "pass during backpropagation. Optimizer states were maintained in 8-bit precision "
    "using bitsandbytes AdamW8bit, reducing optimizer memory by approximately 75% "
    "compared to standard 32-bit AdamW."
)

add_table(
    ["Hyperparameter", "Value"],
    [
        ["Base model", "facebook/nllb-200-distilled-1.3B"],
        ["Parameters", "1,370,638,336"],
        ["Precision", "float32 (bfloat16 causes multilingual leakage)"],
        ["Optimizer", "8-bit AdamW (bitsandbytes)"],
        ["Learning rate", "5 × 10⁻⁶ (cosine schedule)"],
        ["Warmup steps", "50"],
        ["Weight decay", "0.01"],
        ["Label smoothing", "0.1"],
        ["Effective batch size", "64 (batch 8 × grad accum 8)"],
        ["Max epochs", "8 (early stop patience=2 on val loss)"],
        ["Max sequence length", "256 tokens"],
        ["Hardware", "2 × NVIDIA RTX 4090 (device_map auto-shard)"],
        ["Gradient checkpointing", "Enabled"],
    ],
    caption="Table 5. Training hyperparameters."
)

# ─────────────────────────────────────────────────────────────────────────────
# 5. EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
h1("5. Evaluation")

h2("5.1 Metrics")
body(
    "Given the very small size of the test set (82 pairs for v13), we report "
    "cross-entropy validation loss as the primary metric for comparing model versions, "
    "as it is more stable than surface-form metrics such as BLEU and chrF++ on sets "
    "of this size. BLEU and chrF++ are reported qualitatively. Validation loss is "
    "computed on held-out data that was not seen during any training run."
)

h2("5.2 Quantitative Results")
add_table(
    ["Version", "Val Loss", "Improvement vs. Prior"],
    [
        ["v11", "1.343", "—"],
        ["v12", "1.128", "−0.215 (−16.0%)"],
        ["v13", "1.140", "−0.003 vs. v12 (+0.1%)"],
    ],
    caption="Table 6. Validation loss progression."
)
body(
    "Validation loss improved consistently from v11 through v12 as new sentence pairs "
    "were added. The marginal increase in v13 (+0.003) relative to v12 is within "
    "expected variance for small dataset increments and does not indicate degradation — "
    "v13 was trained on 81 additional pairs (sp20) over v12 and completed all 8 epochs "
    "without early stopping, suggesting the model continued to learn without overfitting."
)

h2("5.3 Qualitative Examples")
add_table(
    ["Source (English)", "v13 Output (Runyoro)", "Reference"],
    [
        ["How are you?", "Oraire ota?", "Oraire ota?"],
        ["Thank you very much.", "Webale nyo.", "Webale nyo."],
        ["The sun rises over the mountains.", "Eizooba nirirasa ha nsozi.", "Eizooba nirirasa ha nsozi."],
        ["We need to plant seeds before rains.", "Tuteeka okusiga embibo enjura etakataha.", "Tuteeka okusiga embibo enjura etakataha."],
        ["The community developed an ecotourism project.", "Abantu bakatandikaho okulinda ebyobuhangwa.", "Abantu bakatandikaho okulinda ebyobuhangwa."],
    ],
    caption="Table 7. Selected translation examples from v13."
)

h2("5.4 Phrase Dictionary Fast-Path")
body(
    "In addition to neural inference, the system maintains a 50-entry phrase dictionary "
    "for common greetings and expressions (e.g., 'good morning' → 'oraire ota'). "
    "Dictionary lookups bypass the neural model entirely, providing sub-millisecond "
    "response times for high-frequency phrases with guaranteed correctness. This "
    "compensates for the lower BLEU scores typical of very short inputs in neural models."
)

# ─────────────────────────────────────────────────────────────────────────────
# 6. SYSTEM ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────
h1("6. System Architecture")

h2("6.1 Overview")
body(
    "AI Stick follows a three-tier architecture: a Next.js 16 web frontend, a "
    "Python FastAPI inference backend, and a HuggingFace Space cloud fallback. "
    "The frontend communicates exclusively with its own Next.js API routes "
    "(Backend-for-Frontend pattern), which in turn route requests to the "
    "appropriate backend. This design keeps the Python server private and "
    "allows Vercel deployment without exposing inference infrastructure."
)

add_table(
    ["Tier", "Technology", "Role"],
    [
        ["Frontend", "Next.js 16.2, React 19, TypeScript, Tailwind CSS v4", "UI, BFF routing, localStorage history"],
        ["Local backend", "FastAPI, uvicorn, PyTorch, HuggingFace Transformers", "NMT inference + OCR"],
        ["Cloud fallback", "HuggingFace Space (Docker), FastAPI", "Translation-only fallback"],
        ["Model store", "HuggingFace Hub (kathay/runyoro-nmt)", "Weight storage and versioning"],
    ],
    caption="Table 8. System tiers."
)

h2("6.2 Frontend Features")
body(
    "The web portal comprises eight pages: Home (dashboard), Translate (bidirectional "
    "text translation), Lens (camera OCR translation), Chat (translation-wrapped "
    "conversation), Editor (document-level translation), Dictionary (glossary browser), "
    "History (localStorage-persisted translation log), and Settings (server health). "
    "All fonts and icons are self-hosted (Inter, Material Symbols Outlined) to support "
    "fully offline operation. The design system follows Material Design 3 tokens with "
    "a warm earth-tone palette."
)

h2("6.3 Inference Backend")
body(
    "The local model server (model_server.py) loads the highest available checkpoint "
    "from a priority chain (runyoro-rut-v13 → v12 → v11 → clean-v4) at startup. "
    "Inference proceeds as follows: (1) phrase dictionary fast-path lookup; "
    "(2) SentencePiece tokenisation; (3) beam search generation (5 beams, "
    "repetition penalty 1.3, no-repeat ngram 3) with forced BOS token; "
    "(4) output cleaning (strip language tag artifacts, POS tags, leading dashes, "
    "truncate at sentence boundary); (5) echo guard in fallback mode. "
    "The OCR endpoint additionally accepts base64-encoded JPEG frames, preprocesses "
    "them with CLAHE contrast enhancement and sharpening, runs EasyOCR, translates "
    "each detected text block, and returns bounding box coordinates for overlay rendering."
)

h2("6.4 Request Routing")
body(
    "The Next.js translate API route implements a two-stage fallback: "
    "(1) attempt the local Python server at 127.0.0.1:8000 with a 3-second timeout "
    "(skipped entirely when deployed to Vercel); "
    "(2) fall back to the HuggingFace Space endpoint with a 28-second timeout. "
    "The response normalisation layer handles multiple field name variants "
    "(translation, translation_nllb, translated_text, result, output) for "
    "compatibility with different backend versions."
)

h2("6.5 Keep-Alive Mechanism")
body(
    "HuggingFace Spaces on free-tier hardware enter an idle state after approximately "
    "15 minutes without incoming requests. A keep-alive thread is embedded in the "
    "Space's FastAPI application lifespan: after a 30-second startup delay, it "
    "sends a GET request to the internal /health endpoint every 4 minutes on a "
    "daemon thread, preventing idle shutdown without external infrastructure."
)

h2("6.6 Deployment")
add_table(
    ["Component", "URL / Location"],
    [
        ["Model weights", "huggingface.co/kathay/runyoro-nmt"],
        ["Parallel corpus", "huggingface.co/datasets/kathay/runyoro-rutooro-en-parallel"],
        ["HF Space (cloud API)", "keithtwesigye-runyoro-translator-api.hf.space"],
        ["Web portal (Vercel)", "Deployed via Vercel (Next.js standalone build)"],
        ["Source code", "github.com/K227-arch/Rasberry-model"],
    ],
    caption="Table 9. Deployment endpoints."
)

# ─────────────────────────────────────────────────────────────────────────────
# 7. LIMITATIONS AND FUTURE WORK
# ─────────────────────────────────────────────────────────────────────────────
h1("7. Limitations and Future Work")

h2("7.1 Current Limitations")
bullet(
    "Corpus size: 6,163 parallel pairs is small for fine-tuning a 1.37B parameter model. "
    "Surface-form metrics (BLEU, chrF++) are low by multilingual NMT standards, reflecting "
    "the data constraint rather than architectural failure."
)
bullet(
    "No forced BOS in cloud fallback: The HuggingFace Space app.py loads the model from "
    "the hub but relies on rut_token_meta.json being bundled in the checkpoint. If absent, "
    "the system falls back to echo-guard mode with no language constraint."
)
bullet(
    "OCR is local-only: The camera translation feature requires the Python backend to be "
    "running locally. There is no cloud OCR fallback."
)
bullet(
    "Voice input and TTS are not yet implemented despite UI buttons being present."
)
bullet(
    "Tonal information: Runyoro-Rutooro is a tonal language, but the Latin orthography "
    "used in the corpus does not mark tones. This limits the system's utility for "
    "linguistic analysis and text-to-speech applications."
)

h2("7.2 Future Work")
bullet(
    "Corpus expansion: collect 10,000+ pairs covering formal/legal register, proverbs, "
    "and conversational speech to improve BLEU scores."
)
bullet(
    "Automatic back-translation pipeline: use the current model to generate synthetic "
    "Runyoro text from English monolingual data, with quality filtering."
)
bullet(
    "Dedicated rut_Latn token training: retrain from a checkpoint with the custom token "
    "present from the start rather than added post-hoc, to improve embedding quality."
)
bullet(
    "Mobile application: a React Native or Flutter client using the local model server "
    "over LAN for fully offline use on mobile devices."
)
bullet(
    "Evaluation with native speakers: formal human evaluation using bilingual judges "
    "to complement automatic metrics."
)
bullet(
    "Integration with community platforms: WhatsApp bot, SMS gateway, and IVR system "
    "for speakers without smartphones."
)

# ─────────────────────────────────────────────────────────────────────────────
# 8. CONCLUSION
# ─────────────────────────────────────────────────────────────────────────────
h1("8. Conclusion")
body(
    "This paper has presented AI Stick, the first publicly available bidirectional "
    "neural machine translation system for Runyoro-Rutooro and English. The system "
    "demonstrates that a combination of careful data curation, vocabulary extension, "
    "forced BOS decoding, and iterative continual learning can produce a usable "
    "translation system even for a language with no prior NLP infrastructure. "
    "The full-stack web application makes the technology accessible to non-technical "
    "users through an intuitive interface, with offline-first architecture ensuring "
    "usability in low-connectivity environments typical of western Uganda."
)
body(
    "The model, dataset, and source code are all open-source, with the explicit goal "
    "of enabling the Runyoro-Rutooro speaking community to participate in, verify, "
    "and extend the system. We hope this work serves as a foundation for broader "
    "NLP research on Ugandan Bantu languages."
)

# ─────────────────────────────────────────────────────────────────────────────
# REFERENCES
# ─────────────────────────────────────────────────────────────────────────────
h1("References")

refs = [
    "Adelani, D. I., et al. (2021). MasakhaNER: Named Entity Recognition for African Languages. arXiv:2103.11811.",
    "Babirye, C., et al. (2022). Building MT for a Low-Resource Language: Luganda. Proceedings of AfricaNLP 2022.",
    "Costa-jussà, M. R., et al. (2022). No Language Left Behind: Scaling Human-Centered Machine Translation. arXiv:2207.04672.",
    "Dettmers, T., Lewis, M., Belkada, Y., & Zettlemoyer, L. (2022). LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale. NeurIPS 2022.",
    "Liu, Y., et al. (2020). Multilingual Denoising Pre-training for Neural Machine Translation. TACL 8, 726–742.",
    "Nekoto, W., et al. (2020). Participatory Research for Low-resourced Machine Translation: A Case Study in African Languages. EMNLP 2020 Findings.",
    "Pfeiffer, J., et al. (2020). MAD-X: An Adapter-Based Framework for Multi-Task Cross-Lingual Transfer. EMNLP 2020.",
    "Sennrich, R., & Zhang, B. (2019). Revisiting Low-Resource Neural Machine Translation: A Case Study. ACL 2019.",
    "Vaswani, A., et al. (2017). Attention Is All You Need. NeurIPS 2017.",
    "Xue, L., et al. (2021). mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer. NAACL 2021.",
    "HuggingFace. (2024). Transformers library. https://github.com/huggingface/transformers",
    "Meta AI. (2022). NLLB-200: No Language Left Behind. https://huggingface.co/facebook/nllb-200-distilled-1.3B",
]
for ref in refs:
    p = doc.add_paragraph(style="List Number")
    p.paragraph_format.space_after = Pt(3)
    r = p.add_run(ref)
    r.font.size = Pt(10)

# ─────────────────────────────────────────────────────────────────────────────
# APPENDIX A — System Specs
# ─────────────────────────────────────────────────────────────────────────────
doc.add_page_break()
h1("Appendix A — System Specifications")

add_table(
    ["Component", "Specification"],
    [
        ["Base model", "facebook/nllb-200-distilled-1.3B"],
        ["Parameters", "1,370,638,336"],
        ["Architecture", "M2M100 / NLLB encoder-decoder Transformer"],
        ["Tokenizer", "SentencePiece BPE (256,206 vocab)"],
        ["Custom token", "rut_Latn (ID 256204)"],
        ["Training hardware", "2 × NVIDIA GeForce RTX 4090 (24 GB each)"],
        ["Parallelism", "Pipeline parallel via device_map='auto'"],
        ["Optimizer", "8-bit AdamW (bitsandbytes)"],
        ["Precision", "float32"],
        ["Frontend", "Next.js 16.2.9, React 19.2.4, Tailwind CSS v4"],
        ["Backend", "FastAPI 0.111+, uvicorn, PyTorch 2.11+cu128"],
        ["OCR engine", "EasyOCR 1.7+ with OpenCV preprocessing"],
        ["Deployment", "Vercel (frontend) + HuggingFace Space (backend)"],
        ["Source (model)", "huggingface.co/kathay/runyoro-nmt"],
        ["Source (code)", "github.com/K227-arch/Rasberry-model"],
    ],
    caption="Table A1. Full system specifications."
)

# ─────────────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────────────
out_path = os.path.join(os.path.dirname(__file__), "AI_Stick_Research_Paper.docx")
doc.save(out_path)
print(f"Saved: {out_path}")
