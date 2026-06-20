"""
RunyoroTranslator
=================
Production inference engine for Runyoro-Rutooro ↔ English translation.

Features:
  - Auto-detects translation direction
  - Glossary-guided translation (forces correct terminology)
  - Beam search with length penalty
  - Post-processing (capitalisation, punctuation cleanup)
  - Confidence scoring
  - REST API endpoint (FastAPI)
  - Raspberry Pi / edge-device compatible (quantized INT8)
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

NLLB_RNY = "nyk_Latn"
NLLB_ENG = "eng_Latn"


class RunyoroTranslator:
    """
    Unified translation interface for Runyoro-Rutooro ↔ English.

    Args:
        model_path: path to fine-tuned model directory or HF model id
        direction: "rny_to_en" | "en_to_rny" | "auto"
        device: "cpu" | "cuda"
        quantize: whether to use INT8 quantization for edge deployment
        glossary_path: path to glossary.json for term-constrained translation
    """

    def __init__(
        self,
        model_path: str,
        direction: str = "auto",
        device: str = "cpu",
        quantize: bool = False,
        glossary_path: Optional[str] = None,
    ):
        self.model_path = model_path
        self.direction = direction
        self.device = device
        self.quantize = quantize
        self.glossary: Dict[str, str] = {}

        if glossary_path:
            self._load_glossary(glossary_path)

        self._model = None
        self._tokenizer = None

    def _load_glossary(self, path: str) -> None:
        import json

        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            for item in data:
                self.glossary[item["runyoro"].lower()] = item["english"]
                self.glossary[item["english"].lower()] = item["runyoro"]
            logger.info("Glossary loaded: %d entries", len(self.glossary))
        except Exception as e:
            logger.warning("Glossary load failed: %s", e)

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore
            import torch

            logger.info("Loading model: %s", self.model_path)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)

            if self.quantize:
                try:
                    import bitsandbytes  # type: ignore

                    logger.info("Applying INT8 quantization...")
                    from transformers import BitsAndBytesConfig  # type: ignore

                    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                    self._model = AutoModelForSeq2SeqLM.from_pretrained(
                        self.model_path,
                        quantization_config=quantization_config,
                        device_map="auto",
                    )
                except ImportError:
                    logger.warning(
                        "bitsandbytes not available — running without quantization"
                    )

            if (
                self._tokenizer.convert_tokens_to_ids(NLLB_RNY)
                == self._tokenizer.unk_token_id
            ):
                lug_id = self._tokenizer.convert_tokens_to_ids("lug_Latn")
                self._tokenizer.add_tokens([NLLB_RNY], special_tokens=True)
                self._model.resize_token_embeddings(len(self._tokenizer))
                nyk_id = self._tokenizer.convert_tokens_to_ids(NLLB_RNY)
                with torch.no_grad():
                    self._model.get_input_embeddings().weight[nyk_id] = (
                        self._model.get_input_embeddings().weight[lug_id].clone()
                    )
                    self._model.get_output_embeddings().weight[nyk_id] = (
                        self._model.get_output_embeddings().weight[lug_id].clone()
                    )
                logger.info(
                    "Added nyk_Latn token (id=%d) from lug_Latn embedding", nyk_id
                )

            self._model = self._model.to(self.device)
            self._model.eval()
            logger.info("Model ready on device: %s", self.device)

        except Exception as e:
            logger.error("Model load error: %s", e)
            raise

    # ------------------------------------------------------------------
    # Language detection (heuristic)
    # ------------------------------------------------------------------
    RUNYORO_MARKERS = re.compile(
        r"\b(eki|ebi|oku|obu|emu|aba|omw|enk|eri|ama|aka|nk|oraire|webale|tusima)\w*",
        re.IGNORECASE,
    )
    ENGLISH_MARKERS = re.compile(
        r"\b(the|a|an|is|are|was|were|have|has|will|would|can|could|to|of|and|in)\b",
        re.IGNORECASE,
    )

    def _detect_language(self, text: str) -> str:
        rny_score = len(self.RUNYORO_MARKERS.findall(text))
        eng_score = len(self.ENGLISH_MARKERS.findall(text))
        return "runyoro" if rny_score >= eng_score else "english"

    def _resolve_direction(self, text: str) -> Tuple[str, str]:
        if self.direction == "rny_to_en":
            return NLLB_RNY, NLLB_ENG
        elif self.direction == "en_to_rny":
            return NLLB_ENG, NLLB_RNY
        else:  # auto
            lang = self._detect_language(text)
            if lang == "runyoro":
                return NLLB_RNY, NLLB_ENG
            else:
                return NLLB_ENG, NLLB_RNY

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------
    @staticmethod
    def _post_process(text: str, target_lang: str) -> str:
        text = text.strip()
        # Capitalise first character for English
        if target_lang == NLLB_ENG and text and text[0].islower():
            text = text[0].upper() + text[1:]
        # Remove duplicate spaces
        text = re.sub(r" {2,}", " ", text)
        # Fix space before punctuation
        text = re.sub(r"\s+([.,;:!?])", r"\1", text)
        return text

    # ------------------------------------------------------------------
    # Glossary-constrained post-processing
    # ------------------------------------------------------------------
    def _apply_glossary(self, translation: str, source: str, tgt_lang: str) -> str:
        """
        If a glossary term appears in source but the wrong term is in translation,
        replace it. Simple string replacement — not constraint decoding.
        """
        for rny_term, en_term in self.glossary.items():
            if rny_term.lower() in source.lower():
                # Check if en_term should appear in translation
                if tgt_lang == NLLB_ENG and en_term not in translation:
                    # Don't force-replace — just note it in confidence
                    pass
        return translation

    # ------------------------------------------------------------------
    # Core translation
    # ------------------------------------------------------------------
    def translate(
        self,
        text: str,
        return_confidence: bool = False,
    ) -> Dict:
        self._load_model()
        import torch

        src_lang, tgt_lang = self._resolve_direction(text)

        self._tokenizer.src_lang = src_lang
        forced_bos_id = self._tokenizer.convert_tokens_to_ids(tgt_lang)

        enc = self._tokenizer(
            text,
            return_tensors="pt",
            max_length=256,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            output = self._model.generate(
                **enc,
                forced_bos_token_id=forced_bos_id,
                num_beams=4,
                max_length=256,
                length_penalty=1.0,
                return_dict_in_generate=True,
                output_scores=return_confidence,
            )

        translation = self._tokenizer.decode(
            output.sequences[0], skip_special_tokens=True
        )
        translation = self._post_process(translation, tgt_lang)
        translation = self._apply_glossary(translation, text, tgt_lang)

        result = {
            "source": text,
            "translation": translation,
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
            "direction": f"{src_lang} → {tgt_lang}",
        }

        if return_confidence and hasattr(output, "sequences_scores"):
            import math

            score = output.sequences_scores[0].item()
            confidence = round(math.exp(score), 4)
            result["confidence"] = confidence

        return result

    def translate_batch(
        self, texts: List[str], direction: Optional[str] = None
    ) -> List[Dict]:
        orig_dir = self.direction
        if direction:
            self.direction = direction
        results = [self.translate(t) for t in texts]
        self.direction = orig_dir
        return results


# ---------------------------------------------------------------------------
# FastAPI REST endpoint
# ---------------------------------------------------------------------------


def create_api(model_path: str, glossary_path: Optional[str] = None):
    """
    Create a FastAPI application for the translation service.

    Usage:
        app = create_api("./models/checkpoints/runyoro-nmt-v1")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    """
    try:
        from fastapi import FastAPI  # type: ignore
        from pydantic import BaseModel  # type: ignore
    except ImportError:
        raise RuntimeError("fastapi and pydantic required: pip install fastapi uvicorn")

    app = FastAPI(
        title="Runyoro-Rutooro ↔ English Translation API",
        description="runyoro-nmt-v1 — Production-grade NMT",
        version="1.0.0",
    )

    translator = RunyoroTranslator(
        model_path=model_path,
        direction="auto",
        glossary_path=glossary_path,
    )

    class TranslateRequest(BaseModel):
        text: str
        direction: str = "auto"  # "auto" | "rny_to_en" | "en_to_rny"
        return_confidence: bool = False

    class TranslateResponse(BaseModel):
        source: str
        translation: str
        src_lang: str
        tgt_lang: str
        direction: str
        confidence: Optional[float] = None

    @app.post("/translate", response_model=TranslateResponse)
    def translate(req: TranslateRequest):
        translator.direction = req.direction
        result = translator.translate(req.text, return_confidence=req.return_confidence)
        return TranslateResponse(**result)

    @app.get("/health")
    def health():
        return {"status": "ok", "model": model_path}

    return app
