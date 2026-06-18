"""
ModelQuantizer
==============
Optimises the trained model for edge deployment (Raspberry Pi):

  - INT8 dynamic quantization (PyTorch native)
  - ONNX export for cross-platform inference
  - CTranslate2 conversion for fast CPU inference
  - Model size and speed benchmarking

Outputs deployment-ready model artifacts.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class ModelQuantizer:
    """Convert and quantize a fine-tuned NMT model for edge deployment."""

    def __init__(self, model_path: str, output_dir: str):
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # INT8 dynamic quantization (PyTorch)
    # ------------------------------------------------------------------
    def quantize_int8_pytorch(self) -> Path:
        """
        Apply dynamic INT8 quantization to the encoder and decoder.
        Reduces model size ~2-4x with minimal quality loss.
        """
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore

        logger.info("Loading model for INT8 quantization: %s", self.model_path)
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
        model.eval()

        # Dynamic quantization of linear layers
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8,
        )

        out_path = self.output_dir / "runyoro-nmt-v1-int8"
        out_path.mkdir(parents=True, exist_ok=True)
        quantized_model.save_pretrained(str(out_path))
        tokenizer.save_pretrained(str(out_path))

        # Compare sizes
        orig_size = self._model_size_mb(self.model_path)
        quant_size = self._model_size_mb(str(out_path))
        logger.info(
            "INT8 quantization complete: %.1f MB → %.1f MB (%.1f%% reduction)",
            orig_size, quant_size,
            100 * (1 - quant_size / max(orig_size, 1)),
        )
        return out_path

    # ------------------------------------------------------------------
    # ONNX export
    # ------------------------------------------------------------------
    def export_onnx(self) -> Path:
        """Export the model to ONNX format for cross-runtime inference."""
        try:
            from optimum.onnxruntime import ORTModelForSeq2SeqLM  # type: ignore
            from transformers import AutoTokenizer  # type: ignore

            logger.info("Exporting model to ONNX...")
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            ort_model = ORTModelForSeq2SeqLM.from_pretrained(
                self.model_path, export=True
            )
            out_path = self.output_dir / "runyoro-nmt-v1-onnx"
            ort_model.save_pretrained(str(out_path))
            tokenizer.save_pretrained(str(out_path))
            logger.info("ONNX export complete: %s", out_path)
            return out_path

        except ImportError:
            logger.error(
                "optimum not installed. Run: pip install optimum[onnxruntime]"
            )
            raise

    # ------------------------------------------------------------------
    # CTranslate2 conversion (fastest CPU inference)
    # ------------------------------------------------------------------
    def convert_ctranslate2(self, quantization: str = "int8") -> Path:
        """
        Convert to CTranslate2 format — typically 4-8x faster than PyTorch on CPU.
        Ideal for Raspberry Pi deployment.
        """
        try:
            import ctranslate2  # type: ignore

            out_path = self.output_dir / f"runyoro-nmt-v1-ct2-{quantization}"
            logger.info("Converting to CTranslate2 (%s)...", quantization)

            ctranslate2.converters.OpusMTConverter(self.model_path).convert(
                str(out_path),
                quantization=quantization,
                force=True,
            )
            logger.info("CTranslate2 conversion complete: %s", out_path)
            return out_path

        except ImportError:
            logger.error("ctranslate2 not installed. Run: pip install ctranslate2")
            raise
        except Exception as e:
            # Fall back to the generic HF converter
            logger.warning("OpusMT converter failed (%s) — trying HF converter", e)
            converter = ctranslate2.converters.OpusMTConverter(self.model_path)
            out_path = self.output_dir / f"runyoro-nmt-v1-ct2-{quantization}"
            converter.convert(str(out_path), quantization=quantization, force=True)
            return out_path

    # ------------------------------------------------------------------
    # Speed benchmark
    # ------------------------------------------------------------------
    def benchmark(
        self,
        test_sentences: list,
        model_path: Optional[str] = None,
        n_runs: int = 10,
    ) -> Dict:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore
        import torch

        mp = model_path or self.model_path
        tokenizer = AutoTokenizer.from_pretrained(mp)
        model = AutoModelForSeq2SeqLM.from_pretrained(mp)
        model.eval()

        tokenizer.src_lang = "nyk_Latn"
        forced_bos_id = tokenizer.lang_code_to_id["eng_Latn"]

        latencies = []
        for _ in range(n_runs):
            text = test_sentences[_ % len(test_sentences)]
            enc = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
            start = time.perf_counter()
            with torch.no_grad():
                model.generate(
                    **enc,
                    forced_bos_token_id=forced_bos_id,
                    num_beams=4,
                    max_length=128,
                )
            latencies.append(time.perf_counter() - start)

        avg = sum(latencies) / len(latencies)
        return {
            "model": mp,
            "n_runs": n_runs,
            "avg_latency_s": round(avg, 3),
            "min_latency_s": round(min(latencies), 3),
            "max_latency_s": round(max(latencies), 3),
            "throughput_sentences_per_sec": round(1 / avg, 2),
        }

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------
    @staticmethod
    def _model_size_mb(path: str) -> float:
        total = sum(
            f.stat().st_size
            for f in Path(path).rglob("*")
            if f.is_file()
        )
        return total / (1024 * 1024)
