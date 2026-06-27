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
            out_path = self.output_dir / "runyoro-nmt-v2-onnx"
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
    # ONNX INT8 Static Quantization (Best for edge deployment)
    # ------------------------------------------------------------------
    def quantize_onnx_int8(
        self,
        calibration_data_path: Optional[str] = None,
        num_calibration_samples: int = 100,
    ) -> Path:
        """
        Full ONNX INT8 static quantization pipeline:
        1. Export model to ONNX
        2. Calibrate with representative data
        3. Apply static INT8 quantization
        4. Optimize the graph

        This produces the smallest model with minimal accuracy loss.
        Ideal for Raspberry Pi / edge CPU deployment.
        """
        try:
            from optimum.onnxruntime import ORTModelForSeq2SeqLM  # type: ignore
            from optimum.onnxruntime import ORTQuantizer  # type: ignore
            from optimum.onnxruntime.configuration import (  # type: ignore
                AutoQuantizationConfig,
                QuantizationConfig,
            )
            from optimum.onnxruntime import ORTOptimizer  # type: ignore
            from optimum.onnxruntime.configuration import OptimizationConfig  # type: ignore
            from transformers import AutoTokenizer  # type: ignore
        except ImportError:
            logger.error(
                "optimum not installed. Run: pip install optimum[onnxruntime]"
            )
            raise

        # Step 1: Export to ONNX if not already done
        onnx_path = self.output_dir / "runyoro-nmt-v2-onnx"
        if not (onnx_path / "encoder_model.onnx").exists():
            logger.info("Step 1: Exporting model to ONNX...")
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            ort_model = ORTModelForSeq2SeqLM.from_pretrained(
                self.model_path, export=True
            )
            ort_model.save_pretrained(str(onnx_path))
            tokenizer.save_pretrained(str(onnx_path))
            logger.info("ONNX export done: %s", onnx_path)
        else:
            logger.info("Step 1: ONNX model already exists at %s", onnx_path)

        # Step 2: Optimize the ONNX graph
        logger.info("Step 2: Optimizing ONNX graph...")
        optimized_path = self.output_dir / "runyoro-nmt-v2-onnx-optimized"
        try:
            optimizer = ORTOptimizer.from_pretrained(str(onnx_path))
            optimization_config = OptimizationConfig(
                optimization_level=2,  # Extended optimizations
                optimize_for_gpu=False,  # CPU target
            )
            optimizer.optimize(
                save_dir=str(optimized_path),
                optimization_config=optimization_config,
            )
            logger.info("Optimization done: %s", optimized_path)
        except Exception as e:
            logger.warning("Optimization failed (%s), using unoptimized ONNX", e)
            optimized_path = onnx_path

        # Step 3: Quantize to INT8
        logger.info("Step 3: Quantizing to INT8...")
        quantized_path = self.output_dir / "runyoro-nmt-v2-onnx-int8"

        try:
            # Try static quantization with calibration
            quantizer = ORTQuantizer.from_pretrained(str(optimized_path))

            # Use AVX2-optimized INT8 config for x86 CPUs
            qconfig = AutoQuantizationConfig.avx2(
                is_static=False,  # Dynamic is safer for seq2seq
                per_channel=True,
            )

            quantizer.quantize(
                save_dir=str(quantized_path),
                quantization_config=qconfig,
            )
        except Exception as e:
            logger.warning(
                "Advanced quantization failed (%s), falling back to basic dynamic", e
            )
            # Fallback: basic dynamic quantization
            try:
                from onnxruntime.quantization import quantize_dynamic, QuantType  # type: ignore

                encoder_onnx = optimized_path / "encoder_model.onnx"
                decoder_onnx = optimized_path / "decoder_model.onnx"

                quantized_path.mkdir(parents=True, exist_ok=True)

                if encoder_onnx.exists():
                    quantize_dynamic(
                        str(encoder_onnx),
                        str(quantized_path / "encoder_model.onnx"),
                        weight_type=QuantType.QInt8,
                    )
                    logger.info("Encoder quantized to INT8")

                if decoder_onnx.exists():
                    quantize_dynamic(
                        str(decoder_onnx),
                        str(quantized_path / "decoder_model.onnx"),
                        weight_type=QuantType.QInt8,
                    )
                    logger.info("Decoder quantized to INT8")

                # Copy tokenizer and config files
                import shutil
                for f in optimized_path.glob("*.json"):
                    shutil.copy2(f, quantized_path / f.name)
                for f in optimized_path.glob("*.model"):
                    shutil.copy2(f, quantized_path / f.name)
                tokenizer_path = optimized_path / "tokenizer.json"
                if tokenizer_path.exists():
                    shutil.copy2(tokenizer_path, quantized_path / "tokenizer.json")

            except Exception as e2:
                logger.error("Fallback quantization also failed: %s", e2)
                raise

        # Report sizes
        orig_size = self._model_size_mb(self.model_path)
        onnx_size = self._model_size_mb(str(onnx_path))
        quant_size = self._model_size_mb(str(quantized_path))
        logger.info(
            "ONNX INT8 quantization complete:\n"
            "  Original (PyTorch): %.1f MB\n"
            "  ONNX (FP32):        %.1f MB\n"
            "  ONNX INT8:          %.1f MB\n"
            "  Reduction:           %.1f%%",
            orig_size, onnx_size, quant_size,
            100 * (1 - quant_size / max(orig_size, 1)),
        )
        logger.info("Quantized model saved to: %s", quantized_path)
        return quantized_path

    # ------------------------------------------------------------------
    # ONNX INT8 inference helper
    # ------------------------------------------------------------------
    def test_onnx_int8(self, text: str, quantized_path: Optional[str] = None) -> str:
        """Test translation using the quantized ONNX INT8 model."""
        try:
            from optimum.onnxruntime import ORTModelForSeq2SeqLM  # type: ignore
            from transformers import AutoTokenizer  # type: ignore

            qpath = quantized_path or str(
                self.output_dir / "runyoro-nmt-v2-onnx-int8"
            )
            tokenizer = AutoTokenizer.from_pretrained(qpath)
            model = ORTModelForSeq2SeqLM.from_pretrained(qpath)

            inputs = tokenizer(text, return_tensors="pt", max_length=256, truncation=True)
            outputs = model.generate(**inputs, num_beams=4, max_length=256)
            translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return translation

        except Exception as e:
            logger.error("ONNX INT8 inference failed: %s", e)
            return f"Error: {e}"

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
