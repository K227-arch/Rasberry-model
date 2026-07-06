#!/usr/bin/env python3
"""
fix_int8_quantization.py - Quantize the decoder_with_past_model and test inference
"""
import os
os.environ["USE_TF"] = "0"

from pathlib import Path
from onnxruntime.quantization import quantize_dynamic, QuantType

ROOT = Path(__file__).parent.parent
ONNX_DIR = ROOT / "models" / "exported" / "runyoro-nmt-v2-onnx"
INT8_DIR = ROOT / "models" / "exported" / "runyoro-nmt-v2-onnx-int8"

# Quantize decoder_with_past_model
src = ONNX_DIR / "decoder_with_past_model.onnx"
dst = INT8_DIR / "decoder_with_past_model.onnx"

if src.exists() and not dst.exists():
    print(f"Quantizing decoder_with_past_model.onnx ({src.stat().st_size / 1e6:.0f}MB)...")
    quantize_dynamic(
        str(src),
        str(dst),
        weight_type=QuantType.QInt8,
    )
    print(f"Done! Output: {dst} ({dst.stat().st_size / 1e6:.0f}MB)")
else:
    if dst.exists():
        print("decoder_with_past_model.onnx already quantized")
    else:
        print(f"Source not found: {src}")

# Test inference
print("\nTesting ONNX INT8 inference...")
try:
    from optimum.onnxruntime import ORTModelForSeq2SeqLM
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(INT8_DIR))
    model = ORTModelForSeq2SeqLM.from_pretrained(str(INT8_DIR))

    # Test EN -> RNY
    inputs = tokenizer("I am going to school", return_tensors="pt", max_length=256, truncation=True)
    outputs = model.generate(**inputs, num_beams=4, max_length=256)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"  EN -> RNY: 'I am going to school' → '{result}'")

    # Test RNY -> EN
    inputs = tokenizer("Ndi kugenda ku isomero", return_tensors="pt", max_length=256, truncation=True)
    outputs = model.generate(**inputs, num_beams=4, max_length=256)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"  RNY -> EN: 'Ndi kugenda ku isomero' → '{result}'")

    print("\nONNX INT8 inference working!")
except Exception as e:
    print(f"  Inference error: {e}")

# Report sizes
print("\n--- Model Size Comparison ---")
orig_size = sum(f.stat().st_size for f in (ROOT / "models" / "checkpoints" / "runyoro-nmt-v3").rglob("*") if f.is_file() and "checkpoint-" not in str(f))
int8_size = sum(f.stat().st_size for f in INT8_DIR.rglob("*") if f.is_file())
print(f"  Original FP32:  {orig_size / 1e9:.2f} GB")
print(f"  ONNX INT8:      {int8_size / 1e9:.2f} GB")
print(f"  Reduction:      {100 * (1 - int8_size / orig_size):.1f}%")
