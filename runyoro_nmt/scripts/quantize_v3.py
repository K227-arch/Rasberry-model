#!/usr/bin/env python3
"""
quantize_v3.py - Run ONNX INT8 quantization on the v3 model
"""
import os
import sys
import logging

os.environ["USE_TF"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

from inference.quantizer import ModelQuantizer

MODEL_PATH = str(ROOT / "models" / "checkpoints" / "runyoro-nmt-v3")
OUTPUT_DIR = str(ROOT / "models" / "exported")

print(f"Model: {MODEL_PATH}")
print(f"Output: {OUTPUT_DIR}")
print()

quantizer = ModelQuantizer(model_path=MODEL_PATH, output_dir=OUTPUT_DIR)

# Run full ONNX INT8 pipeline
print("=" * 60)
print("Running ONNX INT8 Quantization Pipeline")
print("=" * 60)
result_path = quantizer.quantize_onnx_int8()

print(f"\nQuantized model saved to: {result_path}")
print("\nTesting inference...")

# Quick test
translation = quantizer.test_onnx_int8("I am going to school")
print(f"  EN -> RNY: 'I am going to school' → '{translation}'")

translation2 = quantizer.test_onnx_int8("Ndi kugenda ku isomero")
print(f"  RNY -> EN: 'Ndi kugenda ku isomero' → '{translation2}'")

print("\nDone!")
