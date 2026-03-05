#!/usr/bin/env python3
"""Quantize a FP32 ONNX model to INT8 using dynamic quantization.

Usage:
    python scripts/quantize_onnx.py \
        --input models/medium1.onnx \
        --output models/medium1_int8.onnx
"""
import argparse
from pathlib import Path
import numpy as np
import onnxruntime
from onnxruntime.quantization import quantize_dynamic, QuantType


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Limit to MatMul/Gemm only: ConvInteger is not implemented in
    # onnxruntime CPUExecutionProvider on macOS (would need QLinearConv
    # from static quantization + calibration data). FC layers still benefit.
    quantize_dynamic(
        str(args.input),
        str(args.output),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm"],
    )
    size_in = args.input.stat().st_size / 1e6
    size_out = args.output.stat().st_size / 1e6
    print(f"Quantized: {args.input.name} ({size_in:.1f} MB) → {args.output.name} ({size_out:.1f} MB)")
    print("Note: Conv ops not quantized (ConvInteger unsupported on CPUExecutionProvider/macOS)")

    # Verify output shapes unchanged
    sess = onnxruntime.InferenceSession(str(args.output))
    dummy = np.zeros((1, 18, 8, 8), dtype=np.float32)
    policy, value = sess.run(None, {"board": dummy})
    assert policy.shape == (1, 4672), f"Unexpected policy shape: {policy.shape}"
    assert value.shape == (1, 1), f"Unexpected value shape: {value.shape}"
    print(f"Verified: policy={policy.shape}, value={value.shape}")


if __name__ == "__main__":
    main()
