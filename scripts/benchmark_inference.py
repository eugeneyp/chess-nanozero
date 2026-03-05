#!/usr/bin/env python3
"""Benchmark PyTorch vs FP32 ONNX vs INT8 ONNX inference speed.

Usage:
    python scripts/benchmark_inference.py \
        --checkpoint checkpoints/step4/epoch_0024.pt \
        --config configs/medium.yaml \
        --onnx-fp32 models/medium1.onnx \
        --onnx-int8 models/medium1_int8.onnx
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import yaml

TIME_BUDGET_S = 2.25  # seconds (tc=60+1 at start of game)
WARMUP = 10
N = 200


def bench(fn, board_t):
    for _ in range(WARMUP):
        fn(board_t)
    t0 = time.perf_counter()
    for _ in range(N):
        fn(board_t)
    return (time.perf_counter() - t0) / N * 1000  # ms/call


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--onnx-fp32", required=True, type=Path)
    parser.add_argument("--onnx-int8", required=True, type=Path)
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))
    board_t = torch.zeros(1, 18, 8, 8)

    from src.agents.alphazero_agent import AlphaZeroAgent
    from src.neural_net.onnx_model import OnnxModel

    agent = AlphaZeroAgent.from_checkpoint(args.checkpoint, config, "cpu")
    torch.set_num_threads(1)

    results = [
        ("PyTorch", lambda b: agent.mcts.model(b)),
        ("FP32 ONNX", OnnxModel(args.onnx_fp32)),
        ("INT8 ONNX", OnnxModel(args.onnx_int8)),
    ]

    pt_ms = None
    print(f"\n{'Mode':<15} {'ms/call':>8} {'sims/2.25s':>12} {'vs PyTorch':>12}")
    print("-" * 52)
    for name, fn in results:
        with torch.no_grad():
            ms = bench(fn, board_t)
        sims = int(TIME_BUDGET_S * 1000 / ms)
        ratio = (pt_ms / ms) if pt_ms else 1.0
        if pt_ms is None:
            pt_ms = ms
        print(f"{name:<15} {ms:>7.2f}ms {sims:>12d} {ratio:>11.2f}x")


if __name__ == "__main__":
    main()
