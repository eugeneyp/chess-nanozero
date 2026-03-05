#!/usr/bin/env python3
"""UCI entry point for chess-nanozero.

Usage:
    python scripts/play_uci.py [--config configs/medium.yaml]
                               [--checkpoint checkpoints/step4/epoch_0024.pt]
                               [--device cpu]
                               [--num-simulations 100]
"""

import argparse
import os
import sys

# Ensure stdout is unbuffered so bestmove reaches fastchess immediately.
# When Python runs as a subprocess with piped stdout, it may default to
# block-buffering even with print(..., flush=True).
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.uci.uci_engine import UCIEngine, run_uci_loop

parser = argparse.ArgumentParser(description="chess-nanozero UCI engine")
parser.add_argument("--config", default="configs/medium.yaml")
parser.add_argument("--checkpoint", default="checkpoints/step4/epoch_0024.pt")
parser.add_argument("--device", default="cpu")
parser.add_argument("--num-simulations", type=int, default=None,
                    help="Override mcts.num_simulations from config")
parser.add_argument("--onnx-model", default=None,
                    help="Path to .onnx model file; uses ONNX Runtime instead of PyTorch")
args = parser.parse_args()

engine = UCIEngine(args.config, args.checkpoint, args.device, args.num_simulations,
                   onnx_path=args.onnx_model)
run_uci_loop(engine)
