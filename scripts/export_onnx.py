"""Export a trained ChessResNet checkpoint to ONNX format.

Usage:
    python scripts/export_onnx.py \
        --checkpoint checkpoints/step4/epoch_0024.pt \
        --config configs/medium.yaml \
        --output models/medium1.onnx
"""

import argparse
from pathlib import Path

import numpy as np
import onnxruntime
import torch
import yaml

from src.agents.alphazero_agent import AlphaZeroAgent


def main() -> None:
    parser = argparse.ArgumentParser(description="Export checkpoint to ONNX")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--output", required=True, help="Output .onnx path")
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))
    agent = AlphaZeroAgent.from_checkpoint(Path(args.checkpoint), config, device="cpu")
    model = agent.mcts.model
    model.eval()

    dummy_input = torch.zeros(1, 18, 8, 8)
    output_path = args.output
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=14,
        input_names=["board"],
        output_names=["policy", "value"],
        dynamic_axes={
            "board":  {0: "batch"},
            "policy": {0: "batch"},
            "value":  {0: "batch"},
        },
    )
    print(f"Exported to {output_path}")

    # Verify with onnxruntime
    sess = onnxruntime.InferenceSession(output_path)
    dummy_np = np.zeros((1, 18, 8, 8), dtype=np.float32)
    policy_out, value_out = sess.run(None, {"board": dummy_np})
    assert policy_out.shape == (1, 4672), f"Unexpected policy shape: {policy_out.shape}"
    assert value_out.shape == (1, 1), f"Unexpected value shape: {value_out.shape}"
    print(f"Verified: policy={policy_out.shape}, value={value_out.shape}")


if __name__ == "__main__":
    main()
