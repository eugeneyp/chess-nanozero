"""OnnxModel: drop-in replacement for ChessResNet using ONNX Runtime.

Wraps an onnxruntime.InferenceSession to match the ChessResNet(board_t) interface
so MCTS can run inference via ONNX without any changes to MCTS code.

Typical speedup vs raw PyTorch on CPU: 3-7x (fp32), up to 10x with INT8.
"""

from __future__ import annotations

import ctypes

import numpy as np
import torch
import onnxruntime


class OnnxModel:
    """ONNX Runtime inference session matching the ChessResNet call interface.

    Usage:
        model = OnnxModel("models/medium1.onnx")
        policy_logits, value = model(board_tensor)  # same as ChessResNet
    """

    def __init__(self, onnx_path: str, device: str = "cpu") -> None:
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )
        self.session = onnxruntime.InferenceSession(str(onnx_path), providers=providers)
        self._device = device

    def __call__(self, board_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run inference.

        Args:
            board_t: (B, 18, 8, 8) float32 torch tensor

        Returns:
            policy_logits: (B, 4672) torch tensor
            value:         (B, 1)    torch tensor
        """
        # torch.Tensor.numpy() is broken with numpy 2.x / torch 2.2.x (C-API mismatch).
        # ctypes.memmove: single C-level copy, no Python list/bytes object overhead.
        # Bypasses torch.Tensor.numpy() which fails with numpy 2.x / torch 2.2.x C-API.
        board_contig = board_t.contiguous().cpu()
        board_np = np.empty(tuple(board_contig.shape), dtype=np.float32)
        ctypes.memmove(
            board_np.ctypes.data,       # destination: numpy data pointer
            board_contig.data_ptr(),    # source: tensor raw data pointer
            board_contig.numel() * 4,   # bytes: n_elements × sizeof(float32)
        )

        policy_np, value_np = self.session.run(None, {"board": board_np})

        # Same frombuffer pattern used throughout the codebase for numpy→torch
        policy_t = torch.frombuffer(
            policy_np.tobytes(), dtype=torch.float32
        ).reshape(policy_np.shape)
        value_t = torch.frombuffer(
            value_np.tobytes(), dtype=torch.float32
        ).reshape(value_np.shape)
        return policy_t, value_t

    def eval(self) -> "OnnxModel":
        """No-op: ONNX Runtime is always in inference mode."""
        return self
