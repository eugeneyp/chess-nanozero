"""ChessResNet: dual-head residual network for AlphaZero-style chess.

Input:  (batch, 18, 8, 8) board encoding
Output: policy logits (batch, 4672), value (batch, 1)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.game.encoding import NUM_PLANES, POLICY_SIZE


class ResBlock(nn.Module):
    """Standard residual block with two 3x3 convolutions and skip connection."""

    def __init__(self, num_filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class ChessResNet(nn.Module):
    """Dual-head ResNet for chess policy and value estimation."""

    def __init__(
        self,
        num_res_blocks: int = 8,
        num_filters: int = 128,
        input_planes: int = NUM_PLANES,
        policy_output_size: int = POLICY_SIZE,
    ):
        super().__init__()
        self.num_res_blocks = num_res_blocks
        self.num_filters = num_filters
        self.policy_output_size = policy_output_size

        # Stem
        self.stem_conv = nn.Conv2d(input_planes, num_filters, 3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(num_filters)

        # Residual tower
        self.res_blocks = nn.ModuleList(
            [ResBlock(num_filters) for _ in range(num_res_blocks)]
        )

        # Policy head: F -> 32 -> 73 planes, then flatten to 4672
        self.policy_conv1 = nn.Conv2d(num_filters, 32, 1, bias=False)
        self.policy_bn1 = nn.BatchNorm2d(32)
        self.policy_conv2 = nn.Conv2d(32, 73, 1, bias=True)

        # Value head: F -> 1 plane, flatten to 64, FC to 1
        self.value_conv = nn.Conv2d(num_filters, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(64, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            x: (B, 18, 8, 8) board encoding

        Returns:
            policy: (B, 4672) raw logits (no softmax)
            value:  (B, 1) in range [-1, +1]
        """
        # Stem
        x = F.relu(self.stem_bn(self.stem_conv(x)))

        # Residual tower
        for block in self.res_blocks:
            x = block(x)

        # Policy head
        p = F.relu(self.policy_bn1(self.policy_conv1(x)))
        p = self.policy_conv2(p)           # (B, 73, 8, 8)
        policy = p.flatten(start_dim=1)    # (B, 4672)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.flatten(start_dim=1)         # (B, 64)
        v = F.relu(self.value_fc1(v))      # (B, 256)
        value = torch.tanh(self.value_fc2(v))  # (B, 1)

        return policy, value

    @classmethod
    def from_config(cls, cfg: dict) -> "ChessResNet":
        """Construct from a config dict (the 'model' section of a YAML config)."""
        m = cfg.get("model", cfg)
        return cls(
            num_res_blocks=m["num_res_blocks"],
            num_filters=m["num_filters"],
            input_planes=m.get("input_planes", NUM_PLANES),
            policy_output_size=m.get("policy_output_size", POLICY_SIZE),
        )


def masked_policy_probs(policy_logits: Tensor, mask: Tensor) -> Tensor:
    """Apply legal-move mask then softmax.

    Args:
        policy_logits: (..., 4672) raw logits
        mask:          (..., 4672) bool, True = legal move

    Returns:
        probs: same shape, sums to 1 over legal moves per sample
    """
    masked = policy_logits.clone().float()
    masked[~mask] = float("-inf")
    return torch.softmax(masked, dim=-1)
