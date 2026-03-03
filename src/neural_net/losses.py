"""AlphaZero combined loss: cross-entropy policy + MSE value."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class AlphaZeroLoss(nn.Module):
    """Combined policy + value loss for AlphaZero-style training.

    Policy loss: cross-entropy against soft target distribution (KL-style).
    Value loss:  MSE between predicted value and game result.
    """

    def __init__(self, value_loss_weight: float = 1.0):
        super().__init__()
        self.value_loss_weight = value_loss_weight

    def forward(
        self,
        policy_logits: Tensor,
        value_pred: Tensor,
        target_policy: Tensor,
        target_value: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            policy_logits : (B, 4672) raw logits
            value_pred    : (B, 1)   predicted value from network
            target_policy : (B, 4672) soft target distribution (MCTS visits or one-hot)
            target_value  : (B, 1)   game result in [-1, +1]

        Returns:
            total_loss, policy_loss, value_loss
        """
        # Policy: KL-style CE against soft target
        log_probs = F.log_softmax(policy_logits, dim=1)
        policy_loss = -(target_policy * log_probs).sum(dim=1).mean()

        # Value: MSE
        value_loss = F.mse_loss(value_pred, target_value)

        total = policy_loss + self.value_loss_weight * value_loss
        return total, policy_loss, value_loss
