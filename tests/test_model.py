"""Phase 2 tests: ChessResNet neural network."""

import chess
import numpy as np
import pytest
import torch

from src.game.encoding import encode_board, get_legal_move_mask, NUM_PLANES, POLICY_SIZE
from src.neural_net.model import ChessResNet, masked_policy_probs
from src.neural_net.losses import AlphaZeroLoss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_tiny() -> ChessResNet:
    return ChessResNet(num_res_blocks=4, num_filters=64)


def random_input(batch: int = 2) -> torch.Tensor:
    return torch.randn(batch, NUM_PLANES, 8, 8)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_forward_pass_shape():
    """(B, 18, 8, 8) -> policy (B, 4672), value (B, 1)."""
    model = make_tiny()
    x = random_input(batch=4)
    policy, value = model(x)
    assert policy.shape == (4, POLICY_SIZE), f"policy shape {policy.shape}"
    assert value.shape == (4, 1), f"value shape {value.shape}"


def test_value_range():
    """Value output must be in [-1, +1] (tanh output)."""
    model = make_tiny()
    x = random_input(batch=8)
    _, value = model(x)
    assert (value >= -1.0).all() and (value <= 1.0).all(), \
        f"value out of range: min={value.min()}, max={value.max()}"


def test_policy_masking():
    """Illegal moves get probability ~0; legal moves sum to ~1."""
    board = chess.Board()
    x = torch.tensor(encode_board(board), dtype=torch.float32).unsqueeze(0)  # (1, 18, 8, 8)
    mask_np = get_legal_move_mask(board)  # (4672,) bool
    mask = torch.tensor(mask_np.astype(np.uint8), dtype=torch.bool).unsqueeze(0)  # (1, 4672)

    model = make_tiny()
    model.eval()
    with torch.no_grad():
        policy, _ = model(x)
    probs = masked_policy_probs(policy, mask)

    illegal_prob_sum = probs[~mask].sum().item()
    legal_prob_sum = probs[mask].sum().item()

    assert illegal_prob_sum < 1e-6, f"illegal moves have prob {illegal_prob_sum}"
    assert abs(legal_prob_sum - 1.0) < 1e-5, f"legal probs sum to {legal_prob_sum}"


def test_different_configs():
    """tiny/medium/large configs all produce correct output shapes."""
    configs = [
        {"num_res_blocks": 4, "num_filters": 64},    # tiny
        {"num_res_blocks": 8, "num_filters": 128},   # medium
        {"num_res_blocks": 12, "num_filters": 256},  # large
    ]
    x = random_input(batch=2)
    for cfg in configs:
        model = ChessResNet(**cfg)
        policy, value = model(x)
        assert policy.shape == (2, POLICY_SIZE), f"policy shape {policy.shape} for {cfg}"
        assert value.shape == (2, 1), f"value shape {value.shape} for {cfg}"


def test_gradient_flow():
    """Backward pass produces non-zero gradients for every parameter."""
    model = make_tiny()
    x = random_input(batch=2)
    policy, value = model(x)
    (policy.sum() + value.sum()).backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"param {name} has no grad"
        assert param.grad.abs().sum() > 0, f"param {name} has all-zero grad"


def test_loss_computation():
    """Loss with known inputs: total > 0, both components > 0."""
    model = make_tiny()
    loss_fn = AlphaZeroLoss(value_loss_weight=1.0)

    x = random_input(batch=4)
    policy_logits, value_pred = model(x)

    # Soft uniform target over 20 random legal-ish indices
    target_policy = torch.zeros(4, POLICY_SIZE)
    indices = torch.randperm(POLICY_SIZE)[:20]
    target_policy[:, indices] = 1.0 / 20.0

    target_value = torch.zeros(4, 1)
    target_value[0] = 1.0
    target_value[1] = -1.0

    total, policy_loss, value_loss = loss_fn(
        policy_logits, value_pred, target_policy, target_value
    )

    assert total.item() > 0, f"total loss {total.item()} not positive"
    assert policy_loss.item() > 0, f"policy loss {policy_loss.item()} not positive"
    assert value_loss.item() > 0, f"value loss {value_loss.item()} not positive"
    assert abs(total.item() - (policy_loss.item() + value_loss.item())) < 1e-5


def test_from_config():
    """from_config() constructs model with correct num_blocks and filters."""
    cfg = {
        "model": {
            "num_res_blocks": 4,
            "num_filters": 64,
            "input_planes": 18,
            "policy_output_size": 4672,
        }
    }
    model = ChessResNet.from_config(cfg)
    assert model.num_res_blocks == 4
    assert model.num_filters == 64

    x = random_input(batch=2)
    policy, value = model(x)
    assert policy.shape == (2, POLICY_SIZE)
    assert value.shape == (2, 1)


def test_batched_masked_probs():
    """masked_policy_probs works correctly on a batch of 4."""
    batch = 4
    logits = torch.randn(batch, POLICY_SIZE)

    # Each sample has a different random subset of 20 legal moves
    masks = torch.zeros(batch, POLICY_SIZE, dtype=torch.bool)
    for i in range(batch):
        idx = torch.randperm(POLICY_SIZE)[:20]
        masks[i, idx] = True

    probs = masked_policy_probs(logits, masks)

    # Shape preserved
    assert probs.shape == (batch, POLICY_SIZE)

    for i in range(batch):
        illegal_sum = probs[i][~masks[i]].sum().item()
        legal_sum = probs[i][masks[i]].sum().item()
        assert illegal_sum < 1e-6, f"sample {i}: illegal prob sum {illegal_sum}"
        assert abs(legal_sum - 1.0) < 1e-5, f"sample {i}: legal prob sum {legal_sum}"
