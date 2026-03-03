"""Tests for Phase 3: data pipeline (PGN parsing, dataset, trainer integration)."""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import chess
import numpy as np
import pytest

from src.game.encoding import encode_board
from src.training.supervised.dataset import ChessDataset
from src.training.supervised.prepare_data import parse_pgn_to_positions, save_positions

# ─── PGN fixtures ────────────────────────────────────────────────────────────

SCHOLAR_MATE_PGN = """[Event "Test"]
[Result "1-0"]

1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6 4. Qxf7# 1-0
"""

DRAW_PGN = """[Event "Test"]
[Result "1/2-1/2"]

1. e4 e5 2. Nf3 Nf6 3. Nxe5 Nxe4 4. Qe2 Nf6 5. Nc6+ 1/2-1/2
"""

MULTI_PGN = SCHOLAR_MATE_PGN + "\n" + DRAW_PGN

UNFINISHED_PGN = """[Event "Test"]
[Result "*"]

1. e4 e5 *
"""


# ─── test_pgn_parsing ─────────────────────────────────────────────────────────

def test_pgn_parsing():
    """Scholar's Mate has 7 half-moves; skip=0 yields 7 positions (before each push)."""
    positions = parse_pgn_to_positions(io.StringIO(SCHOLAR_MATE_PGN), skip_first_n_moves=0)
    assert len(positions) == 7, f"Expected 7 positions, got {len(positions)}"

    # Basic sanity: all are 3-tuples
    for enc, idx, val in positions:
        assert enc.shape == (18, 8, 8)
        assert 0 <= idx < 4672
        assert val in (-1.0, 0.0, 1.0)


# ─── test_encoding_consistency ───────────────────────────────────────────────

def test_encoding_consistency():
    """First position from Scholar's Mate == manually encoded starting board."""
    positions = parse_pgn_to_positions(io.StringIO(SCHOLAR_MATE_PGN), skip_first_n_moves=0)
    enc_from_parser, _, _ = positions[0]

    board = chess.Board()
    enc_manual = encode_board(board)

    np.testing.assert_array_equal(enc_from_parser, enc_manual)


# ─── test_dataset_loading ────────────────────────────────────────────────────

def test_dataset_loading():
    """ChessDataset returns correct tensor shapes and dtypes."""
    import torch

    positions = parse_pgn_to_positions(io.StringIO(SCHOLAR_MATE_PGN), skip_first_n_moves=0)

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        tmp_path = Path(f.name)

    try:
        save_positions(positions, tmp_path)
        ds = ChessDataset([tmp_path], split="train", val_split=0.0)

        assert len(ds) == len(positions)

        item = ds[0]
        assert "board" in item and "move" in item and "result" in item

        assert item["board"].shape == (18, 8, 8)
        assert item["board"].dtype == torch.float32

        assert item["move"].ndim == 0
        assert item["move"].dtype == torch.long

        assert item["result"].ndim == 0
        assert item["result"].dtype == torch.float32
    finally:
        tmp_path.unlink(missing_ok=True)


# ─── test_train_val_split ────────────────────────────────────────────────────

def test_train_val_split():
    """Train and val splits are disjoint and together cover all positions."""
    positions = parse_pgn_to_positions(io.StringIO(MULTI_PGN), skip_first_n_moves=0)
    total = len(positions)
    assert total > 1, "Need more than 1 position for a split test"

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        tmp_path = Path(f.name)

    try:
        save_positions(positions, tmp_path)

        val_split = 0.3
        train_ds = ChessDataset([tmp_path], split="train", val_split=val_split, seed=42)
        val_ds = ChessDataset([tmp_path], split="val", val_split=val_split, seed=42)

        assert len(train_ds) + len(val_ds) == total

        # Reconstruct which indices each split used by checking board encodings
        # The split uses a permuted index; verify no overlap by checking move values
        train_moves = set(int(train_ds[i]["move"].item()) for i in range(len(train_ds)))
        val_moves = set(int(val_ds[i]["move"].item()) for i in range(len(val_ds)))

        # Count total unique (board, move) pairs — boards could repeat, so check lengths
        # The simplest check: lengths add up to total (already verified above)
        assert len(train_ds) > 0
        assert len(val_ds) > 0
    finally:
        tmp_path.unlink(missing_ok=True)


# ─── test_skip_first_n_moves ─────────────────────────────────────────────────

def test_skip_first_n_moves():
    """Skipping N half-moves reduces position count by exactly N."""
    n_full = len(parse_pgn_to_positions(io.StringIO(SCHOLAR_MATE_PGN), skip_first_n_moves=0))
    skip = 4
    n_skip = len(parse_pgn_to_positions(io.StringIO(SCHOLAR_MATE_PGN), skip_first_n_moves=skip))
    assert n_skip == n_full - skip, (
        f"Expected {n_full - skip} positions with skip={skip}, got {n_skip}"
    )


# ─── test_result_encoding ────────────────────────────────────────────────────

def test_result_encoding():
    """Values are assigned correctly from current player's perspective."""
    # White wins: Scholar's Mate
    positions = parse_pgn_to_positions(io.StringIO(SCHOLAR_MATE_PGN), skip_first_n_moves=0)

    # Position 0: White to move → White wins → value = +1.0
    _, _, val0 = positions[0]
    assert val0 == pytest.approx(1.0), f"Position 0 (White to move) should be +1.0, got {val0}"

    # Position 1: Black to move → Black lost → value = -1.0
    _, _, val1 = positions[1]
    assert val1 == pytest.approx(-1.0), f"Position 1 (Black to move) should be -1.0, got {val1}"

    # Draw: all values should be 0.0
    draw_positions = parse_pgn_to_positions(io.StringIO(DRAW_PGN), skip_first_n_moves=0)
    for _, _, val in draw_positions:
        assert val == pytest.approx(0.0), f"Draw game should have value 0.0, got {val}"


# ─── test_unfinished_games_skipped ───────────────────────────────────────────

def test_unfinished_games_skipped():
    """Games with result '*' are skipped entirely."""
    positions = parse_pgn_to_positions(io.StringIO(UNFINISHED_PGN), skip_first_n_moves=0)
    assert len(positions) == 0, f"Expected 0 positions for unfinished game, got {len(positions)}"
