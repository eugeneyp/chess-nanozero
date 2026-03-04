"""Tests for MCTS integration (Phase 4).

All tests use an inline tiny config with a randomly initialized model
— no file I/O or trained checkpoint required.
"""

import time

import chess
import pytest
import torch

from src.game.chess_game import ChessGame
from src.mcts.mcts import MCTS
from src.neural_net.model import ChessResNet

TINY_CONFIG = {
    "model": {
        "num_res_blocks": 2,
        "num_filters": 32,
        "input_planes": 18,
        "policy_output_size": 4672,
    },
    "mcts": {
        "num_simulations": 50,
        "c_puct": 2.0,
        "dirichlet_alpha": 0.3,
        "dirichlet_epsilon": 0.25,
        "temperature": 1.0,
        "temperature_threshold_move": 30,
    },
}


def make_mcts(num_simulations: int | None = None) -> tuple[ChessResNet, MCTS]:
    """Create a tiny random model and MCTS instance."""
    torch.manual_seed(42)
    model = ChessResNet.from_config(TINY_CONFIG)
    model.eval()
    cfg = TINY_CONFIG
    if num_simulations is not None:
        cfg = {**TINY_CONFIG, "mcts": {**TINY_CONFIG["mcts"], "num_simulations": num_simulations}}
    mcts = MCTS(model, cfg, device="cpu")
    return model, mcts


def test_mcts_legal_moves_only():
    """MCTS must only create children for legal moves."""
    _, mcts = make_mcts()
    game = ChessGame()
    legal_moves = set(game.get_legal_moves())

    probs = mcts.get_action_probs(game, temperature=1.0, add_noise=False)

    # Every move in probs must be legal
    for move in probs:
        assert move in legal_moves, f"Illegal move returned: {move}"

    # All probabilities non-negative
    assert all(p >= 0.0 for p in probs.values())


def test_mcts_visit_counts():
    """Probabilities from get_action_probs must sum to ~1 and be non-negative."""
    _, mcts = make_mcts(num_simulations=100)
    game = ChessGame()

    probs = mcts.get_action_probs(game, temperature=1.0, add_noise=False)

    assert len(probs) > 0, "No moves returned"
    total = sum(probs.values())
    assert abs(total - 1.0) < 1e-6, f"Probabilities sum to {total}, expected 1.0"
    assert all(p >= 0.0 for p in probs.values()), "Negative probability found"


def test_mcts_checkmate_in_one():
    """MCTS must find Qxf7# (Scholar's Mate final move) in 200 sims.

    FEN: r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 3
    White plays Qxf7# which is an immediate checkmate — MCTS assigns +1 on the
    first visit and will strongly favor it regardless of network quality.
    """
    torch.manual_seed(0)
    model = ChessResNet.from_config(TINY_CONFIG)
    model.eval()
    cfg = {**TINY_CONFIG, "mcts": {**TINY_CONFIG["mcts"], "num_simulations": 200}}
    mcts = MCTS(model, cfg, device="cpu")

    fen = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 3"
    board = chess.Board(fen)
    game = ChessGame(board)

    probs = mcts.get_action_probs(game, temperature=0.0, add_noise=False)
    best_move = max(probs, key=probs.get)

    mate_move = chess.Move.from_uci("h5f7")
    assert best_move == mate_move, (
        f"Expected Qxf7# ({mate_move}), got {best_move}. "
        f"Top moves: {sorted(probs.items(), key=lambda x: -x[1])[:5]}"
    )


def test_mcts_avoids_blunder():
    """In a position with exactly 1 legal move, MCTS must return that move."""
    # Construct a position where there's only one legal move (king forced to escape).
    # Use a known single-move-forced position: king in check with one escape.
    # FEN: K7/8/8/8/8/8/8/7k b - - 0 1  — not forced; use a constructed one.
    #
    # Simpler approach: find/construct any position with exactly 1 legal move.
    # "8/8/8/8/8/8/6pk/7K w - - 0 1" - White king at h1, black pawn g2, black king h2.
    # White only has Kxg2.
    fen = "8/8/8/8/8/8/6pk/7K w - - 0 1"
    board = chess.Board(fen)
    legal = list(board.legal_moves)
    assert len(legal) == 1, f"Expected 1 legal move, got {len(legal)}: {legal}"

    _, mcts = make_mcts(num_simulations=50)
    game = ChessGame(board)
    probs = mcts.get_action_probs(game, temperature=0.0, add_noise=False)

    assert len(probs) == 1
    returned_move = list(probs.keys())[0]
    assert returned_move == legal[0], f"Expected {legal[0]}, got {returned_move}"


def test_mcts_deterministic():
    """Same model + same seed + no noise => same best move both calls."""
    torch.manual_seed(42)
    model = ChessResNet.from_config(TINY_CONFIG)
    model.eval()
    cfg = {**TINY_CONFIG, "mcts": {**TINY_CONFIG["mcts"], "num_simulations": 50}}
    mcts = MCTS(model, cfg, device="cpu")

    game = ChessGame()

    probs1 = mcts.get_action_probs(game, temperature=0.0, add_noise=False)
    probs2 = mcts.get_action_probs(game, temperature=0.0, add_noise=False)

    best1 = max(probs1, key=probs1.get)
    best2 = max(probs2, key=probs2.get)

    assert best1 == best2, f"Non-deterministic: {best1} vs {best2}"


def test_inference_speed():
    """200 simulations on CPU with tiny model must complete in under 5 seconds.

    A single warmup forward pass is performed first to trigger PyTorch kernel
    compilation, which can otherwise dominate timing on first use.
    """
    torch.manual_seed(42)
    model = ChessResNet.from_config(TINY_CONFIG)
    model.eval()
    cfg = {**TINY_CONFIG, "mcts": {**TINY_CONFIG["mcts"], "num_simulations": 200}}
    mcts = MCTS(model, cfg, device="cpu")

    game = ChessGame()

    # Warmup: one forward pass to trigger PyTorch kernel compilation
    with torch.no_grad():
        dummy = torch.zeros(1, 18, 8, 8)
        model(dummy)

    start = time.time()
    mcts.get_action_probs(game, temperature=1.0, add_noise=False)
    elapsed = time.time() - start

    assert elapsed < 5.0, f"200 sims took {elapsed:.2f}s, expected < 5.0s"
