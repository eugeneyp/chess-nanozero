"""Phase 7 tests: Web interface (FastAPI).

Uses a tiny random checkpoint — no trained model required.
Tests are self-contained via pytest tmp_path_factory.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
import yaml

from src.neural_net.model import ChessResNet


@pytest.fixture(scope="module")
def tiny_web_client(tmp_path_factory):
    """Create tiny checkpoint + config, patch env vars, return TestClient."""
    tmp = tmp_path_factory.mktemp("web_test")

    config = {
        "model": {
            "num_res_blocks": 1,
            "num_filters": 16,
            "input_planes": 18,
            "policy_output_size": 4672,
        },
        "mcts": {
            "num_simulations": 2,
            "c_puct": 2.0,
            "dirichlet_alpha": 0.3,
            "dirichlet_epsilon": 0.25,
            "temperature_threshold_move": 30,
        },
    }
    config_path = tmp / "tiny.yaml"
    config_path.write_text(yaml.dump(config))

    model = ChessResNet.from_config(config)
    ckpt_path = tmp / "tiny.pt"
    torch.save({"epoch": 0, "model_state_dict": model.state_dict()}, ckpt_path)

    # Patch env vars before importing app (agent is module-level singleton)
    os.environ["CHESS_CONFIG"] = str(config_path)
    os.environ["CHESS_CHECKPOINT"] = str(ckpt_path)
    os.environ["NUM_SIMULATIONS"] = "2"

    # Reset the module-level singleton in case it was loaded in a previous test
    import web.app as app_module
    app_module._agent = None

    from fastapi.testclient import TestClient
    client = TestClient(app_module.app)
    return client


def test_health_check(tiny_web_client):
    """GET /api/health → 200, status ok."""
    response = tiny_web_client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_api_move_starting_position(tiny_web_client):
    """POST /api/move with starting FEN → 200, valid UCI move, valid FEN."""
    import chess

    starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    response = tiny_web_client.post(
        "/api/move",
        json={"fen": starting_fen, "time_limit": 1.0},
    )
    assert response.status_code == 200, f"Unexpected status: {response.status_code}, body: {response.text}"

    data = response.json()
    assert "move" in data
    assert "fen" in data
    assert "score" in data
    assert "simulations" in data

    # Move must be a legal UCI move from the starting position
    board = chess.Board(starting_fen)
    legal_ucis = {m.uci() for m in board.legal_moves}
    assert data["move"] in legal_ucis, f"Illegal move returned: {data['move']}"

    # Returned FEN must be parseable
    result_board = chess.Board(data["fen"])
    assert result_board is not None
