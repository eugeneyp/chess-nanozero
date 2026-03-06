"""FastAPI web application for Chess NanoZero.

Stateless REST API: POST /api/move accepts FEN + time_limit, runs MCTS,
returns best move. Serves the chessboard.js frontend via static files.

Configuration via environment variables:
    CHESS_CONFIG      path to YAML config (default: configs/medium.yaml)
    CHESS_CHECKPOINT  path to .pt checkpoint (default: models/medium1.pt)
    NUM_SIMULATIONS   MCTS simulation cap (default: 400)
"""

import logging
import os
import random
import time
from pathlib import Path

import chess
import numpy as np
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

from src.agents.alphazero_agent import AlphaZeroAgent
from src.game.chess_game import ChessGame
from src.game.encoding import encode_board, get_legal_move_mask, index_to_move

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Chess NanoZero", version="1.0.0")

# ---------------------------------------------------------------------------
# Agent singleton — lazy-loaded on first request
# ---------------------------------------------------------------------------

_agent: AlphaZeroAgent | None = None


def get_agent() -> AlphaZeroAgent:
    global _agent
    if _agent is None:
        config_path = os.environ.get("CHESS_CONFIG", "configs/medium.yaml")
        checkpoint_path = os.environ.get("CHESS_CHECKPOINT", "models/medium2.onnx")
        num_simulations = int(os.environ.get("NUM_SIMULATIONS", "400"))

        _log.info("Loading config from %s", config_path)
        config = yaml.safe_load(open(config_path))
        config.setdefault("mcts", {})["num_simulations"] = num_simulations

        _log.info("Loading checkpoint from %s", checkpoint_path)
        if checkpoint_path.endswith(".onnx"):
            _agent = AlphaZeroAgent.from_onnx(Path(checkpoint_path), config)
        else:
            _agent = AlphaZeroAgent.from_checkpoint(
                Path(checkpoint_path), config, device="cpu"
            )
        _log.info("Agent ready (%s).", "ONNX" if checkpoint_path.endswith(".onnx") else "PyTorch")
    return _agent


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class MoveRequest(BaseModel):
    fen: str
    time_limit: float = 3.0

    @field_validator("time_limit")
    @classmethod
    def clamp_time_limit(cls, v: float) -> float:
        return max(0.0, min(v, 10.0))


class MoveResponse(BaseModel):
    move: str
    fen: str
    score: float
    simulations: int


# ---------------------------------------------------------------------------
# API routes (registered BEFORE StaticFiles mount)
# ---------------------------------------------------------------------------


@app.get("/api/health")
def api_health() -> dict:
    return {"status": "ok"}


@app.post("/api/move", response_model=MoveResponse)
def api_move(request: MoveRequest) -> MoveResponse:
    try:
        board = chess.Board(request.fen)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid FEN: {exc}") from exc

    if board.is_game_over():
        raise HTTPException(status_code=400, detail=f"Game is already over: {board.result()}")

    agent = get_agent()

    opening_temp_moves = int(os.environ.get("OPENING_TEMP_MOVES", "15"))
    temperature = 0.5 if board.fullmove_number <= opening_temp_moves else 0.0

    if request.time_limit == 0.0:
        # Pure policy head: single forward pass, no MCTS
        best_move, policy_prob = _policy_move(board, agent, temperature)
        sims_done = 0
    else:
        game = ChessGame(board)
        # Compute deadline AFTER agent is loaded so model-load time doesn't eat into budget
        deadline = time.monotonic() + request.time_limit
        probs = agent.mcts.get_action_probs(
            game, temperature=temperature, add_noise=False, deadline=deadline
        )
        if not probs:
            raise HTTPException(status_code=500, detail="Engine returned no move")
        if temperature == 0.0:
            best_move = max(probs, key=probs.get)
        else:
            moves, weights = zip(*probs.items())
            best_move = random.choices(moves, weights=weights)[0]
        policy_prob = float(probs[best_move])
        root = getattr(agent.mcts, "root", None)
        sims_done = sum(c.visit_count for c in root.children.values()) if root else 0

    _log.info("move=%s sims=%d fen=%s", best_move.uci(), sims_done, request.fen[:40])

    board.push(best_move)
    return MoveResponse(
        move=best_move.uci(),
        fen=board.fen(),
        score=policy_prob,
        simulations=sims_done,
    )


def _policy_move(board: chess.Board, agent: AlphaZeroAgent, temperature: float = 0.0) -> tuple[chess.Move, float]:
    """Pick best move using the policy head only — no MCTS tree search."""
    from src.mcts.mcts import _softmax_masked
    encoding = encode_board(board)
    mask = get_legal_move_mask(board)
    if isinstance(agent.mcts, __import__('src.mcts.mcts', fromlist=['OnnxMCTS']).OnnxMCTS):
        # ONNX path
        policy_np, _ = agent.mcts.sess.run(None, {agent.mcts.input_name: encoding[np.newaxis]})
        probs = _softmax_masked(policy_np[0], mask)
        if temperature > 0.0:
            best_idx = int(np.random.choice(len(probs), p=probs / probs.sum()))
        else:
            best_idx = int(probs.argmax())
        return index_to_move(best_idx, board), float(probs[best_idx])
    else:
        # PyTorch path (lazy import to avoid numpy/torch compat issue at startup)
        import torch
        from src.neural_net.model import masked_policy_probs
        board_t = torch.frombuffer(encoding.tobytes(), dtype=torch.float32).reshape(1, 18, 8, 8)
        with torch.no_grad():
            policy_logits, _ = agent.mcts.model(board_t)
        mask_t = torch.frombuffer(mask.tobytes(), dtype=torch.uint8).bool()
        probs_t = masked_policy_probs(policy_logits[0], mask_t)
        probs_np = probs_t.numpy()
        if temperature > 0.0:
            best_idx = int(np.random.choice(len(probs_np), p=probs_np / probs_np.sum()))
        else:
            best_idx = int(probs_np.argmax())
        return index_to_move(best_idx, board), float(probs_np[best_idx])


@app.get("/", include_in_schema=False)
def serve_root() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


# ---------------------------------------------------------------------------
# Static file mount — MUST be last
# ---------------------------------------------------------------------------

app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")
