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
import time
from pathlib import Path

import chess
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

from src.agents.alphazero_agent import AlphaZeroAgent
from src.game.chess_game import ChessGame

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
        checkpoint_path = os.environ.get("CHESS_CHECKPOINT", "models/medium1.pt")
        num_simulations = int(os.environ.get("NUM_SIMULATIONS", "400"))

        _log.info("Loading config from %s", config_path)
        config = yaml.safe_load(open(config_path))
        config.setdefault("mcts", {})["num_simulations"] = num_simulations

        _log.info("Loading checkpoint from %s", checkpoint_path)
        _agent = AlphaZeroAgent.from_checkpoint(
            Path(checkpoint_path), config, device="cpu"
        )
        _log.info("Agent ready.")
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
        return max(0.5, min(v, 10.0))


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

    game = ChessGame(board)
    deadline = time.monotonic() + request.time_limit
    agent = get_agent()

    probs = agent.mcts.get_action_probs(
        game, temperature=0.0, add_noise=False, deadline=deadline
    )
    if not probs:
        raise HTTPException(status_code=500, detail="Engine returned no move")

    best_move = max(probs, key=probs.get)

    # Count actual simulations: sum of root children visit counts
    root = getattr(agent.mcts, "root", None)
    sims_done = (
        sum(c.visit_count for c in root.children.values()) if root else 0
    )

    _log.info("move=%s sims=%d fen=%s", best_move.uci(), sims_done, request.fen[:40])

    board.push(best_move)
    return MoveResponse(
        move=best_move.uci(),
        fen=board.fen(),
        score=float(probs.get(best_move, 0.0)),
        simulations=sims_done,
    )


@app.get("/", include_in_schema=False)
def serve_root() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


# ---------------------------------------------------------------------------
# Static file mount — MUST be last
# ---------------------------------------------------------------------------

app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")
