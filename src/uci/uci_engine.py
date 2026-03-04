"""UCI protocol engine for chess-nanozero.

Speaks UCI on stdin/stdout; search is driven by AlphaZeroAgent (MCTS).
Threading model: each 'go' spawns a daemon thread; 'stop' joins it.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Optional

import chess
import yaml

from src.agents.alphazero_agent import AlphaZeroAgent
from src.game.chess_game import ChessGame


def _send(msg: str) -> None:
    """Write a line to stdout and flush immediately."""
    print(msg, flush=True)


def _log(msg: str) -> None:
    """Write a debug line to stderr (invisible to GUI/fastchess)."""
    print(msg, file=sys.stderr, flush=True)


class UCIEngine:
    ENGINE_NAME = "chess-nanozero"
    ENGINE_AUTHOR = "eugeneyp"

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: str = "cpu",
        num_simulations: Optional[int] = None,
    ) -> None:
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.num_simulations_override = num_simulations
        self.agent: Optional[AlphaZeroAgent] = None  # lazy-loaded on isready
        self.board = chess.Board()
        self.search_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

    # ------------------------------------------------------------------
    # UCI command handlers
    # ------------------------------------------------------------------

    def handle_uci(self) -> None:
        _send(f"id name {self.ENGINE_NAME}")
        _send(f"id author {self.ENGINE_AUTHOR}")
        _send("uciok")

    def handle_isready(self) -> None:
        if self.agent is None:
            _log(f"Loading model from {self.checkpoint_path} …")
            config = yaml.safe_load(open(self.config_path))
            if self.num_simulations_override is not None:
                config["mcts"]["num_simulations"] = self.num_simulations_override
            self.agent = AlphaZeroAgent.from_checkpoint(
                Path(self.checkpoint_path), config, self.device
            )
            # On macOS, PyTorch's OpenMP threads run slower when spawned from
            # a non-main Python thread (daemon thread scheduling).
            # torch.set_num_threads(1) reduces the overhead (~3x improvement).
            import torch
            torch.set_num_threads(1)

            # Warmup forward pass — eliminates PyTorch first-call overhead.
            # Use frombuffer path to match the exact code path MCTS uses in
            # _expand (torch.zeros uses a different kernel than frombuffer).
            _log("Warming up …")
            from src.game.encoding import NUM_PLANES
            import numpy as np
            dummy_arr = np.zeros(NUM_PLANES * 8 * 8, dtype=np.float32)
            dummy_t = torch.frombuffer(dummy_arr.tobytes(), dtype=torch.float32).reshape(1, NUM_PLANES, 8, 8).to(self.device)
            with torch.no_grad():
                self.agent.mcts.model(dummy_t)
            _log("Ready.")
        _send("readyok")

    def handle_ucinewgame(self) -> None:
        self._stop_search()
        self.board = chess.Board()

    def handle_position(self, tokens: list[str]) -> None:
        """Parse: position startpos [moves m1 m2 ...]
                  position fen <fen> [moves m1 m2 ...]"""
        self._stop_search()
        idx = 0
        if tokens and tokens[idx] == "startpos":
            self.board = chess.Board()
            idx += 1
        elif tokens and tokens[idx] == "fen":
            idx += 1
            fen_parts: list[str] = []
            while idx < len(tokens) and tokens[idx] != "moves":
                fen_parts.append(tokens[idx])
                idx += 1
            self.board = chess.Board(" ".join(fen_parts))
        # replay moves
        if idx < len(tokens) and tokens[idx] == "moves":
            idx += 1
            while idx < len(tokens):
                move = chess.Move.from_uci(tokens[idx])
                self.board.push(move)
                idx += 1

    TIME_USAGE_FRACTION = 0.9  # use 90% of budget (safety margin)

    def _parse_go_time(self, tokens: list[str]) -> Optional[float]:
        """Parse go tokens → seconds to use. Returns None if no time control found."""
        is_white = self.board.turn == chess.WHITE
        i = 0
        movetime = wtime = btime = winc = binc = None
        while i < len(tokens):
            t = tokens[i]
            if t == "movetime" and i + 1 < len(tokens):
                movetime = int(tokens[i + 1]) / 1000.0
            elif t == "wtime" and i + 1 < len(tokens):
                wtime = int(tokens[i + 1]) / 1000.0
            elif t == "btime" and i + 1 < len(tokens):
                btime = int(tokens[i + 1]) / 1000.0
            elif t == "winc" and i + 1 < len(tokens):
                winc = int(tokens[i + 1]) / 1000.0
            elif t == "binc" and i + 1 < len(tokens):
                binc = int(tokens[i + 1]) / 1000.0
            i += 1

        if movetime is not None:
            return movetime * self.TIME_USAGE_FRACTION

        time_left = wtime if is_white else btime
        increment = (winc if is_white else binc) or 0.0
        if time_left is not None:
            return (time_left / 40.0 + increment) * self.TIME_USAGE_FRACTION

        return None  # infinite pondering — run up to num_simulations

    def handle_go(self, tokens: list[str]) -> None:
        """Start search in a daemon thread; send bestmove when done."""
        self._stop_search()
        self.stop_event.clear()

        board_copy = self.board.copy()
        time_budget = self._parse_go_time(tokens)

        def _search() -> None:
            try:
                game = ChessGame(board_copy)
                deadline = (time.monotonic() + time_budget) if time_budget is not None else None
                probs = self.agent.mcts.get_action_probs(
                    game, temperature=0.0, add_noise=False, deadline=deadline
                )
                if probs:
                    move = max(probs, key=probs.get)
                    _send("info depth 1 score cp 0 nodes 1")
                    _send(f"bestmove {move.uci()}")
                else:
                    _send("info depth 1 score cp 0 nodes 1")
                    _send("bestmove 0000")
            except Exception as exc:  # pragma: no cover
                _log(f"Search error: {exc}")
                _send("bestmove 0000")

        self.search_thread = threading.Thread(target=_search, daemon=True)
        self.search_thread.start()

    def handle_stop(self) -> None:
        self._stop_search()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _stop_search(self) -> None:
        """Signal stop and wait for the search thread to finish."""
        self.stop_event.set()
        if self.search_thread is not None and self.search_thread.is_alive():
            self.search_thread.join(timeout=60)
        self.search_thread = None


# ------------------------------------------------------------------
# Main dispatch loop
# ------------------------------------------------------------------

def run_uci_loop(engine: UCIEngine) -> None:
    """Read UCI commands from stdin and dispatch to engine handlers."""
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        tokens = line.split()
        cmd = tokens[0]
        args = tokens[1:]

        try:
            if cmd == "uci":
                engine.handle_uci()
            elif cmd == "isready":
                engine.handle_isready()
            elif cmd == "ucinewgame":
                engine.handle_ucinewgame()
            elif cmd == "position":
                engine.handle_position(args)
            elif cmd == "go":
                engine.handle_go(args)
            elif cmd == "stop":
                engine.handle_stop()
            elif cmd == "quit":
                engine._stop_search()
                break
            # Unknown commands are silently ignored per UCI spec
        except Exception as exc:  # pragma: no cover
            _log(f"Error handling '{cmd}': {exc}")
