"""Chess game wrapper for MCTS interface.

Wraps chess.Board with the same interface as connect4-alphazero,
so the MCTS implementation can be ported with minimal changes.
"""

from __future__ import annotations

import chess
import numpy as np


class ChessGame:
    """Wraps chess.Board with the connect4-alphazero board interface."""

    def __init__(self, board: chess.Board | None = None) -> None:
        self.board = board if board is not None else chess.Board()

    @property
    def current_player(self) -> int:
        """Returns 1 for White, 2 for Black."""
        return 1 if self.board.turn == chess.WHITE else 2

    def get_legal_moves(self) -> list[chess.Move]:
        """Returns list of legal moves in the current position."""
        return list(self.board.legal_moves)

    def make_move(self, move: chess.Move) -> "ChessGame":
        """Returns a new ChessGame with the move applied (MCTS-style, non-mutating)."""
        new_board = self.board.copy()
        new_board.push(move)
        return ChessGame(new_board)

    def push(self, move: chess.Move) -> None:
        """Mutates the board in place by pushing the move (UCI-style)."""
        self.board.push(move)

    def pop(self) -> chess.Move:
        """Undoes the last move and returns it."""
        return self.board.pop()

    def is_terminal(self) -> bool:
        """Returns True if the game is over (checkmate, stalemate, or draw)."""
        return self.board.is_game_over(claim_draw=True)

    def get_winner(self) -> int | None:
        """Returns 1 (White wins), 2 (Black wins), or None (draw or ongoing)."""
        outcome = self.board.outcome(claim_draw=True)
        if outcome is None:
            return None
        if outcome.winner == chess.WHITE:
            return 1
        if outcome.winner == chess.BLACK:
            return 2
        return None  # draw

    def get_result(self, player: int) -> float:
        """Returns +1.0 if player won, -1.0 if lost, 0.0 for draw or ongoing."""
        winner = self.get_winner()
        if winner is None:
            return 0.0
        if winner == player:
            return 1.0
        return -1.0

    def encode(self) -> np.ndarray:
        """Returns the board encoded as (18, 8, 8) float32 tensor."""
        from src.game.encoding import encode_board
        return encode_board(self.board)

    def clone(self) -> "ChessGame":
        """Returns a deep copy of this game."""
        return ChessGame(self.board.copy())

    def __repr__(self) -> str:
        return f"ChessGame(turn={'White' if self.board.turn else 'Black'}, fen={self.board.fen()!r})"
