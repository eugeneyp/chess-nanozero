"""MCTS node for AlphaZero chess search."""

from __future__ import annotations

import math

import chess

from src.game.chess_game import ChessGame


class MCTSNode:
    """Represents a node in the MCTS tree.

    value_sum is always from the perspective of the node's current player.
    Parent negates child.q_value during selection (opponent's perspective).
    """

    def __init__(self, game: ChessGame, prior: float, parent: MCTSNode | None = None):
        self.game = game
        self.prior = prior          # P(s,a) — prior probability from network
        self.visit_count = 0        # N(s,a)
        self.value_sum = 0.0        # W(s,a) — from current player's perspective
        self.children: dict[chess.Move, MCTSNode] = {}
        self.is_expanded = False
        self.parent = parent

    @property
    def q_value(self) -> float:
        """Expected outcome for current player at this node."""
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0

    def ucb_score(self, parent_visit_count: int, c_puct: float) -> float:
        """PUCT formula. Called by parent, so negate q_value (opponent's perspective)."""
        exploration = c_puct * self.prior * math.sqrt(parent_visit_count) / (1 + self.visit_count)
        return -self.q_value + exploration
