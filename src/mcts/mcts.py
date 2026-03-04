"""Monte Carlo Tree Search for AlphaZero chess."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import torch

import chess

from src.game.chess_game import ChessGame
from src.game.encoding import get_legal_move_mask, move_to_index
from src.mcts.node import MCTSNode
from src.neural_net.model import ChessResNet, masked_policy_probs


class MCTS:
    """MCTS with neural network guidance (PUCT selection).

    Uses network policy as priors, value head for leaf evaluation.
    """

    def __init__(self, model: ChessResNet, config: dict, device: str = "cpu"):
        self.model = model
        self.model.eval()  # always in eval mode
        self.device = device
        mcts_cfg = config.get("mcts", {})
        self.num_simulations = mcts_cfg.get("num_simulations", 400)
        self.c_puct = mcts_cfg.get("c_puct", 2.0)
        self.dirichlet_alpha = mcts_cfg.get("dirichlet_alpha", 0.3)
        self.dirichlet_epsilon = mcts_cfg.get("dirichlet_epsilon", 0.25)

    def get_action_probs(
        self, game: ChessGame, temperature: float = 1.0, add_noise: bool = True,
        deadline: Optional[float] = None,
    ) -> dict[chess.Move, float]:
        """Run simulations; return {move: probability} from visit counts.

        temperature=1.0  → sample proportional to N(s,a)
        temperature=0.0  → argmax (greedy, used after move 30)
        add_noise=False  → deterministic (for evaluation, tests)
        deadline         → time.monotonic() deadline; stop early if reached
        """
        root = MCTSNode(game, prior=1.0)
        self._expand(root)
        if add_noise and root.children:
            self._add_dirichlet_noise(root)

        TIME_CHECK_INTERVAL = 10
        for i in range(self.num_simulations):
            self._simulate(root)
            if deadline is not None and i % TIME_CHECK_INTERVAL == (TIME_CHECK_INTERVAL - 1):
                if time.monotonic() >= deadline:
                    break

        self.root = root
        return self._compute_action_probs(root, temperature)

    def _simulate(self, root: MCTSNode) -> None:
        """One MCTS simulation: select leaf → expand/evaluate → backup."""
        path = [root]
        node = root
        while node.is_expanded and not node.game.is_terminal():
            node = self._select_child(node)
            path.append(node)

        if node.game.is_terminal():
            value = node.game.get_result(node.game.current_player)
        else:
            value = self._expand(node)

        self._backup(path, value)

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """Select child with highest UCB score."""
        return max(
            node.children.values(),
            key=lambda c: c.ucb_score(node.visit_count, self.c_puct),
        )

    def _expand(self, node: MCTSNode) -> float:
        """Evaluate with network, create children, mark expanded. Returns value."""
        encoding = node.game.encode()  # (18,8,8) float32
        # Use frombuffer to avoid slow numpy→torch conversion path when numpy C API
        # is incompatible with the installed PyTorch version.
        board_t = torch.frombuffer(encoding.tobytes(), dtype=torch.float32).reshape(1, 18, 8, 8).to(self.device)

        with torch.no_grad():
            policy_logits, value_t = self.model(board_t)  # (1,4672), (1,1)

        mask = get_legal_move_mask(node.game.board)  # (4672,) bool
        mask_t = torch.frombuffer(mask.tobytes(), dtype=torch.uint8).bool().to(self.device)
        probs_t = masked_policy_probs(policy_logits[0], mask_t)  # (4672,) tensor

        for move in node.game.get_legal_moves():
            idx = move_to_index(move, node.game.board)
            child_game = node.game.make_move(move)
            node.children[move] = MCTSNode(child_game, prior=float(probs_t[idx].item()), parent=node)

        node.is_expanded = True
        return float(value_t[0, 0].item())

    def _backup(self, path: list[MCTSNode], value: float) -> None:
        """Propagate value up path, flipping sign at each level."""
        for node in reversed(path):
            node.value_sum += value
            node.visit_count += 1
            value = -value

    def _add_dirichlet_noise(self, root: MCTSNode) -> None:
        """Mix Dirichlet noise into root priors (exploration during self-play)."""
        moves = list(root.children.keys())
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(moves))
        eps = self.dirichlet_epsilon
        for move, eta in zip(moves, noise):
            root.children[move].prior = (1 - eps) * root.children[move].prior + eps * eta

    def _compute_action_probs(
        self, root: MCTSNode, temperature: float
    ) -> dict[chess.Move, float]:
        """Convert visit counts to probabilities."""
        counts = {m: c.visit_count for m, c in root.children.items()}
        if temperature == 0.0:
            best = max(counts, key=counts.get)
            return {m: (1.0 if m == best else 0.0) for m in counts}
        total = sum(counts.values()) or 1
        return {m: n / total for m, n in counts.items()}
