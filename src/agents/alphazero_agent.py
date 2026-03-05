"""AlphaZero agent: wraps MCTS + checkpoint loading."""

from __future__ import annotations

import random
from pathlib import Path

import chess
import torch

from src.game.chess_game import ChessGame
from src.mcts.mcts import MCTS, OnnxMCTS
from src.neural_net.model import ChessResNet


class AlphaZeroAgent:
    """Chess agent that uses MCTS guided by a trained ChessResNet."""

    def __init__(self, model: ChessResNet, config: dict, device: str = "cpu"):
        self.mcts = MCTS(model, config, device)
        self.temp_threshold = config.get("mcts", {}).get("temperature_threshold_move", 30)

    @classmethod
    def from_checkpoint(
        cls, checkpoint_path: Path, config: dict, device: str = "cpu"
    ) -> "AlphaZeroAgent":
        """Load model weights from a checkpoint and construct the agent."""
        model = ChessResNet.from_config(config)
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()
        return cls(model, config, device)

    @classmethod
    def from_onnx(cls, onnx_path: Path, config: dict) -> "AlphaZeroAgent":
        """Load ONNX model for faster CPU inference (no PyTorch overhead)."""
        import onnxruntime as ort
        sess = ort.InferenceSession(str(onnx_path))
        agent = cls.__new__(cls)
        agent.mcts = OnnxMCTS(sess, config)
        agent.temp_threshold = config.get("mcts", {}).get("temperature_threshold_move", 30)
        return agent

    def select_move(
        self, game: ChessGame, move_number: int = 0, add_noise: bool = False
    ) -> chess.Move:
        """Select best move using MCTS.

        Temperature drops to 0 (greedy) after move_number >= temp_threshold.
        """
        temperature = 1.0 if move_number < self.temp_threshold else 0.0
        probs = self.mcts.get_action_probs(
            game, temperature=temperature, add_noise=add_noise
        )
        if temperature == 0.0:
            return max(probs, key=probs.get)
        moves, weights = zip(*probs.items())
        return random.choices(moves, weights=weights)[0]
