"""ChessDataset: PyTorch Dataset for supervised chess training.

Loads .npz files produced by prepare_data.py and provides train/val splits.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class ChessDataset(Dataset):
    """Dataset of chess positions for supervised training.

    Each item is a dict with:
        board  : (18, 8, 8) float32 tensor — encoded board state
        move   : long scalar               — policy index of move played
        result : float32 scalar            — game result from current player's view
    """

    def __init__(
        self,
        npz_files: list[str | Path],
        split: str = "train",
        val_split: float = 0.05,
        seed: int = 42,
    ):
        """
        Args:
            npz_files: List of .npz file paths to load.
            split: "train" or "val".
            val_split: Fraction of data to use as validation (0.0 = all train).
            seed: Random seed for reproducible shuffling.
        """
        boards_list, moves_list, results_list = [], [], []

        for path in npz_files:
            data = np.load(path)
            boards_list.append(data["boards"])
            moves_list.append(data["moves"])
            results_list.append(data["results"])

        boards_all = np.concatenate(boards_list, axis=0)
        moves_all = np.concatenate(moves_list, axis=0)
        results_all = np.concatenate(results_list, axis=0)

        n = len(boards_all)
        rng = np.random.default_rng(seed)
        indices = rng.permutation(n)

        val_size = int(n * val_split)
        if split == "val":
            idx = indices[:val_size]
        else:
            idx = indices[val_size:]

        self.boards = boards_all[idx]
        self.moves = moves_all[idx]
        self.results = results_all[idx]

    def __len__(self) -> int:
        return len(self.boards)

    def __getitem__(self, idx: int) -> dict:
        return {
            "board": torch.tensor(self.boards[idx], dtype=torch.float32),
            "move": torch.tensor(int(self.moves[idx]), dtype=torch.long),
            "result": torch.tensor(float(self.results[idx]), dtype=torch.float32),
        }
