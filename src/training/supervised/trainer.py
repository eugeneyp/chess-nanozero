"""SupervisedTrainer: training loop for AlphaZero-style chess with supervised data."""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.game.encoding import POLICY_SIZE
from src.neural_net.losses import AlphaZeroLoss
from src.neural_net.model import ChessResNet


class SupervisedTrainer:
    """Trains ChessResNet on supervised data from Lichess Elite Database."""

    def __init__(self, model: ChessResNet, config: dict, device: str = "cpu"):
        """
        Args:
            model: ChessResNet instance.
            config: Full config dict (reads from 'supervised_training' section).
            device: 'cpu', 'cuda', or 'mps'.
        """
        self.model = model.to(device)
        self.device = device

        cfg = config.get("supervised_training", config)
        self.lr = cfg.get("learning_rate", 1e-3)
        self.weight_decay = cfg.get("weight_decay", 1e-4)
        self.value_loss_weight = cfg.get("value_loss_weight", 1.0)
        self.num_epochs = cfg.get("num_epochs", 10)

        self.optimizer = AdamW(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.num_epochs)
        self.loss_fn = AlphaZeroLoss(value_loss_weight=self.value_loss_weight)

    def _move_to_one_hot(self, moves: Tensor) -> Tensor:
        """Convert move indices (B,) to one-hot policy targets (B, POLICY_SIZE)."""
        batch_size = moves.size(0)
        target = torch.zeros(batch_size, POLICY_SIZE, device=self.device)
        target.scatter_(1, moves.unsqueeze(1), 1.0)
        return target

    def _compute_accuracy(self, logits: Tensor, moves: Tensor) -> tuple[float, float]:
        """Compute top-1 and top-5 accuracy."""
        with torch.no_grad():
            top1 = (logits.argmax(dim=1) == moves).float().mean().item()
            _, top5_indices = logits.topk(5, dim=1)
            top5 = (top5_indices == moves.unsqueeze(1)).any(dim=1).float().mean().item()
        return top1, top5

    def train_epoch(self, loader: DataLoader) -> dict:
        """Run one training epoch.

        Returns dict with: loss, policy_loss, value_loss, top1_acc, top5_acc, samples_per_sec
        """
        self.model.train()
        total_loss = total_policy = total_value = 0.0
        total_top1 = total_top5 = 0.0
        total_samples = 0
        t0 = time.time()

        for batch in loader:
            boards = batch["board"].to(self.device)
            moves = batch["move"].to(self.device)
            results = batch["result"].to(self.device).unsqueeze(1)

            policy_logits, value_pred = self.model(boards)
            target_policy = self._move_to_one_hot(moves)
            total, p_loss, v_loss = self.loss_fn(policy_logits, value_pred, target_policy, results)

            self.optimizer.zero_grad()
            total.backward()
            self.optimizer.step()

            bs = boards.size(0)
            top1, top5 = self._compute_accuracy(policy_logits, moves)
            total_loss += total.item() * bs
            total_policy += p_loss.item() * bs
            total_value += v_loss.item() * bs
            total_top1 += top1 * bs
            total_top5 += top5 * bs
            total_samples += bs

        elapsed = time.time() - t0
        n = max(total_samples, 1)
        return {
            "loss": total_loss / n,
            "policy_loss": total_policy / n,
            "value_loss": total_value / n,
            "top1_acc": total_top1 / n,
            "top5_acc": total_top5 / n,
            "samples_per_sec": total_samples / max(elapsed, 1e-6),
        }

    def validate(self, loader: DataLoader) -> dict:
        """Run validation pass (no grad, no optimizer step).

        Returns same keys as train_epoch.
        """
        self.model.eval()
        total_loss = total_policy = total_value = 0.0
        total_top1 = total_top5 = 0.0
        total_samples = 0
        t0 = time.time()

        with torch.no_grad():
            for batch in loader:
                boards = batch["board"].to(self.device)
                moves = batch["move"].to(self.device)
                results = batch["result"].to(self.device).unsqueeze(1)

                policy_logits, value_pred = self.model(boards)
                target_policy = self._move_to_one_hot(moves)
                total, p_loss, v_loss = self.loss_fn(policy_logits, value_pred, target_policy, results)

                bs = boards.size(0)
                top1, top5 = self._compute_accuracy(policy_logits, moves)
                total_loss += total.item() * bs
                total_policy += p_loss.item() * bs
                total_value += v_loss.item() * bs
                total_top1 += top1 * bs
                total_top5 += top5 * bs
                total_samples += bs

        elapsed = time.time() - t0
        n = max(total_samples, 1)
        return {
            "loss": total_loss / n,
            "policy_loss": total_policy / n,
            "value_loss": total_value / n,
            "top1_acc": total_top1 / n,
            "top5_acc": total_top5 / n,
            "samples_per_sec": total_samples / max(elapsed, 1e-6),
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        num_epochs: int | None = None,
        checkpoint_dir: Path | None = None,
    ) -> None:
        """Full training loop with optional checkpointing.

        Args:
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data (None = skip validation).
            num_epochs: Number of epochs (defaults to self.num_epochs).
            checkpoint_dir: Directory to save checkpoints (None = no checkpoints).
        """
        if num_epochs is None:
            num_epochs = self.num_epochs

        if checkpoint_dir is not None:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_epoch(train_loader)
            self.scheduler.step()

            val_str = ""
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                val_str = (
                    f" | val_loss={val_metrics['loss']:.4f}"
                    f" val_top1={val_metrics['top1_acc']:.3f}"
                )

            print(
                f"Epoch {epoch}/{num_epochs}"
                f" | loss={train_metrics['loss']:.4f}"
                f" policy={train_metrics['policy_loss']:.4f}"
                f" value={train_metrics['value_loss']:.4f}"
                f" top1={train_metrics['top1_acc']:.3f}"
                f" top5={train_metrics['top5_acc']:.3f}"
                f" {train_metrics['samples_per_sec']:.0f} samp/s"
                + val_str
            )

            if checkpoint_dir is not None:
                ckpt_path = checkpoint_dir / f"epoch_{epoch:04d}.pt"
                self.save_checkpoint(ckpt_path, epoch)

    def save_checkpoint(self, path: Path, epoch: int) -> None:
        """Save model, optimizer, and scheduler state to file."""
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
            },
            path,
        )

    def load_checkpoint(self, path: Path) -> int:
        """Load checkpoint from file.

        Returns:
            epoch: The epoch number stored in the checkpoint.
        """
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        return ckpt["epoch"]
