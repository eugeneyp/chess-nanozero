#!/usr/bin/env python3
"""Step 5: Fine-tune medium model on existing 5M dataset with ReduceLROnPlateau.

Loads model weights only from a trained checkpoint (not optimizer/scheduler),
starts a fresh AdamW at lower LR, and reduces LR when val_top1 plateaus.
This tests whether the model has exhausted learning on the 5M dataset.

Usage (GCP — same data files as step 4):
    python scripts/train_step5.py \
        --checkpoint models/medium1.pt \
        --data data/lichess_elite/sample_5m_part*.npz \
        --checkpoint-dir checkpoints/step5/

    # Or single-file (for testing):
    python scripts/train_step5.py \
        --checkpoint models/medium1.pt \
        --data data/lichess_elite/sample_50k.npz \
        --checkpoint-dir checkpoints/step5_test/ \
        --num-epochs 3
"""

from __future__ import annotations

import argparse
import csv
import datetime
import random
from pathlib import Path

import torch
import torch.optim as optim
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 5: fine-tune with ReduceLROnPlateau")
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="Trained .pt checkpoint to load model weights from")
    parser.add_argument("--config", type=Path, default=Path("configs/medium.yaml"))
    parser.add_argument("--data", required=True, nargs="+", type=Path,
                        help="One or more .npz data files (glob-expanded by shell)")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/step5"))
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Initial learning rate (default 1e-4, 10× lower than step 4)")
    parser.add_argument("--patience", type=int, default=2,
                        help="ReduceLROnPlateau patience (epochs with no improvement)")
    parser.add_argument("--factor", type=float, default=0.5,
                        help="LR reduction factor when plateau detected")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--log-file", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    # Log file
    log_file = args.log_file
    if log_file is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = Path("logs") / f"step5_medium_5m_{ts}.csv"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"Logging to: {log_file}")

    # Build model and load weights only (skip optimizer/scheduler from step 4)
    from src.neural_net.model import ChessResNet
    model = ChessResNet.from_config(config)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    loaded_epoch = ckpt.get("epoch", "?")
    print(f"Loaded weights from {args.checkpoint} (step 4 epoch {loaded_epoch})")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Fresh optimizer + ReduceLROnPlateau (maximise val_top1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=args.factor, patience=args.patience,
    )
    print(f"Optimizer: AdamW lr={args.lr:.1e}  ReduceLROnPlateau patience={args.patience} factor={args.factor}")

    # Build trainer, then override its optimizer (keep its loss_fn, device setup, etc.)
    from src.training.supervised.trainer import SupervisedTrainer
    trainer = SupervisedTrainer(model, config, device=device)
    trainer.optimizer = optimizer  # replace cosine-linked AdamW with ours

    train_cfg = config.get("supervised_training", config)
    batch_size = train_cfg.get("batch_size", 2048)
    val_split = train_cfg.get("val_split", 0.05)
    num_workers = 0 if device == "cpu" else min(4, torch.get_num_threads())

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # CSV logging (append so it can be resumed)
    csv_fields = [
        "timestamp", "epoch", "lr",
        "train_loss", "train_policy_loss", "train_value_loss",
        "train_top1", "train_top5", "samples_per_sec",
        "val_loss", "val_policy_loss", "val_value_loss",
        "val_top1", "val_top5",
    ]
    write_header = not log_file.exists()
    log_fh = open(log_file, "a", newline="")
    writer = csv.DictWriter(log_fh, fieldnames=csv_fields)
    if write_header:
        writer.writeheader()
        log_fh.flush()

    # Fixed val set from last chunk (consistent across epochs)
    from src.training.supervised.dataset import ChessDataset
    val_ds = ChessDataset([args.data[-1]], split="val", val_split=val_split)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device != "cpu"),
    ) if len(val_ds) > 0 else None
    print(f"Val: {len(val_ds):,} positions (from last chunk: {args.data[-1].name})")

    best_val_top1 = 0.0

    try:
        for epoch in range(1, args.num_epochs + 1):
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"\n--- Step 5 Epoch {epoch}/{args.num_epochs}  LR={current_lr:.2e} ---", flush=True)

            # Shuffle chunk order each epoch
            chunk_order = list(range(len(args.data)))
            random.shuffle(chunk_order)

            # Weighted accumulation across chunks
            total_loss = total_policy = total_value = 0.0
            total_top1 = total_top5 = total_speed = 0.0
            total_samples = 0

            for ci_num, ci in enumerate(chunk_order, 1):
                chunk_ds = ChessDataset([args.data[ci]], split="train", val_split=0.0)
                chunk_loader = torch.utils.data.DataLoader(
                    chunk_ds, batch_size=batch_size, shuffle=True,
                    num_workers=num_workers, pin_memory=(device != "cpu"),
                )
                m = trainer.train_epoch(chunk_loader)
                n = len(chunk_ds)
                total_loss  += m["loss"] * n
                total_policy += m["policy_loss"] * n
                total_value  += m["value_loss"] * n
                total_top1   += m["top1_acc"] * n
                total_top5   += m["top5_acc"] * n
                total_speed  += m["samples_per_sec"] * n
                total_samples += n
                print(
                    f"  chunk [{ci_num}/{len(args.data)}]"
                    f" top1={m['top1_acc']:.3f}"
                    f" {m['samples_per_sec']:.0f} samp/s",
                    flush=True,
                )

            n = max(total_samples, 1)
            train_metrics = {
                "loss":            total_loss / n,
                "policy_loss":     total_policy / n,
                "value_loss":      total_value / n,
                "top1_acc":        total_top1 / n,
                "top5_acc":        total_top5 / n,
                "samples_per_sec": total_speed / n,
            }

            val_metrics: dict = {}
            val_str = ""
            if val_loader is not None:
                val_metrics = trainer.validate(val_loader)
                val_top1 = val_metrics["top1_acc"]
                scheduler.step(val_top1)  # ReduceLROnPlateau checks for improvement
                marker = " ★" if val_top1 > best_val_top1 else ""
                if val_top1 > best_val_top1:
                    best_val_top1 = val_top1
                val_str = (
                    f" | val_loss={val_metrics['loss']:.4f}"
                    f" val_top1={val_top1:.3f}"
                    f" val_top5={val_metrics['top5_acc']:.3f}"
                    f"{marker}"
                )

            print(
                f"Epoch {epoch}/{args.num_epochs}"
                f" | loss={train_metrics['loss']:.4f}"
                f" policy={train_metrics['policy_loss']:.4f}"
                f" value={train_metrics['value_loss']:.4f}"
                f" top1={train_metrics['top1_acc']:.3f}"
                f" top5={train_metrics['top5_acc']:.3f}"
                f" {train_metrics['samples_per_sec']:.0f} samp/s"
                + val_str
            )

            row = {
                "timestamp":        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "epoch":            epoch,
                "lr":               f"{current_lr:.2e}",
                "train_loss":       f"{train_metrics['loss']:.6f}",
                "train_policy_loss": f"{train_metrics['policy_loss']:.6f}",
                "train_value_loss": f"{train_metrics['value_loss']:.6f}",
                "train_top1":       f"{train_metrics['top1_acc']:.6f}",
                "train_top5":       f"{train_metrics['top5_acc']:.6f}",
                "samples_per_sec":  f"{train_metrics['samples_per_sec']:.1f}",
                "val_loss":         f"{val_metrics.get('loss', 0):.6f}" if val_metrics else "",
                "val_policy_loss":  f"{val_metrics.get('policy_loss', 0):.6f}" if val_metrics else "",
                "val_value_loss":   f"{val_metrics.get('value_loss', 0):.6f}" if val_metrics else "",
                "val_top1":         f"{val_metrics.get('top1_acc', 0):.6f}" if val_metrics else "",
                "val_top5":         f"{val_metrics.get('top5_acc', 0):.6f}" if val_metrics else "",
            }
            writer.writerow(row)
            log_fh.flush()

            ckpt_path = args.checkpoint_dir / f"epoch_{epoch:04d}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }, ckpt_path)
            print(f"Saved {ckpt_path}  (best val_top1={best_val_top1:.3f})")

    finally:
        log_fh.close()


if __name__ == "__main__":
    main()
