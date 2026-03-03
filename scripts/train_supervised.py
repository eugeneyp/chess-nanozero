#!/usr/bin/env python3
"""CLI entry point for supervised training.

Usage:
    # Step 1: tiny model on 1K positions
    python scripts/train_supervised.py \\
        --config configs/tiny.yaml \\
        --data data/sample_1k.npz \\
        --checkpoint-dir checkpoints/step1/

    # Step 2: tiny model on 50K positions
    python scripts/train_supervised.py \\
        --config configs/tiny.yaml \\
        --data data/sample_50k.npz \\
        --checkpoint-dir checkpoints/step2/

    # Step 3+: medium model on larger data
    python scripts/train_supervised.py \\
        --config configs/medium.yaml \\
        --data data/lichess_elite_2023_01.npz \\
        --checkpoint-dir checkpoints/medium/
"""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path

import torch
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supervised training for chess AlphaZero")
    parser.add_argument("--config", required=True, type=Path, help="YAML config file")
    parser.add_argument(
        "--data", required=True, nargs="+", type=Path,
        help="One or more .npz data files"
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None, help="Resume from checkpoint")
    parser.add_argument("--num-epochs", type=int, default=None, help="Override num_epochs")
    parser.add_argument("--device", type=str, default=None, help="cpu / cuda / mps")
    parser.add_argument(
        "--log-file", type=Path, default=None,
        help="CSV log file path. Defaults to logs/<config_stem>_<timestamp>.csv"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Select device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Resolve log file path
    log_file = args.log_file
    if log_file is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = Path("logs") / f"{args.config.stem}_{timestamp}.csv"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"Logging to: {log_file}")

    # Build model
    from src.neural_net.model import ChessResNet
    model = ChessResNet.from_config(config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Build datasets
    from src.training.supervised.dataset import ChessDataset
    train_cfg = config.get("supervised_training", config)
    val_split = train_cfg.get("val_split", 0.05)

    train_ds = ChessDataset(args.data, split="train", val_split=val_split)
    val_ds = ChessDataset(args.data, split="val", val_split=val_split)
    print(f"Train: {len(train_ds):,} positions | Val: {len(val_ds):,} positions")

    batch_size = train_cfg.get("batch_size", 256)
    # num_workers=0 on CPU: avoids multiprocessing overhead and NumPy compat
    # warnings when workers fail to spawn. Switch to 4+ on GPU (GCP).
    num_workers = 0 if device == "cpu" else min(4, torch.get_num_threads())

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device != "cpu"),
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device != "cpu"),
    ) if len(val_ds) > 0 else None

    # Build trainer
    from src.training.supervised.trainer import SupervisedTrainer
    trainer = SupervisedTrainer(model, config, device=device)

    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
        print(f"Resumed from epoch {start_epoch}")

    num_epochs = args.num_epochs or train_cfg.get("num_epochs", 10)

    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        checkpoint_dir=args.checkpoint_dir,
        log_file=log_file,
    )


if __name__ == "__main__":
    main()
