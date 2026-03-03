#!/usr/bin/env python3
"""CLI entry point for supervised training.

Usage:
    # Single file (standard mode — all data loaded into RAM)
    python3 scripts/train_supervised.py \\
        --config configs/tiny.yaml \\
        --data data/lichess_elite/sample_50k.npz \\
        --checkpoint-dir checkpoints/step3/

    # Multiple chunk files (chunked mode — one file at a time, avoids OOM)
    python3 scripts/train_supervised.py \\
        --config configs/medium.yaml \\
        --data data/lichess_elite/sample_5m_part*.npz \\
        --checkpoint-dir checkpoints/step4/
"""

from __future__ import annotations

import argparse
import csv
import datetime
import random
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

    train_cfg = config.get("supervised_training", config)
    batch_size = train_cfg.get("batch_size", 256)
    val_split = train_cfg.get("val_split", 0.05)
    num_epochs = args.num_epochs or train_cfg.get("num_epochs", 10)
    # num_workers=0 on CPU avoids multiprocessing/NumPy compat warnings
    num_workers = 0 if device == "cpu" else min(4, torch.get_num_threads())

    from src.training.supervised.trainer import SupervisedTrainer
    trainer = SupervisedTrainer(model, config, device=device)

    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
        print(f"Resumed from epoch {start_epoch}")

    if args.checkpoint_dir is not None:
        Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    if len(args.data) > 1:
        # Chunked mode: load one file at a time to avoid OOM.
        # 5M positions = ~22 GB boards — too large to concatenate in RAM.
        # Each 500K-position chunk is ~2.2 GB; we process one per sub-epoch.
        print(f"Chunked mode: {len(args.data)} files, one loaded at a time")
        _train_chunked(
            trainer, args.data, batch_size, val_split, num_epochs, start_epoch,
            num_workers, device, args.checkpoint_dir, log_file,
        )
    else:
        # Standard mode: load all data into RAM (fine for ≤500K positions)
        from src.training.supervised.dataset import ChessDataset
        train_ds = ChessDataset(args.data, split="train", val_split=val_split)
        val_ds = ChessDataset(args.data, split="val", val_split=val_split)
        print(f"Train: {len(train_ds):,} positions | Val: {len(val_ds):,} positions")

        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=(device != "cpu"),
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=(device != "cpu"),
        ) if len(val_ds) > 0 else None

        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            checkpoint_dir=args.checkpoint_dir,
            log_file=log_file,
        )


def _train_chunked(
    trainer,
    data_files: list[Path],
    batch_size: int,
    val_split: float,
    num_epochs: int,
    start_epoch: int,
    num_workers: int,
    device: str,
    checkpoint_dir: Path | None,
    log_file: Path | None,
) -> None:
    """Train epoch-by-epoch, loading one chunk file at a time.

    Each real epoch does one pass over every chunk (shuffled order).
    Memory peak: ~2.2 GB per chunk + val set (~110 MB) — fits in 16 GB RAM.
    Val set comes from the last chunk's val_split fraction (consistent across epochs).
    """
    from src.training.supervised.dataset import ChessDataset

    # Fixed val set from the last chunk (consistent across all epochs)
    val_ds = ChessDataset([data_files[-1]], split="val", val_split=val_split)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device != "cpu"),
    ) if len(val_ds) > 0 else None
    print(f"Val: {len(val_ds):,} positions (from last chunk)")

    csv_fields = [
        "timestamp", "epoch",
        "train_loss", "train_policy_loss", "train_value_loss",
        "train_top1", "train_top5", "samples_per_sec",
        "val_loss", "val_policy_loss", "val_value_loss",
        "val_top1", "val_top5",
    ]
    if log_file is not None:
        write_header = not Path(log_file).exists()
        log_fh = open(log_file, "a", newline="")
        writer = csv.DictWriter(log_fh, fieldnames=csv_fields)
        if write_header:
            writer.writeheader()
            log_fh.flush()
    else:
        log_fh = None
        writer = None

    try:
        for epoch in range(start_epoch + 1, num_epochs + 1):
            # Shuffle chunk order so each epoch sees data in a different sequence
            chunk_order = list(range(len(data_files)))
            random.shuffle(chunk_order)

            # Weighted accumulation of metrics across all chunks
            total_loss = total_policy = total_value = 0.0
            total_top1 = total_top5 = total_speed = 0.0
            total_samples = 0

            for chunk_num, ci in enumerate(chunk_order, 1):
                chunk_ds = ChessDataset([data_files[ci]], split="train", val_split=0.0)
                chunk_loader = torch.utils.data.DataLoader(
                    chunk_ds, batch_size=batch_size, shuffle=True,
                    num_workers=num_workers, pin_memory=(device != "cpu"),
                )
                m = trainer.train_epoch(chunk_loader)
                n = len(chunk_ds)
                total_loss += m["loss"] * n
                total_policy += m["policy_loss"] * n
                total_value += m["value_loss"] * n
                total_top1 += m["top1_acc"] * n
                total_top5 += m["top5_acc"] * n
                total_speed += m["samples_per_sec"] * n
                total_samples += n
                print(
                    f"  chunk [{chunk_num}/{len(data_files)}]"
                    f" top1={m['top1_acc']:.3f}"
                    f" {m['samples_per_sec']:.0f} samp/s",
                    flush=True,
                )

            trainer.scheduler.step()

            n = max(total_samples, 1)
            train_metrics = {
                "loss": total_loss / n,
                "policy_loss": total_policy / n,
                "value_loss": total_value / n,
                "top1_acc": total_top1 / n,
                "top5_acc": total_top5 / n,
                "samples_per_sec": total_speed / n,
            }

            val_metrics: dict = {}
            val_str = ""
            if val_loader is not None:
                val_metrics = trainer.validate(val_loader)
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

            if writer is not None:
                row = {
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "epoch": epoch,
                    "train_loss": f"{train_metrics['loss']:.6f}",
                    "train_policy_loss": f"{train_metrics['policy_loss']:.6f}",
                    "train_value_loss": f"{train_metrics['value_loss']:.6f}",
                    "train_top1": f"{train_metrics['top1_acc']:.6f}",
                    "train_top5": f"{train_metrics['top5_acc']:.6f}",
                    "samples_per_sec": f"{train_metrics['samples_per_sec']:.1f}",
                    "val_loss": f"{val_metrics.get('loss', ''):.6f}" if val_metrics else "",
                    "val_policy_loss": f"{val_metrics.get('policy_loss', ''):.6f}" if val_metrics else "",
                    "val_value_loss": f"{val_metrics.get('value_loss', ''):.6f}" if val_metrics else "",
                    "val_top1": f"{val_metrics.get('top1_acc', ''):.6f}" if val_metrics else "",
                    "val_top5": f"{val_metrics.get('top5_acc', ''):.6f}" if val_metrics else "",
                }
                writer.writerow(row)
                log_fh.flush()

            if checkpoint_dir is not None:
                ckpt_path = Path(checkpoint_dir) / f"epoch_{epoch:04d}.pt"
                trainer.save_checkpoint(ckpt_path, epoch)
    finally:
        if log_fh is not None:
            log_fh.close()


if __name__ == "__main__":
    main()
