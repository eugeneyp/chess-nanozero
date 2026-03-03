#!/usr/bin/env python3
"""Convert a Lichess Elite PGN file into a .npz dataset for training.

Usage examples:

  # Step 1: 1K positions for pipeline smoke test
  python scripts/prepare_data.py \\
      --pgn data/lichess_elite/lichess_elite_2020-08.pgn \\
      --output data/lichess_elite/sample_1k.npz \\
      --max-positions 1000

  # Steps 2-3: 50K positions for local/GCP training
  python scripts/prepare_data.py \\
      --pgn data/lichess_elite/lichess_elite_2020-08.pgn \\
      --output data/lichess_elite/sample_50k.npz \\
      --max-positions 50000

  # Step 4: 5M+ positions for full training run
  python scripts/prepare_data.py \\
      --pgn data/lichess_elite/lichess_elite_2020-08.pgn \\
      --output data/lichess_elite/full_5m.npz \\
      --max-positions 5000000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert PGN to .npz training dataset")
    parser.add_argument("--pgn", required=True, type=Path, help="Input PGN file")
    parser.add_argument("--output", required=True, type=Path, help="Output .npz file")
    parser.add_argument(
        "--max-positions", type=int, default=None,
        help="Stop after collecting this many positions (default: all)"
    )
    parser.add_argument(
        "--max-games", type=int, default=None,
        help="Stop after parsing this many games (alternative limit)"
    )
    parser.add_argument(
        "--skip-first-n-moves", type=int, default=6,
        help="Skip first N half-moves of each game (default: 6)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.pgn.exists():
        print(f"Error: PGN file not found: {args.pgn}")
        raise SystemExit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    from src.training.supervised.prepare_data import parse_pgn_to_positions, save_positions

    print(f"Parsing {args.pgn} ...")
    print(f"  skip_first_n_moves = {args.skip_first_n_moves}")
    if args.max_positions:
        print(f"  max_positions      = {args.max_positions:,}")
    if args.max_games:
        print(f"  max_games          = {args.max_games:,}")

    t0 = time.time()
    positions = parse_pgn_to_positions(
        args.pgn,
        skip_first_n_moves=args.skip_first_n_moves,
        max_games=args.max_games,
        max_positions=args.max_positions,
    )

    elapsed = time.time() - t0
    print(f"Parsed {len(positions):,} positions in {elapsed:.1f}s")

    print(f"Saving to {args.output} ...")
    save_positions(positions, args.output)

    size_mb = args.output.stat().st_size / 1_048_576
    print(f"Done. File size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
