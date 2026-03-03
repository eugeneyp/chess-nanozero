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

    import numpy as np
    from src.training.supervised.prepare_data import stream_pgn_positions

    print(f"Parsing {args.pgn} ...")
    print(f"  skip_first_n_moves = {args.skip_first_n_moves}")
    if args.max_positions:
        print(f"  max_positions      = {args.max_positions:,}")
    if args.max_games:
        print(f"  max_games          = {args.max_games:,}")

    t0 = time.time()

    if args.max_positions:
        # Pre-allocate arrays — avoids building a Python list of N numpy objects
        # Memory: 5M positions ≈ 5.8 GB boards + 40 MB moves/results = ~5.8 GB peak
        n = args.max_positions
        boards = np.empty((n, 18, 8, 8), dtype=np.float32)
        moves = np.empty(n, dtype=np.int32)
        results = np.empty(n, dtype=np.float32)

        count = 0
        last_report = 0
        for encoding, move_idx, value in stream_pgn_positions(
            args.pgn,
            skip_first_n_moves=args.skip_first_n_moves,
            max_games=args.max_games,
            max_positions=args.max_positions,
        ):
            boards[count] = encoding
            moves[count] = move_idx
            results[count] = value
            count += 1
            if count - last_report >= 100_000:
                elapsed = time.time() - t0
                print(f"  {count:,} positions  ({count / elapsed:.0f} pos/s)", flush=True)
                last_report = count

        # Trim to actual count (in case fewer positions were found)
        boards = boards[:count]
        moves = moves[:count]
        results = results[:count]
    else:
        # No limit — collect into list then stack (fine for small extractions)
        pos_list = list(stream_pgn_positions(
            args.pgn,
            skip_first_n_moves=args.skip_first_n_moves,
            max_games=args.max_games,
        ))
        count = len(pos_list)
        boards = np.stack([p[0] for p in pos_list], axis=0).astype(np.float32)
        moves = np.array([p[1] for p in pos_list], dtype=np.int32)
        results = np.array([p[2] for p in pos_list], dtype=np.float32)

    elapsed = time.time() - t0
    print(f"Parsed {count:,} positions in {elapsed:.1f}s")

    print(f"Saving to {args.output} ...")
    np.savez_compressed(args.output, boards=boards, moves=moves, results=results)

    size_mb = args.output.stat().st_size / 1_048_576
    print(f"Done. File size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
