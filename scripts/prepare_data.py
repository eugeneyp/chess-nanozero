#!/usr/bin/env python3
"""Convert a Lichess Elite PGN file into a .npz dataset for training.

For large extractions (>500K positions) the output is split into chunk files
to stay within available RAM. Pass all chunk files to train_supervised.py:

  python3 scripts/train_supervised.py \\
      --config configs/medium.yaml \\
      --data data/lichess_elite/sample_5m_part*.npz

Usage examples:

  # Step 1: 1K positions for pipeline smoke test
  python3 scripts/prepare_data.py \\
      --pgn data/lichess_elite/lichess_elite_2025-11.pgn \\
      --output data/lichess_elite/sample_1k.npz \\
      --max-positions 1000

  # Steps 2-3: 50K positions for local/GCP training
  python3 scripts/prepare_data.py \\
      --pgn data/lichess_elite/lichess_elite_2025-11.pgn \\
      --output data/lichess_elite/sample_50k.npz \\
      --max-positions 50000

  # Step 4: 5M positions (auto-chunked into ~500K-position files)
  python3 scripts/prepare_data.py \\
      --pgn data/lichess_elite/lichess_elite_2025-11.pgn \\
      --output data/lichess_elite/sample_5m.npz \\
      --max-positions 5000000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

# Each chunk: 500K × 18×8×8 × 4 bytes ≈ 2.2 GB — fits comfortably in RAM
DEFAULT_CHUNK_SIZE = 500_000


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
    parser.add_argument(
        "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
        help=f"Positions per output file for large extractions (default: {DEFAULT_CHUNK_SIZE:,})"
    )
    return parser.parse_args()


def save_chunk(
    path: Path,
    boards,   # np.ndarray (n, 18, 8, 8) float32
    moves,    # np.ndarray (n,) int32
    results,  # np.ndarray (n,) float32
    count: int,
) -> None:
    import numpy as np
    np.savez_compressed(
        path,
        boards=boards[:count],
        moves=moves[:count],
        results=results[:count],
    )


def main() -> None:
    args = parse_args()

    if not args.pgn.exists():
        print(f"Error: PGN file not found: {args.pgn}")
        raise SystemExit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    import numpy as np
    from src.training.supervised.prepare_data import stream_pgn_positions

    chunk_size = args.chunk_size
    max_pos = args.max_positions

    print(f"Parsing {args.pgn} ...")
    print(f"  skip_first_n_moves = {args.skip_first_n_moves}")
    if max_pos:
        print(f"  max_positions      = {max_pos:,}")
    if args.max_games:
        print(f"  max_games          = {args.max_games:,}")
    print(f"  chunk_size         = {chunk_size:,}  (~{chunk_size * 18 * 8 * 8 * 4 / 1e9:.1f} GB/chunk)")

    t0 = time.time()

    # Allocate one chunk-sized buffer — reused across chunks
    buf_boards = np.empty((chunk_size, 18, 8, 8), dtype=np.float32)
    buf_moves = np.empty(chunk_size, dtype=np.int32)
    buf_results = np.empty(chunk_size, dtype=np.float32)

    chunk_idx = 1
    pos_in_chunk = 0
    total = 0
    saved_paths: list[Path] = []

    for encoding, move_idx, value in stream_pgn_positions(
        args.pgn,
        skip_first_n_moves=args.skip_first_n_moves,
        max_games=args.max_games,
        max_positions=max_pos,
    ):
        buf_boards[pos_in_chunk] = encoding
        buf_moves[pos_in_chunk] = move_idx
        buf_results[pos_in_chunk] = value
        pos_in_chunk += 1
        total += 1

        if pos_in_chunk == chunk_size:
            elapsed = time.time() - t0
            print(f"  {total:,} positions  ({total / elapsed:.0f} pos/s)", flush=True)
            part_path = args.output.parent / f"{args.output.stem}_part{chunk_idx:03d}.npz"
            save_chunk(part_path, buf_boards, buf_moves, buf_results, pos_in_chunk)
            saved_paths.append(part_path)
            print(f"  -> saved {part_path}", flush=True)
            chunk_idx += 1
            pos_in_chunk = 0

    # Save final partial chunk
    if pos_in_chunk > 0:
        if chunk_idx == 1:
            # Only one chunk — save directly with the requested output name
            part_path = args.output
        else:
            part_path = args.output.parent / f"{args.output.stem}_part{chunk_idx:03d}.npz"
        save_chunk(part_path, buf_boards, buf_moves, buf_results, pos_in_chunk)
        saved_paths.append(part_path)

    elapsed = time.time() - t0
    print(f"\nDone. {total:,} positions in {elapsed:.1f}s ({total / elapsed:.0f} pos/s)")

    if len(saved_paths) == 1:
        size_mb = saved_paths[0].stat().st_size / 1_048_576
        print(f"Output: {saved_paths[0]}  ({size_mb:.1f} MB)")
    else:
        total_mb = sum(p.stat().st_size for p in saved_paths) / 1_048_576
        print(f"Output: {len(saved_paths)} chunk files  ({total_mb:.1f} MB total)")
        for p in saved_paths:
            mb = p.stat().st_size / 1_048_576
            print(f"  {p}  ({mb:.1f} MB)")
        stem = args.output.stem
        print(f"\nTo train on all chunks:")
        print(f"  python3 scripts/train_supervised.py \\")
        print(f"      --config configs/medium.yaml \\")
        print(f"      --data {args.output.parent}/{stem}_part*.npz \\")
        print(f"      --checkpoint-dir checkpoints/step4/")


if __name__ == "__main__":
    main()
