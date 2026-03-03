"""PGN → .npz encoder for supervised training.

Parses Lichess Elite PGN files, encodes each position as an (18, 8, 8) board
tensor, maps the played move to a policy index, and assigns a game result value
from the current player's perspective.
"""

from __future__ import annotations

import io
from pathlib import Path

import chess
import chess.pgn
import numpy as np

from src.game.encoding import encode_board, move_to_index


def parse_pgn_to_positions(
    pgn_path: str | Path | io.StringIO,
    skip_first_n_moves: int = 6,
    max_games: int | None = None,
    max_positions: int | None = None,
) -> list[tuple[np.ndarray, int, float]]:
    """Parse PGN → list of (board_encoding, move_index, value) triples.

    Args:
        pgn_path: Path to PGN file or StringIO object with PGN content.
        skip_first_n_moves: Number of half-moves (plies) to skip from game start.
        max_games: Stop after parsing this many games (None = all).
        max_positions: Stop after collecting this many positions (None = all).
            Stops mid-file as soon as the limit is reached — efficient for
            extracting small samples from large PGN files.

    Returns:
        List of (board_encoding, move_index, value) where:
            board_encoding : (18, 8, 8) float32 — from current player's perspective
            move_index     : int in [0, 4671]   — policy index of the move played
            value          : +1.0 / 0.0 / -1.0 — from current player's perspective
    """
    if isinstance(pgn_path, (str, Path)):
        pgn_file = open(pgn_path, encoding="utf-8", errors="ignore")
        should_close = True
    else:
        pgn_file = pgn_path
        should_close = False

    positions = []
    games_parsed = 0

    try:
        while True:
            if max_games is not None and games_parsed >= max_games:
                break
            if max_positions is not None and len(positions) >= max_positions:
                break

            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break

            result_str = game.headers.get("Result", "*")
            if result_str == "1-0":
                white_result = 1.0
            elif result_str == "0-1":
                white_result = -1.0
            elif result_str == "1/2-1/2":
                white_result = 0.0
            else:
                # Skip unfinished games
                games_parsed += 1
                continue

            board = game.board()
            ply = 0

            for move in game.mainline_moves():
                if max_positions is not None and len(positions) >= max_positions:
                    break

                if ply >= skip_first_n_moves:
                    # Value from current player's perspective
                    if board.turn == chess.WHITE:
                        value = white_result
                    else:
                        value = -white_result

                    encoding = encode_board(board)
                    move_idx = move_to_index(move, board)
                    positions.append((encoding, move_idx, float(value)))

                board.push(move)
                ply += 1

            games_parsed += 1
    finally:
        if should_close:
            pgn_file.close()

    return positions


def save_positions(
    positions: list[tuple[np.ndarray, int, float]],
    output_path: str | Path,
) -> None:
    """Save positions list to compressed .npz.

    Keys: boards (N,18,8,8) float32, moves (N,) int32, results (N,) float32.
    """
    if not positions:
        raise ValueError("No positions to save")

    boards = np.stack([p[0] for p in positions], axis=0).astype(np.float32)
    moves = np.array([p[1] for p in positions], dtype=np.int32)
    results = np.array([p[2] for p in positions], dtype=np.float32)

    np.savez_compressed(output_path, boards=boards, moves=moves, results=results)
