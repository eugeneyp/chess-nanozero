"""Tests for chess board and move encoding."""

import chess
import numpy as np
import pytest
import random

from src.game.encoding import (
    encode_board,
    move_to_index,
    index_to_move,
    get_legal_move_mask,
    POLICY_SIZE,
)


# ---------------------------------------------------------------------------
# Board encoding tests
# ---------------------------------------------------------------------------

def test_encode_starting_position():
    """Starting position: verify piece planes are correctly populated."""
    board = chess.Board()
    planes = encode_board(board)

    assert planes.shape == (18, 8, 8)
    assert planes.dtype == np.float32

    # White's turn: no flip
    # Plane 0 = White pawns: rank 1 (index 1), all 8 files
    assert np.all(planes[0, 1, :] == 1.0), "White pawns on rank 1"
    assert np.sum(planes[0]) == 8.0

    # Plane 5 = White king: e1 = rank 0, file 4
    assert planes[5, 0, 4] == 1.0, "White king on e1"
    assert np.sum(planes[5]) == 1.0

    # Plane 4 = White queen: d1 = rank 0, file 3
    assert planes[4, 0, 3] == 1.0, "White queen on d1"

    # Plane 6 = Black pawns (opponent): rank 6, all 8 files
    assert np.all(planes[6, 6, :] == 1.0), "Black pawns on rank 6"
    assert np.sum(planes[6]) == 8.0

    # Plane 11 = Black king: e8 = rank 7, file 4
    assert planes[11, 7, 4] == 1.0, "Black king on e8"

    # Castling: both sides have all rights at start
    assert planes[12, 0, 0] == 1.0, "White kingside castling"
    assert planes[13, 0, 0] == 1.0, "White queenside castling"
    assert planes[14, 0, 0] == 1.0, "Black kingside castling"
    assert planes[15, 0, 0] == 1.0, "Black queenside castling"

    # No en passant at start
    assert np.sum(planes[16]) == 0.0


def test_encode_perspective_flip():
    """After 1.e4, it's Black's turn; board should be flipped for Black."""
    board = chess.Board()
    white_planes = encode_board(board)

    board.push(chess.Move.from_uci("e2e4"))
    black_planes = encode_board(board)

    # From Black's perspective, Black's own pawns appear at "bottom" (rank 1)
    # Black pawns are at rank 6 in standard coords; after flip, they appear at rank 1
    # planes[0] = current player (Black) pawns
    assert np.all(black_planes[0, 1, :] == 1.0), "Black pawns at rank 1 from Black's perspective"
    assert np.sum(black_planes[0]) == 8.0

    # Black king (current player) after flip: e8 mirrors to e1 = rank 0, file 4
    assert black_planes[5, 0, 4] == 1.0, "Black king at rank 0 from Black's perspective"

    # Opponent (White) pawns from Black's perspective:
    # White pawns at rank 1 + e-pawn moved to rank 3; after flip rank 1->6, rank 3->4
    # 7 pawns at rank 6, 1 pawn at rank 4 (e4 -> e5 perspective)
    assert np.sum(black_planes[6, 6, :]) == 7.0, "7 white pawns at rank 6 from Black's view"
    # e4 square from Black's perspective: e4 = rank 3, file 4; flipped = rank 4, file 4
    assert black_planes[6, 4, 4] == 1.0, "White e4 pawn at rank 4 from Black's view"


def test_encode_castling_rights():
    """Verify castling right planes with various right configurations."""
    # Only White kingside castling
    board = chess.Board(fen="r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQ - 0 1")
    # Remove Black's castling rights
    board.castling_rights = chess.BB_H1 | chess.BB_A1  # only White
    planes = encode_board(board)

    assert planes[12, 0, 0] == 1.0, "White kingside"
    assert planes[13, 0, 0] == 1.0, "White queenside"
    assert planes[14, 0, 0] == 0.0, "Black kingside absent"
    assert planes[15, 0, 0] == 0.0, "Black queenside absent"

    # No castling rights at all
    board2 = chess.Board(fen="r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w - - 0 1")
    planes2 = encode_board(board2)
    assert planes2[12, 0, 0] == 0.0
    assert planes2[13, 0, 0] == 0.0
    assert planes2[14, 0, 0] == 0.0
    assert planes2[15, 0, 0] == 0.0


def test_encode_en_passant():
    """En passant square is correctly encoded in plane 16."""
    # 1.e4 e5 2.d4 (now en passant on d3 is not set)
    # Set up a position where en passant is available
    # After 1.e4 e5 2.d4, en passant available at d3? No, d4 was White's move.
    # Actually: 1.e4 e5 2.d4 sets ep at d3 (d pawn passed through d3)
    board = chess.Board()
    board.push(chess.Move.from_uci("e2e4"))
    board.push(chess.Move.from_uci("e7e5"))
    board.push(chess.Move.from_uci("d2d4"))
    # Now Black can capture en passant on d3
    # board.ep_square should be d3 = square 19 (rank 2, file 3)
    assert board.ep_square is not None
    assert board.ep_square == chess.D3

    planes = encode_board(board)
    # It's Black's turn, so board is flipped
    # d3 = rank 2, file 3; after flip: rank 5, file 3
    assert planes[16, 5, 3] == 1.0, "En passant square at d3 (flipped to rank 5 for Black)"
    assert np.sum(planes[16]) == 1.0


def test_encode_no_en_passant():
    """Plane 16 is zero when no en passant is available."""
    board = chess.Board()
    board.push(chess.Move.from_uci("e2e4"))
    board.push(chess.Move.from_uci("e7e5"))
    # After two half-moves, no en passant (e5 ep was available after e7e5 but now gone)
    # Actually after e7e5, ep_square = e6 for White. Then after any other White move it's gone.
    # Let's push a non-pawn move
    board.push(chess.Move.from_uci("g1f3"))
    planes = encode_board(board)
    # No en passant after knight move
    # Note: after e7e5, Black's ep_square was e6, but now it's White's turn after Nf3
    # actually after g1f3 (White's move), ep_square should be None
    # Wait - e7e5 sets ep at e6. Then g1f3 is White's response - ep is still for White to capture.
    # After g1f3, it's Black's turn - ep is cleared.
    assert board.ep_square is None
    assert np.sum(planes[16]) == 0.0


# ---------------------------------------------------------------------------
# Move encoding tests
# ---------------------------------------------------------------------------

def test_move_encoding_round_trip():
    """For multiple positions, encode then decode all legal moves and verify match."""
    positions = [
        chess.Board(),  # starting position
        chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"),  # Italian game
        chess.Board("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),  # complex pos
        chess.Board("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),  # endgame
        chess.Board("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1"),  # castling available
    ]

    # Add promotion position (White pawn on 7th rank)
    promo_board = chess.Board("8/P7/8/8/8/8/8/k6K w - - 0 1")
    positions.append(promo_board)

    # Add Black promotion position
    promo_board_black = chess.Board("k6K/8/8/8/8/8/p7/8 b - - 0 1")
    positions.append(promo_board_black)

    # Add en passant position
    ep_board = chess.Board()
    ep_board.push(chess.Move.from_uci("e2e4"))
    ep_board.push(chess.Move.from_uci("d7d5"))
    ep_board.push(chess.Move.from_uci("e4e5"))
    ep_board.push(chess.Move.from_uci("f7f5"))
    positions.append(ep_board)  # en passant exf6 available

    for board in positions:
        legal_moves = list(board.legal_moves)
        assert len(legal_moves) > 0, f"No legal moves in position: {board.fen()}"

        for move in legal_moves:
            idx = move_to_index(move, board)
            decoded = index_to_move(idx, board)
            assert decoded == move, (
                f"Round-trip failed for {move.uci()} in {board.fen()}: "
                f"encoded to {idx}, decoded to {decoded.uci()}"
            )


def test_all_legal_moves_encodable():
    """move_to_index must not raise for any legal move in 20+ random positions."""
    random.seed(42)
    board = chess.Board()
    positions_tested = 0

    while positions_tested < 20:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            board = chess.Board()
            continue

        # Verify all legal moves in this position can be encoded
        for move in legal_moves:
            idx = move_to_index(move, board)
            assert 0 <= idx < POLICY_SIZE, (
                f"Index {idx} out of range for move {move.uci()} in {board.fen()}"
            )

        # Make a random move to advance
        move = random.choice(legal_moves)
        board.push(move)

        if board.is_game_over():
            board = chess.Board()

        positions_tested += 1


def test_no_duplicate_indices():
    """No two distinct legal moves in a position share the same policy index."""
    positions = [
        chess.Board(),
        chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"),
        chess.Board("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1"),
        chess.Board("8/P7/8/8/8/8/8/k6K w - - 0 1"),  # promotion position
        chess.Board("k6K/8/8/8/8/8/p7/8 b - - 0 1"),  # Black promotion
    ]

    for board in positions:
        legal_moves = list(board.legal_moves)
        indices = [move_to_index(m, board) for m in legal_moves]
        assert len(indices) == len(set(indices)), (
            f"Duplicate indices in {board.fen()}: "
            f"{[(str(m), i) for m, i in zip(legal_moves, indices)]}"
        )


def test_queen_promotion_uses_queen_planes():
    """Queen promotions must use planes 0-55 (queen move planes)."""
    # White pawn on a7, king positions ensure legality
    board = chess.Board("8/P7/8/8/8/8/8/k6K w - - 0 1")

    for move in board.legal_moves:
        if move.promotion == chess.QUEEN:
            idx = move_to_index(move, board)
            plane = idx // 64
            assert plane < 56, (
                f"Queen promotion {move.uci()} incorrectly uses plane {plane} (expected < 56)"
            )


def test_underpromotion_encoding():
    """Knight/bishop/rook promotions must use planes 64-72."""
    board = chess.Board("8/P7/8/8/8/8/8/k6K w - - 0 1")

    for move in board.legal_moves:
        if move.promotion in (chess.KNIGHT, chess.BISHOP, chess.ROOK):
            idx = move_to_index(move, board)
            plane = idx // 64
            assert 64 <= plane <= 72, (
                f"Underpromotion {move.uci()} uses plane {plane} (expected 64-72)"
            )


def test_underpromotion_black():
    """Black underpromotions also use planes 64-72."""
    board = chess.Board("k6K/8/8/8/8/8/p7/8 b - - 0 1")

    for move in board.legal_moves:
        if move.promotion in (chess.KNIGHT, chess.BISHOP, chess.ROOK):
            idx = move_to_index(move, board)
            plane = idx // 64
            assert 64 <= plane <= 72, (
                f"Black underpromotion {move.uci()} uses plane {plane} (expected 64-72)"
            )


def test_knight_moves_use_planes_56_63():
    """Knight moves must use planes 56-63."""
    board = chess.Board()
    # Get knight moves from starting position (e.g., Ng1f3)
    knight_moves = [m for m in board.legal_moves
                    if board.piece_at(m.from_square) and
                    board.piece_at(m.from_square).piece_type == chess.KNIGHT]

    assert len(knight_moves) > 0, "No knight moves in starting position"

    for move in knight_moves:
        idx = move_to_index(move, board)
        plane = idx // 64
        assert 56 <= plane <= 63, (
            f"Knight move {move.uci()} uses plane {plane} (expected 56-63)"
        )


def test_get_legal_move_mask():
    """Legal move mask has correct shape and count."""
    board = chess.Board()
    mask = get_legal_move_mask(board)

    assert mask.shape == (POLICY_SIZE,)
    assert mask.dtype == bool
    assert np.sum(mask) == len(list(board.legal_moves))


def test_castling_encoding():
    """Castling moves are encoded as king moves (queen-move planes)."""
    board = chess.Board("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1")

    castling_moves = [m for m in board.legal_moves
                      if board.is_castling(m)]

    assert len(castling_moves) == 2, f"Expected 2 castling moves, got {len(castling_moves)}"

    for move in castling_moves:
        idx = move_to_index(move, board)
        plane = idx // 64
        # Castling = king slides 2 squares: should be in queen-move planes 0-55
        assert plane < 56, f"Castling {move.uci()} uses plane {plane} (expected < 56)"

    # Round-trip
    for move in castling_moves:
        idx = move_to_index(move, board)
        decoded = index_to_move(idx, board)
        assert decoded == move, f"Castling round-trip failed: {move.uci()} -> {decoded.uci()}"


def test_index_range():
    """All legal move indices are in valid range [0, 4672)."""
    board = chess.Board()
    for move in board.legal_moves:
        idx = move_to_index(move, board)
        assert 0 <= idx < POLICY_SIZE, f"Index {idx} out of range for {move.uci()}"
