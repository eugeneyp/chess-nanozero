"""Tests for ChessGame wrapper."""

import chess
import pytest
from src.game.chess_game import ChessGame


def test_legal_moves():
    """ChessGame.get_legal_moves() matches board.legal_moves."""
    game = ChessGame()
    game_moves = set(str(m) for m in game.get_legal_moves())
    board_moves = set(str(m) for m in game.board.legal_moves)
    assert game_moves == board_moves
    assert len(game_moves) == 20  # starting position has 20 legal moves


def test_legal_moves_mid_game():
    """Verify legal moves match after several moves."""
    game = ChessGame()
    game.push(chess.Move.from_uci("e2e4"))
    game.push(chess.Move.from_uci("e7e5"))
    game.push(chess.Move.from_uci("g1f3"))
    game_moves = set(str(m) for m in game.get_legal_moves())
    board_moves = set(str(m) for m in game.board.legal_moves)
    assert game_moves == board_moves


def test_game_result_checkmate():
    """Fool's mate: Black wins after 4 moves."""
    game = ChessGame()
    # Fool's mate
    game.push(chess.Move.from_uci("f2f3"))
    game.push(chess.Move.from_uci("e7e5"))
    game.push(chess.Move.from_uci("g2g4"))
    game.push(chess.Move.from_uci("d8h4"))

    assert game.is_terminal()
    assert game.get_winner() == 2  # Black wins
    assert game.get_result(1) == -1.0  # White loses
    assert game.get_result(2) == 1.0   # Black wins


def test_game_result_stalemate():
    """Known stalemate position."""
    # Classic stalemate: Black king has no legal moves, not in check
    # Set up: White Kh6, White Qg6, Black Kh8
    board = chess.Board(fen="7k/8/6QK/8/8/8/8/8 b - - 0 1")
    game = ChessGame(board)

    assert game.is_terminal()
    assert game.get_winner() is None  # draw
    assert game.get_result(1) == 0.0
    assert game.get_result(2) == 0.0


def test_game_result_50_move():
    """50-move rule triggers is_terminal."""
    board = chess.Board()
    board.halfmove_clock = 100  # triggers 50-move rule with claim_draw=True
    game = ChessGame(board)

    assert game.is_terminal()
    assert game.get_winner() is None
    assert game.get_result(1) == 0.0
    assert game.get_result(2) == 0.0


def test_make_undo_move():
    """push/pop preserves board state correctly."""
    game = ChessGame()
    original_fen = game.board.fen()

    move = chess.Move.from_uci("e2e4")
    game.push(move)
    assert game.board.fen() != original_fen
    assert game.current_player == 2  # Black's turn

    popped = game.pop()
    assert str(popped) == "e2e4"
    assert game.board.fen() == original_fen
    assert game.current_player == 1  # White's turn again


def test_make_move_non_mutating():
    """make_move returns a new game without modifying original."""
    game = ChessGame()
    original_fen = game.board.fen()

    move = chess.Move.from_uci("e2e4")
    new_game = game.make_move(move)

    # Original unchanged
    assert game.board.fen() == original_fen
    assert game.current_player == 1

    # New game has move applied
    assert new_game.board.fen() != original_fen
    assert new_game.current_player == 2


def test_clone():
    """clone() creates independent copy."""
    game = ChessGame()
    game.push(chess.Move.from_uci("e2e4"))
    cloned = game.clone()

    assert cloned.board.fen() == game.board.fen()

    # Modifying clone doesn't affect original
    cloned.push(chess.Move.from_uci("e7e5"))
    assert cloned.board.fen() != game.board.fen()


def test_current_player():
    """current_player alternates correctly."""
    game = ChessGame()
    assert game.current_player == 1  # White starts

    game.push(chess.Move.from_uci("e2e4"))
    assert game.current_player == 2  # Black's turn

    game.push(chess.Move.from_uci("e7e5"))
    assert game.current_player == 1  # White's turn again


def test_encode_shape():
    """encode() returns correct shape."""
    game = ChessGame()
    encoded = game.encode()
    assert encoded.shape == (18, 8, 8)
    assert encoded.dtype.name == "float32"
