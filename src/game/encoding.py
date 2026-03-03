"""Chess board and move encoding for the AlphaZero policy representation.

Board encoding: 18 planes of 8x8
Move encoding: 73 planes of 8x8 = 4672 total policy indices

All encoding is from the perspective of the current player.
When it's Black's turn, the board is flipped vertically (rank mirror)
so that the current player's pieces always appear at the "bottom" ranks.
"""

from __future__ import annotations

import chess
import numpy as np

# Board dimensions
NUM_PLANES = 18
BOARD_SIZE = 8
POLICY_PLANES = 73
POLICY_SIZE = POLICY_PLANES * BOARD_SIZE * BOARD_SIZE  # 4672

# Piece order for planes 0-5 (current) and 6-11 (opponent)
PIECE_ORDER = [
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
]

# 8 queen-move directions: (delta_rank, delta_file)
# N, NE, E, SE, S, SW, W, NW
QUEEN_DIRS = [
    (1, 0),   # 0: N
    (1, 1),   # 1: NE
    (0, 1),   # 2: E
    (-1, 1),  # 3: SE
    (-1, 0),  # 4: S
    (-1, -1), # 5: SW
    (0, -1),  # 6: W
    (1, -1),  # 7: NW
]

# 8 knight move deltas: (delta_rank, delta_file)
KNIGHT_DELTAS = [
    (2, 1),
    (2, -1),
    (1, 2),
    (1, -2),
    (-2, 1),
    (-2, -1),
    (-1, 2),
    (-1, -2),
]

# Underpromotion pieces: N=0, B=1, R=2
UNDERPROMOTION_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]

# Underpromotion directions: left-capture=0, straight=1, right-capture=2
UNDERPROMOTION_DFILES = [-1, 0, 1]


def _flip_sq(sq: int) -> int:
    """Mirror a square vertically (rank flip). chess.square_mirror() equivalent."""
    return sq ^ 56


def encode_board(board: chess.Board) -> np.ndarray:
    """Encode board state into (18, 8, 8) float32 array.

    From current player's perspective: board is flipped when it's Black's turn,
    so current player's pieces always appear at the "bottom" (rank 0-1 area).
    """
    planes = np.zeros((NUM_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    flip = board.turn == chess.BLACK
    current_color = board.turn
    opponent_color = not current_color

    # Planes 0-5: current player pieces; 6-11: opponent pieces
    for i, pt in enumerate(PIECE_ORDER):
        for sq in board.pieces(pt, current_color):
            sq_p = _flip_sq(sq) if flip else sq
            planes[i, sq_p >> 3, sq_p & 7] = 1.0
        for sq in board.pieces(pt, opponent_color):
            sq_p = _flip_sq(sq) if flip else sq
            planes[6 + i, sq_p >> 3, sq_p & 7] = 1.0

    # Planes 12-15: castling rights
    planes[12] = float(board.has_kingside_castling_rights(current_color))
    planes[13] = float(board.has_queenside_castling_rights(current_color))
    planes[14] = float(board.has_kingside_castling_rights(opponent_color))
    planes[15] = float(board.has_queenside_castling_rights(opponent_color))

    # Plane 16: en passant target square
    if board.ep_square is not None:
        ep_p = _flip_sq(board.ep_square) if flip else board.ep_square
        planes[16, ep_p >> 3, ep_p & 7] = 1.0

    # Plane 17: halfmove clock normalized
    planes[17] = board.halfmove_clock / 100.0

    return planes


def move_to_index(move: chess.Move, board: chess.Board) -> int:
    """Encode a chess move as a policy index in [0, 4672).

    Uses AlphaZero action representation:
    - Planes 0-55: queen moves (8 dirs x 7 distances), also covers pawn/king moves
    - Planes 56-63: knight moves (8 L-shaped jumps)
    - Planes 64-72: underpromotions (3 pieces x 3 directions)
    - Queen promotions use queen-move planes (same as normal queen sliding move)
    """
    flip = board.turn == chess.BLACK

    from_sq = move.from_square
    to_sq = move.to_square

    from_sq_p = _flip_sq(from_sq) if flip else from_sq
    to_sq_p = _flip_sq(to_sq) if flip else to_sq

    from_rank_p = from_sq_p >> 3
    from_file_p = from_sq_p & 7
    to_rank_p = to_sq_p >> 3
    to_file_p = to_sq_p & 7

    dr = to_rank_p - from_rank_p
    df = to_file_p - from_file_p

    # 1. Check for underpromotion (N, B, R)
    if move.promotion is not None and move.promotion != chess.QUEEN:
        piece_idx = UNDERPROMOTION_PIECES.index(move.promotion)
        dir_idx = UNDERPROMOTION_DFILES.index(df)
        plane = 64 + piece_idx * 3 + dir_idx
        return plane * 64 + from_sq_p

    # 2. Check for knight move
    if (dr, df) in KNIGHT_DELTAS:
        knight_dir_idx = KNIGHT_DELTAS.index((dr, df))
        plane = 56 + knight_dir_idx
        return plane * 64 + from_sq_p

    # 3. Queen move (includes pawn pushes, king moves, queen promotions, castling)
    # Find direction
    if dr == 0 and df == 0:
        raise ValueError(f"Invalid move with no displacement: {move}")

    # Normalize direction to unit vector
    max_dist = max(abs(dr), abs(df))
    unit_dr = dr // max_dist if dr != 0 else 0
    unit_df = df // max_dist if df != 0 else 0

    dir_idx = QUEEN_DIRS.index((unit_dr, unit_df))
    distance = max_dist  # for queen moves, dr and df are equal in abs value OR one is 0

    plane = dir_idx * 7 + (distance - 1)
    return plane * 64 + from_sq_p


def index_to_move(index: int, board: chess.Board) -> chess.Move:
    """Decode a policy index back to a chess.Move.

    Inverse of move_to_index. May produce illegal moves if index doesn't
    correspond to a legal move; callers should validate with board.legal_moves.
    """
    flip = board.turn == chess.BLACK

    plane = index // 64
    from_sq_p = index % 64

    from_sq = _flip_sq(from_sq_p) if flip else from_sq_p

    from_rank_p = from_sq_p >> 3
    from_file_p = from_sq_p & 7

    if plane < 56:
        # Queen move
        dir_idx = plane // 7
        distance = plane % 7 + 1
        unit_dr, unit_df = QUEEN_DIRS[dir_idx]

        to_rank_p = from_rank_p + unit_dr * distance
        to_file_p = from_file_p + unit_df * distance

        if not (0 <= to_rank_p < 8 and 0 <= to_file_p < 8):
            raise ValueError(f"Index {index} decodes to out-of-bounds square")

        to_sq_p = to_rank_p * 8 + to_file_p
        to_sq = _flip_sq(to_sq_p) if flip else to_sq_p

        # Detect queen promotion: pawn reaching rank 7 (from perspective)
        piece = board.piece_at(from_sq)
        promotion = None
        if piece is not None and piece.piece_type == chess.PAWN and to_rank_p == 7:
            promotion = chess.QUEEN

        return chess.Move(from_sq, to_sq, promotion=promotion)

    elif plane < 64:
        # Knight move
        knight_dir_idx = plane - 56
        dr, df = KNIGHT_DELTAS[knight_dir_idx]

        to_rank_p = from_rank_p + dr
        to_file_p = from_file_p + df

        if not (0 <= to_rank_p < 8 and 0 <= to_file_p < 8):
            raise ValueError(f"Index {index} decodes to out-of-bounds square")

        to_sq_p = to_rank_p * 8 + to_file_p
        to_sq = _flip_sq(to_sq_p) if flip else to_sq_p

        return chess.Move(from_sq, to_sq)

    else:
        # Underpromotion
        promo_idx = plane - 64
        piece_idx = promo_idx // 3
        dir_idx = promo_idx % 3

        df = UNDERPROMOTION_DFILES[dir_idx]
        to_rank_p = from_rank_p + 1
        to_file_p = from_file_p + df

        if not (0 <= to_rank_p < 8 and 0 <= to_file_p < 8):
            raise ValueError(f"Index {index} decodes to out-of-bounds square")

        to_sq_p = to_rank_p * 8 + to_file_p
        to_sq = _flip_sq(to_sq_p) if flip else to_sq_p

        promotion = UNDERPROMOTION_PIECES[piece_idx]
        return chess.Move(from_sq, to_sq, promotion=promotion)


def get_legal_move_mask(board: chess.Board) -> np.ndarray:
    """Returns a boolean mask of shape (4672,) with True at each legal move index.

    Used by the neural network to mask illegal moves before softmax.
    """
    mask = np.zeros(POLICY_SIZE, dtype=bool)
    for move in board.legal_moves:
        idx = move_to_index(move, board)
        mask[idx] = True
    return mask
