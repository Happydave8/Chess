"""112-plane "classical" board encoder, matching Lc0's INPUT_CLASSICAL_112_PLANE.

Faithful reimplementation of lc0's `src/neural/encoder.cc` for the classical
format, cross-checked against the training-data writer used by the Maia team
(`move_prediction/maia_chess_backend/maia/chunkparser.py`).

Plane layout (all planes 8x8, side-to-move orientation):
  0..103  8 history positions x 13 planes:
            0-5   our P,N,B,R,Q,K        (us = side to move of that position)
            6-11  their P,N,B,R,Q,K
            12    repetition plane (1 if the position occurred earlier)
  104     our queenside castling right  (current position)
  105     our kingside  castling right
  106     their queenside castling right
  107     their kingside  castling right
  108     side-to-move flag (1 if black to move)
  109     rule-50 ply counter (raw; the released net weights are pre-scaled so
          that feeding the raw count equals the /99 normalization used in
          training, exactly like the lc0 C++ client)
  110     move-count plane (always 0 at inference, as in lc0)
  111     all-ones plane (board-edge hint)
"""

from __future__ import annotations

import chess
import numpy as np

HISTORY = 8                       # lc0 kMoveHistory
PLANES_PER_BOARD = 13             # 12 piece planes + 1 repetition plane
AUX_BASE = HISTORY * PLANES_PER_BOARD  # 104
INPUT_PLANES = 112

_PIECE_ORDER = [
    chess.PAWN, chess.KNIGHT, chess.BISHOP,
    chess.ROOK, chess.QUEEN, chess.KING,
]


def _repetition_key(board: chess.Board) -> str:
    """Position identity ignoring the half-move clock (like lc0's Position==)."""
    parts = board.fen().split()
    return " ".join(parts[:4])


def encode_position(boards: list[chess.Board]) -> np.ndarray:
    """Encode a game history into the 112-plane input tensor.

    Args:
        boards: list of python-chess boards from the start of the game up to
            the current position (current position last).

    Returns:
        float32 ndarray of shape (112, 8, 8).
    """
    planes = np.zeros((INPUT_PLANES, 8, 8), dtype=np.float32)

    history = boards[-HISTORY:]
    n = len(history)
    keys = [_repetition_key(b) for b in boards]
    cur = boards[-1]

    for j in range(n):
        b = history[n - 1 - j]          # j == 0 -> current position
        base = j * PLANES_PER_BOARD
        # Every history position is rendered from the CURRENT side-to-move's
        # perspective: the current stm's pieces always live in planes 0-5 at
        # the bottom. lc0 stores each board with its own stm at the bottom and
        # applies Mirror() (vertical flip + color swap) on odd history plies;
        # in python-chess terms that is a vertical flip whenever the history
        # position's turn differs from the current turn.
        flip = (b.turn != cur.turn)

        for sq in range(64):
            piece = b.piece_at(sq)
            if piece is None:
                continue
            rank, file_ = divmod(sq, 8)
            if flip:
                rank = 7 - rank
            plane = _PIECE_ORDER.index(piece.piece_type)
            if piece.color != cur.turn:
                plane += 6
            planes[base + plane, rank, file_] = 1.0

        # Repetition plane: 1 if this position occurred at least twice in the
        # game up to and including this point.
        idx = len(boards) - 1 - j
        if keys[: idx + 1].count(keys[idx]) >= 2:
            planes[base + 12] = 1.0

    # Auxiliary planes come from the *current* position only.
    we, they = cur.turn, not cur.turn
    if cur.has_queenside_castling_rights(we):
        planes[AUX_BASE + 0] = 1.0
    if cur.has_kingside_castling_rights(we):
        planes[AUX_BASE + 1] = 1.0
    if cur.has_queenside_castling_rights(they):
        planes[AUX_BASE + 2] = 1.0
    if cur.has_kingside_castling_rights(they):
        planes[AUX_BASE + 3] = 1.0
    if we == chess.BLACK:
        planes[AUX_BASE + 4] = 1.0
    planes[AUX_BASE + 5] = float(cur.halfmove_clock)  # raw rule50 count
    planes[AUX_BASE + 6] = 0.0
    planes[AUX_BASE + 7] = 1.0

    return planes
