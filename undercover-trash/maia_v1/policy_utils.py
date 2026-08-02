"""Shared policy helpers for Maia samplers (torch, numpy and lc0 backends)."""

from __future__ import annotations

import chess
import numpy as np

from .policy_index import policy_index

_MOVE_TO_INDEX = {uci: i for i, uci in enumerate(policy_index)}


def mirror_move(move_uci: str) -> str:
    """Mirror a white-perspective UCI move to the real board (rank 1 <-> 8)."""
    if len(move_uci) > 4:
        return (move_uci[0] + str(9 - int(move_uci[1]))
                + move_uci[2] + str(9 - int(move_uci[3]))
                + move_uci[4:])
    return (move_uci[0] + str(9 - int(move_uci[1]))
            + move_uci[2] + str(9 - int(move_uci[3])))


def legal_mask(board: chess.Board) -> np.ndarray:
    """bool ndarray (1858,) marking legal moves in the policy index space."""
    mask = np.zeros(len(policy_index), dtype=bool)
    for move in board.legal_moves:
        uci = move.uci() if board.turn == chess.WHITE else mirror_move(move.uci())
        idx = _MOVE_TO_INDEX.get(uci)
        if idx is not None:
            mask[idx] = True
    return mask


def candidates_from_logits(board: chess.Board, logits: np.ndarray,
                           top_k: int, prob_cutoff: float = 0.0):
    """Top-k legal candidates from (1858,) logits, best first.

    Returns a list of (move, prob, rank).
    """
    mask = legal_mask(board)
    logits = np.where(mask, logits, -1e9)
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()

    order = np.argsort(-probs)
    out = []
    for rank, idx in enumerate(order[:top_k]):
        if prob_cutoff > 0 and probs[idx] < prob_cutoff:
            continue
        uci = policy_index[int(idx)]
        if board.turn == chess.BLACK:
            uci = mirror_move(uci)
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            continue
        if move not in board.legal_moves:
            continue
        out.append((move, float(probs[idx]), rank))
    return out
