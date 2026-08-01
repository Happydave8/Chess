"""Maia v1 sampler: exact policy of the released maia-1100..1900 Lc0 nets.

The net is loaded from its .pb.gz file into PyTorch (see maia_v1/lc0net.py)
and the 1858-dim policy logits are masked to legal moves and converted to a
probability distribution, exactly like running lc0 with `go nodes 1` on the
same net (the official Maia setup).
"""

from __future__ import annotations

import logging

import chess
import numpy as np
import torch

from .base import MoveCandidate, Sampler, SamplerError
from maia_v1.encoder import encode_position
from maia_v1.lc0net import Lc0SeNet
from maia_v1.policy_index import policy_index

log = logging.getLogger("trojan.sampler.maia_v1")


def mirror_move(move_uci: str) -> str:
    """Mirror a white-perspective UCI move to the real board (rank 1 <-> 8)."""
    if len(move_uci) > 4:
        return (move_uci[0] + str(9 - int(move_uci[1]))
                + move_uci[2] + str(9 - int(move_uci[3]))
                + move_uci[4:])
    return (move_uci[0] + str(9 - int(move_uci[1]))
            + move_uci[2] + str(9 - int(move_uci[3])))


class MaiaV1Sampler(Sampler):
    name = "maia_v1"

    def __init__(self, cfg: dict, paths):
        super().__init__(cfg, paths)
        self.weights_path = paths.resolve(cfg.get("weights", "weights/maia-1100.pb.gz"))
        self.device = cfg.get("device", "cpu")
        self._net = None

    def _load(self):
        if self._net is None:
            log.info("loading Maia v1 net from %s", self.weights_path)
            self._net = Lc0SeNet(self.weights_path, device=self.device)
            log.info("Maia v1 net loaded (%d residual blocks)", self._net.num_residual)
        return self._net

    def ready(self) -> bool:
        try:
            self._load()
            return True
        except Exception as exc:  # noqa: BLE001
            log.error("Maia v1 model failed to load: %s", exc)
            return False

    def policy(self, boards: list[chess.Board]):
        """Return (logits (1858,), legal_mask (1858,) as bool ndarray)."""
        net = self._load()
        x = torch.from_numpy(encode_position(boards)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            policy_logits, value_logits = net(x)
        logits = policy_logits[0].cpu().numpy()

        board = boards[-1]
        legal_mask = np.zeros(len(policy_index), dtype=bool)
        for move in board.legal_moves:
            uci = move.uci() if board.turn == chess.WHITE else mirror_move(move.uci())
            try:
                idx = _MOVE_TO_INDEX[uci]
            except KeyError:
                continue
            legal_mask[idx] = True
        return logits, legal_mask

    def sample(self, boards, top_k: int) -> list[MoveCandidate]:
        logits, legal_mask = self.policy(boards)
        # Numerical safety: keep logits finite.
        logits = np.where(legal_mask, logits, -1e9)
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()

        board = boards[-1]
        order = np.argsort(-probs)
        candidates = []
        for rank, idx in enumerate(order[:top_k]):
            uci = policy_index[int(idx)]
            if board.turn == chess.BLACK:
                uci = mirror_move(uci)
            try:
                move = chess.Move.from_uci(uci)
            except ValueError:
                continue
            if move not in board.legal_moves:
                continue
            candidates.append(MoveCandidate(move=move, prob=float(probs[idx]), rank=rank))
        return candidates


# index lookup for the 1858 policy list
_MOVE_TO_INDEX = {uci: i for i, uci in enumerate(policy_index)}
