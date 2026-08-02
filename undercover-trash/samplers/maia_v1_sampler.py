"""Maia v1 sampler (PyTorch backend): exact policy of the released nets.

The net is loaded from its .pb.gz file into PyTorch (see maia_v1/lc0net.py)
and the 1858-dim policy logits are masked to legal moves and converted to a
probability distribution, exactly like running lc0 with `go nodes 1` on the
same net (the official Maia setup).

If PyTorch is not installed, use sampler type `maia_numpy` instead (same
network, pure-numpy inference — see samplers/maia_numpy_sampler.py).
"""

from __future__ import annotations

import logging

import chess
import numpy as np
import torch

from .base import MoveCandidate, Sampler, SamplerError
from maia_v1.encoder import encode_position
from maia_v1.lc0net import Lc0SeNet
from maia_v1.policy_utils import candidates_from_logits

log = logging.getLogger("trojan.sampler.maia_v1")


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

        from maia_v1.policy_utils import legal_mask
        return logits, legal_mask(boards[-1])

    def sample(self, boards, top_k: int) -> list[MoveCandidate]:
        logits, _ = self.policy(boards)
        board = boards[-1]
        cutoff = float(self.cfg.get("min_prob", 0.0))
        return [
            MoveCandidate(move=m, prob=p, rank=r)
            for m, p, r in candidates_from_logits(board, logits, top_k, cutoff)
        ]
