"""Maia v1 sampler with a pure-numpy backend (no torch, no protobuf).

Identical network and policy to the PyTorch sampler (maia_v1), but inference
runs on plain numpy with im2col convolutions — designed for platforms where
PyTorch is not available, e.g. Termux/Android phones.

Dependencies: numpy + python-chess only.
"""

from __future__ import annotations

import logging
import time

import chess
import numpy as np

from .base import MoveCandidate, Sampler
from maia_v1.encoder import encode_position
from maia_v1.numpy_net import NumpyNet
from maia_v1.policy_utils import candidates_from_logits

log = logging.getLogger("trojan.sampler.maia_numpy")


class MaiaNumpySampler(Sampler):
    name = "maia_numpy"

    def __init__(self, cfg: dict, paths):
        super().__init__(cfg, paths)
        self.weights_path = paths.resolve(cfg.get("weights", "weights/maia-1100.pb.gz"))
        self._net = None

    def _load(self):
        if self._net is None:
            t0 = time.time()
            log.info("loading Maia v1 net (numpy backend) from %s", self.weights_path)
            self._net = NumpyNet(self.weights_path)
            log.info("numpy net loaded in %.2fs (%d residual blocks)",
                     time.time() - t0, len(self._net.arr.res))
        return self._net

    def ready(self) -> bool:
        try:
            self._load()
            return True
        except Exception as exc:  # noqa: BLE001
            log.error("numpy Maia net failed to load: %s", exc)
            return False

    def policy(self, boards: list[chess.Board]):
        net = self._load()
        x = encode_position(boards)
        t0 = time.time()
        policy_logits, _value_logits = net(x)
        self._last_ms = (time.time() - t0) * 1000.0
        return policy_logits

    def sample(self, boards, top_k: int) -> list[MoveCandidate]:
        logits = self.policy(boards)
        board = boards[-1]
        cutoff = float(self.cfg.get("min_prob", 0.0))
        return [
            MoveCandidate(move=m, prob=p, rank=r)
            for m, p, r in candidates_from_logits(board, logits, top_k, cutoff)
        ]
