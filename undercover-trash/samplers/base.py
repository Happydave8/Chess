"""Sampler interface: turns a position into Maia-style "human" move candidates."""

from __future__ import annotations

import chess
from dataclasses import dataclass


@dataclass(frozen=True)
class MoveCandidate:
    """A candidate move with its human (Maia) probability.

    Candidates are always sorted best (highest human probability) first;
    `rank` is the 0-based index in that ordering.
    """
    move: chess.Move
    prob: float
    rank: int


class SamplerError(RuntimeError):
    pass


class Sampler:
    """Base class. Subclasses must implement `sample()`."""

    name = "base"

    def __init__(self, cfg: dict, paths: "PathResolver"):
        self.cfg = cfg
        self.paths = paths

    def sample(self, boards: list[chess.Board], top_k: int) -> list[MoveCandidate]:
        """Return up to `top_k` legal candidate moves, best first."""
        raise NotImplementedError

    def close(self):
        """Release resources (subprocesses, GPU memory)."""
        pass

    def ready(self) -> bool:
        """True once the underlying model is loaded."""
        return True
