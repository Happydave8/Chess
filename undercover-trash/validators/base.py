"""Validator interface: deep search evaluation of candidate moves."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MoveEval:
    """Evaluation of a root move, from the side-to-move's perspective."""
    move_uci: str
    score_cp: float          # centipawns (mate converted to +-100000)
    win_prob: float          # 0..1 win probability (engine-reported or derived)
    mate: int | None = None  # plies to mate if the line is a forced mate
    wdl: tuple[int, int, int] | None = None  # engine WDL in permille, if given


@dataclass
class SearchResult:
    best_move: str | None
    best_score_cp: float
    best_win_prob: float
    evals: list[MoveEval]    # multipv list, best first
    depth: int = 0


class ValidatorError(RuntimeError):
    pass


class Validator:
    """Base class. Subclasses implement `analyse()`."""

    name = "base"

    def __init__(self, cfg: dict, paths):
        self.cfg = cfg
        self.paths = paths

    def new_game(self):
        pass

    def analyse(self, fen: str, *, depth: int = 0, movetime_ms: int = 0,
                multipv: int = 1, searchmoves: list[str] | None = None
                ) -> SearchResult:
        raise NotImplementedError

    def stop(self):
        pass

    def close(self):
        pass
