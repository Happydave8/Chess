"""Maia-3 sampler: Elo-conditioned Chessformer policy.

Uses the official `maia3` Python package (https://github.com/CSSLab/maia3).
The model predicts the move distribution of a human of a given Elo
(`elo`, default 1100), which is exactly the "low-rating human" signal we
want for candidate generation. The checkpoint is downloaded from Hugging Face
on first use (or loaded from a local path).
"""

from __future__ import annotations

import logging
from collections import deque

import chess
import torch

from .base import MoveCandidate, Sampler, SamplerError

log = logging.getLogger("trojan.sampler.maia3")

# Aliases understood by maia3.model_registry
MODEL_ALIASES = ("maia3-3m-ablation", "maia3-5m", "maia3-23m", "maia3-79m")


class Maia3Sampler(Sampler):
    name = "maia3"

    def __init__(self, cfg: dict, paths):
        super().__init__(cfg, paths)
        self.elo = int(cfg.get("elo", 1100))
        self.model_spec = cfg.get("model", "maia3-5m")
        self.checkpoint = cfg.get("checkpoint", "") or None
        self.device = cfg.get("device", "cpu")
        self.temperature = float(cfg.get("temperature", 1.0))
        self._model = None
        self._all_moves = None
        self._all_moves_dict = None

    # ------------------------------------------------------------------
    def _load(self):
        if self._model is not None:
            return self._model

        try:
            from maia3 import dataset, utils  # noqa: F401  (import check)
            from maia3.model_registry import (apply_model_config,
                                              resolve_checkpoint_path,
                                              resolve_model_spec)
            from maia3.models import MAIA3Model
        except ImportError as exc:
            raise SamplerError(
                "maia3 package is not installed. Run scripts/setup.sh or "
                "`pip install -e <path-to-CSSLab/maia3>`.") from exc

        cfg = SimpleNamespace(
            device=self.device,
            checkpoint_path=self.checkpoint,
            trust_checkpoint=False,
        )
        if self.cfg.get("random_weights", False):
            # TESTING ONLY: build the architecture with random weights so the
            # sampler pipeline can be validated without a checkpoint.
            spec = resolve_model_spec(self.model_spec or "maia3-5m")
            apply_model_config(cfg, spec)
            self._model = MAIA3Model(cfg).to(self.device)
            self._model.eval()
        else:
            if self.checkpoint is None:
                spec = resolve_model_spec(self.model_spec)
                apply_model_config(cfg, spec)
                cfg.model_spec = spec
                cfg.checkpoint_path = resolve_checkpoint_path(spec)
            else:
                # Local checkpoint: use the 5M preset architecture unless the
                # user provides explicit architecture flags.
                spec = resolve_model_spec("maia3-5m")
                apply_model_config(cfg, spec)

            from maia3.uci import load_model
            self._model = load_model(cfg)
            self._model.to(self.device)
            self._model.eval()

        self._all_moves = utils.get_all_possible_moves()
        self._all_moves_dict = {m: i for i, m in enumerate(self._all_moves)}
        self._dataset = dataset
        log.info("Maia3 model ready (spec=%s, elo=%d)", self.model_spec, self.elo)
        return self._model

    def ready(self) -> bool:
        try:
            self._load()
            return True
        except Exception as exc:  # noqa: BLE001
            log.error("Maia3 model failed to load: %s", exc)
            return False

    # ------------------------------------------------------------------
    def sample(self, boards, top_k: int) -> list[MoveCandidate]:
        model = self._load()
        dataset = self._dataset
        board = boards[-1]

        # Replay the real game history (like the official engine's
        # --use-uci-history mode): the model sees the last `history`
        # positions; `get_historical_tokens` pads with the earliest one.
        history_len = int(self.cfg.get("history", 8))
        hist = deque([dataset.tokenize_board(b) for b in boards[-history_len:]],
                     maxlen=history_len)
        input_tokens = dataset.get_historical_tokens(
            hist, SimpleNamespace(history=history_len, include_time_info=False),
            base=0.0, inc=0.0, clk_left_before=0.0, clk_ponder=0.0)
        input_tokens = input_tokens.unsqueeze(0).to(self.device)

        self_elos = torch.tensor([self.elo], dtype=torch.long, device=self.device)
        oppo_elos = torch.tensor([self.elo], dtype=torch.long, device=self.device)

        with torch.no_grad():
            logits_move, _, _ = model(input_tokens, self_elos, oppo_elos)
        logits = logits_move[0].float()

        legal_mask = dataset.get_legal_moves_mask(board, self._all_moves_dict).to(self.device)
        logits = logits.masked_fill(~legal_mask, float("-inf"))

        probs = torch.softmax(logits / max(self.temperature, 1e-9), dim=-1)
        top_count = min(top_k, int(legal_mask.sum().item()))
        top_probs, top_idxs = torch.topk(probs, k=top_count)

        candidates = []
        for rank, (p, idx) in enumerate(zip(top_probs.tolist(), top_idxs.tolist())):
            move_uci = self._all_moves[int(idx)]
            if board.turn == chess.BLACK:
                from maia3.utils import mirror_move
                move_uci = mirror_move(move_uci)
            try:
                move = chess.Move.from_uci(move_uci)
            except ValueError:
                continue
            if move not in board.legal_moves:
                continue
            candidates.append(MoveCandidate(move=move, prob=float(p), rank=rank))
        return candidates


class SimpleNamespace:
    """Minimal attribute bag (avoids importing types.SimpleNamespace in hot path)."""

    def __init__(self, **kw):
        self.__dict__.update(kw)
