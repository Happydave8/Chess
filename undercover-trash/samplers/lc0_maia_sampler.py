"""Optional Maia-v1 sampler that drives a real lc0 binary.

This is the official CSSLab way of querying a Maia net: run

    lc0 --weights=maia-1100.pb.gz --verbose-move-stats

and read the per-move policy values `(P: ...)` from the UCI output at
`go nodes 1`. It is functionally equivalent to the built-in torch sampler
(`maia_v1`) and is only useful if you want to double-check parity with lc0
or you already have an lc0 binary around.
"""

from __future__ import annotations

import logging
import re
import subprocess
import threading

import chess

from .base import MoveCandidate, Sampler, SamplerError

log = logging.getLogger("trojan.sampler.lc0_maia")

_P_RE = re.compile(r"\(P: +([^)]+)\)")


class Lc0MaiaSampler(Sampler):
    name = "lc0_maia"

    def __init__(self, cfg: dict, paths):
        super().__init__(cfg, paths)
        self.lc0_path = paths.resolve(cfg.get("lc0_path", "lc0"))
        self.weights = paths.resolve(cfg.get("weights", "weights/maia-1100.pb.gz"))
        self.nodes = int(cfg.get("nodes", 1))
        self.threads = int(cfg.get("threads", 2))
        self._proc = None
        self._lock = threading.Lock()

    def _ensure_proc(self):
        if self._proc is not None and self._proc.poll() is None:
            return self._proc
        try:
            self._proc = subprocess.Popen(
                [self.lc0_path, f"--weights={self.weights}",
                 "--verbose-move-stats", f"--threads={self.threads}"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True, bufsize=1)
        except FileNotFoundError as exc:
            raise SamplerError(
                f"lc0 binary not found at {self.lc0_path!r}. Install lc0 "
                "(https://lczero.org) or switch sampler.type to maia_v1.") from exc
        self._proc.stdin.write("uci\n")
        self._proc.stdin.flush()
        while True:
            line = self._proc.stdout.readline()
            if line.strip() == "uciok":
                break
            if self._proc.poll() is not None:
                raise SamplerError("lc0 exited during UCI handshake")
        self._proc.stdin.write("isready\n")
        self._proc.stdin.flush()
        while True:
            line = self._proc.stdout.readline()
            if line.strip() == "readyok":
                break
        return self._proc

    def sample(self, boards, top_k: int) -> list[MoveCandidate]:
        proc = self._ensure_proc()
        board = boards[-1]
        with self._lock:
            proc.stdin.write(f"position fen {board.fen()}\n")
            proc.stdin.write(f"go nodes {self.nodes}\n")
            proc.stdin.flush()

            moves = []          # (prob, uci)
            while True:
                line = proc.stdout.readline()
                if not line:
                    raise SamplerError("lc0 closed its output stream")
                line = line.strip()
                if line.startswith("bestmove"):
                    break
                m = _P_RE.search(line)
                if not m or " pv " not in line:
                    continue
                try:
                    prob = float(m.group(1))
                except ValueError:
                    continue
                pv = line.split(" pv ", 1)[1].split()[0]
                moves.append((prob, pv))

        moves.sort(key=lambda t: -t[0])
        candidates = []
        for rank, (prob, uci) in enumerate(moves[:top_k]):
            try:
                move = chess.Move.from_uci(uci)
            except ValueError:
                continue
            if move not in board.legal_moves:
                continue
            candidates.append(MoveCandidate(move=move, prob=prob, rank=rank))
        return candidates

    def close(self):
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.stdin.write("quit\n")
                self._proc.stdin.flush()
                self._proc.wait(timeout=3)
            except Exception:  # noqa: BLE001
                self._proc.kill()
