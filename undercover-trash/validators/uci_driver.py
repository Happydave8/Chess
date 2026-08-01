"""A small, robust UCI subprocess driver for Stockfish / lc0.

Speaks the UCI protocol directly over pipes so we can request multipv
searches, restrict root moves with `searchmoves`, read WDL/winrate output and
abort searches on demand (`stop`). Used as the deep validator backend.
"""

from __future__ import annotations

import logging
import queue
import re
import subprocess
import threading
import time

from .base import MoveEval, SearchResult, Validator, ValidatorError

log = logging.getLogger("trojan.validator.uci")

_WDL_RE = re.compile(r"\bwdl (\d+) (\d+) (\d+)")
_WINRATE_RE = re.compile(r"\bwinrate ([0-9.]+)")

MATE_SCORE = 100000.0


def logistic_win_prob(score_cp: float) -> float:
    """Centipawns -> win probability (standard Elo logistic, 400 cp/decade)."""
    return 1.0 / (1.0 + 10.0 ** (-score_cp / 400.0))


class UciValidator(Validator):
    def __init__(self, cfg: dict, paths, *, engine: str):
        super().__init__(cfg, paths)
        self.engine = engine                       # "stockfish" | "lc0"
        self.binary = paths.resolve(cfg.get("path", "stockfish"))
        self.threads = int(cfg.get("threads", 2))
        self.hash_mb = int(cfg.get("hash_mb", 128))
        self._proc = None
        self._queue: queue.Queue = queue.Queue()
        self._reader: threading.Thread | None = None

    # ------------------------------------------------------------------
    def _ensure_proc(self):
        if self._proc is not None and self._proc.poll() is None:
            return self._proc
        cmd = [self.binary]
        if self.engine == "lc0":
            weights = self.paths.resolve(self.cfg.get("weights", ""))
            if weights:
                cmd.append(f"--weights={weights}")
        try:
            self._proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True, bufsize=1)
        except FileNotFoundError as exc:
            raise ValidatorError(
                f"validator binary not found at {self.binary!r}") from exc

        self._reader = threading.Thread(
            target=self._reader_loop, daemon=True)
        self._reader.start()

        self._send("uci")
        if self._read_until("uciok", timeout=10) is None:
            raise ValidatorError(f"{self.engine} UCI handshake failed")
        self._send(f"setoption name Threads value {self.threads}")
        if self.engine == "stockfish":
            self._send(f"setoption name Hash value {self.hash_mb}")
            self._send("setoption name UCI_ShowWDL value true")
            # Keep output deterministic enough for filtering; engine may
            # still choose between equal moves.
            self._send("setoption name UCI_AnalyseMode value true")
        self._send("isready")
        if self._read_until("readyok", timeout=10) is None:
            raise ValidatorError(f"{self.engine} isready failed")
        log.info("%s validator ready (%s)", self.engine, self.binary)
        return self._proc

    def _reader_loop(self):
        try:
            for line in self._proc.stdout:
                self._queue.put(line)
        except Exception:  # noqa: BLE001
            pass
        finally:
            self._queue.put(None)  # EOF marker

    def _readline(self, timeout: float = 0.1):
        try:
            item = self._queue.get(timeout=timeout)
        except queue.Empty:
            return None
        return item

    def _read_until(self, token: str, timeout: float):
        end = time.time() + timeout
        while time.time() < end:
            line = self._readline(0.2)
            if line is None:
                if self._proc.poll() is not None:
                    return None
                continue
            if token in line:
                return line
        return None

    # ------------------------------------------------------------------
    def _send(self, text: str):
        proc = self._ensure_proc()
        proc.stdin.write(text + "\n")
        proc.stdin.flush()

    # ------------------------------------------------------------------
    def analyse(self, fen: str, *, depth: int = 0, movetime_ms: int = 0,
                nodes: int = 0, multipv: int = 1,
                searchmoves: list[str] | None = None) -> SearchResult:
        self._ensure_proc()
        # MultiPV is a UCI *option* in both Stockfish and lc0 (not a go
        # token), so set it around the search and restore it afterwards.
        if multipv > 1:
            self._send(f"setoption name MultiPV value {multipv}")
        self._send(f"position fen {fen}")
        go = ["go"]
        if depth > 0:
            go.append(f"depth {depth}")
        if movetime_ms > 0:
            go.append(f"movetime {movetime_ms}")
        if nodes > 0:
            go.append(f"nodes {nodes}")
        if searchmoves:
            go.append("searchmoves " + " ".join(searchmoves))
        self._send(" ".join(go))

        pvs: dict[int, dict] = {}
        bestmove = None
        start = time.time()
        while True:
            line = self._readline(timeout=0.05)
            if line is None:
                if self._proc.poll() is not None:
                    raise ValidatorError(f"{self.engine} process died during search")
                if time.time() - start > 600:
                    raise ValidatorError(f"{self.engine} search hung")
                continue
            line = line.strip()
            if not line:
                continue
            if line.startswith("info"):
                self._parse_info(line, pvs)
            elif line.startswith("bestmove"):
                parts = line.split()
                bestmove = parts[1] if len(parts) > 1 else None
                break

        if multipv > 1:
            self._send("setoption name MultiPV value 1")

        evals = []
        best = None
        for idx in sorted(pvs):
            pv = pvs[idx]
            score_cp, mate = pv["score"]
            win_prob = pv.get("win_prob")
            wdl = pv.get("wdl")
            if win_prob is None:
                if mate and mate > 0:
                    win_prob = 1.0
                elif mate and mate < 0:
                    win_prob = 0.0
                else:
                    win_prob = logistic_win_prob(score_cp)
            root_move = pv.get("root_move") or (bestmove or "")
            ev = MoveEval(move_uci=root_move, score_cp=score_cp,
                          win_prob=win_prob, mate=mate, wdl=wdl)
            evals.append(ev)
            if best is None or (score_cp, win_prob) > (best["score_cp"], best["win_prob"]):
                best = {"score_cp": score_cp, "win_prob": win_prob,
                        "move": root_move}
        if best is None:
            raise ValidatorError(f"{self.engine} produced no evaluation")

        return SearchResult(
            best_move=best["move"] or bestmove,
            best_score_cp=best["score_cp"],
            best_win_prob=best["win_prob"],
            evals=evals,
            depth=max(pv.get("depth", 0) for pv in pvs.values()) if pvs else 0,
        )

    # ------------------------------------------------------------------
    def _parse_info(self, line: str, pvs: dict):
        """Parse one `info ...` line into the pvs[multipv] dict.

        Handles both Stockfish (`score cp ... wdl W D L ... pv ...`) and lc0
        (`score cp ... winrate 0.xx ... pv ...`) output, in any token order.
        """
        m = re.search(r"\bmultipv (\d+)", line)
        multipv = int(m.group(1)) if m else 1
        m = re.search(r"\bdepth (\d+)", line)
        depth = int(m.group(1)) if m else 0
        m = re.search(r"\bscore (cp|mate) (-?\d+)", line)
        if not m:
            return
        kind, value = m.group(1), int(m.group(2))
        if kind == "mate":
            score_cp = MATE_SCORE if value > 0 else -MATE_SCORE
            mate = value
        else:
            score_cp = float(value)
            mate = None

        win_prob = None
        wdl = None
        m = _WDL_RE.search(line)
        if m:
            win, draw, loss = (int(x) for x in m.groups())
            wdl = (win, draw, loss)
            win_prob = win / 1000.0
        else:
            m = _WINRATE_RE.search(line)
            if m:
                win_prob = float(m.group(1))

        m = re.search(r"\bpv (.+)$", line)
        root_move = m.group(1).split()[0] if m else None

        entry = pvs.setdefault(multipv, {"depth": 0})
        entry["score"] = (score_cp, mate)
        entry["depth"] = max(entry["depth"], depth)
        if win_prob is not None:
            entry["win_prob"] = win_prob
        if wdl is not None:
            entry["wdl"] = wdl
        if root_move:
            entry["root_move"] = root_move

    # ------------------------------------------------------------------
    def stop(self):
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.stdin.write("stop\n")
                self._proc.stdin.flush()
            except Exception:  # noqa: BLE001
                pass

    def new_game(self):
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.stdin.write("ucinewgame\n")
                self._proc.stdin.flush()
            except Exception:  # noqa: BLE001
                pass

    def close(self):
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.stdin.write("quit\n")
                self._proc.stdin.flush()
                self._proc.wait(timeout=3)
            except Exception:  # noqa: BLE001
                self._proc.kill()
