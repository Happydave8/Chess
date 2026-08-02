#!/usr/bin/env python3
"""TrojanHorse — the "undercover trash" hybrid UCI engine.

Pipeline per move:

    1. CANDIDATE SAMPLER  (human brain)
       A low-rated human model (Maia-1100 .pb.gz in torch, Maia-3 with an Elo
       condition, or lc0 + a Maia net) proposes the top-K moves a weak human
       would actually play, with human probabilities.

    2. DEEP VALIDATOR     (safety net)
       Stockfish (or lc0) evaluates every candidate with a deep multipv
       search and reports centipawn loss and win probability per candidate.

    3. SELECTION          (the Trojan Horse filter)
       Pick the highest Maia-ranked candidate that:
         * keeps the position winning:      win_prob >= win_prob_threshold
         * does not blunder:                cp loss <= max_cp_loss
         * still *looks* imperfect:         cp loss >= min_cp_loss (optional)
         * is not the engine's best move    (optional)
       If nothing qualifies, the engine plays the validator's best move, so
       it never actually loses games through its own "trash" act.

Speaks plain UCI (uci / isready / ucinewgame / setoption / position / go /
stop / ponderhit / quit), so it plugs straight into lichess-bot or any GUI.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import threading
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import chess
import yaml

from samplers.base import SamplerError
from validators.base import ValidatorError
from validators.uci_driver import UciValidator, logistic_win_prob

ENGINE_NAME = "TrojanHorse 1.0"
ENGINE_AUTHOR = "undercover-trash project"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULTS = {
    "sampler": {
        "type": "maia_v1",              # maia_v1 | maia_numpy | maia3 | lc0_maia
        "candidate_count": 12,
        "temperature": 0.0,             # maia3 sampling temperature (0 = argmax)
        "min_prob": 0.0,                # drop candidates below this human prob
        "maia_v1": {"weights": "weights/maia-1100.pb.gz", "device": "cpu"},
        "maia_numpy": {"weights": "weights/maia-1100.pb.gz"},
        "maia3": {"model": "maia3-5m", "elo": 1100, "checkpoint": "", "device": "cpu"},
        "lc0_maia": {"lc0_path": "lc0", "weights": "weights/maia-1100.pb.gz",
                     "nodes": 1, "threads": 2},
    },
    "validator": {
        "type": "stockfish",            # stockfish | lc0 | both
        "stockfish": {"path": "bin/stockfish", "threads": 2, "hash_mb": 128},
        "lc0": {"path": "lc0", "weights": "", "threads": 2},
    },
    "filter": {
        "win_prob_threshold": 0.45,     # minimum win probability to play a candidate
        "win_prob_mode": "expected",    # expected = win + draw/2 (recommended);
                                        # win_only = literal WDL win share
        "max_cp_loss": 60,              # candidate may be at most this many cp worse than best
        "min_cp_loss": 15,              # candidate must look at least this imperfect (0 = off)
        "require_not_best": True,       # never play the validator's #1 move
        "selection": "rank",            # rank | softmax
        "softmax_temp": 1.5,
        "top_j": 3,                     # softmax: sample among top-J qualifying candidates
        "fallback": "best",             # best | none
    },
    "time": {
        "movetime_scale": 0.5,          # fraction of the per-move budget used
        "movetime_min_ms": 150,
        "movetime_max_ms": 15000,
        "quick_depth": 12,              # pass A depth (best-move discovery)
        "depth_cap": 22,                # pass B depth ceiling
        "pass_a_share": 0.30,           # fraction of budget spent on pass A
    },
    "logging": {"level": "info", "file": "logs/trojan.log"},
}

UCI_OPTION_DEFAULTS = {
    "SamplerElo": "1100",
    "SamplerModel": "maia3-5m",
    "SamplerWeights": "weights/maia-1100.pb.gz",
    "CandidateCount": "12",
    "MinWinProb": "0.45",
    "MaxCpLoss": "60",
    "MinCpLoss": "15",
    "RequireNotBest": "true",
    "SelectionStyle": "rank",
    "DepthCap": "22",
    "QuickDepth": "12",
    "Threads": "2",
    "Hash": "128",
    "LogFile": "logs/trojan.log",
}


class PathResolver:
    """Resolves relative paths against the config file's directory."""

    def __init__(self, base: Path):
        self.base = base

    def resolve(self, path: str) -> str:
        p = Path(path)
        if not p.is_absolute():
            candidate = self.base / p
            # Bare command names (no directory separator) that don't exist in
            # the config dir fall back to $PATH — handy for system installs
            # (e.g. `pkg install stockfish` on Termux).
            if not candidate.exists() and "/" not in path:
                import shutil
                which = shutil.which(path)
                if which:
                    return which
            return str(candidate)
        return str(p)


def deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(config_path: str | None) -> tuple[dict, PathResolver]:
    cfg = json.loads(json.dumps(DEFAULTS))  # deep copy
    base = HERE
    if config_path:
        p = Path(config_path).expanduser()
        if not p.exists():
            print(f"info string [trojan] WARNING: config file not found: {p}", flush=True)
        else:
            with open(p, "r", encoding="utf-8") as f:
                user_cfg = yaml.safe_load(f) or {}
            cfg = deep_merge(cfg, user_cfg)
            base = p.parent
    return cfg, PathResolver(base)


# ---------------------------------------------------------------------------
# Selection logic (pure, unit-testable)
# ---------------------------------------------------------------------------

def select_move(candidates, best_move_uci, best_score_cp, candidate_scores,
                flt: dict, rng: random.Random):
    """Apply the Trojan Horse filter.

    candidates:        list of MoveCandidate (Maia order, best first)
    best_move_uci:     validator's best move (pass A)
    best_score_cp:     its score, side-to-move perspective
    candidate_scores:  {move_uci: (score_cp, win_prob)} from the multipv pass

    Returns (move_uci, kind, loss_cp, win_prob) where kind is one of
    "trash" (a qualifying human-looking move), "fallback" (validator best).
    """
    threshold = float(flt.get("win_prob_threshold", 0.45))
    max_loss = float(flt.get("max_cp_loss", 60))
    min_loss = float(flt.get("min_cp_loss", 0))
    require_not_best = bool(flt.get("require_not_best", True))
    selection = flt.get("selection", "rank")
    temp = float(flt.get("softmax_temp", 1.5))
    top_j = int(flt.get("top_j", 3))

    qualified = []
    ref_best = best_score_cp
    for cand in candidates:
        item = candidate_scores.get(cand.move.uci())
        if item is None:
            continue
        score_cp, win_prob = item
        loss = ref_best - score_cp
        if win_prob < threshold:
            continue
        if loss > max_loss:
            continue
        if min_loss > 0 and loss < min_loss:
            continue
        if require_not_best and cand.move.uci() == best_move_uci:
            continue
        qualified.append((cand, score_cp, win_prob, loss))

    if not qualified:
        return best_move_uci, "fallback", 0.0, None

    if selection == "softmax":
        pool = qualified[:top_j]
        weights = [cand.prob ** (1.0 / max(temp, 1e-6)) for cand, _, _, _ in pool]
        total = sum(weights)
        r = rng.random() * total
        acc = 0.0
        for (cand, score, wp, loss), w in zip(pool, weights):
            acc += w
            if r <= acc:
                return cand.move.uci(), "trash", loss, wp
        cand, score, wp, loss = pool[-1]
        return cand.move.uci(), "trash", loss, wp

    cand, score, wp, loss = qualified[0]
    return cand.move.uci(), "trash", loss, wp


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class TrojanEngine:
    def __init__(self, config_path: str | None = None):
        self.cfg, self.paths = load_config(config_path)
        self.boards: list[chess.Board] = [chess.Board()]
        self.sampler = None
        self.validators: list[UciValidator] = []
        self._search_thread: threading.Thread | None = None
        self._searching = threading.Event()
        self._stop_req = threading.Event()
        self._rng = random.Random()
        self._setup_logging()

    # ------------------------------------------------------------------
    def _setup_logging(self):
        level = getattr(logging, str(self.cfg["logging"].get("level", "info")).upper(),
                        logging.INFO)
        root = logging.getLogger("trojan")
        root.setLevel(level)
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        try:
            logfile = Path(self.paths.resolve(self.cfg["logging"].get("file", "logs/trojan.log")))
            logfile.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(logfile)
            fh.setFormatter(fmt)
            root.addHandler(fh)
        except Exception:  # noqa: BLE001
            pass
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(fmt)
        root.addHandler(sh)
        self.log = logging.getLogger("trojan.engine")

    # ------------------------------------------------------------------
    def load_components(self):
        if self.sampler is not None:
            return
        scfg = self.cfg["sampler"]
        stype = scfg.get("type", "maia_v1")
        try:
            if stype == "maia_v1":
                try:
                    import torch  # noqa: F401
                except ImportError:
                    self.log.warning(
                        "torch not available; falling back to the numpy Maia "
                        "backend (sampler behaves identically)")
                    stype = "maia_numpy"
                else:
                    from samplers.maia_v1_sampler import MaiaV1Sampler
                    self.sampler = MaiaV1Sampler(scfg.get("maia_v1", {}), self.paths)
            elif stype == "maia_numpy":
                from samplers.maia_numpy_sampler import MaiaNumpySampler
                self.sampler = MaiaNumpySampler(
                    scfg.get("maia_numpy", scfg.get("maia_v1", {})), self.paths)
            elif stype == "maia3":
                from samplers.maia3_sampler import Maia3Sampler
                self.sampler = Maia3Sampler(scfg.get("maia3", {}), self.paths)
            elif stype == "lc0_maia":
                from samplers.lc0_maia_sampler import Lc0MaiaSampler
                self.sampler = Lc0MaiaSampler(scfg.get("lc0_maia", {}), self.paths)
            else:
                raise SamplerError(f"unknown sampler type: {stype}")
        except Exception as exc:  # noqa: BLE001
            self.log.error("sampler %s failed to initialise: %s", stype, exc)
            self.sampler = None

        vcfg = self.cfg["validator"]
        vtype = vcfg.get("type", "stockfish")
        engines = []
        if vtype in ("stockfish", "both"):
            engines.append(("stockfish", vcfg.get("stockfish", {})))
        if vtype in ("lc0", "both"):
            engines.append(("lc0", vcfg.get("lc0", {})))
        for name, ecfg in engines:
            try:
                self.validators.append(UciValidator(ecfg, self.paths, engine=name))
            except ValidatorError as exc:
                self.log.error("validator %s failed: %s", name, exc)
        if not self.validators:
            raise ValidatorError("no validator available (install Stockfish)")

    def validator(self) -> UciValidator:
        return self.validators[0]

    # ------------------------------------------------------------------
    # UCI handlers
    # ------------------------------------------------------------------
    def cmd_uci(self):
        print(f"id name {ENGINE_NAME}", flush=True)
        print(f"id author {ENGINE_AUTHOR}", flush=True)
        for name, default in UCI_OPTION_DEFAULTS.items():
            print(f"option name {name} type string default {default}", flush=True)
        print("uciok", flush=True)

    def cmd_isready(self):
        # Load everything so the first `go` is fast.
        try:
            self.load_components()
            if self.sampler is not None and not self.sampler.ready():
                self.log.warning("sampler not ready; engine will play validator-best only")
        except Exception as exc:  # noqa: BLE001
            self.log.error("isready init failed: %s", exc)
        # Wait for an in-flight search so the GUI state is consistent.
        if self._searching.is_set():
            self._join_search(5.0)
        print("readyok", flush=True)

    def cmd_ucinewgame(self):
        self.boards = [chess.Board()]
        for v in self.validators:
            v.new_game()

    def cmd_position(self, args: list[str]):
        boards: list[chess.Board] = []
        if args and args[0] == "startpos":
            boards = [chess.Board()]
            rest = args[1:]
        elif args and args[0] == "fen":
            fen_parts = []
            i = 1
            while i < len(args) and args[i] != "moves":
                fen_parts.append(args[i])
                i += 1
            rest = args[i:]
            boards = [chess.Board(" ".join(fen_parts))]
        else:
            return
        if rest and rest[0] == "moves":
            for uci in rest[1:]:
                try:
                    move = chess.Move.from_uci(uci)
                except ValueError:
                    break
                if move in boards[-1].legal_moves:
                    b = boards[-1].copy(stack=False)
                    b.push(move)
                    boards.append(b)
        self.boards = boards

    def cmd_go(self, args: list[str]):
        go = {}
        it = iter(args)
        for tok in it:
            if tok in ("wtime", "btime", "winc", "binc", "movetime", "depth",
                       "nodes", "mate", "searchmoves"):
                try:
                    go[tok] = int(next(it))
                except StopIteration:
                    pass
            elif tok == "infinite":
                go[tok] = True
            elif tok == "ponder":
                go[tok] = True
        if self._searching.is_set():
            self.log.warning("go while searching; ignoring")
            return
        self._stop_req.clear()
        self._search_thread = threading.Thread(
            target=self._run_search, args=(go,), daemon=True)
        self._search_thread.start()

    def cmd_stop(self):
        self._stop_req.set()
        for v in self.validators:
            v.stop()
        self._join_search(3.0)

    def cmd_setoption(self, args: list[str]):
        # setoption name X value Y [value Z ...]
        name = None
        value_parts = []
        i = 0
        while i < len(args):
            if args[i] == "name" and name is None:
                name = args[i + 1]
                i += 2
            elif args[i] == "value":
                value_parts = args[i + 1:]
                break
            else:
                i += 1
        if name is None:
            return
        value = " ".join(value_parts)
        self._apply_option(name, value)

    def _apply_option(self, name: str, value: str):
        self.log.info("setoption %s = %s", name, value)
        low = name.lower()
        try:
            if low in ("samplerelo", "elo"):
                self.cfg["sampler"]["maia3"]["elo"] = int(float(value))
            elif low == "samplermodel":
                self.cfg["sampler"]["maia3"]["model"] = value
            elif low == "samplerweights":
                self.cfg["sampler"]["maia_v1"]["weights"] = value
            elif low == "candidatecount":
                self.cfg["sampler"]["candidate_count"] = int(value)
            elif low == "minwinprob":
                self.cfg["filter"]["win_prob_threshold"] = float(value)
            elif low == "maxcploss":
                self.cfg["filter"]["max_cp_loss"] = float(value)
            elif low == "mincploss":
                self.cfg["filter"]["min_cp_loss"] = float(value)
            elif low == "requirenotbest":
                self.cfg["filter"]["require_not_best"] = value.lower() in ("true", "1", "yes")
            elif low == "selectionstyle":
                self.cfg["filter"]["selection"] = value
            elif low == "depthcap":
                self.cfg["time"]["depth_cap"] = int(value)
            elif low == "quickdepth":
                self.cfg["time"]["quick_depth"] = int(value)
            elif low == "threads":
                self.cfg["validator"]["stockfish"]["threads"] = int(value)
                self.cfg["validator"]["lc0"]["threads"] = int(value)
            elif low == "hash":
                self.cfg["validator"]["stockfish"]["hash_mb"] = int(value)
            # Unknown options are accepted and ignored (UCI requires this
            # tolerance so GUIs/lichess-bot can push generic options).
        except (TypeError, ValueError):
            self.log.warning("bad value for option %s: %r", name, value)

    # ------------------------------------------------------------------
    # Search pipeline
    # ------------------------------------------------------------------
    def _budget(self, go: dict) -> tuple[int, int, int]:
        """Return (movetime_ms, depth, nodes) for pass B, plus pass A time."""
        t = self.cfg["time"]
        if go.get("movetime"):
            budget = int(go["movetime"])
        elif "wtime" in go or "btime" in go:
            white = self.boards[-1].turn == chess.WHITE
            remaining = go.get("wtime" if white else "btime", 0) or 0
            inc = go.get("winc" if white else "binc", 0) or 0
            budget = int(remaining / 40 + inc * 0.75)
            budget = max(t["movetime_min_ms"], min(t["movetime_max_ms"], budget))
        else:
            budget = 0  # infinite / fixed-depth mode
        budget = int(budget * t["movetime_scale"])
        depth = go.get("depth", 0) or 0
        nodes = go.get("nodes", 0) or 0
        return budget, depth, nodes

    def _run_search(self, go: dict):
        self._searching.set()
        try:
            self._search_inner(go)
        except Exception:  # noqa: BLE001
            self.log.error("search crashed:\n%s", traceback.format_exc())
            try:
                print("bestmove 0000", flush=True)
            except Exception:  # noqa: BLE001
                pass
        finally:
            self._searching.clear()

    def _search_inner(self, go: dict):
        board = self.boards[-1]
        if board.is_game_over():
            print("bestmove 0000", flush=True)
            return

        self.load_components()
        t = self.cfg["time"]
        budget, depth, nodes = self._budget(go)
        quick_depth = min(t["quick_depth"], depth) if depth else t["quick_depth"]
        pass_a_ms = int(budget * t["pass_a_share"]) if budget else 0
        pass_b_ms = int(budget * (1 - t["pass_a_share"])) if budget else 0
        depth_cap = min(t["depth_cap"], depth) if depth else t["depth_cap"]

        fen = board.fen()

        # ---- candidates -------------------------------------------------
        candidates = []
        try:
            if self.sampler is not None:
                candidates = self.sampler.sample(
                    self.boards, int(self.cfg["sampler"].get("candidate_count", 12)))
        except Exception as exc:  # noqa: BLE001
            self.log.error("sampling failed: %s", exc)
            candidates = []
        self._print_info(f"candidates={len(candidates)} "
                         + " ".join(f"{c.move.uci()}({c.prob:.3f})" for c in candidates[:8]))

        # ---- pass A: best move -----------------------------------------
        try:
            res_a = self.validator().analyse(fen, depth=quick_depth,
                                             movetime_ms=pass_a_ms, nodes=nodes)
        except ValidatorError as exc:
            self.log.error("pass A (best move) failed: %s", exc)
            self._print_info("validator unavailable; playing no move")
            raise
        best_move = res_a.best_move or ""
        best_score = res_a.best_score_cp
        # Report a score so lichess-bot can resign/draw on our behalf.
        print(f"info depth {res_a.depth} score cp {int(round(best_score))} "
              f"pv {best_move}", flush=True)

        # ---- pass B: deep evaluation of every candidate ------------------
        # B1: one multipv search covers the engine's own top moves (which
        # usually overlap with Maia's); B2: any candidate not in that list is
        # evaluated individually with `searchmoves` so Maia's "blunders" that
        # the engine ranks low still get a precise score.
        mode = self.cfg["filter"].get("win_prob_mode", "expected")

        def eff(score_cp, win_prob, wdl):
            """WDL -> effective win probability per win_prob_mode."""
            if wdl is not None:
                win, draw, _ = wdl
                if mode == "win_only":
                    return win / 1000.0
                return (win + 0.5 * draw) / 1000.0
            return win_prob

        candidate_scores = {}
        if candidates:
            want = len(candidates) + 1
            try:
                res_b = self.validator().analyse(
                    fen, depth=depth_cap, movetime_ms=pass_b_ms, nodes=nodes,
                    multipv=min(want, 20))
                for ev in res_b.evals:
                    candidate_scores[ev.move_uci] = (
                        ev.score_cp, eff(ev.score_cp, ev.win_prob, ev.wdl))
            except Exception as exc:  # noqa: BLE001
                self.log.error("multipv pass failed: %s", exc)

            missing = [c for c in candidates if c.move.uci() not in candidate_scores]
            if missing:
                per_move = min(pass_b_ms // max(1, len(missing)), 3000) if pass_b_ms else 0
                ind_depth = min(depth_cap, quick_depth + 4) if depth_cap else 0
                for c in missing:
                    try:
                        r = self.validator().analyse(
                            fen, depth=ind_depth, movetime_ms=per_move, nodes=nodes,
                            searchmoves=[c.move.uci()])
                        if r.best_move == c.move.uci():
                            candidate_scores[c.move.uci()] = (
                                r.best_score_cp,
                                eff(r.best_score_cp, r.best_win_prob, r.evals[0].wdl if r.evals else None))
                    except Exception as exc:  # noqa: BLE001
                        self.log.error("individual eval of %s failed: %s",
                                       c.move.uci(), exc)

            if best_move and best_move not in candidate_scores:
                candidate_scores[best_move] = (
                    best_score, eff(best_score, res_a.best_win_prob, None))

        for c in candidates:
            item = candidate_scores.get(c.move.uci())
            if item:
                self._print_info(f"cand {c.move.uci()} maia#{c.rank} p={c.prob:.3f} "
                                 f"score={item[0]:+.0f} wp={item[1]:.3f}")

        # ---- selection ---------------------------------------------------
        flt = self.cfg["filter"]
        chosen, kind, loss, wp = select_move(
            candidates, best_move, best_score, candidate_scores, flt, self._rng)

        if self._stop_req.is_set() and not go.get("infinite"):
            self.log.info("stopped before selection; playing best move")

        self._print_info(
            f"chosen={chosen} kind={kind} best={best_move}({best_score:+.0f}) "
            + (f"loss={loss:+.0f}cp wp={wp:.3f}" if wp is not None else ""))
        print(f"bestmove {chosen}", flush=True)

    def _print_info(self, msg: str):
        print(f"info string [trojan] {msg}", flush=True)
        self.log.info("%s", msg)

    def _join_search(self, timeout: float):
        th = self._search_thread
        if th is not None and th.is_alive():
            th.join(timeout)

    def shutdown(self):
        if self._searching.is_set():
            self.cmd_stop()
        for v in self.validators:
            v.close()
        if self.sampler is not None:
            try:
                self.sampler.close()
            except Exception:  # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
# UCI main loop
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=ENGINE_NAME + " UCI engine")
    parser.add_argument("--config", default=os.environ.get("TROJAN_CONFIG", ""),
                        help="path to engine.yml")
    args = parser.parse_args(argv)

    engine = TrojanEngine(args.config or None)

    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cmd = parts[0]
            try:
                if cmd == "uci":
                    engine.cmd_uci()
                elif cmd == "isready":
                    engine.cmd_isready()
                elif cmd == "ucinewgame":
                    engine.cmd_ucinewgame()
                elif cmd == "position":
                    engine.cmd_position(parts[1:])
                elif cmd == "go":
                    engine.cmd_go(parts[1:])
                elif cmd == "stop":
                    engine.cmd_stop()
                elif cmd == "ponderhit":
                    pass
                elif cmd == "setoption":
                    engine.cmd_setoption(parts[1:])
                elif cmd == "quit":
                    break
            except Exception:  # noqa: BLE001
                engine.log.error("error handling %r:\n%s", line, traceback.format_exc())
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
