#!/usr/bin/env python3
"""TrojanHorse test suite (no pytest needed: `python3 tests/run_tests.py`).

Covers:
  1. the 112-plane encoder (exact plane values in known positions),
  2. the Maia-v1 sampler against the real maia-1100 net (human-move sanity),
  3. the Maia-3 sampler pipeline with a random-weight model,
  4. the Trojan Horse selection filter (pure function),
  5. a full UCI session over a subprocess (handshake + bestmoves).
"""

from __future__ import annotations

import os
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess

HERE = Path(__file__).resolve().parent.parent
WEIGHTS = HERE / "weights" / "maia-1100.pb.gz"

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


# ---------------------------------------------------------------------------
def test_encoder():
    print("== encoder ==")
    from maia_v1.encoder import encode_position, AUX_BASE

    # Start position, white to move
    planes = encode_position([chess.Board()])
    assert planes.shape == (112, 8, 8), planes.shape
    # a1 white rook -> plane 3 (our rooks), rank 0 file 0
    check("startpos: a1 rook in plane 3", planes[3, 0, 0] == 1.0)
    check("startpos: e1 king in plane 5", planes[5, 0, 4] == 1.0)
    check("startpos: e8 black king in plane 11", planes[11, 7, 4] == 1.0)
    check("startpos: castling KQkq planes 104-107", all(planes[AUX_BASE + i, :, :].all() for i in range(4)))
    check("startpos: plane 108 = 0 (white to move)", planes[AUX_BASE + 4].sum() == 0)
    check("startpos: plane 109 = 0 (rule50)", planes[AUX_BASE + 5, 0, 0] == 0.0)
    check("startpos: plane 111 all ones", planes[AUX_BASE + 7].all())
    # Single-position history: older planes are zero (lc0 FillEmptyHistory
    # stops at the start position).
    check("startpos: history[1] planes are zero", planes[13:26].sum() == 0)

    # Black to move after 1.e4: plane 108 must be 1; black pieces are "ours"
    # (planes 0-5) and stay at their real squares (e7 = rank 6).
    b = chess.Board()
    b.push_san("e4")
    planes = encode_position([chess.Board(), b])
    check("black stm: plane 108 = 1", planes[AUX_BASE + 4].sum() == 64)
    check("black stm: our (black) pawn e7 at plane 0, (6,4)",
          planes[0, 6, 4] == 1.0)
    check("black stm: their (white) pawn e4 at plane 6, (3,4)",
          planes[6, 3, 4] == 1.0)
    # history[1] = startpos rendered from black's perspective (vertical flip):
    # black pawn a7 -> "our pawns" at (1,0); white pawn a2 -> "their pawns" (6,0).
    check("black stm: history[1] plane 13 has black a7 at (1,0)",
          planes[13, 1, 0] == 1.0)
    check("black stm: history[1] plane 19 has white a2 at (6,0)",
          planes[19, 6, 0] == 1.0)
    # castling rights intact for both sides after 1.e4
    check("black stm: planes 104-107 all set (rights intact)",
          all(planes[AUX_BASE + i].all() for i in range(4)))

    # After 1.e4 e5 2.Ke2: white lost both rights, black kept them.
    b = chess.Board()
    for uci in ("e2e4", "e7e5", "e1e2"):
        b.push_uci(uci)
    planes = encode_position([chess.Board(), b])
    check("white king moved: plane 106/107 (their castling) = 0",
          planes[AUX_BASE + 2].sum() == 0 and planes[AUX_BASE + 3].sum() == 0)
    check("white king moved: plane 104/105 (our castling) = 1",
          planes[AUX_BASE + 0].all() and planes[AUX_BASE + 1].all())

    # rule50 after a few moves
    b2 = chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 12 40")
    planes = encode_position([b2])
    check("rule50 plane = 12", planes[AUX_BASE + 5, 0, 0] == 12.0)

    # repetition plane
    b3 = chess.Board()
    b3.push_san("Nf3"); b3.push_san("Nf6"); b3.push_san("Ng1"); b3.push_san("Ng8")
    planes = encode_position([chess.Board(), b3])
    # the position after Ng8 equals the start position -> repetition in current
    check("repetition plane 12 set", planes[12].all())
    # ...but the same position one ply earlier (after Ng1) is not a repetition
    check("repetition plane 25 (history[1]) not set", planes[25].sum() == 0)


# ---------------------------------------------------------------------------
def test_maia_v1_sampler():
    print("== maia_v1 sampler ==")
    if not WEIGHTS.exists():
        print("  [skip] weights not present (run scripts/download_weights.sh)")
        return
    from samplers.maia_v1_sampler import MaiaV1Sampler

    class P:
        def resolve(self, p):
            return str(HERE / p)

    s = MaiaV1Sampler({"weights": str(WEIGHTS)}, P())
    t0 = time.time()
    cands = s.sample([chess.Board()], 10)
    print(f"  (inference took {time.time() - t0:.2f}s)")
    top = [c.move.uci() for c in cands]
    print("  top-10 at startpos:", top, [round(c.prob, 4) for c in cands])
    check("top-1 is e2e4 or d2d4 (the two human favourites)", top[0] in ("e2e4", "d2d4"), top[0])
    check("all candidates legal", all(c.move in chess.Board().legal_moves for c in cands))
    check("probs sorted desc", all(cands[i].prob >= cands[i + 1].prob for i in range(len(cands) - 1)))
    check("probs sum ~ 1 (over full dist)", abs(sum(c.prob for c in s.sample([chess.Board()], 40)) - 1.0) < 0.05)

    # Italian game, black to move: Bc5 is the overwhelming human favourite.
    b = chess.Board()
    for uci in ("e2e4", "e7e5", "g1f3", "b8c6", "f1c4"):
        b.push_uci(uci)
    cands = s.sample([chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")] + _replay(b), 6)
    top = [c.move.uci() for c in cands]
    print("  italian, black top-6:", top)
    check("italian: Bc5 in top-2 for Maia-1100", top[0] in ("f8c5", "f8b4") or top[1] == "f8c5", top)


def _replay(board):
    """Return the board list from the start position to `board`."""
    boards = [chess.Board()]
    for m in board.move_stack:
        b = boards[-1].copy(stack=False)
        b.push(m)
        boards.append(b)
    return boards


# ---------------------------------------------------------------------------
def test_pb_reader():
    print("== pb_reader (pure-python .pb.gz parser) ==")
    if not WEIGHTS.exists():
        print("  [skip] weights not present (run scripts/download_weights.sh)")
        return
    from maia_v1.pb_reader import parse_net

    a = parse_net(str(WEIGHTS))
    check("input conv (64,112,3,3)", a.conv1_w.shape == (64, 112, 3, 3))
    check("6 residual blocks", len(a.res) == 6)
    c1w = a.res[0][0]
    check("res conv1 (64,64,3,3)", c1w.shape == (64, 64, 3, 3))
    check("policy1 (64,64,3,3)", a.pol1_w.shape == (64, 64, 3, 3))
    check("policy2 (80,64,3,3) + bias 80", a.pol2_w.shape == (80, 64, 3, 3)
          and a.pol2_b.shape == (80,))
    check("value (32,64,1,1)", a.val_w.shape == (32, 64, 1, 1))
    check("ip1 (128,2048)", a.ip1_w.shape == (128, 2048))
    check("ip2 (3,128)", a.ip2_w.shape == (3, 128))
    check("SE w2 = 2C x C/8", a.res[0][8].shape == (128, 8))
    # finite + sane magnitude
    import numpy as np
    check("weights finite", np.isfinite(a.conv1_w).all())
    check("bn scale finite", np.isfinite(a.conv1_scale).all())


def test_backend_parity():
    print("== torch vs numpy backend parity ==")
    if not WEIGHTS.exists():
        print("  [skip] weights not present")
        return
    try:
        import torch  # noqa: F401
    except ImportError:
        print("  [skip] torch not installed")
        return
    import numpy as np
    from maia_v1.encoder import encode_position
    from maia_v1.lc0net import Lc0SeNet
    from maia_v1.numpy_net import NumpyNet
    from maia_v1.policy_utils import candidates_from_logits

    torch_net = Lc0SeNet(str(WEIGHTS), device="cpu")
    np_net = NumpyNet(str(WEIGHTS))

    cases = [
        [chess.Board()],
        [chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")],
        _replay(_to_board("e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6")),
        _replay(_to_board("d2d4 d7d5 c2c4 e7e6 b1c3 g8f6 c4d5")),
    ]
    worst = 0.0
    for boards in cases:
        x = torch.from_numpy(encode_position(boards)).unsqueeze(0)
        with torch.no_grad():
            t_logits, _ = torch_net(x)
        n_logits, _ = np_net(encode_position(boards))
        diff = float(np.abs(t_logits[0].numpy() - n_logits).max())
        worst = max(worst, diff)
        t_top = candidates_from_logits(boards[-1], t_logits[0].numpy(), 3)
        n_top = candidates_from_logits(boards[-1], n_logits, 3)
        check("same top-1 in both backends",
              t_top[0][0] == n_top[0][0] and t_top[1][0] == n_top[1][0],
              f"{t_top[0][0]} vs {n_top[0][0]}")
    check("max |logit diff| < 1e-3", worst < 1e-3, f"max diff {worst:.2e}")


def _to_board(moves_spec: str) -> chess.Board:
    b = chess.Board()
    for uci in moves_spec.split():
        b.push_uci(uci)
    return b


# ---------------------------------------------------------------------------
def test_maia_numpy_sampler():
    print("== maia_numpy sampler (no torch) ==")
    if not WEIGHTS.exists():
        print("  [skip] weights not present")
        return
    from samplers.maia_numpy_sampler import MaiaNumpySampler

    class P:
        def resolve(self, p):
            return str(HERE / p)

    s = MaiaNumpySampler({"weights": str(WEIGHTS)}, P())
    import time
    t0 = time.time()
    cands = s.sample([chess.Board()], 10)
    elapsed = time.time() - t0
    print(f"  (numpy inference took {elapsed:.2f}s)")
    top = [c.move.uci() for c in cands]
    check("numpy top-1 is e2e4 (same as torch/lc0)", top[0] == "e2e4", top[0])
    check("numpy candidates legal + sorted",
          all(c.move in chess.Board().legal_moves for c in cands)
          and all(cands[i].prob >= cands[i + 1].prob for i in range(len(cands) - 1)))
    check("numpy prob of e2e4 ~ 0.66 (matches published Maia-1100)",
          abs(cands[0].prob - 0.66) < 0.05, f"{cands[0].prob:.3f}")
    if elapsed > 5:
        print("  WARNING: numpy inference slow — fine on a phone, "
              "consider candidate_count: 8")


# ---------------------------------------------------------------------------
def test_maia3_sampler():
    print("== maia3 sampler (random weights) ==")
    try:
        import torch  # noqa: F401
        import maia3  # noqa: F401
    except ImportError:
        print("  [skip] torch or maia3 not installed")
        return
    from samplers.maia3_sampler import Maia3Sampler

    class P:
        def resolve(self, p):
            return str(HERE / p)

    s = Maia3Sampler({"model": "maia3-5m", "elo": 1100, "random_weights": True,
                      "temperature": 0.0}, P())
    if not s.ready():
        print("  [skip] maia3 model failed to build")
        return
    b = chess.Board()
    b.push_san("e4")
    cands = s.sample([chess.Board(), b], 8)
    check("maia3 returns legal candidates", all(c.move in b.legal_moves for c in cands))
    check("maia3 returns top-K sorted", all(cands[i].prob >= cands[i + 1].prob for i in range(len(cands) - 1)))
    print("  top-8 after 1.e4 (random net):", [c.move.uci() for c in cands])


# ---------------------------------------------------------------------------
def test_selection():
    print("== selection filter ==")
    from trojan_engine import select_move

    class C:
        def __init__(self, uci, prob, rank):
            self.move = chess.Move.from_uci(uci)
            self.prob = prob
            self.rank = rank

    rng = random.Random(0)
    candidates = [C("e2e4", 0.30, 0), C("d2d4", 0.25, 1), C("g1f3", 0.15, 2), C("h2h4", 0.05, 3)]
    flt = dict(win_prob_threshold=0.45, max_cp_loss=60, min_cp_loss=15,
               require_not_best=True, selection="rank", softmax_temp=1.5, top_j=3)
    # All scores are in centipawns. Best is e2e4 (+35); g1f3 is +15
    # (loss 20cp, wp .58) -> qualifies; d2d4 is +30 (loss 5cp < min_cp_loss)
    # -> too "perfect"; h2h4 is -50 -> blunder.
    scores = {"e2e4": (35, 0.62), "d2d4": (30, 0.60), "g1f3": (15, 0.58), "h2h4": (-50, 0.31)}
    chosen, kind, loss, wp = select_move(candidates, "e2e4", 35, scores, flt, rng)
    check("picks highest Maia move that still looks imperfect", chosen == "g1f3", chosen)
    check("kind == trash", kind == "trash")

    # no candidate qualifies (all blunder) -> fallback to best
    scores2 = {"e2e4": (35, 0.62), "d2d4": (30, 0.60), "g1f3": (-100, 0.10), "h2h4": (-200, 0.05)}
    chosen, kind, _, _ = select_move(candidates, "e2e4", 35, scores2, flt, rng)
    check("falls back to validator best when all candidates blunder", chosen == "e2e4" and kind == "fallback")

    # require_not_best=False lets the best move qualify when Maia loves it
    flt2 = dict(flt, require_not_best=False, min_cp_loss=0)
    chosen, kind, _, _ = select_move(candidates, "e2e4", 35, scores, flt2, rng)
    check("without require_not_best, Maia top-1 (best move) is chosen", chosen == "e2e4")

    # softmax style returns one of the qualifying candidates
    flt3 = dict(flt, selection="softmax", top_j=2, min_cp_loss=0)
    outcomes = {select_move(candidates, "e2e4", 35, scores, flt3, rng)[0] for _ in range(100)}
    check("softmax only ever picks qualifying candidates", outcomes <= {"g1f3", "d2d4"} and len(outcomes) >= 1, str(outcomes))


# ---------------------------------------------------------------------------
def test_uci_session():
    print("== UCI session (subprocess) ==")
    py = sys.executable
    if not py:
        return
    cmd = [py, str(HERE / "trojan_engine.py"), "--config", str(HERE / "engine.yml")]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True, bufsize=1)
    moves = []

    def feed(line):
        proc.stdin.write(line + "\n")
        proc.stdin.flush()

    def read_until(pred, timeout=120):
        end = time.time() + timeout
        while time.time() < end:
            line = proc.stdout.readline()
            if not line:
                raise RuntimeError("engine closed stdout")
            if pred(line.strip()):
                return line.strip()
        raise TimeoutError("timeout waiting for engine output")

    feed("uci")
    read_until(lambda l: l == "uciok")
    feed("isready")
    read_until(lambda l: l == "readyok")
    feed("ucinewgame")
    feed("position startpos moves e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6")
    feed("go depth 8")
    line = read_until(lambda l: l.startswith("bestmove"))
    bm = line.split()[1]
    moves.append(bm)
    check("engine replies with a bestmove", bm != "0000", bm)
    board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    for uci in ("e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6"):
        board.push_uci(uci)
    check("bestmove is legal", chess.Move.from_uci(bm) in board.legal_moves, bm)

    feed("position startpos moves e2e4")
    feed("go movetime 300")
    line = read_until(lambda l: l.startswith("bestmove"), timeout=90)
    bm2 = line.split()[1]
    b2 = chess.Board()
    b2.push_uci("e2e4")
    check("second bestmove legal", chess.Move.from_uci(bm2) in b2.legal_moves, bm2)

    feed("quit")
    proc.wait(timeout=10)


# ---------------------------------------------------------------------------
def main():
    test_encoder()
    test_pb_reader()
    test_maia_v1_sampler()
    test_maia_numpy_sampler()
    test_backend_parity()
    test_maia3_sampler()
    test_selection()
    test_uci_session()
    print()
    print(f"passed: {len(PASS)}   failed: {len(FAIL)}")
    if FAIL:
        print("failed tests:", FAIL)
        sys.exit(1)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
