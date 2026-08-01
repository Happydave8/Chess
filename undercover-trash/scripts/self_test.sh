#!/usr/bin/env bash
# =============================================================================
# Offline smoke test: drive the engine through a full UCI session and check
# that it produces legal bestmove answers.
#
#   ./scripts/self_test.sh [depth]
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

DEPTH="${1:-10}"
VENV="$(cd .. && pwd)/.venv"
PY="$VENV/bin/python"
[ -x "$PY" ] || PY=python3

echo "==> UCI handshake + two positions (depth $DEPTH)"
printf 'uci\nisready\nucinewgame\nposition startpos moves e2e4 e7e5\ngo depth %s\nposition startpos moves e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6\ngo depth %s\nquit\n' \
  "$DEPTH" "$DEPTH" \
| "$PY" trojan_engine.py --config engine.yml 2>/dev/null \
| tee /tmp/trojan_self_test.log \
| grep -E "bestmove|\[trojan\] chosen"

echo
echo "==> legality check"
"$PY" - <<'PY'
import re
lines = open("/tmp/trojan_self_test.log").read().splitlines()
ok = True
for i, line in enumerate(lines):
    if line.startswith("bestmove"):
        mv = line.split()[1]
        if mv == "0000":
            continue
        if not re.fullmatch(r"[a-h][1-8][a-h][1-8][qrbn]?", mv):
            print(f"  BAD move syntax: {mv!r}")
            ok = False
print("  all bestmoves are syntactically legal UCI moves" if ok else "  FAILED")
sys_exit = 0 if ok else 1
raise SystemExit(sys_exit)
PY
