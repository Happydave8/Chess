#!/usr/bin/env bash
# =============================================================================
# One-shot TrojanHorse installer for Termux (Android).
#
# Run INSIDE Termux:
#     pkg install -y git && git clone https://github.com/Happydave8/Chess.git
#     cd Chess && ./undercover-trash/scripts/install_termux.sh
#
# What it does:
#   1. installs python, git, stockfish, clang + numpy via pkg
#   2. pip-installs python-chess + pyyaml (pure/wheel, no torch needed)
#   3. downloads the Maia-1100 weights
#   4. asks for your Lichess token (bot:play scope) and stores it in
#      undercover-trash/lichess/.token.env  (gitignored, mode 600)
#   5. prints the exact commands to upgrade the account and start the bot
#
# Then keep the bot alive with:
#     pkg install -y tmux termux-api
#     termux-wake-lock
#     tmux new-session -s bot './undercover-trash/scripts/run_bot.sh'
# (detach with Ctrl+B D; reattach with tmux attach -t bot)
# =============================================================================
set -euo pipefail

if [ -z "${PREFIX:-}" ] || [ ! -d "${PREFIX:-}" ]; then
  echo "error: this script must run inside Termux (PREFIX is not set)." >&2
  echo "Open Termux and clone the repo there first." >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PROJ="$REPO_ROOT/undercover-trash"

echo "==> TrojanHorse installer for Termux"
echo "    repo: $REPO_ROOT"

echo "==> updating Termux packages"
pkg update -y && pkg upgrade -y

echo "==> installing packages (python, stockfish, numpy, build tools)"
pkg install -y python git stockfish clang binutils python-numpy || {
  echo "  (python-numpy may not be in your repo; falling back to pip)"
  pkg install -y python git stockfish clang binutils
}

echo "==> installing python modules"
python -m pip install --upgrade pip
python -m pip install python-chess pyyaml || {
  echo "  pip install failed; trying pkg fallbacks..."
  pkg install -y python-chess 2>/dev/null || true
}

echo "==> verifying the engine's python deps"
python - <<'PY'
import chess, numpy, yaml
print(f"  python-chess {chess.__version__}, numpy {numpy.__version__}, yaml ok")
try:
    import torch  # noqa
    print("  (torch present — not needed on Termux, maia_numpy is used)")
except ImportError:
    print("  torch not present — expected; using the numpy Maia backend")
PY

echo "==> downloading Maia-1100 weights (~1.3 MB)"
mkdir -p "$PROJ/weights"
if [ ! -s "$PROJ/weights/maia-1100.pb.gz" ]; then
  curl -fL --retry 3 -o "$PROJ/weights/maia-1100.pb.gz" \
    "https://github.com/CSSLab/maia-chess/releases/download/v1.0/maia-1100.pb.gz"
fi
ls -l "$PROJ/weights/maia-1100.pb.gz"

echo "==> Lichess token"
if [ -f "$PROJ/lichess/.token.env" ] && grep -q "lip_" "$PROJ/lichess/.token.env"; then
  echo "  token already present in lichess/.token.env"
else
  read -r -p "  Paste your Lichess bot:play token (starts with lip_): " TOKEN
  if [ -z "$TOKEN" ]; then
    echo "  no token given — you can add it later:"
    echo "  echo 'LICHESS_BOT_TOKEN=lip_...' > $PROJ/lichess/.token.env"
  else
    umask 177
    printf 'LICHESS_BOT_TOKEN=%s\n' "$TOKEN" > "$PROJ/lichess/.token.env"
    chmod 600 "$PROJ/lichess/.token.env"
    echo "  stored in lichess/.token.env (gitignored, mode 600)"
  fi
fi

echo "==> fixing config paths for this device"
cd "$REPO_ROOT"
sed -i "s|/home/user/Chess|$REPO_ROOT|g" "$PROJ/lichess/config.yml"

echo
echo "=================================================================="
echo " Termux setup complete."
echo
echo " Next steps:"
echo "  1. Upgrade the account to a bot account (irreversible, once):"
echo "       ./undercover-trash/scripts/run_bot.sh -u"
echo "  2. Start playing (needs the token in lichess/.token.env):"
echo "       ./undercover-trash/scripts/run_bot.sh"
echo "     (the engine config for phones is engine_termux.yml; run_bot.sh"
echo "      already uses it when TROJAN_CONFIG=engine_termux.yml — export it:"
echo "       export TROJAN_CONFIG=\$PWD/undercover-trash/engine_termux.yml)"
echo "  3. Keep it alive while the screen is off:"
echo "       pkg install -y tmux termux-api"
echo "       termux-wake-lock"
echo "       tmux new-session -s bot -c $REPO_ROOT './undercover-trash/scripts/run_bot.sh'"
echo "     (detach: Ctrl+B then D   |   reattach: tmux attach -t bot)"
echo "=================================================================="
