#!/usr/bin/env bash
# =============================================================================
# One-shot setup for the TrojanHorse bot.
#
#   1. python venv + pip dependencies (torch CPU by default; set
#      TORCH_INDEX=https://download.pytorch.org/whl/cu128 to get a CUDA build)
#   2. clones lichess-bot and CSSLab/maia3 into <repo>/third_party/
#   3. installs/acquires Stockfish (existing binary > official release >
#      source build)
#   4. downloads Maia weights (skipped if already present)
#   5. writes the lichess-bot config (lichess/config.yml) if missing
#
# Afterwards:
#   * create the Lichess bot account + token (see lichess/config.yml header)
#   * ./scripts/run_bot.sh -u      (upgrade account to bot — once)
#   * ./scripts/run_bot.sh         (start playing)
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"          # <repo>/
PROJ="$REPO_ROOT/undercover-trash"
THIRD="$REPO_ROOT/third_party"
VENV="$REPO_ROOT/.venv"
PY="${PYTHON:-python3}"

echo "==> TrojanHorse setup"
echo "    repo:   $REPO_ROOT"
echo "    project: $PROJ"
echo "    venv:   $VENV"

mkdir -p "$THIRD" "$PROJ/bin" "$PROJ/weights" "$PROJ/logs"

# ------------------------------------------------------------------ venv
if [ ! -x "$VENV/bin/python" ]; then
  echo "==> creating virtualenv"
  "$PY" -m venv "$VENV"
fi
"$VENV/bin/pip" install -q --upgrade pip
echo "==> installing python dependencies"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cpu}"
"$VENV/bin/pip" install -q torch --index-url "$TORCH_INDEX" || \
  "$VENV/bin/pip" install -q torch
"$VENV/bin/pip" install -q -r "$PROJ/requirements.txt"

# ------------------------------------------------------------ lichess-bot
if [ ! -d "$THIRD/lichess-bot" ]; then
  echo "==> cloning lichess-bot"
  git clone -q --depth 1 https://github.com/lichess-bot/lichess-bot.git "$THIRD/lichess-bot"
fi
# lichess-bot pins old python-chess; install it first so the engine's
# requirements (installed above) win in the end.
"$VENV/bin/pip" install -q -r "$THIRD/lichess-bot/requirements.txt" || true

# ------------------------------------------------------------- maia3 pkg
if [ ! -d "$THIRD/maia3" ]; then
  echo "==> cloning CSSLab/maia3"
  git clone -q --depth 1 https://github.com/CSSLab/maia3.git "$THIRD/maia3"
fi
"$VENV/bin/pip" install -q -e "$THIRD/maia3" || \
  "$VENV/bin/pip" install -q -r "$THIRD/maia3/requirements.txt"

# -------------------------------------------------------------- stockfish
SF="$PROJ/bin/stockfish"
if [ ! -x "$SF" ]; then
  if command -v stockfish >/dev/null 2>&1; then
    echo "==> using system stockfish"
    ln -sf "$(command -v stockfish)" "$SF"
  else
    echo "==> downloading official Stockfish release"
    SF_TAG="$(curl -fsSL https://api.github.com/repos/official-stockfish/Stockfish/releases/latest \
               | python3 -c 'import json,sys; print(json.load(sys.stdin)["tag_name"])')"
    ARCH="x86-64-avx2"
    curl -fL -o /tmp/sf.tar "https://github.com/official-stockfish/Stockfish/releases/download/${SF_TAG}/stockfish-ubuntu-${ARCH}.tar"
    tar -xf /tmp/sf.tar -C /tmp
    cp "$(find /tmp -path '*/stockfish-ubuntu-*' -type f | head -1)" "$SF"
    chmod +x "$SF"
  fi
fi
echo "    stockfish: $SF"

# ---------------------------------------------------------------- weights
echo "==> Maia weights"
"$PROJ/scripts/download_weights.sh" maia_v1 || true

echo
echo "=================================================================="
echo " Setup complete."
echo
echo " Next steps:"
echo "  1. Lichess bot account + token: create a bot:play token at"
echo "     https://lichess.org/account/oauth/token/create?scopes[]=bot:play"
echo "     and store it in $PROJ/lichess/.token.env as"
echo "         LICHESS_BOT_TOKEN=lip_..."
echo "  2. Upgrade the account once (irreversible):"
echo "     $PROJ/scripts/run_bot.sh -u"
echo "  3. Play (or install the systemd unit for 24/7):"
echo "     $PROJ/scripts/run_bot.sh"
echo "=================================================================="
