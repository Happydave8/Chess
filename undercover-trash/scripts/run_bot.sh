#!/usr/bin/env bash
# =============================================================================
# Run the TrojanHorse bot on Lichess.
#
#   ./scripts/run_bot.sh            start the bot
#   ./scripts/run_bot.sh -u         upgrade the account to a bot (once)
#   ./scripts/run_bot.sh -v         verbose logging
#
# Config: lichess/config.yml (token + challenge settings)
# =============================================================================
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PROJ="$REPO_ROOT/undercover-trash"
VENV="$REPO_ROOT/.venv"
CONFIG="${CONFIG:-$PROJ/lichess/config.yml}"

if [ ! -f "$CONFIG" ]; then
  echo "config not found: $CONFIG  (copy lichess/config.yml.default first)" >&2
  exit 1
fi

UPGRADE=""
if [ "${1:-}" = "-u" ]; then
  UPGRADE="-u"
  shift
fi

cd "$REPO_ROOT/third_party/lichess-bot"
exec "$VENV/bin/python" lichess-bot.py $UPGRADE -c "$CONFIG" "$@"
