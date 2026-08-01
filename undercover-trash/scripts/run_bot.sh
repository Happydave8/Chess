#!/usr/bin/env bash
# =============================================================================
# Run the TrojanHorse bot on Lichess.
#
#   ./scripts/run_bot.sh            start the bot
#   ./scripts/run_bot.sh -u         upgrade the account to a bot account (once)
#   ./scripts/run_bot.sh -v         verbose logging
#   ./scripts/run_bot.sh --check    show the connected account status
#
# Config: lichess/config.yml (challenge settings etc.)
# Token : lichess/.token.env  (LICHESS_BOT_TOKEN=lip_..., gitignored) or the
#         LICHESS_BOT_TOKEN environment variable.
# =============================================================================
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PROJ="$REPO_ROOT/undercover-trash"
VENV="$REPO_ROOT/.venv"
CONFIG="${CONFIG:-$PROJ/lichess/config.yml}"

# Load the bot token from the gitignored token file if present (never edit
# the tracked config.yml to put the token in).
if [ -f "$PROJ/lichess/.token.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$PROJ/lichess/.token.env"
  set +a
fi
if [ -z "${LICHESS_BOT_TOKEN:-}" ] || [ "$LICHESS_BOT_TOKEN" = "REPLACE_WITH_YOUR_BOT_PLAY_TOKEN" ]; then
  echo "error: no Lichess token found." >&2
  echo "  Put LICHESS_BOT_TOKEN=lip_... into $PROJ/lichess/.token.env" >&2
  exit 1
fi

if [ ! -f "$CONFIG" ]; then
  echo "config not found: $CONFIG" >&2
  exit 1
fi

ACTION=""
case "${1:-}" in
  -u) ACTION="-u"; shift ;;
  --check) ACTION="--check"; shift ;;
esac

cd "$REPO_ROOT/third_party/lichess-bot"

if [ "$ACTION" = "--check" ]; then
  exec "$VENV/bin/python" - "$LICHESS_BOT_TOKEN" <<'PY'
import json, sys, urllib.request
req = urllib.request.Request("https://lichess.org/api/account",
                             headers={"Authorization": f"Bearer {sys.argv[1]}"})
try:
    with urllib.request.urlopen(req, timeout=20) as r:
        a = json.load(r)
    print(f"account: {a['username']}  (created {a['createdAt']})")
    print(f"bot account: {a.get('bot', False)}")
    print(f"title: {a.get('title', '-')}")
    print(f"playTime: {a.get('playTime', 0)}s")
except urllib.error.HTTPError as e:
    print(f"HTTP {e.code}: {e.read().decode()[:300]}")
    sys.exit(1)
except Exception as e:
    print(f"error: {e}")
    sys.exit(1)
PY
fi

exec "$VENV/bin/python" lichess-bot.py $ACTION --config "$CONFIG" "$@"
