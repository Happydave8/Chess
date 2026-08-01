#!/usr/bin/env bash
# Launcher that lichess-bot uses as `engine.name`. Resolves the project venv
# (falling back to any python3) and forwards CLI args such as --config=...
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${SCRIPT_DIR}/../.venv/bin/python"
if [ ! -x "$PY" ]; then
  PY="$(command -v python3)"
fi
exec "$PY" "$SCRIPT_DIR/trojan_engine.py" "$@"
