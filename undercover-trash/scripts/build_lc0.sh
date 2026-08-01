#!/usr/bin/env bash
# =============================================================================
# Build lc0 from source (needed only if you want the optional lc0 validator
# or the lc0_maia sampler).
#
#   ./scripts/build_lc0.sh
#
# Result: <repo>/third_party/lc0/build/release/lc0
# Set validator.lc0.path (engine.yml) to that binary and run
#   ./scripts/download_weights.sh lc0net
# to fetch a network for it.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."

mkdir -p third_party
if [ ! -d third_party/lc0 ]; then
  git clone --depth 1 https://github.com/LeelaChessZero/lc0.git third_party/lc0
fi
cd third_party/lc0

# Build dependencies: meson + ninja (pip), OpenBLAS + zlib (apt).
if ! command -v meson >/dev/null 2>&1 || ! command -v ninja >/dev/null 2>&1; then
  echo "==> installing meson and ninja via pip"
  pip install --user meson ninja || pip install meson ninja
fi
if ! ldconfig -p 2>/dev/null | grep -q openblas; then
  echo "==> OpenBLAS not found; trying apt (needs sudo)"
  sudo apt-get update -qq && sudo apt-get install -y -qq libopenblas-dev zlib1g-dev || \
    { echo "!! Could not install OpenBLAS. Install it manually, then rerun."; exit 1; }
fi

echo "==> building lc0 (CPU backend, release)"
./build.sh -Dbuildtype=release 2>/dev/null || ./build.sh

BIN="$(pwd)/build/release/lc0"
echo
echo " lc0 binary: $BIN"
echo " test:       printf 'uci\\nquit\\n' | \"$BIN\""
