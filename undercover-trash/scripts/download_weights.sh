#!/usr/bin/env bash
# =============================================================================
# Download all model weights for TrojanHorse.
#
#   ./scripts/download_weights.sh [maia3] [maia_v1] [lc0net]
#
#   maia_v1  : maia-1100..maia-1900 .pb.gz nets (CSSLab/maia-chess release)
#              -> undercover-trash/weights/
#   maia3    : Maia-3 checkpoints from Hugging Face (needed for sampler maia3)
#   lc0net   : a current lc0 net for the optional lc0 validator (Lc0 style)
#              (only needed if validator.type is lc0 or both)
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

MAIA_BASE="https://github.com/CSSLab/maia-chess/releases/download/v1.0"
MAIA3_ALIAS="${MAIA3_ALIAS:-maia3-5m}"
RATINGS="1100 1200 1300 1400 1500 1600 1700 1800 1900"

mkdir -p weights

fetch_maia_v1() {
  local rating
  for rating in $RATINGS; do
    local out="weights/maia-${rating}.pb.gz"
    if [ -s "$out" ]; then
      echo "  [skip] $out already present"
      continue
    fi
    echo "  [get ] $out"
    curl -fL --retry 3 -o "$out" "${MAIA_BASE}/maia-${rating}.pb.gz" \
      || { echo "  [FAIL] maia-${rating} download failed"; rm -f "$out"; }
  done
}

fetch_maia3() {
  if ! python3 -c "import huggingface_hub" 2>/dev/null; then
    echo "  [FAIL] huggingface_hub missing — run scripts/setup.sh first"
    return 1
  fi
  echo "  [get ] Maia3 checkpoint: $MAIA3_ALIAS (Hugging Face, ~1st run downloads)"
  python3 - "$MAIA3_ALIAS" <<'PY'
import sys
from huggingface_hub import snapshot_download
alias = sys.argv[1]
specs = {
    "maia3-5m": "UofTCSSLab/Maia3-5M",
    "maia3-23m": "UofTCSSLab/Maia3-23M",
    "maia3-79m": "UofTCSSLab/Maia3-79M",
}
repo = specs.get(alias, alias)
path = snapshot_download(repo_id=repo)
print(f"  [ ok ] Maia3 cached at {path}")
print("        set sampler.maia3.model in engine.yml if you want a different size")
PY
}

fetch_lc0net() {
  local url="${LC0_NET_URL:-https://training.lczero.org/get_network?download=latest}"
  echo "  [get ] latest lc0 net from $url"
  curl -fL --retry 3 -o weights/lc0-latest.pb.gz "$url" \
    && echo "  [ ok ] weights/lc0-latest.pb.gz  (set validator.lc0.weights to it)"
}

what="maia_v1"
[ $# -gt 0 ] && what="$*"

case "$what" in
  *maia_v1*|all) echo "== Maia v1 nets =="; fetch_maia_v1 ;;
esac
case "$what" in
  *maia3*|all) echo "== Maia-3 =="; fetch_maia3 ;;
esac
case "$what" in
  *lc0net*|all) echo "== lc0 net =="; fetch_lc0net ;;
esac

echo "done."
