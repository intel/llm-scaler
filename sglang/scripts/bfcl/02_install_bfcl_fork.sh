#!/usr/bin/env bash
# Install zhangYi's gorilla fork (has the Qwen3.6 model entry) as editable
# bfcl_eval into the sglang container's venv. Same fork as the llama.cpp kit
# (commit 6ee7b771) — the harness is backend-agnostic; only the server differs.
#
# OFFLINE-FIRST: the fork's berkeley-function-call-leaderboard subdir is VENDORED
# into this kit (vendor/), so install needs NO network and does not depend on
# the third-party personal fork repo staying alive. Set USE_NETWORK=1 to force a
# fresh `git clone` of the upstream fork instead (e.g. to pick up a newer commit).
#
# Runs INSIDE the container (or wrap with: docker exec <ctr> bash <this>).
set -euo pipefail

KIT_ROOT="${KIT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
VENDOR_PKG="$KIT_ROOT/vendor/berkeley-function-call-leaderboard"
VENV="${VENV:-/opt/venv}"
FORK_COMMIT="${FORK_COMMIT:-6ee7b7718d9c7498b26e043635db6381a6583593}"
USE_NETWORK="${USE_NETWORK:-0}"
# Where to clone if USE_NETWORK=1 (and the editable path in that case)
FORK_DIR="${FORK_DIR:-/workspace/bfcl_kit/gorilla}"

source "$VENV/bin/activate"

if [[ "$USE_NETWORK" == "1" ]]; then
  echo "USE_NETWORK=1 → cloning upstream fork at $FORK_COMMIT"
  if [[ ! -d "$FORK_DIR/.git" ]]; then
    git clone https://github.com/zhangYiIntel/gorilla.git "$FORK_DIR"
  fi
  git -C "$FORK_DIR" fetch origin --tags 2>/dev/null || true
  git -C "$FORK_DIR" checkout "$FORK_COMMIT"
  PKG_DIR="$FORK_DIR/berkeley-function-call-leaderboard"
else
  # Offline: install the vendored copy. pip install -e needs a writable dir
  # (it drops an .egg-link / .egg-info); the vendor dir under the kit is fine.
  [[ -f "$VENDOR_PKG/pyproject.toml" ]] || {
    echo "ERROR: vendored package missing at $VENDOR_PKG"
    echo "       (re-vendor, or run with USE_NETWORK=1 to clone upstream)"; exit 1; }
  echo "Installing VENDORED bfcl_eval (offline) from $VENDOR_PKG"
  echo "  (vendored from gorilla fork commit: $(cat "$KIT_ROOT/vendor/GORILLA_FORK_COMMIT.txt" 2>/dev/null))"
  PKG_DIR="$VENDOR_PKG"
fi

# Editable install + the one missing transitive dep.
pip install -e "$PKG_DIR"
pip install soundfile   # qwen-agent transitive, missing from package deps

# CRITICAL (kit README quirk #3): a stale non-fork bfcl_eval in site-packages
# would shadow the editable install. Verify `import bfcl_eval` resolves to OUR
# package dir and the Qwen3.6 entry routes through QwenFCHandler.
PKG_DIR="$PKG_DIR" python3 - <<'PY'
import bfcl_eval, os
p = os.path.realpath(bfcl_eval.__file__)
want = os.path.realpath(os.environ["PKG_DIR"])
print("bfcl_eval resolves to:", p)
assert p.startswith(want), (
    f"WRONG: {p} is not the installed package {want} — a stale "
    f"site-packages/bfcl_eval/ is shadowing it. `pip uninstall bfcl_eval` "
    f"(maybe twice) and rm the stale dir, then re-run.")
from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
m = MODEL_CONFIG_MAPPING["Qwen/Qwen3.6-35B-A3B-FC"]
print("handler:", m.model_handler.__name__, "| hf model name:", m.model_name)
assert m.model_handler.__name__ == "QwenFCHandler", "fork pinning didn't take"
print("OK — fork installed correctly")
PY
