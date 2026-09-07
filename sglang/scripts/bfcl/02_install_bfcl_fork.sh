#!/usr/bin/env bash
# Install the pinned gorilla fork with Qwen3.6 and Gemma4 model entries as
# editable bfcl_eval into the sglang container's venv.
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
FORK_REPO="${FORK_REPO:-https://github.com/liu-shaojun/gorilla.git}"
FORK_COMMIT="${FORK_COMMIT:-9d49adb45bd765794fd6b0a0f8006e0b31b997d2}"
USE_NETWORK="${USE_NETWORK:-0}"
# Where to clone if USE_NETWORK=1 (and the editable path in that case)
FORK_DIR="${FORK_DIR:-/workspace/bfcl_kit/gorilla}"

source "$VENV/bin/activate"

if [[ "$USE_NETWORK" == "1" ]]; then
  echo "USE_NETWORK=1 → cloning upstream fork at $FORK_COMMIT"
  if [[ ! -d "$FORK_DIR/.git" ]]; then
    git clone "$FORK_REPO" "$FORK_DIR"
  fi
  git -C "$FORK_DIR" remote set-url origin "$FORK_REPO"
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
# package dir and all four model entries route through the native SGLang
# function-calling handlers.
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
expected = {
    "Qwen/Qwen3.6-27B-FC": "Qwen36FCHandler",
    "Qwen/Qwen3.6-35B-A3B-FC": "Qwen36FCHandler",
    "google/gemma-4-31B-it-FC": "Gemma4FCHandler",
    "google/gemma-4-26B-A4B-it-FC": "Gemma4FCHandler",
}
for model_id, handler_name in expected.items():
    model = MODEL_CONFIG_MAPPING[model_id]
    print(model_id, "→", model.model_handler.__name__)
    assert model.model_handler.__name__ == handler_name, (
        f"{model_id} did not resolve to {handler_name}"
    )
print("OK — fork installed correctly")
PY
