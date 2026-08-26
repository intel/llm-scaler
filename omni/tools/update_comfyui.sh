#!/usr/bin/env bash
set -euo pipefail

COMFYUI_ROOT="${COMFYUI_ROOT:-/llm/ComfyUI}"
COMFYUI_UPGRADE_REF="${COMFYUI_UPGRADE_REF:-master}"

cd "${COMFYUI_ROOT}"

# Preserve user edits instead of hiding them in an implicit stash. Runtime
# model, input, output, and user directories must be mounted outside this Git
# checkout and selected through ComfyUI's path arguments; mounting over its
# tracked directories correctly makes the checkout appear modified.
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "Refusing to upgrade a ComfyUI checkout with tracked local changes" >&2
    exit 1
fi

git fetch --depth 1 origin "${COMFYUI_UPGRADE_REF}"
git checkout --detach FETCH_HEAD
test "$(git rev-parse HEAD)" = "$(git rev-parse FETCH_HEAD)"

# The focused image exports PIP_CONSTRAINT for the Torch/XPU ABI and provider
# distributions. Official ComfyUI, Kitchen, and AIMDO requirements may move;
# incompatible XPU providers remain installed but are skipped at prestartup.
python -m pip install --upgrade -r requirements.txt
python -m pip check
