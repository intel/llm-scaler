#!/usr/bin/env bash
set -euo pipefail

export no_proxy="${no_proxy:-localhost,127.0.0.1}"

# ComfyUI's default smart-memory calculation reserves enough space to load a
# model's weights, but not necessarily enough for its peak activations.  The
# official LTX 2.3 template exposes this on 32 GiB BMG: after diffusion runs,
# re-executing the 12B XPU text encoder can leave only ~0.7 GiB for activations
# and fail at the final concatenation.  Keep a configurable 4 GiB runtime
# reserve so model switching unloads enough diffusion weights while the text
# encoder remains on XPU.
reserve_vram_gb="${OMNI_COMFYUI_RESERVE_VRAM_GB:-4}"
if [[ ! "$reserve_vram_gb" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "OMNI_COMFYUI_RESERVE_VRAM_GB must be a nonnegative number" >&2
    exit 2
fi

exec python /llm/ComfyUI/main.py \
    --listen 0.0.0.0 \
    --port 8188 \
    --reserve-vram "$reserve_vram_gb" \
    "$@"
