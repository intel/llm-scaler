#!/usr/bin/env bash
set -euo pipefail

export no_proxy="${no_proxy:-localhost,127.0.0.1}"

# Keep runtime VRAM headroom configurable for model switching.
reserve_vram_gb="${OMNI_COMFYUI_RESERVE_VRAM_GB:-4}"
if [[ ! "$reserve_vram_gb" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "OMNI_COMFYUI_RESERVE_VRAM_GB must be a nonnegative number" >&2
    exit 2
fi

exec python /llm/ComfyUI/main.py \
    --listen 0.0.0.0 \
    --port 8188 \
    --extra-model-paths-config /llm/configs/comfyui_host_models.yaml \
    --input-directory /data/input \
    --output-directory /data/output \
    --user-directory /data/user \
    --reserve-vram "$reserve_vram_gb" \
    --enable-dynamic-vram \
    --enable-manager \
    "$@"
