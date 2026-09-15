#!/usr/bin/env bash
set -euo pipefail

# Native AIMDO interposition must be present before the validation Python
# process starts. Keep this resolver identical to start_comfyui.sh while
# allowing the image checker to run as its own container command.
native_preload_path=$(python \
    /llm/ComfyUI/custom_nodes/ComfyUI-OmniXPU/runtime_bootstrap.py \
    --allocator-preload-path)
if [[ -z "$native_preload_path" ]]; then
    echo "native_hook provider returned an empty preload path" >&2
    exit 2
fi
export AIMDO_XPU_ALLOCATOR_MODE=native_hook
export LD_PRELOAD="${native_preload_path}${LD_PRELOAD:+:$LD_PRELOAD}"

exec python /llm/tools/validate_comfyui_image.py "$@"
