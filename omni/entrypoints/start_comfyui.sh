#!/usr/bin/env bash
set -euo pipefail

export no_proxy="${no_proxy:-localhost,127.0.0.1}"

# Keep runtime VRAM headroom configurable for model switching.
reserve_vram_gb="${OMNI_COMFYUI_RESERVE_VRAM_GB:-4}"
if [[ ! "$reserve_vram_gb" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "OMNI_COMFYUI_RESERVE_VRAM_GB must be a nonnegative number" >&2
    exit 2
fi

# An explicit memory-mode argument replaces the entrypoint default. ComfyUI
# gives --enable-dynamic-vram precedence if both flags reach its parser.
dynamic_vram_arguments=(--enable-dynamic-vram)
explicit_dynamic_vram_argument=""
for argument in "$@"; do
    case "$argument" in
        --enable-dynamic-vram|--disable-dynamic-vram)
            if [[ -n "$explicit_dynamic_vram_argument" && \
                  "$explicit_dynamic_vram_argument" != "$argument" ]]; then
                echo "choose only one DynamicVRAM enable/disable argument" >&2
                exit 2
            fi
            explicit_dynamic_vram_argument="$argument"
            dynamic_vram_arguments=()
            ;;
    esac
done

if [[ "${AIMDO_XPU_ALLOCATOR_MODE:-}" == "native_hook" ]]; then
    if [[ "$explicit_dynamic_vram_argument" == "--disable-dynamic-vram" ]]; then
        echo "native_hook requires DynamicVRAM" >&2
        exit 2
    fi
    native_preload_path=$(python \
        /llm/ComfyUI/custom_nodes/ComfyUI-OmniXPU/runtime_bootstrap.py \
        --native-preload-path)
    if [[ -z "$native_preload_path" ]]; then
        echo "native_hook provider returned an empty preload path" >&2
        exit 2
    fi
    export LD_PRELOAD="${native_preload_path}${LD_PRELOAD:+:$LD_PRELOAD}"
fi

exec python /llm/ComfyUI/main.py \
    --listen 0.0.0.0 \
    --port 8188 \
    --extra-model-paths-config /llm/configs/comfyui_host_models.yaml \
    --input-directory /data/input \
    --output-directory /data/output \
    --user-directory /data/user \
    --reserve-vram "$reserve_vram_gb" \
    "${dynamic_vram_arguments[@]}" \
    --enable-manager \
    "$@"
