#!/usr/bin/env bash

set -euo pipefail

log() {
    echo "[vllm-preflight] $*"
}

is_truthy() {
    local value="${1:-}"
    [[ "$value" == "1" || "$value" == "true" || "$value" == "TRUE" || "$value" == "yes" || "$value" == "YES" ]]
}

should_run_preflight() {
    if [[ $# -eq 0 ]]; then
        return 0
    fi

    if [[ "$1" == "serve" ]]; then
        return 0
    fi

    if [[ "$1" == "vllm" && "${2:-}" == "serve" ]]; then
        return 0
    fi

    if [[ "$1" == "python" || "$1" == "python3" ]] && [[ "${2:-}" == "-m" ]] && [[ "${3:-}" == "vllm.entrypoints.openai.api_server" ]]; then
        return 0
    fi

    return 1
}

preflight_intel_gpu() {
    local has_render_nodes=0
    local has_sycl_gpu=0
    local sycl_output=""

    if compgen -G "/dev/dri/renderD*" > /dev/null; then
        has_render_nodes=1
    fi

    if command -v sycl-ls > /dev/null 2>&1; then
        sycl_output="$(sycl-ls 2>&1 || true)"
        if echo "$sycl_output" | grep -Eqi "(level_zero|ext_oneapi_level_zero).*(gpu|xpu)|opencl.*(gpu|xpu)"; then
            has_sycl_gpu=1
        fi
    fi

    if [[ $has_render_nodes -eq 1 && $has_sycl_gpu -eq 0 ]]; then
        log "Found /dev/dri render nodes, but sycl-ls did not report a usable Intel GPU/XPU."
        log "sycl-ls output:"
        echo "$sycl_output"
    fi

    if [[ $has_render_nodes -eq 1 || $has_sycl_gpu -eq 1 ]]; then
        log "Intel GPU preflight passed."
        return 0
    fi

    log "ERROR: No Intel GPU/XPU detected before starting vLLM OpenAI server."
    if [[ -f "/.dockerenv" ]]; then
        log "Container runtime hint: map GPU devices and required flags."
        log "Example docker run flags:"
        log "  --device /dev/dri --group-add render --group-add video"
        log "Example docker compose stanza:"
        log "  devices: [\"/dev/dri:/dev/dri\"]"
    fi
    log "Install oneAPI runtime tools (for sycl-ls) and verify host GPU drivers."

    if is_truthy "${VLLM_PREFLIGHT_CPU_FALLBACK:-0}"; then
        export VLLM_TARGET_DEVICE=cpu
        log "WARNING: Falling back to CPU backend (VLLM_TARGET_DEVICE=cpu). Performance will be significantly reduced."
        return 0
    fi

    log "Set VLLM_PREFLIGHT_CPU_FALLBACK=1 to force CPU fallback instead of exiting."
    exit 64
}

main() {
    if should_run_preflight "$@"; then
        preflight_intel_gpu
    fi

    if [[ $# -eq 0 || "$1" == "serve" ]]; then
        if [[ "${1:-}" == "serve" ]]; then
            shift
        fi
        exec vllm serve "$@"
    fi

    if [[ "$1" == "vllm" && "${2:-}" == "serve" ]]; then
        shift 2
        exec vllm serve "$@"
    fi

    exec "$@"
}

main "$@"
