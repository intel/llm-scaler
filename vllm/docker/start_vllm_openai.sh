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


extract_sycl_xpu_count() {
    local sycl_output="${1:-}"
    local count=0

    count=$(echo "$sycl_output" | grep -Eio "(level_zero|ext_oneapi_level_zero).*(gpu|xpu)" | wc -l || true)
    if [[ "$count" -eq 0 ]]; then
        count=$(echo "$sycl_output" | grep -Eio "opencl.*(gpu|xpu)" | wc -l || true)
    fi

    echo "$count"
}

extract_int_arg() {
    local key="$1"
    shift
    local args=("$@")
    local i=0

    while [[ $i -lt ${#args[@]} ]]; do
        local token="${args[$i]}"
        if [[ "$token" == "$key" ]]; then
            if [[ $((i + 1)) -lt ${#args[@]} ]]; then
                echo "${args[$((i + 1))]}"
                return 0
            fi
        elif [[ "$token" =~ ^${key}=(.+)$ ]]; then
            echo "${BASH_REMATCH[1]}"
            return 0
        fi
        ((i += 1))
    done

    echo ""
}

validate_parallelism_settings() {
    local sycl_xpu_count="$1"
    shift

    local dp_raw tp_raw dp tp total_required
    dp_raw="$(extract_int_arg "--dp" "$@")"
    tp_raw="$(extract_int_arg "--tensor-parallel-size" "$@")"

    if [[ -z "$tp_raw" ]]; then
        tp_raw="$(extract_int_arg "-tp" "$@")"
    fi

    if [[ "$dp_raw" =~ ^[0-9]+$ ]]; then
        dp="$dp_raw"
    else
        dp=1
    fi

    if [[ "$tp_raw" =~ ^[0-9]+$ ]]; then
        tp="$tp_raw"
    else
        tp=1
    fi

    total_required=$((dp * tp))
    if [[ "$sycl_xpu_count" -gt 0 && "$total_required" -gt "$sycl_xpu_count" ]]; then
        log "ERROR: Requested parallelism requires $total_required XPU devices (dp=$dp, tp=$tp), but sycl-ls reports $sycl_xpu_count."
        log "Adjust --dp/--tensor-parallel-size (or -tp), or fix device filtering env vars like ZE_AFFINITY_MASK/ONEAPI_DEVICE_SELECTOR."
        exit 65
    fi
}
preflight_intel_gpu() {
    local sycl_xpu_count=0
    local has_render_nodes=0
    local has_sycl_gpu=0
    local sycl_output=""

    if compgen -G "/dev/dri/renderD*" > /dev/null; then
        has_render_nodes=1
    fi

    if command -v sycl-ls > /dev/null 2>&1; then
        sycl_output="$(sycl-ls 2>&1 || true)"
        sycl_xpu_count="$(extract_sycl_xpu_count "$sycl_output")"
        if [[ "$sycl_xpu_count" -gt 0 ]]; then
            has_sycl_gpu=1
        fi
    fi

    if [[ $has_render_nodes -eq 1 && $has_sycl_gpu -eq 0 ]]; then
        log "Found /dev/dri render nodes, but sycl-ls did not report a usable Intel GPU/XPU."
        log "sycl-ls output:"
        echo "$sycl_output"
    fi

    if [[ $has_sycl_gpu -eq 1 ]]; then
        validate_parallelism_settings "$sycl_xpu_count" "$@"
        log "Intel GPU preflight passed (detected $sycl_xpu_count XPU device(s) via sycl-ls)."
        return 0
    fi

    if [[ $has_render_nodes -eq 1 ]]; then
        log "ERROR: Render nodes are visible, but oneAPI cannot detect a usable Intel GPU/XPU device."
    else
        log "ERROR: No Intel GPU/XPU detected before starting vLLM OpenAI server."
    fi
    if [[ -f "/.dockerenv" ]]; then
        log "Container runtime hint: map GPU devices and required flags."
        log "Example docker run flags:"
        log "  --device /dev/dri --group-add render --group-add video"
        log "Example docker compose stanza:"
        log "  devices: [\"/dev/dri:/dev/dri\"]"
    fi
    log "Install oneAPI runtime tools (for sycl-ls) and verify host GPU drivers."
    log "If running non-interactively, make sure oneAPI env is sourced: source /opt/intel/oneapi/setvars.sh --force"

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
        preflight_intel_gpu "$@"
    fi

    if [[ $# -eq 0 || "$1" == "serve" ]]; then
        if [[ "${1:-}" == "serve" ]]; then
            shift
        fi
        # Use sockets (not pidfd) for Level Zero IPC — pidfd requires the
        # pidfd_getfd syscall which is blocked by container seccomp on many
        # hosts and causes "No device of requested type" SYCL crashes.
        export CCL_ZE_IPC_EXCHANGE="${CCL_ZE_IPC_EXCHANGE:-sockets}"
        exec vllm serve "$@"
    fi

    if [[ "$1" == "vllm" && "${2:-}" == "serve" ]]; then
        shift 2
        exec vllm serve "$@"
    fi

    exec "$@"
}

main "$@"
