#!/usr/bin/env bash
# Launch Gemma4-26B-A4B online FP8 on Intel BMG, TP=2, in eager mode.

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/llm/models/gemma-4-26B-A4B-it}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
FP8_DTYPE="${SGLANG_FP8_DTYPE:-e4m3}"
WORKLOAD_PROFILE="${GEMMA4_WORKLOAD_PROFILE:-serving}"
TP_SIZE="${TP_SIZE:-2}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.82}"
SWA_FULL_TOKENS_RATIO="${SWA_FULL_TOKENS_RATIO:-0.05}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"

ensure_oneccl_render_links() {
    if compgen -G "/dev/dri/by-path/*-render" >/dev/null; then
        return
    fi

    mkdir -p /dev/dri/by-path
    local render device_path bdf node
    local created=0
    for render in /sys/class/drm/renderD*; do
        [[ -e "${render}" ]] || continue
        device_path="$(readlink -f "${render}/device")"
        bdf="${device_path##*/}"
        node="${render##*/}"
        [[ -e "/dev/dri/${node}" ]] || continue
        ln -sf "../${node}" "/dev/dri/by-path/pci-${bdf}-render"
        created=1
    done

    if [[ "${created}" -ne 1 ]]; then
        echo "Cannot create oneCCL render links under /dev/dri/by-path" >&2
        return 1
    fi
}

ensure_oneccl_render_links

case "${WORKLOAD_PROFILE}" in
    serving|bfcl)
        default_radix_cache=1
        default_max_running_requests=1
        ;;
    canonical_bench)
        default_radix_cache=0
        default_max_running_requests=1
        ;;
    *)
        echo "GEMMA4_WORKLOAD_PROFILE must be serving, bfcl, or canonical_bench, got: ${WORKLOAD_PROFILE}" >&2
        exit 2
        ;;
esac

MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-${default_max_running_requests}}"
RADIX_CACHE="${RADIX_CACHE:-${default_radix_cache}}"

case "${FP8_DTYPE}" in
    e4m3|e5m2) ;;
    *)
        echo "SGLANG_FP8_DTYPE must be e4m3 or e5m2, got: ${FP8_DTYPE}" >&2
        exit 2
        ;;
esac

case "${RADIX_CACHE}" in
    1) radix_args=() ;;
    0) radix_args=(--disable-radix-cache) ;;
    *)
        echo "RADIX_CACHE must be 0 or 1, got: ${RADIX_CACHE}" >&2
        exit 2
        ;;
esac

export ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-0,1}"
export SGLANG_USE_SGL_XPU=1
export SGLANG_SKIP_VISION_GPU=1
export SGLANG_FP8_IGNORED_LAYERS=vision_tower,embed_vision
export SGLANG_SPLITK_G="${SGLANG_SPLITK_G:-64}"
export SGLANG_XPU_FP8_W8A16_PREFILL="${SGLANG_XPU_FP8_W8A16_PREFILL:-1}"
export SGLANG_FP8_DTYPE="${FP8_DTYPE}"
export CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD="${CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD:-4294967296}"
export CCL_SYCL_REDUCE_SCATTER_SIMPLE_THRESHOLD="${CCL_SYCL_REDUCE_SCATTER_SIMPLE_THRESHOLD:-4294967296}"
export CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD="${CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD:-4294967296}"
export CCL_SYCL_ALLTOALL_TMP_BUF="${CCL_SYCL_ALLTOALL_TMP_BUF:-1}"

if [[ "${TP_SIZE}" -lt 1 ]]; then
    echo "TP_SIZE must be at least 1, got: ${TP_SIZE}" >&2
    exit 2
fi

echo "Gemma4 profile=${WORKLOAD_PROFILE} fp8=${FP8_DTYPE} radix_cache=${RADIX_CACHE} max_running_requests=${MAX_RUNNING_REQUESTS}" >&2

exec python3 -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --device xpu \
    --tp "${TP_SIZE}" \
    --quantization fp8 \
    --dtype float16 \
    --load-format layered_fp8 \
    --attention-backend intel_xpu \
    --page-size 64 \
    --mem-fraction-static "${MEM_FRACTION_STATIC}" \
    --swa-full-tokens-ratio "${SWA_FULL_TOKENS_RATIO}" \
    --chunked-prefill-size 1024 \
    "${radix_args[@]}" \
    --max-running-requests "${MAX_RUNNING_REQUESTS}" \
    --context-length "${CONTEXT_LENGTH}" \
    --disable-cuda-graph \
    --skip-server-warmup \
    --watchdog-timeout 3600 \
    --trust-remote-code \
    --model-impl sglang \
    --tool-call-parser gemma4 \
    --reasoning-parser gemma4 \
    --host "${HOST}" \
    --port "${PORT}"
