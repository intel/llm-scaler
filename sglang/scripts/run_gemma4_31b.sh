#!/usr/bin/env bash
# Launch Gemma4-31B online FP8 on Intel BMG, TP=2, in eager mode.

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/models/gemma-4-31B-it}"
SPECULATIVE_DRAFT_MODEL_PATH="${SPECULATIVE_DRAFT_MODEL_PATH:-}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"
SWA_FULL_TOKENS_RATIO="${SWA_FULL_TOKENS_RATIO:-0.05}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-1}"

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

export ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-0,1}"
export SGLANG_USE_SGL_XPU=1
export SGLANG_SKIP_VISION_GPU=1
export SGLANG_FP8_IGNORED_LAYERS=vision_tower,embed_vision
export SGLANG_SPLITK_G="${SGLANG_SPLITK_G:-64}"
# NOTE: the gate sglang actually reads is SGLANG_XPU_FP8_W8A16_PREFILL
# (python/sglang/srt/layers/quantization/fp8_utils.py); it defaults to true.
# The old SGL_XPU_FP8_W8A16_PREFILL spelling is a no-op.
export SGLANG_XPU_FP8_W8A16_PREFILL="${SGLANG_XPU_FP8_W8A16_PREFILL:-1}"
# Decode runs eager: XPU graph is known-broken at TP>1 on this stack.
export SGL_XPU_ENABLE_GRAPH="${SGL_XPU_ENABLE_GRAPH:-0}"

speculative_args=()
if [[ -n "${SPECULATIVE_DRAFT_MODEL_PATH}" ]]; then
    speculative_args=(
        --speculative-algorithm NEXTN
        --speculative-draft-model-path "${SPECULATIVE_DRAFT_MODEL_PATH}"
        --speculative-draft-model-quantization unquant
        --speculative-num-steps 3
        --speculative-num-draft-tokens 4
        --speculative-eagle-topk 1
    )
fi

exec python3 -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --device xpu \
    --tp 2 \
    --quantization fp8 \
    --dtype float16 \
    --load-format layered_fp8 \
    --attention-backend intel_xpu \
    --page-size 64 \
    --mem-fraction-static "${MEM_FRACTION_STATIC}" \
    --swa-full-tokens-ratio "${SWA_FULL_TOKENS_RATIO}" \
    --chunked-prefill-size 1024 \
    --disable-radix-cache \
    --max-running-requests "${MAX_RUNNING_REQUESTS}" \
    --context-length 70000 \
    --disable-cuda-graph \
    --skip-server-warmup \
    --watchdog-timeout 3600 \
    --trust-remote-code \
    --model-impl sglang \
    "${speculative_args[@]}" \
    --host "${HOST}" \
    --port "${PORT}"
