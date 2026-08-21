#!/usr/bin/env bash
# Launch Gemma4-26B-A4B online FP8 on Intel BMG, TP=2, in eager mode.

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/models/gemma-4-26B-A4B-it}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
FP8_DTYPE="${SGLANG_FP8_DTYPE:-e4m3}"
TP_SIZE="${TP_SIZE:-2}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.82}"
SWA_FULL_TOKENS_RATIO="${SWA_FULL_TOKENS_RATIO:-0.05}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-1}"

case "${FP8_DTYPE}" in
    e4m3|e5m2) ;;
    *)
        echo "SGLANG_FP8_DTYPE must be e4m3 or e5m2, got: ${FP8_DTYPE}" >&2
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

if [[ "${TP_SIZE}" -lt 1 ]]; then
    echo "TP_SIZE must be at least 1, got: ${TP_SIZE}" >&2
    exit 2
fi

echo "Launching Gemma4-26B-A4B TP=${TP_SIZE} fp8=${FP8_DTYPE}" >&2

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
