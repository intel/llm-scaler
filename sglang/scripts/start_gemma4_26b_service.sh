#!/usr/bin/env bash
# Launch Gemma4-26B-A4B FP8 or GGUF on Intel BMG, TP=2, in eager mode.
#
# FP8 example:
#   MODEL_PATH=/models/gemma-4-26B-A4B-it ZE_AFFINITY_MASK=0,1 \
#   TP_SIZE=2 HOST=0.0.0.0 PORT=30000 \
#     bash scripts/start_gemma4_26b_service.sh
#
# GGUF example:
#   MODEL_PATH=/models/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf \
#   GGUF_CFG_DIR=/models/gemma-4-26B-A4B-it ZE_AFFINITY_MASK=0,1 TP_SIZE=2 \
#   HOST=0.0.0.0 PORT=30000 \
#     bash scripts/start_gemma4_26b_service.sh

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
IS_GGUF=0

if [[ "${MODEL_PATH}" == *.gguf ]]; then
    IS_GGUF=1
    if [[ ! -f "${MODEL_PATH}" ]]; then
        echo "MODEL_PATH must point to an existing .gguf file." >&2
        exit 2
    fi
    if [[ -z "${GGUF_CFG_DIR:-}" || ! -f "${GGUF_CFG_DIR}/config.json" ]]; then
        echo "GGUF_CFG_DIR must point to the matching HF model directory containing config.json." >&2
        exit 2
    fi
fi

if [[ ${IS_GGUF} == 0 ]]; then
    case "${FP8_DTYPE}" in
        e4m3|e5m2) ;;
        *)
            echo "SGLANG_FP8_DTYPE must be e4m3 or e5m2, got: ${FP8_DTYPE}" >&2
            exit 2
            ;;
    esac
fi

export ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-0,1}"
export SGLANG_USE_SGL_XPU=1
export SGLANG_SKIP_VISION_GPU=1
export SGLANG_FP8_IGNORED_LAYERS=vision_tower,embed_vision
export SGLANG_SPLITK_G="${SGLANG_SPLITK_G:-64}"
export SGLANG_XPU_FP8_W8A16_PREFILL="${SGLANG_XPU_FP8_W8A16_PREFILL:-1}"

format_args=()
if [[ ${IS_GGUF} == 1 ]]; then
    export SGLANG_GGUF_HF_CONFIG_DIR="${GGUF_CFG_DIR}"
    format_args=(
        --tokenizer-path "${GGUF_CFG_DIR}"
        --served-model-name "${SERVED_MODEL_NAME:-${GGUF_CFG_DIR}}"
    )
else
    export SGLANG_FP8_DTYPE="${FP8_DTYPE}"
    format_args=(
        --quantization fp8
        --load-format layered_fp8
    )
fi

if [[ "${TP_SIZE}" -lt 1 ]]; then
    echo "TP_SIZE must be at least 1, got: ${TP_SIZE}" >&2
    exit 2
fi

if [[ ${IS_GGUF} == 1 ]]; then
    echo "Launching Gemma4-26B-A4B TP=${TP_SIZE} gguf=${MODEL_PATH}" >&2
else
    echo "Launching Gemma4-26B-A4B TP=${TP_SIZE} fp8=${FP8_DTYPE}" >&2
fi

exec python3 -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    "${format_args[@]}" \
    --device xpu \
    --tp "${TP_SIZE}" \
    --dtype float16 \
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
