#!/usr/bin/env bash
# Launch Gemma4-31B FP8 or GGUF on Intel BMG, TP=2, in eager mode.
#
# FP8 example:
#   MODEL_PATH=/models/gemma-4-31B-it ZE_AFFINITY_MASK=0,1 \
#   TP_SIZE=2 HOST=0.0.0.0 PORT=30000 \
#     bash scripts/start_gemma4_31b_service.sh
#
# GGUF example:
#   MODEL_PATH=/models/gemma-4-31B-it-GGUF/gemma-4-31B-it-Q4_K_M.gguf \
#   GGUF_CFG_DIR=/models/gemma-4-31B-it ZE_AFFINITY_MASK=0,1 TP_SIZE=2 \
#   HOST=0.0.0.0 PORT=30000 \
#     bash scripts/start_gemma4_31b_service.sh

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/models/gemma-4-31B-it}"
SPECULATIVE_DRAFT_MODEL_PATH="${SPECULATIVE_DRAFT_MODEL_PATH:-}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"
SWA_FULL_TOKENS_RATIO="${SWA_FULL_TOKENS_RATIO:-0.05}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-1}"
TP_SIZE="${TP_SIZE:-2}"
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
export SGLANG_XPU_ENABLE_GRAPH="${SGLANG_XPU_ENABLE_GRAPH:-0}"

format_args=()
if [[ ${IS_GGUF} == 1 ]]; then
    export SGLANG_GGUF_HF_CONFIG_DIR="${GGUF_CFG_DIR}"
    format_args=(
        --tokenizer-path "${GGUF_CFG_DIR}"
        --served-model-name "${SERVED_MODEL_NAME:-${GGUF_CFG_DIR}}"
    )
else
    format_args=(
        --quantization fp8
        --load-format layered_fp8
    )
fi

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
    "${format_args[@]}" \
    --device xpu \
    --tp "${TP_SIZE}" \
    --dtype float16 \
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
    --tool-call-parser gemma4 \
    --reasoning-parser gemma4 \
    "${speculative_args[@]}" \
    --host "${HOST}" \
    --port "${PORT}"
