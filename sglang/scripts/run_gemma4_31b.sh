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

export ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-0,1}"
export SGLANG_USE_SGL_XPU=1
export SGLANG_SKIP_VISION_GPU=1
export SGLANG_FP8_IGNORED_LAYERS=vision_tower,embed_vision
export SGLANG_SPLITK_G="${SGLANG_SPLITK_G:-64}"

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
