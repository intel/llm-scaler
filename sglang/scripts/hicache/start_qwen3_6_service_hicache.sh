#!/usr/bin/env bash
# Launch Qwen3.6-35B-A3B (hybrid GDN+attention) on Intel BMG with HiCache, TP=2.
# FP8 online quantization only (no GGUF, no MTP).
#
# Supported models:
#   - Qwen3.6-35B-A3B (hybrid GDN + full-attention, uses Mamba cache)
#   - Dense models work too but Mamba flags are ignored
#
# Usage:
#   MODEL_PATH=/models/Qwen3.6-35B-A3B ZE_AFFINITY_MASK=0,1 \
#     bash start_qwen3_6_service_hicache.sh
#
# HiCache tuning:
#   HICACHE_RATIO=1.0        # L2 pool as multiple of device pool (default 1.0)
#   HICACHE_WRITE_POLICY=write_back  # write_back (default) or write_through
#   HICACHE_L3_DIR=/tmp/hicache      # Set to enable L3 disk tier (default: disabled)
#   MAX_TOTAL_TOKENS=262144  # Pin token capacity (required for ratio sizing)
#
# Optional: TP_SIZE=2 PORT=30000 HOST=127.0.0.1 MEM_FRACTION_STATIC=0.9

set -euo pipefail

# Bypass proxy for loopback traffic (warmup self-requests)
export no_proxy="127.0.0.1,localhost,0.0.0.0"
export NO_PROXY="127.0.0.1,localhost,0.0.0.0"

MODEL_PATH="${MODEL_PATH:-/models/Qwen3.6-35B-A3B}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
TP_SIZE="${TP_SIZE:-2}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.9}"

# --- HiCache settings ---
HICACHE_RATIO="${HICACHE_RATIO:-1.0}"
# write_back: only write to L2 on eviction (no overhead when device pool isn't full)
# write_through: write every decode step (constant overhead, faster eviction)
HICACHE_WRITE_POLICY="${HICACHE_WRITE_POLICY:-write_through}"
# L3 disk backend: set HICACHE_L3_DIR to enable, leave empty/unset to disable L3
export HICACHE_L3_DIR="${HICACHE_L3_DIR-}"
# max_total_tokens: if unset, auto-sizes to fill device memory (like baseline).
# Set explicitly only if you need to control L2 pool size (L2 = device_pool * hicache_ratio).
# WARNING: Setting too low causes performance regression due to smaller device pool.
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-}"

# --- device selection ---
export ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-0,1}"

# --- triton-xpu fp16 mismatch workaround ---
export SGLANG_MAMBA_CONV_DTYPE=float16
export SGLANG_MAMBA_SSM_DTYPE=float16

# --- ESIMD fast-path gates ---
export SGL_XPU_ESIMD_DECODE=1
export SGL_XPU_ESIMD_MOE=1
export SGL_XPU_ESIMD_MOE_FULL=1
export SGL_XPU_ESIMD_MOE_PREFILL=1
export SGL_XPU_FA_ESIMD_QKV=1
export SGL_XPU_GDN_ESIMD=1
export SGL_XPU_GDN_EXTEND_ESIMD=1
export SGL_XPU_PREFILL_DPAS=1

# --- XPU Graph disabled for accuracy ---
export SGL_XPU_ENABLE_GRAPH=0

# --- e5m2 online-quant + fused decode kernels ---
export SGLANG_FP8_DTYPE=e5m2
export SGL_XPU_GDN_NORM_GEMV=1
export SGL_XPU_GDN_INPROJ_FUSED2=0
export SGL_XPU_GDN_RESADD_NORM=1
export SGL_XPU_FA_RESADD_NORM=1
export SGL_XPU_MOE_ROUTER_FP8=1
export SGLANG_XPU_TP_SYNC_TOKENS=0

export TORCHDYNAMO_DISABLE=1

# Build optional args
OPTIONAL_ARGS=()
if [[ -n "${MAX_TOTAL_TOKENS:-}" ]]; then
    OPTIONAL_ARGS+=(--max-total-tokens "${MAX_TOTAL_TOKENS}")
fi

# L3 disk storage: enable file backend if HICACHE_L3_DIR is set
if [[ -n "${HICACHE_L3_DIR:-}" ]]; then
    export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR="${HICACHE_L3_DIR}"
    OPTIONAL_ARGS+=(--hicache-storage-backend file)
    echo "L3 storage enabled: ${HICACHE_L3_DIR}"
fi

exec python3 -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --tp "${TP_SIZE}" \
    --dtype float16 \
    --quantization fp8 \
    --load-format layered_fp8 \
    --attention-backend intel_xpu \
    --trust-remote-code \
    --mem-fraction-static "${MEM_FRACTION_STATIC}" \
    --max-mamba-cache-size 64 \
    --page-size 64 \
    --mamba-scheduler-strategy extra_buffer \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --enable-cache-report \
    --enable-metrics \
    --enable-hierarchical-cache \
    --hicache-ratio "${HICACHE_RATIO}" \
    --hicache-write-policy "${HICACHE_WRITE_POLICY}" \
    --hicache-io-backend kernel \
    --host "${HOST}" \
    --port "${PORT}" \
    ${OPTIONAL_ARGS[@]+"${OPTIONAL_ARGS[@]}"}
