#!/usr/bin/env bash
# Launch SGLang server for Qwen3.6-35B-A3B online fp8 on Intel BMG, TP=2.
#
# Golden fp8 + full-ESIMD + XPU-graph config (matches the sgl-fp8-perf setup).
# All ESIMD fast-paths + prefill fast-paths enabled. Required env knobs are
# documented inline.

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/models/Qwen3.6-35B-A3B}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
TP_SIZE="${TP_SIZE:-2}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.9}"

# --- device selection ---
# Pin to the last two BMG cards (physical 0,1). After masking, sglang sees
# them as XPU 0,1 so TP=2 maps onto exactly these two devices.
export ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-0,1}"

# --- triton-xpu fp16 mismatch workaround ---
# Mamba state pool defaults to bf16; force fp16 so it matches the activation
# dtype when running --dtype float16 (otherwise causal_conv1d_update kernel
# fails with "Mismatched type for col0 (bf16 vs fp16)").
export SGLANG_MAMBA_CONV_DTYPE=float16
export SGLANG_MAMBA_SSM_DTYPE=float16

# --- ESIMD fast-path gates ---
# decode attn split-K (sglang_decode_attn): mandatory for online perf
export SGLANG_ENABLE_XPU_ESIMD_DECODE=1
# MoE silu routed kernel (replaces triton fused_moe on XPU)
export SGLANG_ENABLE_ESIMD_MOE=1
# MoE prefill ESIMD (M-tiled DPAS fp8 MoE prefill)
export SGLANG_ENABLE_ESIMD_MOE_PREFILL=1
# Full-attention fused QKV split + RMSNorm + RoPE (Qwen3.5/3.6)
export SGL_XPU_FA_ESIMD_QKV=1
# GDN conv fused_seq for the linear-attention decode path
export SGL_XPU_GDN_ESIMD=1
# GDN chunk_gated_delta_rule prefill (extend) — ESIMD M-tiled kernel.
# This is the prefill TTFT lever: triton GDN recurrence is the prefill
# bottleneck. The kernel was extended to accept fp16 ssm-state to match
# this fp16 model's mamba pool.
export SGL_XPU_GDN_EXTEND_ESIMD=1
# Prefill SDPA via DPAS/XMX (AOT-compiled, doubleGRF)
export SGL_XPU_PREFILL_DPAS=1

# --- XPU Graph (CUDA-graph-equivalent) ---
# Captures the decode forward graph for a TPOT speedup at BS=1.
# Safe to leave on with the kernels in this image; falls back to eager
# replay on sequences > 16384 tokens (kernel MAX_SPLITS cap).
export SGLANG_XPU_ENABLE_GRAPH=1

# --load-format layered_fp8: build on CPU, load the full bf16 checkpoint into
# host RAM, then move + quantize each module onto the device one at a time.
# Peak device memory is fp8 weights + one module's bf16, so a TP=2 split
# (only two cards) fits where the default loader would OOM on the full bf16.
# --mamba-scheduler-strategy extra_buffer + --page-size 64: hybrid GDN
# scheduler tuning that keeps the radix prefix-cache stable on this model
# (so radix cache is left ENABLED for prefill reuse).
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
    --reasoning-parser qwen3 \
    --enable-cache-report \
    --enable-metrics \
    --host "${HOST}" \
    --port "${PORT}"
