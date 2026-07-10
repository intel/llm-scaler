#!/usr/bin/env bash
# Launch SGLang server for Qwen3.6-35B-A3B online fp8 on Intel BMG, TP=2.
#
# All ESIMD fast-paths enabled + XPU Graph capture.
# Required env knobs are documented inline.

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/models/Qwen3.6-35B-A3B}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
TP_SIZE="${TP_SIZE:-2}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.9}"

# --- device selection ---
# Pin to the last two BMG cards (physical 2,3). After masking, sglang sees
# them as XPU 0,1 so TP=2 maps onto exactly these two devices.
export ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-2,3}"

# --- ESIMD fast-path gates ---
# decode attn split-K (sglang_decode_attn): mandatory for online perf
export SGLANG_ENABLE_XPU_ESIMD_DECODE=1
# MoE silu routed kernel (replaces triton fused_moe on XPU)
export SGLANG_ENABLE_ESIMD_MOE=1
# Full-attention fused QKV split + RMSNorm + RoPE (Qwen3.5/3.6)
export SGL_XPU_FA_ESIMD_QKV=1
# GDN conv fused_seq for the linear-attention decode path
export SGL_XPU_GDN_ESIMD=1
# GDN chunk_gated_delta_rule prefill (extend) — ESIMD M-tiled kernel.
# This is the prefill TTFT lever: triton GDN recurrence is the prefill
# bottleneck (~6.7x TTFT speedup, 13s->2s at 2k tokens). The kernel was
# extended to accept fp16 ssm-state to match this fp16 model's mamba pool.
export SGL_XPU_GDN_EXTEND_ESIMD=1

# --- triton-xpu fp16 mismatch workaround ---
# Mamba state pool defaults to bf16; force fp16 so it matches the activation
# dtype when running --dtype float16 (otherwise causal_conv1d_update kernel
# fails with "Mismatched type for col0 (bf16 vs fp16)").
export SGLANG_MAMBA_CONV_DTYPE=float16
export SGLANG_MAMBA_SSM_DTYPE=float16

# --- XPU Graph (CUDA-graph-equivalent) ---
# Captures the decode forward graph for ~3x TPOT speedup at BS=1.
# Safe to leave on with the kernels in this image; falls back to eager
# replay on sequences > 16384 tokens (kernel MAX_SPLITS cap).
export SGLANG_XPU_ENABLE_GRAPH=1

# --- python path for vendored ESIMD package ---
# Setup.py at /workspace/custom-esimd-kernels/setup.py installs the single
# merged custom_esimd_kernels_sglang package; nothing extra to set normally.

# --disable-radix-cache: the MambaRadixCache prefix-cache path can deadlock
# the TP schedulers on this hybrid GDN model (cache_prefix / zero-token
# prefill insert spins the cross-rank sync). Disabling it keeps the server
# stable for multi-request workloads; per-request decode perf is unaffected.
# --load-format layered_fp8: build on CPU, load the full bf16 checkpoint into
# host RAM, then move + quantize each module onto the device one at a time.
# Peak device memory is fp8 weights + one module's bf16, so a TP=2 split
# (only two cards) fits where the default loader would OOM on the full bf16.
exec python3 -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --tp "${TP_SIZE}" \
    --dtype float16 \
    --quantization fp8 \
    --load-format layered_fp8 \
    --attention-backend triton \
    --trust-remote-code \
    --disable-radix-cache \
    --mem-fraction-static "${MEM_FRACTION_STATIC}" \
    --host "${HOST}" \
    --port "${PORT}"
