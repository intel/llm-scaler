#!/usr/bin/env bash
# Launch SGLang server for Qwen3.6-35B-A3B online fp8 on Intel BMG, TP=4.
#
# All ESIMD fast-paths enabled + XPU Graph capture.
# Required env knobs are documented inline.

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/models/Qwen3.6-35B-A3B}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
TP_SIZE="${TP_SIZE:-4}"

# --- ESIMD fast-path gates ---
# decode attn split-K (sglang_decode_attn): mandatory for online perf
export SGLANG_ENABLE_XPU_ESIMD_DECODE=1
# MoE silu routed kernel (replaces triton fused_moe on XPU)
export SGLANG_ENABLE_ESIMD_MOE=1
# Full-attention fused QKV split + RMSNorm + RoPE (Qwen3.5/3.6)
export SGL_XPU_FA_ESIMD_QKV=1
# GDN conv fused_seq for the linear-attention decode path
export SGL_XPU_GDN_ESIMD=1

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

# --- python path for vendored ESIMD packages ---
# Setup.py at /workspace/custom-esimd-kernels {-, -sglang}/setup.py
# installs both via package_dir; nothing extra to set normally.

exec python3 -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --tp "${TP_SIZE}" \
    --dtype float16 \
    --quantization fp8 \
    --attention-backend triton \
    --trust-remote-code \
    --host "${HOST}" \
    --port "${PORT}"
