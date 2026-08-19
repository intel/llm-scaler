#!/usr/bin/env bash
# Launch SGLang server for Qwen3.6-35B-A3B online fp8 on Intel BMG, TP=2.
#
# e5m2 online-fp8 + full-ESIMD config, XPU-graph DISABLED (accuracy).
# All ESIMD fast-paths + prefill fast-paths + e5m2 fused decode kernels enabled.
# Required env knobs are documented inline.

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
# All ESIMD/XPU fast-path gates use the SGL_XPU_* prefix.
# decode attn split-K (sglang_decode_attn): mandatory for online perf
export SGL_XPU_ESIMD_DECODE=1
# MoE silu routed kernel (replaces triton fused_moe on XPU)
export SGL_XPU_ESIMD_MOE=1
# Full decode MoE fusion: router topk + routed + shared + gate -> 1 dispatch.
# e5m2 only (SGLANG_FP8_DTYPE=e5m2 below). Reads native N-major w13 (no
# transposed weight copy). This is the main decode TPOT lever for this model.
export SGL_XPU_ESIMD_MOE_FULL=1
# MoE prefill ESIMD (M-tiled DPAS fp8 MoE prefill)
export SGL_XPU_ESIMD_MOE_PREFILL=1
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
# DISABLED: xpu-graph accuracy is unstable on this model, so decode runs eager.
# The e5m2 MoE-full fusion + resadd-norm fusions below recover the per-step
# host-dispatch cost that the graph would otherwise have hidden.
export SGL_XPU_ENABLE_GRAPH=0

# --- e5m2 online-quant + fused decode kernels ---
# Quantize online fp8 to e5m2 (the fused MoE-full decode kernels require e5m2).
export SGLANG_FP8_DTYPE=e5m2
# GDN gated-RMSNorm as an ESIMD GEMV (decode).
export SGL_XPU_GDN_NORM_GEMV=1
# Superseded by GDN_RESADD_NORM (which fuses in_proj qkvz+ba WITH input_layernorm);
# the standalone in_proj fused2 gave no e2e gain -> keep OFF.
export SGL_XPU_GDN_INPROJ_FUSED2=0
# Fuse input_layernorm (resadd+rmsnorm) + GDN in_proj (qkvz+ba) into one ESIMD GEMV.
export SGL_XPU_GDN_RESADD_NORM=1
# Fuse full-attention input_layernorm (resadd+rmsnorm) into qkv_proj (decode).
export SGL_XPU_FA_RESADD_NORM=1
# MoE router as fp8 ESIMD GEMV instead of fp16 aten::mm (saves a host launch per
# MoE layer). Perturbs top-8 routing on a fraction of tokens -> gsm8k A/B before
# trusting; set to 0 to fall back to the accurate fp16 gate.
export SGL_XPU_MOE_ROUTER_FP8=1
# Skip the per-step TP token-count sync (host overhead) on this single-node TP setup.
export SGLANG_XPU_TP_SYNC_TOKENS=0

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
