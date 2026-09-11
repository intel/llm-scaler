#!/usr/bin/env bash
# Launch Qwen3.6-27B / 35B-A3B on Intel BMG, TP=2 by default.
#
# Run INSIDE the container with the oneAPI environment initialized:
#   cd /llm-scaler/sglang
#
# FP8 (HF directory; loaded and quantized online to E5M2):
#   MODEL_PATH=/models/Qwen3.6-27B ZE_AFFINITY_MASK=6,7 \
#     bash scripts/start_qwen3_6_service.sh
#   MODEL_PATH=/models/Qwen3.6-35B-A3B ZE_AFFINITY_MASK=6,7 \
#     bash scripts/start_qwen3_6_service.sh
#
# GGUF (detected automatically from the .gguf suffix):
#   MODEL_PATH=/models/Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_M.gguf \
#     GGUF_CFG_DIR=/models/Qwen3.6-27B ZE_AFFINITY_MASK=6,7 \
#     bash scripts/start_qwen3_6_service.sh
#   MODEL_PATH=/models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
#     GGUF_CFG_DIR=/models/Qwen3.6-35B-A3B ZE_AFFINITY_MASK=6,7 \
#     bash scripts/start_qwen3_6_service.sh
#
# MTP / speculative decoding (NEXTN), opt-in for either format by pointing
# SPEC_DRAFT_PATH at a checkpoint that carries the mtp.* tensors. For GGUF that
# is an MTP-enabled .gguf, which is also its own draft model:
#   MODEL_PATH=/models/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
#     GGUF_CFG_DIR=/models/Qwen3.6-35B-A3B \
#     SPEC_DRAFT_PATH="$MODEL_PATH" ZE_AFFINITY_MASK=6,7 \
#     bash scripts/start_qwen3_6_service.sh
#
# Run one model at a time on the same GPUs/port.
# GGUF_CFG_DIR must contain the matching HF config, tokenizer and safetensors
# index (or HF shards for weight-name mapping). Tested GGUF flavor: Q4_K_M.
# Optional: TP_SIZE=2 PORT=30000 HOST=127.0.0.1 MEM_FRACTION_STATIC=0.8.
# MEM_FRACTION_STATIC defaults lower with MTP on, because the draft model needs
# headroom the non-speculative defaults do not leave. Raising it back for an MTP
# run surfaces as UR_RESULT_ERROR_OUT_OF_RESOURCES in the draft extend, or as a
# misleading oneCCL "unknown memory type" during warmup.
# HOST defaults to 0.0.0.0; use 127.0.0.1 for local-only access.
# Optional MTP tuning: SPEC_NUM_STEPS=3 SPEC_TOPK=1 SPEC_NUM_DRAFT_TOKENS=4.
# The XPU GDN verify kernels support linear chains only (SPEC_TOPK=1).
# For GGUF the served model id defaults to GGUF_CFG_DIR so that OpenAI-style
# clients can use it as a tokenizer id; override with SERVED_MODEL_NAME.
# Long-context runs (e.g. BFCL multi-turn) also need
# MAMBA_TRACK_INTERVAL=8192 CHUNKED_PREFILL_SIZE=8192.

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/models/Qwen3.6-35B-A3B}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
TP_SIZE="${TP_SIZE:-2}"

# --- device selection ---
# Pin to the last two BMG cards (physical 0,1). After masking, sglang sees
# them as XPU 0,1 so TP=2 maps onto exactly these two devices.
export ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-0,1}"

# --- MTP / speculative decoding (NEXTN) ---
# Off unless SPEC_DRAFT_PATH is set. The draft model is the mtp.* branch of the
# checkpoint, so for GGUF it is the same file as MODEL_PATH. Verify runs the
# target model on num_draft_tokens rows at once, which is what the batched
# ESIMD M-tile GEMVs are tuned for. XPU graph capture cannot express the
# speculative control flow, so decode stays eager here too.
SPEC_ARGS=()
SPEC_ON=0
if [[ -n "${SPEC_DRAFT_PATH:-}" ]]; then
    SPEC_ON=1
    if [[ "${SPEC_TOPK:-1}" != 1 ]]; then
        echo "XPU GDN MTP verification requires SPEC_TOPK=1." >&2
        exit 2
    fi
    # Use the existing XPU kernels for verification and rollback snapshots,
    # just as decode/extend use their XPU kernels. Ordinary launches leave
    # this MTP-only switch unset; an explicit override is preserved.
    export SGLANG_XPU_MTP_GDN_VERIFY="${SGLANG_XPU_MTP_GDN_VERIFY:-${SGL_XPU_GDN_VERIFY_ESIMD:-1}}"
    # Forward the resolved value for SGLang revisions that read the old name.
    export SGL_XPU_GDN_VERIFY_ESIMD="$SGLANG_XPU_MTP_GDN_VERIFY"
    SPEC_ARGS=(
        --speculative-algorithm NEXTN
        --speculative-draft-model-path "$SPEC_DRAFT_PATH"
        --speculative-num-steps "${SPEC_NUM_STEPS:-3}"
        --speculative-eagle-topk "${SPEC_TOPK:-1}"
        --speculative-num-draft-tokens "${SPEC_NUM_DRAFT_TOKENS:-4}"
        --disable-cuda-graph
    )
fi

# Keep the GGUF path separate from FP8-only quantization and fusion settings.
if [[ "$MODEL_PATH" == *.gguf ]]; then
    if [[ ! -f "$MODEL_PATH" ]]; then
        echo "MODEL_PATH must point to an existing .gguf file." >&2
        exit 2
    fi
    if [[ -z "${GGUF_CFG_DIR:-}" || ! -f "$GGUF_CFG_DIR/config.json" ]]; then
        echo "GGUF_CFG_DIR must point to the matching HF model directory containing config.json." >&2
        exit 2
    fi
    export SGLANG_GGUF_HF_CONFIG_DIR="$GGUF_CFG_DIR"
    # The draft weights, KV cache and per-token GDN snapshots need headroom
    # beyond the target's static allocation. Preserve user memory settings.
    if [[ -z "${MEM_FRACTION_STATIC:-}" ]]; then
        if [[ $SPEC_ON == 1 ]]; then
            MEM_FRACTION_STATIC=0.65
        else
            MEM_FRACTION_STATIC=0.8
        fi
    fi
    export SGLANG_MAMBA_CONV_DTYPE=float16
    export SGLANG_MAMBA_SSM_DTYPE=float16
    export SGL_XPU_ESIMD_DECODE=1
    export SGL_XPU_FA_ESIMD_QKV=1
    export SGL_XPU_GDN_ESIMD=1
    export SGL_XPU_GDN_EXTEND_ESIMD=1
    export SGL_XPU_PREFILL_DPAS=1
    export SGL_XPU_ENABLE_GRAPH=0
    export SGLANG_XPU_ENABLE_GRAPH=0
    # GGUF-native decode fusions. These are the GGUF counterparts of the fp8
    # MoE-full / resadd-norm fusions below; the fp8 ones never fire here because
    # the attention and GDN projections are q8_0 GEMVs.
    # Full GGUF MoE fusion: router topk + routed Q4_K/Q5_K experts + Q8_0 shared
    # expert in one dispatch. Defaults OFF in the loader, and it is the main
    # decode TPOT lever for GGUF, so turn it on here.
    export SGL_XPU_GGUF_MOE_FULL="${SGL_XPU_GGUF_MOE_FULL:-1}"
    # Q8_0 shared-expert kernel, used on the steps the full fusion declines.
    export SGL_XPU_GGUF_MOE_SHARED="${SGL_XPU_GGUF_MOE_SHARED:-1}"
    # Folds GemmaRMSNorm(input_layernorm) + the q8_0 in_proj/qkv GEMV + the fp16
    # in_proj_ba GEMV into one op. Set to 0 for the unfused fallback.
    export SGL_XPU_GGUF_RESADD_NORM="${SGL_XPU_GGUF_RESADD_NORM:-1}"
    # GGUF k-quant-only fusions. They default to off in SGLang so FP8 launches
    # do not probe GGUF layouts; enable them explicitly for the GGUF path.
    export SGL_XPU_GGUF_RESADD_NORM_KQ="${SGL_XPU_GGUF_RESADD_NORM_KQ:-1}"
    export SGL_XPU_GGUF_MLP_SILU="${SGL_XPU_GGUF_MLP_SILU:-1}"
    export SGL_XPU_GGUF_NORM_OUT_Q5K="${SGL_XPU_GGUF_NORM_OUT_Q5K:-1}"
    # Largest decode batch the fusions handle. Must cover the MTP verify batch
    # (concurrency x num_draft_tokens), so leave it at the default 64.
    export SGL_XPU_GGUF_FUSE_MAX_M="${SGL_XPU_GGUF_FUSE_MAX_M:-64}"

    # Report the HF config dir as the model id instead of the .gguf path.
    # OpenAI-style clients reuse the id from /v1/models as a tokenizer id, and
    # AutoTokenizer cannot load a .gguf ("not a valid JSON file"). sglang's own
    # bench_serving hits this unless it is passed an explicit --tokenizer.
    exec python3 -m sglang.launch_server \
        --model-path "$MODEL_PATH" \
        --tokenizer-path "$GGUF_CFG_DIR" \
        --served-model-name "${SERVED_MODEL_NAME:-$GGUF_CFG_DIR}" \
        --device xpu \
        --tp "$TP_SIZE" \
        --dtype float16 \
        --attention-backend intel_xpu \
        --trust-remote-code \
        --mem-fraction-static "${MEM_FRACTION_STATIC:-0.8}" \
        --page-size 64 \
        --max-mamba-cache-size 64 \
        --disable-overlap-schedule \
        --mamba-scheduler-strategy extra_buffer \
        --mamba-track-interval "${MAMBA_TRACK_INTERVAL:-64}" \
        --chunked-prefill-size "${CHUNKED_PREFILL_SIZE:-8192}" \
        --tool-call-parser qwen3_coder \
        --reasoning-parser qwen3 \
        --enable-cache-report \
        --enable-metrics \
        --skip-server-warmup \
        --host "$HOST" \
        --port "$PORT" \
        ${SPEC_ARGS[@]+"${SPEC_ARGS[@]}"}
fi

if [[ -z "${MEM_FRACTION_STATIC:-}" ]]; then
    if [[ $SPEC_ON == 1 ]]; then
        MEM_FRACTION_STATIC=0.75
    else
        MEM_FRACTION_STATIC=0.9
    fi
fi

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
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --enable-cache-report \
    --enable-metrics \
    --host "${HOST}" \
    --port "${PORT}" \
    ${SPEC_ARGS[@]+"${SPEC_ARGS[@]}"}
