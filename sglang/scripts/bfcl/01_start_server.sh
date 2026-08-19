#!/usr/bin/env bash
# ============================================================================
# REFERENCE server config — ONE example, not the harness contract.
# ============================================================================
# This is the bundled reference: Qwen3.6-35B-A3B-Q4_K_M GGUF on sglang + Intel
# XPU (PTL Xe3), fp16, radix prefix-cache ENABLED. It needs the GDN track-buffer
# fix (sglang qwen3_5 + sgl-kernel-xpu gdn_attn inter_ssm/inter_conv; see README).
#
# The BFCL harness (steps 02-05) is model/format/backend-AGNOSTIC — it only needs
# an OpenAI-compatible /v1 endpoint. To eval a different model/format/backend,
# REPLACE this script (or use SKIP_START=1 in 04_run.sh with your own server) and
# set MODEL_ID to the matching BFCL entry. Everything GGUF/mamba/GDN/XPU-specific
# below is particular to THIS target, not required by the harness.
#
# Runs INSIDE the GPU container. The server stays foreground here; background it
# yourself (this kit's 04_run.sh launches it detached). All values are env-var
# overridable so this ports to a differently-laid-out box.
#
# radix-ON recipe (notes #136): DROP --disable-radix-cache, ADD
# --mamba-scheduler-strategy extra_buffer (no_buffer needs page_size=1 which the
# intel_xpu XE FMHA decode kernel rejects; extra_buffer snapshots intermediate
# state every track_interval -> page-granular 64 reuse). The track-buffer fix
# makes that snapshot actually correct on XPU (else multi-turn restores 0 state).
set -e

GGUF="${GGUF:-/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}"
GGUF_CFG_DIR="${GGUF_CFG_DIR:-/models/Qwen3.5-35B-A3B-cfg}"   # HF config dir for the GGUF arch
PORT="${PORT:-9010}"
ZE_MASK="${ZE_MASK:-0}"                                       # which XPU tile
TRACK_INTERVAL="${TRACK_INTERVAL:-64}"                        # MUST divide page_size cleanly
PAGE_SIZE="${PAGE_SIZE:-64}"
MEM_FRAC="${MEM_FRAC:-0.65}"
ONEAPI_SETVARS="${ONEAPI_SETVARS:-/opt/intel/oneapi/setvars.sh}"

[ -f "$ONEAPI_SETVARS" ] && source "$ONEAPI_SETVARS" --force >/dev/null 2>&1

export ZE_AFFINITY_MASK="$ZE_MASK"
export ONEAPI_DEVICE_SELECTOR="level_zero:${ZE_MASK}"
# XPU GDN/FMHA ESIMD switches (canonical PTL fp16 config)
export SGL_XPU_FA_ESIMD_QKV=1       SGLANG_XPU_FA_ESIMD_QKV=1
export SGL_XPU_ESIMD_DECODE=1       SGLANG_XPU_ESIMD_DECODE=1
export SGL_XPU_GDN_EXTEND_ESIMD=1   SGLANG_XPU_GDN_EXTEND_ESIMD=1
export SGLANG_XPU_ENABLE_GRAPH=1
export SGLANG_XPU_GDN_FAST_PATH=1
export SGL_XPU_FA_FALLBACK=1
export SGLANG_GGUF_HF_CONFIG_DIR="$GGUF_CFG_DIR"
export SGLANG_MAMBA_CONV_DTYPE=float16

echo "[start] GGUF=$GGUF port=$PORT ZE_MASK=$ZE_MASK track_interval=$TRACK_INTERVAL"
env | grep -E 'SGLANG_XPU|SGL_XPU|SGLANG_GGUF|ZE_AFFINITY|ONEAPI_DEVICE' | sort

exec python3 -m sglang.launch_server \
    --model-path "$GGUF" \
    --trust-remote-code \
    --device xpu \
    --attention-backend intel_xpu \
    --dtype float16 \
    --page-size "$PAGE_SIZE" \
    --mem-fraction-static "$MEM_FRAC" \
    --disable-overlap-schedule \
    --mamba-scheduler-strategy extra_buffer \
    --mamba-track-interval "$TRACK_INTERVAL" \
    --enable-cache-report \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --skip-server-warmup \
    --watchdog-timeout 3600 \
    --host 0.0.0.0 --port "$PORT" \
    --cuda-graph-bs 1 --cuda-graph-max-bs 1
