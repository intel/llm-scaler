#!/bin/bash
# ==============================================================================
# Lunar Lake vLLM Serving Script
# ------------------------------------------------------------------------------
# Launch vLLM on Intel Core Ultra (Lunar Lake) with Arc 140V iGPU.
# Configures memory-aware settings for shared LPDDR5x memory.
#
# Usage:
#   ./lunar_lake_serve.sh <model_path_or_name> [extra vllm args...]
#
# Examples:
#   ./lunar_lake_serve.sh Qwen/Qwen3-8B --quantization fp8
#   ./lunar_lake_serve.sh /models/DeepSeek-R1-Distill-Qwen-7B --quantization int4
#   ./lunar_lake_serve.sh Qwen/Qwen3.5-35B-A3B --quantization int4 --max-model-len 8192
# ==============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# === Source oneAPI ===
if [ -f /opt/intel/oneapi/setvars.sh ]; then
    set +euo pipefail
    source /opt/intel/oneapi/setvars.sh --force 2>/dev/null || true
    set -euo pipefail
fi

# === Fix MKL library path ===
# PyTorch bundles MKL stubs with relative RPATHs that break in venvs.
# Preload the real oneAPI MKL libraries to avoid "Cannot load libmkl_core.so" errors.
if [ -n "${MKLROOT:-}" ] && [ -f "$MKLROOT/lib/libmkl_core.so.2" ]; then
    export LD_PRELOAD="${MKLROOT}/lib/libmkl_core.so.2:${MKLROOT}/lib/libmkl_intel_thread.so.2:${MKLROOT}/lib/libmkl_intel_lp64.so.2${LD_PRELOAD:+:$LD_PRELOAD}"
    echo -e "${GREEN}[Lunar Lake vLLM]${NC} MKL preloaded from $MKLROOT"
fi

# === Validate args ===
if [ $# -lt 1 ]; then
    echo -e "${RED}Usage: $0 <model_path_or_name> [extra vllm args...]${NC}"
    echo ""
    echo "Recommended models for Lunar Lake (32GB shared memory):"
    echo "  Small  (fits easily):  Qwen/Qwen3-8B --quantization fp8"
    echo "  Medium (fits tight):   Qwen/Qwen3-14B --quantization int4"
    echo "  Large  (requires int4): Qwen/Qwen3.5-35B-A3B --quantization int4 --max-model-len 8192"
    echo ""
    echo "Notes:"
    echo "  - Always use --quantization (fp8 or int4) to fit in shared memory"
    echo "  - Use --max-model-len to limit context and reduce KV cache memory"
    echo "  - INT4 is recommended for models >14B on 32GB systems"
    exit 1
fi

MODEL="$1"
shift

# === Detect available memory ===
TOTAL_MEM_GB=$(free -g | awk '/^Mem:/{print $2}')
AVAIL_MEM_GB=$(free -g | awk '/^Mem:/{print $7}')

echo -e "${GREEN}[Lunar Lake vLLM]${NC} System memory: ${TOTAL_MEM_GB}GB total, ${AVAIL_MEM_GB}GB available"
echo -e "${GREEN}[Lunar Lake vLLM]${NC} Model: $MODEL"

# === Memory warnings ===
if [ "$AVAIL_MEM_GB" -lt 8 ]; then
    echo -e "${RED}WARNING: Only ${AVAIL_MEM_GB}GB available. Close other applications.${NC}"
    echo -e "${RED}Minimum 8GB free recommended for inference.${NC}"
elif [ "$AVAIL_MEM_GB" -lt 16 ]; then
    echo -e "${YELLOW}NOTE: ${AVAIL_MEM_GB}GB available. Use INT4 quantization and limit context length.${NC}"
fi

# === Environment for iGPU ===
export VLLM_TARGET_DEVICE=xpu
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export PYTORCH_ALLOC_CONF="expandable_segments:True"
# USM mode for shared memory (no P2P needed)
export CCL_TOPO_P2P_ACCESS=0
# Skip profile_run() during KV cache init — the dummy forward pass hangs
# indefinitely on Lunar Lake iGPU (Xe2/BMG). Instead, estimate peak memory
# from current allocation. Requires the corresponding xpu_worker.py patch.
export VLLM_SKIP_PROFILE_RUN=1

# === CCL single-GPU workaround ===
# oneCCL's KVS init tries to resolve a network interface even for single-GPU.
# On laptops/handhelds without wired Ethernet this can fail with
# "fill_local_host_ip: can't find non-loopback interface".
# These env vars force CCL to use a local TCP transport instead.
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=${MASTER_PORT:-29500}
export CCL_ZE_ENABLE=0
export CCL_ATL_TRANSPORT=ofi
export FI_PROVIDER=tcp
# Use WiFi interface if available, fallback to loopback
if ip link show wlo1 &>/dev/null; then
    export CCL_SOCKET_IFNAME=wlo1
elif ip link show wlan0 &>/dev/null; then
    export CCL_SOCKET_IFNAME=wlan0
else
    export CCL_SOCKET_IFNAME=lo
fi

echo -e "${GREEN}[Lunar Lake vLLM]${NC} Launching vLLM serve..."
echo "───────────────────────────────────────────────────────────────────────────────"

# === Launch vLLM ===
# Device is set via VLLM_TARGET_DEVICE=xpu (not a CLI flag)
# --tensor-parallel-size 1:  Single GPU (integrated)
# --gpu-memory-utilization:  Conservative for shared memory (leave room for OS + KV cache)
# --enforce-eager:           Disable CUDA graphs (not supported on XPU)
exec vllm serve "$MODEL" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.7 \
    --enforce-eager \
    "$@"
