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
    source /opt/intel/oneapi/setvars.sh --force 2>/dev/null
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

echo -e "${GREEN}[Lunar Lake vLLM]${NC} Launching vLLM serve..."
echo "───────────────────────────────────────────────────────────────────────────────"

# === Launch vLLM ===
# --device xpu:              Use Intel XPU (Xe2 iGPU)
# --tensor-parallel-size 1:  Single GPU (integrated)
# --gpu-memory-utilization:  Conservative for shared memory (leave room for OS + KV cache)
# --enforce-eager:           Disable CUDA graphs (not supported on XPU)
exec vllm serve "$MODEL" \
    --device xpu \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.7 \
    --enforce-eager \
    "$@"
