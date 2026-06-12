#!/bin/bash
# ==============================================================================
# vLLM Benchmark Recipe
# ------------------------------------------------------------------------------
# Standard benchmark suite for Lunar Lake iGPU (Arc 140V / Xe2).
# Tests single-user latency and batched throughput at multiple context lengths.
#
# Usage:
#   ./benchmark.sh <model_path> <port>
#
# Examples:
#   ./benchmark.sh /shared/models/qwen3.5-4b-int4-autoround 8082
#   ./benchmark.sh /shared/models/qwen3.5-9b-claude-4.6-opus-reasoning-distilled 8090
#   ./benchmark.sh openai/gpt-oss-20b 8000
# ==============================================================================

set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

if [ $# -lt 2 ]; then
    echo "Usage: $0 <model_path> <port>"
    exit 1
fi

MODEL="$1"
PORT="$2"

echo -e "${GREEN}[Benchmark]${NC} Model: $MODEL"
echo -e "${GREEN}[Benchmark]${NC} Port:  $PORT"
echo ""

# ==============================================================================
# Single-user tests (--max-concurrency 1)
# Most relevant for interactive chat / agent use (OpenClaw/Lyra)
# ==============================================================================

echo -e "${YELLOW}═══ Single-User Tests (--max-concurrency 1) ═══${NC}"
echo ""

echo -e "${GREEN}--- 128 in / 128 out — single user ---${NC}"
vllm bench serve \
    --model "$MODEL" \
    --dataset-name random \
    --random-input-len 128 \
    --random-output-len 128 \
    --ignore-eos \
    --num-prompt 5 \
    --max-concurrency 1 \
    --backend vllm \
    --port "$PORT"

echo ""
echo -e "${GREEN}--- 1024 in / 1024 out — single user ---${NC}"
vllm bench serve \
    --model "$MODEL" \
    --dataset-name random \
    --random-input-len 1024 \
    --random-output-len 1024 \
    --ignore-eos \
    --num-prompt 5 \
    --max-concurrency 1 \
    --backend vllm \
    --port "$PORT"

echo ""
echo -e "${GREEN}--- 2048 in / 2048 out — single user ---${NC}"
vllm bench serve \
    --model "$MODEL" \
    --dataset-name random \
    --random-input-len 2048 \
    --random-output-len 2048 \
    --ignore-eos \
    --num-prompt 5 \
    --max-concurrency 1 \
    --backend vllm \
    --port "$PORT"

# ==============================================================================
# Long context test (optional — only for models with large max_model_len)
# Uncomment if the model supports 40K+ context
# ==============================================================================

# echo ""
# echo -e "${GREEN}--- 20480 in / 20480 out — single user ---${NC}"
# vllm bench serve \
#     --model "$MODEL" \
#     --dataset-name random \
#     --random-input-len 20480 \
#     --random-output-len 20480 \
#     --ignore-eos \
#     --num-prompt 5 \
#     --max-concurrency 1 \
#     --backend vllm \
#     --port "$PORT"

# ==============================================================================
# Batched tests (5 concurrent, --request-rate inf)
# Measures aggregate throughput under load
# ==============================================================================

echo ""
echo -e "${YELLOW}═══ Batched Tests (5 concurrent, --request-rate inf) ═══${NC}"
echo ""

echo -e "${GREEN}--- 128 in / 128 out — batched ---${NC}"
vllm bench serve \
    --model "$MODEL" \
    --dataset-name random \
    --random-input-len 128 \
    --random-output-len 128 \
    --ignore-eos \
    --num-prompt 5 \
    --request-rate inf \
    --backend vllm \
    --port "$PORT"

echo ""
echo -e "${GREEN}--- 1024 in / 1024 out — batched ---${NC}"
vllm bench serve \
    --model "$MODEL" \
    --dataset-name random \
    --random-input-len 1024 \
    --random-output-len 1024 \
    --ignore-eos \
    --num-prompt 5 \
    --request-rate inf \
    --backend vllm \
    --port "$PORT"

echo ""
echo -e "${GREEN}--- 2048 in / 2048 out — batched ---${NC}"
vllm bench serve \
    --model "$MODEL" \
    --dataset-name random \
    --random-input-len 2048 \
    --random-output-len 2048 \
    --ignore-eos \
    --num-prompt 5 \
    --request-rate inf \
    --backend vllm \
    --port "$PORT"

echo ""
echo -e "${GREEN}[Benchmark]${NC} All tests complete."
