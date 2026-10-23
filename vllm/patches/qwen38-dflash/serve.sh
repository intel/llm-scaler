#!/bin/bash
# Reference serving command for qwen38-fp8-dspark:v8 + drafter-fp8-v5
# Usage: TARGET_DIR=/path/to/qwen3.8-27b-fp8 DRAFTER_DIR=/path/to/drafter-fp8-v5 ./serve.sh
set -euo pipefail
TARGET_DIR="${TARGET_DIR:?set TARGET_DIR}"
DRAFTER_DIR="${DRAFTER_DIR:?set DRAFTER_DIR}"
PORT="${PORT:-8003}"

exec docker run --rm --name qwen38-dspark \
  --device /dev/dri/card1 --device /dev/dri/card2 \
  --device /dev/dri/renderD128 --device /dev/dri/renderD129 \
  --mount type=bind,source=/dev/dri/by-path,target=/dev/dri/by-path,readonly \
  --network host --shm-size 32g --ipc=host \
  -e ZE_AFFINITY_MASK=0,1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=1 \
  -e VLLM_USE_V2_MODEL_RUNNER=0 \
  -e VLLM_USE_AOT_COMPILE=0 \
  -e VLLM_XPU_ENABLE_XPU_GRAPH=0 \
  -e CCL_TOPO_P2P_ACCESS=1 -e CCL_ZE_IPC_EXCHANGE=drmfd \
  -e CCL_SYCL_ALLGATHERV_TMP_BUF=0 -e CCL_SYCL_ALLREDUCE_TMP_BUF=0 \
  -e CCL_ENABLE_SYCL_KERNELS=1 \
  -e CCL_SYCL_ALLGATHERV_SMALL_THRESHOLD=131072 \
  -e CCL_SYCL_ALLGATHERV_SCALEOUT_THRESHOLD=1048576 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -v "${TARGET_DIR}":/models/target:ro \
  -v "${DRAFTER_DIR}":/models/drafter:ro \
  --entrypoint /opt/venv/bin/vllm \
  qwen38-fp8-dspark:v8 \
  serve --host 127.0.0.1 --port "$PORT" \
    --model /models/target \
    --served-model-name qwen3.8-27b-fp8 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --max-num-batched-tokens 4096 \
    --max-num-seqs 1 \
    --block-size 64 \
    --dtype bfloat16 \
    --mamba-ssm-cache-dtype float16 \
    --async-scheduling \
    --speculative-config "{\"method\":\"dflash\",\"model\":\"/models/drafter\",\"num_speculative_tokens\":4}"
