#!/usr/bin/env bash
# Reproduction harness for the alt-recipe qwen38-27b-fp8-r272 stack.
# Captured verbatim from the live production service (qwen38-vllm-fp8-tp2)
# via `docker inspect .Config.Cmd` and `.Config.Env` before any change.
# Requires: 2x Intel Arc Pro B70 with `xe` driver, docker, and the image
#   neural-download/vllm-openai-xpu:qwen38-fp8-gdn544-ptr683
# 
# This script starts nothing by default. `--record` only writes a launch plan.
# Set MODEL_HOST before running if /home/dom/models/Qwen3.8-27B-FP8 differs.
set -Eeuo pipefail
docker run --rm \
  --name qwen38-vllm-fp8-tp2 \
  --hostname qwen38-vllm-fp8-tp2 \
  --privileged \
  --network host \
  --device /dev/dri \
  --shm-size 32g \
  --ulimit memlock=-1:-1 \
  --log-driver journald \
  --log-opt tag=qwen38-vllm-fp8-tp2 \
  --health-cmd 'curl -fsS http://127.0.0.1:8001/health || exit 1' \
  --health-interval 30s --health-timeout 10s --health-start-period 45m --health-retries 3 \
  --mount "type=bind,source=${MODEL_HOST-/home/dom/models/Qwen3.8-27B-FP8},target=/model,readonly" \
  --mount "type=bind,source=${CACHE_HOST-/home/dom/.cache/vllm/qwen38-r187},target=/root/.cache/vllm" \
  --mount "type=bind,source=${LOCAL_GPU_MODEL_RUNNER-/home/dom/fixes/qwen38-vllm/gpu_model_runner.py},target=/workspace/vllm/vllm/v1/worker/gpu_model_runner.py,readonly" \
  --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  --env VLLM_TARGET_DEVICE=xpu \
  --env VLLM_WORKER_MULTIPROC_METHOD=spawn \
  --env ZE_AFFINITY_MASK=0,1 \
  --env ONEAPI_DEVICE_SELECTOR=level_zero:0,1 \
  --env VLLM_XPU_ENABLE_XPU_GRAPH=0 \
  --env VLLM_XPU_FP8_BLOCK_W8A16=1 \
  --env VLLM_XPU_QWEN_GEMMA_RMSNORM_PACKED_SERIAL_EXACT=1 \
  --env VLLM_XPU_GDN_SPEC_PERSISTENT_SCRATCH=0 \
  --env VLLM_XPU_GDN_NATIVE_FALLBACK=1 \
  --env VLLM_XPU_GDN_SPLIT_MIXED=1 \
  --env VLLM_XPU_DRAFT_LM_HEAD_INT4=0 \
  --env VLLM_XPU_DRAFT_LM_HEAD_INT4_GROUP_SIZE=128 \
  --env VLLM_XPU_DRAFT_LM_HEAD_INT4_SCALE_DTYPE=bf16 \
  --env VLLM_XPU_DRAFT_LM_HEAD_INT4_CHUNK_ROWS=2048 \
  --env TORCHINDUCTOR_DETERMINISTIC=1 \
  --env VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE=0 \
  --env VLLM_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING=0 \
  --env PYTHONHASHSEED=0 \
  --env PYTORCH_ALLOC_CONF=expandable_segments:True \
  --env CCL_ATL_TRANSPORT=ofi \
  --env FI_PROVIDER=tcp \
  --env FI_TCP_IFACE=lo \
  --env CCL_ZE_IPC_EXCHANGE=pidfd \
  --env CCL_SEND=direct \
  --env CCL_RECV=direct \
  --env CCL_TOPO_P2P_ACCESS=1 \
  --env CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD=4294967296 \
  --env CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD=4294967296 \
  --env CCL_SYCL_REDUCE_SCATTER_SIMPLE_THRESHOLD=4294967296 \
  neural-download/vllm-openai-xpu:qwen38-fp8-gdn544-ptr683 \
  --model /model \
  --served-model-name Qwen38-27b-0 \
  --host 0.0.0.0 --port 8001 \
  --dtype float16 \
  --tensor-parallel-size 2 \
  --quantization fp8 \
  --kv-cache-dtype fp8_e4m3 \
  --mamba-ssm-cache-dtype bfloat16 \
  --max-model-len 200000 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 8192 \
  --block-size 64 \
  --gpu-memory-utilization 0.95 \
  --cpu-offload-gb 0 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --disable-sliding-window \
  --enable-prompt-tokens-details \
  --language-model-only \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --compilation-config '{"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[1],"max_cudagraph_capture_size":1,"splitting_ops":[],"inductor_compile_config":{"combo_kernels":false,"benchmark_combo_kernel":false,"deterministic":true,"triton.autotune_pointwise":false,"benchmark_epilogue_fusion":false}}' \
  --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":3}' \
  --trust-remote-code
