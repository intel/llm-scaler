# Qwen3.8-27B FP8 + DSpark Drafter — B70 XPU Serving Stack

Achieves **72.2 tok/s median (85.9 peak)** on the isolated C1 benchmark
(temp=0, pp=0, n=16) on 2× Intel Arc Pro B70 (TP=2), vs 32.4 tok/s FP8
no-spec and 54.67 tok/s FP8+MTP2.

## Contents

- `qwen38-fp8-dspark:v8` — vLLM 0.21.1.dev0 XPU image with:
  - DSpark/DFlash draft model classes (qwen3_dflash.py, registry.py)
  - **Kernel readout fix** (dflash.py, utils.py): SpecForge-trained drafters
    were off-by-one in the sampling offsets, capping acceptance at ~24%.
  - Adaptive-mode list handling (gpu_model_runner.py)
- `drafter-fp8-v5/` — trained 1.36B DSpark drafter (HF format)
- `serve.sh` — reference serving command

## Quick start

```bash
docker run --rm --name qwen38-dspark \
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
  -v /path/to/qwen3.8-27b-fp8:/models/target:ro \
  -v /path/to/drafter-fp8-v5:/models/drafter:ro \
  --entrypoint /opt/venv/bin/vllm \
  qwen38-fp8-dspark:v8 \
  serve --host 127.0.0.1 --port 8003 \
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
```

## The kernel fix (why this matters)

SpecForge DSpark trains output position j to predict token anchor+j+1
(LM-style). The stock vLLM dflash kernel sampled draft outputs at query
offsets 1..k (BERT-style), so every draft was off by one position and
acceptance collapsed to ~24% for any SpecForge-trained drafter.

Fix (3 lines):
- `dflash.py:487`: `num_query_per_req = k` (was `k+1`)
- `utils.py:557`: `is_sample = is_query` (was `is_query & (query_off > 0)`)
- `utils.py:558`: `sample_out_idx = req*k + off` (was `req*k + off - 1`)

Effect on the released RadixArk drafter: 24% → 66% pos-0 acceptance.
With our fine-tuned drafter-fp8-v5: 62-74% pos-0, mean acceptance length 2.5-3.5.

## Quality

Greedy spec decode is lossless by construction. Verified: 4/5 byte-identical
outputs vs target-only baseline; the one divergence is a dtype tie-break
(fp16 vs bf16), not a drafter effect. Cross-checked against an independent
bf16 reference endpoint: drafter adds zero divergence.

## Requirements

- 2× Intel Arc Pro B70 (Battlemage G31, 8086:e223) or equivalent XPU
- intel/level-zero driver stack matching the image
- Qwen/Qwen3.8-27B-FP8 checkpoint (block FP8 e4m3, [128,128])

## Known limitations

- bf16 target-only serving (no drafter) crashes: ESIMD attention kernel
  asserts fp16. Use fp16 dtype for baseline comparisons.
- XPU graphs must stay disabled (VLLM_XPU_ENABLE_XPU_GRAPH=0); piecewise
  capture hangs the xe engines on this build.
- Adaptive block truncation (DSPARK_ADAPTIVE_BLOCK=1) runs but is slower
  than fixed k=4 on single-request workloads.
