# SGLang on Intel BMG

End-to-end recipe for running Qwen3.6-35B-A3B online fp8 inference on Intel
Battlemage (BMG) GPUs with the optimized ESIMD kernel fast-paths.

## What's in here

```
sgl/
├── Dockerfile                       # builds the full image
├── scripts/run_qwen3_6.sh           # launches the TP=4 fp8 server
├── custom-esimd-kernels/            # ESIMD kernels: decode attn, fp8 GEMM, fp8 MoE
└── custom-esimd-kernels-sglang/     # ESIMD kernels: fused QKV, GDN conv fused_seq
```

The patched **sglang** and **sgl-kernel-xpu** sources are pulled from the
`dev-bmg` branches of analytics-zoo on build, so this directory does not
vendor them.

## Build

```bash
docker build -t llm-scaler-sgl:bmg \
    --build-arg http_proxy=${http_proxy:-} \
    --build-arg https_proxy=${https_proxy:-} \
    llm-scaler/sgl
```

Time: ~25 min on a workstation (cold), dominated by the ESIMD AOT compile
and the sgl-kernel-xpu cmake build.

## Run

```bash
docker run --rm -it \
    --device=/dev/dri \
    --shm-size=16g \
    -v /home/intel/LLM/models/Qwen3.6-35B-A3B:/models/Qwen3.6-35B-A3B:ro \
    -p 30000:30000 \
    llm-scaler-sgl:bmg \
    /workspace/scripts/run_qwen3_6.sh
```

## Performance

BS=1, 1k-in / 256-out, fp16 + fp8 weights:

| Config                          | TPOT (ms) |
|---------------------------------|-----------|
| Baseline (no XPU Graph)         | ~45       |
| **With XPU Graph (this image)** | **~15**   |

## Fast-paths enabled

Each is gated by an env var (set by `run_qwen3_6.sh`):

| Env var                          | Path                                   |
|----------------------------------|----------------------------------------|
| `SGLANG_ENABLE_XPU_ESIMD_DECODE` | Decode SDPA (split-K, flat NHD KV)     |
| `SGLANG_ENABLE_ESIMD_MOE`        | FP8 MoE silu routed kernel             |
| `SGL_XPU_FA_ESIMD_QKV`           | Full-attention fused QKV+RMSNorm+RoPE  |
| `SGL_XPU_GDN_ESIMD`              | GDN conv fused_seq decode              |
| `SGLANG_XPU_ENABLE_GRAPH`        | XPU device-graph capture/replay        |

In addition `SGLANG_MAMBA_{CONV,SSM}_DTYPE=float16` is required when running
the model with `--dtype float16` so the mamba state pool matches activation
dtype (the triton causal_conv1d_update kernel rejects mismatches).
