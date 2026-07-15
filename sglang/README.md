# SGLang on Intel BMG

End-to-end recipe for running Qwen3.6-35B-A3B online fp8 inference on Intel
Battlemage (BMG) GPUs with the optimized ESIMD kernel fast-paths.

## What's in here

```
sglang/
├── docker/
│   └── Dockerfile                   # builds the full image
├── scripts/
│   ├── build_image.sh               # wrapper around `docker buildx build`
│   ├── run_qwen3_6.sh               # launches the TP=2 fp8 server
│   └── run_gsm8k.py                 # standalone GSM8K accuracy harness
├── patches/                         # sglang / sgl-kernel-xpu source patches
└── custom-esimd-kernels/            # merged ESIMD kernel package:
                                     #   decode attn, fp8 GEMM, fp8 MoE (silu + prefill),
                                     #   fused QKV, GDN conv fused_seq, RMSNormGated
```


## Build

```bash
llm-scaler/sglang/scripts/build_image.sh
```

The script resolves `docker/Dockerfile` relative to itself, forwards
`http_proxy` / `https_proxy` from the environment, and bumps
`SGLANG_CACHEBUST` each run. Override the tag with `IMAGE_TAG=...`.

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

## Fast-paths enabled

Each is gated by an env var (set by `run_qwen3_6.sh`):

| Env var                            | Path                                   |
|------------------------------------|----------------------------------------|
| `SGL_XPU_ESIMD_DECODE`             | Decode SDPA (split-K, flat NHD KV)     |
| `SGL_XPU_ESIMD_MOE`                | FP8 MoE silu routed kernel             |
| `SGL_XPU_ESIMD_MOE_PREFILL`        | FP8 MoE prefill (M-tiled DPAS)         |
| `SGL_XPU_FA_ESIMD_QKV`             | Full-attention fused QKV+RMSNorm+RoPE  |
| `SGL_XPU_GDN_ESIMD`                | GDN conv fused_seq decode              |
| `SGL_XPU_GDN_EXTEND_ESIMD`         | GDN chunk_gated_delta_rule prefill     |
| `SGL_XPU_PREFILL_DPAS`             | Prefill SDPA via DPAS/XMX              |
| `SGL_XPU_ENABLE_GRAPH`             | XPU device-graph capture/replay        |

> **Note:** all ESIMD/XPU fast-path gates use the `SGL_XPU_*` prefix.

In addition `SGLANG_MAMBA_{CONV,SSM}_DTYPE=float16` is required when running
the model with `--dtype float16` so the mamba state pool matches activation
dtype (the triton causal_conv1d_update kernel rejects mismatches).

## Accuracy check (GSM8K)

`scripts/run_gsm8k.py` is a standalone harness (stdlib only) that hits the
running server's OpenAI-compatible endpoint with full sampling-parameter
control, then reports accuracy and classifies failures
(correct / wrong_answer / empty_output / runaway_len / error).

```bash
# non-thinking chat, greedy, 200 questions (cleanest kernel-debug signal)
python3 scripts/run_gsm8k.py \
    --base-url http://localhost:30000 \
    --num-questions 200 \
    --no-thinking \
    --temperature 0

# thinking mode with the Qwen3-recommended sampling params
python3 scripts/run_gsm8k.py \
    --base-url http://localhost:30000 \
    --num-questions 200 \
    --thinking \
    --temperature 0.6 --top-p 0.95 --top-k 20 --repetition-penalty 1.05
```

Key flags: `--thinking/--no-thinking` (explicitly sets `enable_thinking`),
`--chat-stop/--no-chat-stop` (adds `Question:` stops to prevent fake-question
continuation), `--api chat|completion`, plus the full sampling set
(`--temperature --top-p --top-k --min-p --repetition-penalty
--frequency-penalty --presence-penalty --max-tokens`). Outputs
`<prefix>_examples.jsonl` and `<prefix>_summary.txt`.
