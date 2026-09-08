# SGLang on Intel BMG

End-to-end recipes for running Qwen3.6 and Gemma4 online FP8 inference on
Intel Battlemage (BMG) GPUs with optimized ESIMD kernel fast-paths.

## What's in here

```
sglang/
├── docker/
│   ├── Dockerfile                   # builds the release image
│   └── Dockerfile.dev               # retains compiler and build sources
├── scripts/
│   ├── build_image.sh               # wrapper around `docker buildx build`
│   ├── run_container.sh             # starts a service-free background container
│   ├── start_qwen3_6_service.sh     # launches 27B/35B online FP8 or GGUF
│   ├── run_gemma4_26b_moe.sh        # Gemma4-26B-A4B TP=2, e4m3/e5m2
│   ├── run_gemma4_31b.sh             # Gemma4-31B TP=2 online FP8
│   ├── run_gsm8k.py                 # standalone GSM8K accuracy harness
│   └── bfcl/                         # BFCL setup and evaluation workflow
├── patches/                         # sglang / sgl-kernel-xpu source patches
└── custom-esimd-kernels/            # merged ESIMD kernel package:
                                     #   decode attn, fp8 GEMM, fp8 MoE (silu + prefill),
                                     #   fused QKV, GDN conv fused_seq, RMSNormGated
```


## Build

```bash
cd llm-scaler/sglang
bash scripts/build_image.sh release
bash scripts/build_image.sh dev
```

The script defaults to `release` (`docker/Dockerfile`); `dev` selects
`docker/Dockerfile.dev`. Paths resolve relative to the script, so it can run
from any directory. Default image names are
`intel/llm-scaler-sglang:YYYYMMDD-release` and
`intel/llm-scaler-sglang:YYYYMMDD-dev`, using the host's current local date.
Override the full image reference with `IMAGE_TAG=...`.
It forwards `http_proxy`, `https_proxy`, and `no_proxy`, and bumps
`SGLANG_CACHEBUST` each run.

Cold builds take a while, dominated by the ESIMD AOT compile
and the sgl-kernel-xpu cmake build.

## Run

### Container launcher (dev or release)

Start a background container with the host model root mounted read-only at `/models`:

```bash
IMAGE_TAG=llm-scaler-sglang:dev-0907 \
MODEL_DIR=/path/to/models CONTAINER_NAME=sglang-dev \
  bash scripts/run_container.sh
```

Set `IMAGE_TAG` to the full image reference, for either dev or release.
There is no `dev/release` argument for the container launcher.
This launcher only starts the container in the background; it does not
enter the container, start a model service, or accept a service command.
Run the desired model script separately inside the container.

`IMAGE_TAG` and `MODEL_DIR` are required.
Other overrides: `CONTAINER_NAME` (default `sglang-container`)
and `SHM_SIZE` (default `16g`).
The launcher exposes `/dev/dri` without setting a GPU affinity mask.
It also mounts `/dev/dri/by-path` read-only for oneCCL's device discovery
during tensor-parallel communication.
Set `ZE_AFFINITY_MASK` when launching a service inside the container.
The container uses `--net=host`, so no port mapping is needed. Choose free
GPUs and an unused host port when starting a service inside the container.
Services share the host network; their bind address controls external access.

Containers always run detached; enter manually with
`docker exec -it sglang-dev bash`.
Containers are retained on exit, preserving dev changes.
Restart a stopped dev container with `docker start sglang-dev`;
use a different `CONTAINER_NAME` to create another container. The launcher
overrides the release image's `sglang serve` entrypoint and keeps a background
Bash shell running for both image types.

### Qwen3.6-27B / 35B-A3B: FP8 or GGUF

Inside the container, `scripts/start_qwen3_6_service.sh` selects GGUF when
`MODEL_PATH` ends in `.gguf`; otherwise it uses the existing online FP8 path.
The script header includes examples for both models and formats.

```bash
cd /llm-scaler/sglang

# Online FP8
MODEL_PATH=/models/Qwen3.6-27B ZE_AFFINITY_MASK=6,7 \
  bash scripts/start_qwen3_6_service.sh

# Q4_K_M GGUF (stop the previous service first)
MODEL_PATH=/models/Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_M.gguf \
GGUF_CFG_DIR=/models/Qwen3.6-27B ZE_AFFINITY_MASK=6,7 \
  bash scripts/start_qwen3_6_service.sh
```

GGUF requires a matching HF directory containing configuration, tokenizer
files, and the safetensors index (or HF shards) for weight-name mapping.
It does not use `--quantization fp8` or `--load-format layered_fp8`.
The GGUF branch uses the exercised Q4_K_M configuration: graph disabled,
overlap scheduling disabled, and default static memory fraction 0.8.
FP8 retains its existing fusion settings and default memory fraction 0.9.
Both accept `TP_SIZE`, `HOST`, `PORT`, and `MEM_FRACTION_STATIC` overrides.

For function-calling evaluation of the four FP8 models, see
[`scripts/bfcl/README.md`](scripts/bfcl/README.md).

### Gemma4-26B-A4B

```bash
cd /llm-scaler/sglang

MODEL_PATH=/models/gemma-4-26B-A4B-it \
ZE_AFFINITY_MASK=6,7 TP_SIZE=2 SGLANG_FP8_DTYPE=e4m3 \
HOST=127.0.0.1 PORT=30000 \
  bash scripts/run_gemma4_26b_moe.sh
```

### Gemma4-31B

```bash
cd /llm-scaler/sglang

MODEL_PATH=/models/gemma-4-31B-it \
ZE_AFFINITY_MASK=6,7 HOST=127.0.0.1 PORT=30000 \
  bash scripts/run_gemma4_31b.sh
```

## Fast-paths enabled

Each is gated by an env var (set by `start_qwen3_6_service.sh`):

| Env var                            | Path                                   |
|------------------------------------|----------------------------------------|
| `SGL_XPU_ESIMD_DECODE`             | Decode SDPA (split-K, flat NHD KV)     |
| `SGL_XPU_ESIMD_MOE`                | FP8 MoE silu routed kernel             |
| `SGL_XPU_ESIMD_MOE_FULL`          | Full decode MoE fusion (router+routed+shared+gate, e5m2, native N-major w13) |
| `SGL_XPU_ESIMD_MOE_PREFILL`        | FP8 MoE prefill (M-tiled DPAS)         |
| `SGL_XPU_FA_ESIMD_QKV`             | Full-attention fused QKV+RMSNorm+RoPE  |
| `SGL_XPU_FA_RESADD_NORM`           | Fuse FA input_layernorm (resadd+rmsnorm) into qkv_proj (decode) |
| `SGL_XPU_GDN_ESIMD`                | GDN conv fused_seq decode              |
| `SGL_XPU_GDN_EXTEND_ESIMD`         | GDN chunk_gated_delta_rule prefill     |
| `SGL_XPU_GDN_NORM_GEMV`            | GDN gated-RMSNorm as ESIMD GEMV (decode) |
| `SGL_XPU_GDN_RESADD_NORM`          | Fuse GDN input_layernorm + in_proj (qkvz+ba) into one GEMV |
| `SGL_XPU_MOE_ROUTER_FP8`           | MoE router as fp8 ESIMD GEMV (vs fp16 aten::mm) |
| `SGL_XPU_PREFILL_DPAS`             | Prefill SDPA via DPAS/XMX              |
| `SGL_XPU_ENABLE_GRAPH`             | XPU device-graph capture/replay (kept **0** here) |

> **Note:** all ESIMD/XPU fast-path gates use the `SGL_XPU_*` prefix.

The full decode MoE fusion (`SGL_XPU_ESIMD_MOE_FULL`) and the MoE router fp8
path require online fp8 to be quantized as **e5m2** — set `SGLANG_FP8_DTYPE=e5m2`
(the script does). The e5m2 fused MoE kernel reads the native N-major `w13`
weight directly (no transposed weight copy), so it needs no extra device
memory for a transposed copy. `SGL_XPU_MOE_ROUTER_FP8=1` perturbs top-8 routing
on a fraction of tokens — A/B against GSM8K before trusting it (set to 0 for the
accurate fp16 gate).

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
    --host 127.0.0.1 --port 30000 \
    --num-examples 200 --num-threads 8 \
    --no-thinking \
    --temperature 0

# thinking mode with the Qwen3-recommended sampling params
python3 scripts/run_gsm8k.py \
    --host 127.0.0.1 --port 30000 \
    --num-examples 200 --num-threads 8 \
    --thinking \
    --temperature 0.6 --top-p 0.95 --top-k 20 --repetition-penalty 1.05
```

Key flags: `--thinking/--no-thinking` (explicitly sets `enable_thinking`),
`--chat-stop/--no-chat-stop` (adds `Question:` stops to prevent fake-question
continuation), `--api chat|completion`, plus the full sampling set
(`--temperature --top-p --top-k --min-p --repetition-penalty
--frequency-penalty --presence-penalty --max-tokens`). Outputs
`<prefix>_examples.jsonl` and `<prefix>_summary.txt`.
