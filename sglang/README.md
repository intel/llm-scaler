# SGLang on Intel BMG

End-to-end recipe for running Qwen3.6-35B-A3B online fp8 (e5m2) inference on
Intel Battlemage (BMG) GPUs with the optimized ESIMD kernel fast-paths. Decode
runs eager (XPU graph disabled for accuracy); the e5m2 fused decode kernels
recover the per-step host-dispatch cost.

## What's in here

```
sglang/
├── docker/
│   └── Dockerfile                   # builds the full image
├── scripts/
│   ├── build_image.sh               # wrapper around `docker buildx build`
│   ├── start_qwen3_6_service.sh     # launches the TP=2 e5m2 fp8 server
│   ├── run_gemma4_26b_moe.sh        # Gemma4-26B-A4B TP=2, e4m3/e5m2
│   ├── run_gemma4_26b_moe_bfcl.sh   # radix-on BFCL profile
│   ├── run_gemma4_26b_moe_canonical.sh # radix-off latency profile
│   └── run_gsm8k.py                 # standalone GSM8K accuracy harness
├── patches/                         # sglang / sgl-kernel-xpu source patches
│                                    # (base BMG + Gemma4/KVCacheIO delta)
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

Cold builds take a while, dominated by the ESIMD AOT compile
and the sgl-kernel-xpu cmake build.

## Run

```bash
docker run --rm -it \
    --device=/dev/dri \
    --shm-size=16g \
    -v /home/intel/LLM/models/Qwen3.6-35B-A3B:/models/Qwen3.6-35B-A3B:ro \
    -p 30000:30000 \
    llm-scaler-sgl:bmg \
    /llm-scaler/sglang/scripts/start_qwen3_6_service.sh
```

The Gemma4 launch script creates the `/dev/dri/by-path/*-render` links needed
by oneCCL 2021.15 when Docker exposes only the render nodes through
`--device=/dev/dri`; privileged mode is not required.

### Gemma4-26B-A4B

The Gemma4 MoE path dynamically quantizes the BF16 expert weights and supports
both FP8 formats. E4M3 is the default; select E5M2 explicitly:

```bash
MODEL_PATH=/llm/models/gemma-4-26B-A4B-it \
ZE_AFFINITY_MASK=0,1 \
SGLANG_FP8_DTYPE=e4m3 \
scripts/run_gemma4_26b_moe.sh

SGLANG_FP8_DTYPE=e5m2 scripts/run_gemma4_26b_moe.sh
```

Both formats use the routed GELU-tanh kernel for decode and the M-tiled kernel
for prefill. The GELU exponent is saturated before `exp` to avoid `inf/inf`
for large positive gates. Radix cache is enabled by default for multi-turn
prefix reuse; set `RADIX_CACHE=0` for an uncached comparison. XPU graph remains
disabled for TP=2.

This path requires the Dockerfile-pinned **oneCCL 2021.15.9** runtime. The
Dockerfile removes the pip-provided oneCCL package and links the copy loaded by
`libtorch_xpu.so` to `/opt/intel/oneapi/ccl/2021.15`. Do not install a package
that restores oneCCL 2021.17 after the image build: 2021.17 caused delayed
non-finite prefill outputs during long BFCL runs. Verify the scheduler mappings
in `/proc/<scheduler-pid>/maps` when deriving another image.

For the exact TP=2 model shape, ESIMD fast paths also fuse input RMSNorm+QKV,
the dense branch, router normalization, logits-to-MoE decode, and the final
dual-branch RMSNorm chain. Prefill uses the production ESIMD top-k and batched
dual RMSNorm. The XPU KV-index builder and request-cache write path do not use
Triton; the acceptance trace contains no Triton kernels.

#### Workload profiles

Do not reuse a server across BFCL and canonical latency measurement. Restart
the whole container before switching profiles.

| Profile | Launch script | Radix | SWA full ratio | Max running | Intended use |
|---|---|---:|---:|---:|---|
| Serving | `run_gemma4_26b_moe.sh` | on | 0.05 | 1 | normal interactive requests |
| BFCL | `run_gemma4_26b_moe_bfcl.sh` | **on** | **0.2** | 1 | cumulative multi-turn prompts |
| Canonical | `run_gemma4_26b_moe_canonical.sh` | **off** | 0.05 | 1 | isolated 1K/2K/4K/8K latency |

BFCL repeatedly sends the full cumulative conversation, commonly 6K-12K input
tokens per generation step. Running it on the canonical radix-off profile
re-prefills that history at every step and is invalid as a BFCL throughput
configuration. Its larger full-attention pool ratio also preserves headroom for
long cumulative histories. The BFCL harness should use `--num-threads 1` with
the validated profile above; concurrency changes require a separate
correctness/performance validation.

#### Validated Gemma4-26B results

The final TP=2 path was measured from the reviewed image with oneCCL 2021.15.9,
W8A16 prefill enabled, M-tiled MoE prefill, eager decode, and no diagnostic
synchronization:

| FP8 | BFCL v4 `multi_turn_base` | 1K / 2K / 4K / 8K TPOT |
|---|---:|---|
| E4M3 | 128/200 = **64.00%** | 17.37 / 17.54 / 17.60 / 17.47 ms |
| E5M2 | 128/200 = **64.00%** | 17.16 / 17.21 / 17.27 / 17.34 ms |

Both full BFCL runs had zero inference errors and left the server healthy.
Canonical latency used bsz=1, two warmups, three trials, 256 output tokens, and
the median of each set. The exact image also passed 21 Gemma4 ESIMD tests and
229 KVCacheIO tests; E4M3 and E5M2 chat sanity both returned `42`.

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
