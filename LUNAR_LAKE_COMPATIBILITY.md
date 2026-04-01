# Lunar Lake Xe2 140V Compatibility

**Status: EXPERIMENTAL SUPPORT**

This document describes how to run llm-scaler on Intel Lunar Lake platforms with the Arc 140V (Xe2) integrated GPU.

## Architecture Compatibility

Deep analysis of the codebase revealed that the core SYCL ESIMD kernels are **fully portable** to Xe2 — no hardcoded device IDs, no BMG-specific constants, and all operations use generic ESIMD intrinsics. The required changes are in the infrastructure layer (scripts, Docker, memory configuration).

| Layer | Status | Notes |
|-------|--------|-------|
| SYCL ESIMD kernels (dequant, norm, RoPE) | **Portable** | Generic ESIMD ops, adaptive SLM, no device-specific code |
| oneDNN INT4 GEMM | **Portable** | Uses abstracted `dnnl::engine` API, device-agnostic |
| vLLM XPU backend (`torch.xpu`) | **Works** | Single GPU with `--tensor-parallel-size 1` |
| Platform scripts | **Adapted** | New Lunar Lake evaluation script added |
| Docker image | **Adapted** | Lightweight `Dockerfile.lunar-lake` variant |

## Key Differences from B60 Discrete GPU

| Aspect | Arc Pro B60 (discrete) | Arc 140V Lunar Lake (integrated) |
|--------|------------------------|----------------------------------|
| **Memory** | 20-50GB dedicated VRAM | ~24GB shared from 32GB LPDDR5x |
| **Bandwidth** | PCIe x16 + GDDR6 | 136.5 GB/s LPDDR5x (shared with CPU) |
| **Multi-GPU** | TP=2+ via P2P/CCL | Single GPU only (TP=1) |
| **Firmware** | BMG GuC/HuC required | xe driver built-in (no extra firmware) |
| **CPU** | Intel Xeon | Core Ultra 7 258V |
| **Quantization** | FP16, FP8, INT4 | **INT4 recommended** (memory constrained) |

## Quick Start

### Option A: Native Install (recommended for Lunar Lake)

```bash
# 1. Install oneAPI (Fedora/Nobara)
sudo dnf install intel-oneapi-basekit

# 2. Install Level-Zero for Xe2 iGPU
sudo dnf install level-zero level-zero-devel

# 3. Source oneAPI
source /opt/intel/oneapi/setvars.sh --force

# 4. Install PyTorch XPU
pip install torch==2.10.0+xpu --extra-index-url=https://download.pytorch.org/whl/xpu

# 5. Install vLLM with XPU support
cd llm-scaler/vllm
pip install -r patches/requirements-lunar-lake.txt  # if available
# Or follow standard vLLM XPU install

# 6. Run evaluation
./tools/platform/evaluation/lunar_lake_evaluation.sh

# 7. Serve a model
./scripts/lunar_lake_serve.sh Qwen/Qwen3-8B --quantization fp8
```

### Option B: Docker

```bash
# Build Lunar Lake image
cd llm-scaler/vllm
docker build -f docker/Dockerfile.lunar-lake -t llm-scaler-lunar-lake .

# Run (note: --device=/dev/dri maps the iGPU)
docker run -it --privileged \
    --device=/dev/dri \
    --group-add video \
    -v /path/to/models:/llm/models \
    --shm-size="16g" \
    llm-scaler-lunar-lake bash

# Inside container:
source /root/.bashrc
./lunar_lake_serve.sh /llm/models/Qwen3-8B --quantization fp8
```

## Model Compatibility on Xe2 (Lunar Lake)

### Critical Blockers

Not all model architectures work on Lunar Lake XPU. Key blockers discovered during testing:

| Blocker | Affected Models | Root Cause |
|---------|----------------|------------|
| **~~Triton XPU backend broken on Xe2~~** RESOLVED | Qwen3.5 (all sizes) | **Root cause: packaging bug, NOT hardware limitation.** The `install_lunar_lake.sh` script installed `triton-xpu` without first uninstalling the plain `triton` package (pulled in as a transitive dependency by vllm-xpu-kernels). Plain `triton`'s `libtriton.so` lacks the Intel backend, causing `ImportError: cannot import name 'intel'` and `TypeError: 'function' object is not subscriptable`. **Fix:** `pip uninstall triton triton-xpu -y && pip install triton-xpu==3.6.0`. Also requires `oneapi-level-zero-devel` for Triton's JIT compilation of `driver.c → spirv_utils.so` (needs `level_zero/ze_api.h`). Install scripts updated. Intel's Docker always did the uninstall-first pattern — that's why it worked. |
| **Marlin kernels are CUDA-only** | AWQ, GPTQ (compressed-tensors format) | `gptq_marlin_repack` is an NVIDIA CUDA kernel. AWQ/GPTQ MoE models route to `CompressedTensorsWNA16MarlinMoEMethod`. **Note:** Intel has an `ipex marlin` backend for MXFP4 (used by gpt-oss-20b), but this is a separate implementation — GPTQ/AWQ Marlin is NOT ported to XPU. |
| **Shared memory ceiling ~13 GiB** | AutoRound/GPTQ models >14B | Both AutoRound and sym_int4 use layer-by-layer `process_weights_after_loading` (peak ≈ initial load + one layer overhead, NOT 2x). However, models with >13 GiB loaded weights OOM due to IPEX kernel buffers + KV cache pre-allocation + OS overhead competing for the same 32GB shared pool. gpt-oss-20b (13.27 GiB) is the largest model that fits. |
| **transformers 5.x `max_pixels` rename** | Qwen3.5 multimodal models | vLLM pins `transformers<5`. Upgrading to 5.x enables `qwen3_5`/`glm4_moe_lite` architecture recognition but renames `image_processor.max_pixels` → `size["longest_edge"]`. Fix: `getattr()` fallback in `qwen2_vl.py` (see limitation #12). Intel's Docker image applies `vllm_for_multi_arc.patch` which includes full Qwen3.5 support. |

### What DOES Work

**Practical model size limit for sym_int4 on 32GB Lunar Lake:** ≤ ~10B dense params. sym_int4 requires loading the full BF16 model into CPU RAM for quantization — on the Claw, CPU and GPU share the same 32GB. A 10B model = ~20 GiB BF16, leaving room for OS + KV cache. Models >10B need pre-quantized formats (AutoRound/GGUF) or llama.cpp with Vulkan.

Only models meeting **all three** criteria work on Lunar Lake XPU:
1. **Standard attention** or **fla/linear attention with Triton** (requires correct `triton-xpu` install — see blocker fix above)
2. **FP16/BF16 base weights** with **online quantization** (`--quantization fp8` or `--quantization int4`), OR **AutoRound INT4** pre-quantized (for models ≤8B)
3. **Steady-state VRAM ≤ ~20GB** (leaving room for OS on 32GB shared memory)

### Recommended Models for 32GB Lunar Lake

| Model | Quantization | VRAM Needed | Context | Notes |
|-------|-------------|-------------|---------|-------|
| **openai/gpt-oss-20b** | MXFP4 (pre-quant) | **13.27 GiB** | 32k | **Tested & working** — 13B actual params. **22.5 tok/s single-user**, 70 tok/s peak batched. Uses `ipex marlin` XPU backend. Requires `VLLM_SKIP_PROFILE_RUN=1` patch. Supports tool calling + reasoning (3 thinking levels). See [benchmarks](#benchmark-results-gpt-oss-20b-mxfp4) and [running recipe](#running-gpt-oss-20b-on-lunar-lake) below. |
| **Qwen3.5-4B-int4-AutoRound** | AutoRound INT4 | **3.68 GiB** | 4k | **Tested & working** — 23.4 tok/s single, 159 tok/s peak batched. Hybrid Mamba+attention (Triton fla/ops). |
| **Intel/Qwen3-8B-int4-AutoRound** | AutoRound INT4 | **5.69 GiB** | 8k | **Tested & working** — 18.6 tok/s single, 90 tok/s peak batched |
| Qwen3-8B | FP8 (online) | ~10GB | 32k | Standard attention, online quantization |
| DeepSeek-R1-Distill-Qwen-7B | FP8 (online) | ~8GB | 32k | Good reasoning |
| Qwen3-14B | INT4 (online) | ~10GB | 16k | Needs `--quantization int4` |
| Qwen3-8B | FP16 | ~18GB | 16k | No quantization loss |
| qwen3.5-9b-claude-distilled | BF16 (none) | **17.66 GiB** | 32k | **Tested — too slow** (5 tok/s single-user, 205ms TPOT). ~18B total Mamba hybrid params. Not viable for interactive chat. |
| qwen3.5-9b-claude-distilled | FP8 (online) | **11.22 GiB** | 32k | **Tested — still slow** (~8.5 tok/s est. single-user, 117ms TPOT batched). 37% smaller than BF16. FP8 is the only working online quantization on XPU native. |
| qwen3.5-9b-claude-distilled | sym_int4 (online) | **8.11 GiB** | 8k | **Tested — best 9B result** 14.7 tok/s single-user, 68ms TPOT. 54% smaller than BF16, 2.9x faster. Requires `vllm_int4_for_multi_arc.so` + `VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=1`. |
| Intel/Qwen3.5-9B-int4-AutoRound | AutoRound INT4 | **~9 GiB** | 32k | Untested. Would still be ~2x slower than 4B due to larger Mamba state. |

### Models That Do NOT Work

| Model | Format | Failure Mode |
|-------|--------|-------------|
| ~~Qwen3.5-* (any size)~~ | Any | ~~Triton kernel crash~~ **RESOLVED & VERIFIED** — was a packaging bug (plain `triton` shadowing `triton-xpu`). Qwen3.5-4B now runs successfully with correct triton-xpu install + `torch.cuda→torch.xpu` patches + `oneapi-level-zero-devel`. See benchmark results below. |
| GLM-4.7-flash AWQ | AWQ 4-bit | `CompressedTensorsWNA16MarlinMoEMethod` → `gptq_marlin_repack` CUDA kernel missing. Dense layers work via `XPUwNa16LinearKernel` but MoE layers still route to Marlin. compressed-tensors MoE has no XPU redirect. |
| GLM-4.7-flash AutoRound INT4 | AutoRound | IPEX routing works (no Marlin error) but **OOM → DEVICE_LOST** during weight init. 30B-A3B MoE model — loaded weights exceed ~13 GiB practical ceiling on 32GB shared memory. Confirmed with 32GB swap enabled. |
| Qwen3.5-35B-A3B GPTQ | GPTQ INT4 | OOM + GPU DEVICE_LOST at 79% loading |
| Qwen3.5-35B-A3B AutoRound | AutoRound INT4 | OOM — ~18 GiB loaded weights exceed ~13 GiB practical ceiling on 32GB shared memory |
| Qwen3-30B-A3B GPTQ INT4 | GPTQ INT4 | Loads 15.7 GiB via IPEX, OOM during MoE expert weight shuffle → DEVICE_LOST |
| Qwen3-Coder-30B-A3B AWQ | AWQ 4-bit | Same compressed-tensors MoE Marlin issue as GLM AWQ + 30B OOM risk |
| ~~Qwen3.5-4B AutoRound INT4~~ | AutoRound | **NOW WORKING** — 3.68 GiB, 23.4 tok/s single-user, 159 tok/s batched. Moved to recommended models. Required fixes: (1) `triton-xpu` clean install, (2) `oneapi-level-zero-devel`, (3) `torch.cuda→torch.xpu` patches, (4) `getattr()` fix for `max_pixels`. |
| ~~gpt-oss-20b~~ | MXFP4 (pre-quant) | **RESOLVED** — was hanging during `profile_run()` in KV cache init. Fixed by setting `VLLM_SKIP_PROFILE_RUN=1` (see patch `vllm_xpu_worker_skip_profile.patch`). Now loads and serves successfully. See "Recommended Models" table above. |
| LFM2-24B-A2B AWQ | AWQ 4-bit | Custom Liquid AI tokenizer (`TokenizersBackend`) not supported |
| Any MLX format | MLX | Apple Silicon only (Metal GPU framework) |

## Environment Variables

```bash
# Required for Lunar Lake
export VLLM_TARGET_DEVICE=xpu
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=1
export PYTORCH_ALLOC_CONF="expandable_segments:True"

# Shared memory mode (no P2P)
export CCL_TOPO_P2P_ACCESS=0

# Skip profile_run() — required for Lunar Lake iGPU (hangs during dummy forward pass)
export VLLM_SKIP_PROFILE_RUN=1

# Source oneAPI
source /opt/intel/oneapi/setvars.sh --force
```

## vLLM Launch Flags

```bash
# Device is set via environment variable, NOT a CLI flag
export VLLM_TARGET_DEVICE=xpu

vllm serve <model> \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.7 \
    --enforce-eager \
    --quantization int4 \
    --max-model-len 8192
```

Key flags:
- `VLLM_TARGET_DEVICE=xpu` — **Environment variable** (NOT `--device xpu`, which is not a valid CLI flag)
- `--tensor-parallel-size 1` — Single iGPU (no multi-GPU)
- `--gpu-memory-utilization 0.7` — Conservative for shared memory (leave room for OS)
- `--enforce-eager` — Disable CUDA graphs (XPU uses eager mode)
- `--quantization int4` — Online INT4 quantization for fitting larger models in shared memory
- `--allow-deprecated-quantization` — Required for pre-quantized AutoRound/GPTQ models

## CCL Single-GPU Workaround

On Lunar Lake handhelds/laptops without wired Ethernet, oneCCL's internal KVS initialization fails with:

```
fill_local_host_ip: can't find non-loopback interface
```

This happens because CCL tries to find a network interface for collective communications, even for single-GPU (TP=1). The `lunar_lake_serve.sh` script handles this automatically, but if launching manually, set these env vars:

```bash
export MASTER_ADDR=127.0.0.1
export CCL_ZE_ENABLE=0
export CCL_ATL_TRANSPORT=ofi
export FI_PROVIDER=tcp
export CCL_SOCKET_IFNAME=wlo1  # or your WiFi interface name
```

Additionally, the `all_reduce` warmup in `vllm/v1/worker/xpu_worker.py` (lines ~201-203) must be patched out for single-GPU. The `install_lunar_lake.sh` script does this automatically.

## Benchmark Results: gpt-oss-20b (MXFP4)

**Model:** openai/gpt-oss-20b (13.27 GiB loaded, MXFP4 pre-quantized)
**Architecture:** Dense transformer, 13B actual parameters (despite "20b" name)
**Backend:** `ipex marlin` XPU backend (MXFP4-specific, not the CUDA GPTQ/AWQ Marlin)
**Server config:** `--max-model-len 8192 --gpu-memory-utilization 0.7 --enforce-eager --num-gpu-blocks-override 256`
**Patch required:** `VLLM_SKIP_PROFILE_RUN=1` (profile_run() hangs on Lunar Lake Xe2 iGPU)
**KV cache:** 256 blocks = ~8,192 tokens

### Single-User Performance (`--max-concurrency 1`)

| Workload | TPOT (median) | tok/s | TTFT (median) | Notes |
|----------|:---:|:---:|:---:|-------|
| **128/128** | **44.4 ms** | **22.5 tok/s** | 225 ms | Fastest decode — excellent for interactive chat |
| **1024/1024** | **58.6 ms** | **17.1 tok/s** | 1,485 ms | Stable, no degradation vs steady state |
| **2048/2048** | **58.5 ms** | **17.1 tok/s** | 2,486 ms | Rock steady — identical to 1024 |

Decode speed is remarkably consistent at longer contexts (58.5ms at 2048 = same as 1024). Only short-context (128) is faster at 44ms due to smaller KV cache reads.

### Batched Throughput (5 concurrent, `--request-rate inf`)

| Workload | Output tok/s | Peak tok/s | TPOT (median) | TTFT (median) | Notes |
|----------|:----------:|:----------:|:---:|:---:|-------|
| **128/128** | 55.2 | 70 | 74.7 ms | 2,168 ms | Good aggregate throughput |
| **1024/1024** | 45.5 | 75 | 105.3 ms | 4,791 ms | KV cache fills to 100%, starts queuing |
| **2048/2048** | 33.3 | 75 | 113.8 ms | 7,802 ms | Degrades as requests compete for KV cache |

### Comparison: gpt-oss-20b vs Qwen3.5-9B sym_int4

| Metric | 9B sym_int4 (8.11 GiB) | gpt-oss-20b MXFP4 (13.27 GiB) | Winner |
|--------|:---:|:---:|:---:|
| Single 128 tok/s | 14.7 | **22.5** | gpt-oss-20b (1.53x) |
| Single 1024 tok/s | 10.6 | **17.1** | gpt-oss-20b (1.61x) |
| Single 2048 tok/s | 10.5 | **17.1** | gpt-oss-20b (1.63x) |
| Batched 128 TPOT | 82 ms | **74.7 ms** | gpt-oss-20b |
| Batched 1024 TPOT | **100 ms** | 105.3 ms | 9B sym_int4 |
| Batched 2048 TPOT | 129 ms | **113.8 ms** | gpt-oss-20b |
| Model size | **8.11 GiB** | 13.27 GiB | 9B sym_int4 (39% smaller) |

**Key insight:** gpt-oss-20b is significantly faster in single-user despite being 64% larger. The MXFP4 + ipex marlin backend is more efficient than sym_int4's IPEX WOQ path. At 22.5 tok/s single-user, it's comfortably above the 15-20 tok/s interactive chat threshold.

## Running gpt-oss-20b on Lunar Lake

The `openai/gpt-oss-20b` model ships pre-quantized in MXFP4 format (13B actual parameters, 13.27 GiB). It runs on Lunar Lake's Arc 140V via the `ipex marlin` XPU backend.

### Prerequisites

1. Apply the `vllm_xpu_worker_skip_profile.patch` to your vLLM installation:
   ```bash
   cd /path/to/vllm
   git apply /path/to/llm-scaler/vllm/patches/vllm_xpu_worker_skip_profile.patch
   ```

2. Ensure oneAPI is installed and sourced.

### Why the patch is needed

vLLM's XPU worker runs a dummy forward pass (`profile_run()`) during startup to measure peak GPU memory for KV cache sizing. On Lunar Lake's Xe2 iGPU, this forward pass **hangs indefinitely** (EngineCore at 100% CPU, no progress). The patch adds `VLLM_SKIP_PROFILE_RUN=1` support which skips the dummy forward pass and estimates peak memory from current allocation + 20% overhead instead.

### Quick start (using lunar_lake_serve.sh)

```bash
# Basic (benchmarking / simple inference)
./vllm/scripts/lunar_lake_serve.sh /shared/models/gpt-oss-20b \
    --max-model-len 32768

# With tool calling + reasoning (for OpenClaw agent)
./vllm/scripts/lunar_lake_serve.sh /shared/models/gpt-oss-20b \
    --max-model-len 32768 \
    --enable-auto-tool-choice \
    --tool-call-parser openai \
    --reasoning-parser openai_gptoss
```

The script automatically sets all required environment variables including `VLLM_SKIP_PROFILE_RUN=1`.

### Manual launch

```bash
source /opt/intel/oneapi/setvars.sh --force

export VLLM_TARGET_DEVICE=xpu
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_SKIP_PROFILE_RUN=1
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export CCL_TOPO_P2P_ACCESS=0

vllm serve /shared/models/gpt-oss-20b \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.7 \
    --enforce-eager \
    --max-model-len 32768 \
    --enable-auto-tool-choice \
    --tool-call-parser openai \
    --reasoning-parser openai_gptoss
```

### Startup log (expected)

```
Model loading took 13.27 GiB memory and ~25 seconds
[VLLM_SKIP_PROFILE_RUN] Skipping profile_run, estimating peak from allocated memory
[Memory Profiling Analysis]
  > Peak Allocated (Real Need)  : 15.94 GB
  > Model memory usage          : 13.27 GB
GPU KV cache size: 88,576 tokens
Maximum concurrency for 32,768 tokens per request: ~2.7x
Application startup complete.
```

The server listens on `http://127.0.0.1:8000` (OpenAI-compatible API).

### Tool calling and reasoning

gpt-oss-20b supports three thinking levels controlled via system prompt:

| System Prompt | Thinking Level | Use Case |
|---|---|---|
| `Reasoning: low` | Minimal reasoning | Fast responses, simple tasks |
| `Reasoning: medium` | Balanced (default) | General agent use |
| `Reasoning: high` | Deep reasoning | Complex multi-step problems |

**Flags:**
- `--enable-auto-tool-choice` — lets the model decide when to call tools (required for OpenClaw agent)
- `--tool-call-parser openai` — parses native OpenAI tool call format
- `--reasoning-parser openai_gptoss` — parses gpt-oss thinking blocks (NOT `gptoss` — full name is `openai_gptoss`)

### Key notes

- **No `--quantization` flag needed** — MXFP4 is auto-detected from the model's `config.json`
- **No `--device xpu` flag** — device is set via `VLLM_TARGET_DEVICE=xpu` environment variable (NOT a CLI flag)
- **No `--num-gpu-blocks-override` needed** — vLLM auto-allocates 88,576 KV cache tokens at `--gpu-memory-utilization 0.7`, enough for ~2.7x concurrent 32K conversations
- **`--dtype bfloat16`** is the default; MXFP4 rejects FP16
- **sym_int4 does NOT work** on this model (quantization method mismatch — already MXFP4)

## Known Limitations

1. **Memory pressure** — GPU and CPU share the same LPDDR5x. Running a 20GB model leaves little room for the OS. Monitor with `free -h` or `nvtop`.
2. **No multi-GPU** — Single iGPU only. Tensor parallelism is not available.
3. **No PCIe P2P** — CCL collective operations run in USM mode, not P2P.
4. **Bandwidth-bound TG** — Token generation speed is limited by LPDDR5x bandwidth (136.5 GB/s), similar to the Vulkan path.
5. **Platform installer** — The B60 offline installer (BMG firmware, Xeon kernel) does not apply. Use native oneAPI install instead.
6. **~13 GiB practical model ceiling** — Both AutoRound INT4 and sym_int4 use layer-by-layer `process_weights_after_loading` (peak ≈ initial load + one layer overhead, NOT 2x). However, models with loaded weights exceeding ~13 GiB OOM because IPEX kernel buffers, KV cache pre-allocation, MoE expert weight shuffling, and OS overhead all compete for the same 32GB shared pool. gpt-oss-20b (13.27 GiB) is the largest confirmed working model. Qwen3.5-35B-A3B (~18 GiB), GLM-4.7-flash (~15-17 GiB), and Qwen3-30B-A3B (~17.5 GiB) all fail with DEVICE_LOST.
7. **GPU crash requires reboot** — If vLLM OOMs and the GPU enters `UR_RESULT_ERROR_DEVICE_LOST` state, a full system reboot is required to reset the GPU.
8. **Build time** — vllm-xpu-kernels compilation (933 SYCL files) takes **1.5-2 hours** on Lunar Lake. Ensure the device is plugged in and sleep is disabled (`systemctl mask sleep.target suspend.target`).
9. **~~Triton XPU broken on Xe2~~ RESOLVED — packaging bug** — The native install scripts installed `triton-xpu` without first removing the plain `triton` package. The `vllm-xpu-kernels` build pulls in plain `triton` as a transitive dependency, and its `libtriton.so` (which lacks the Intel backend) shadows `triton-xpu`'s version. This caused `ImportError: cannot import name 'intel' from 'triton._C.libtriton'`, which made `@triton.jit` kernels degrade to plain functions (`TypeError: 'function' object is not subscriptable`). **Fix:** `pip uninstall triton triton-xpu -y && pip install triton-xpu==3.6.0`. Additionally, `oneapi-level-zero-devel` must be installed for Triton's Intel backend to JIT-compile `driver.c → spirv_utils.so` (requires `level_zero/ze_api.h` header). Intel's Docker images always did the uninstall-first pattern — the bug was only in the native install scripts. The `triton-xpu` source code explicitly recognizes `lnl` (Lunar Lake) as a valid architecture — there is NO iGPU exclusion. Install scripts have been updated.
10. **Marlin kernels CUDA-only** — AWQ and GPTQ models using compressed-tensors format route to Marlin repack kernels (`_C.gptq_marlin_repack`), which are NVIDIA CUDA kernels with no XPU equivalent. Pre-quantized AWQ/GPTQ models cannot be used on XPU.
11. **Download FP16 base models** — For models without pre-quantized INT4 weights, download FP16/BF16 base weights and use vLLM's online quantization (`--quantization fp8` or `--quantization int4`). Both online and pre-quantized (AutoRound) paths use layer-by-layer weight processing. Pre-quantized INT4 (AutoRound) is preferred when available — smaller initial load (~9 GiB vs ~18 GiB for BF16), same inference speed.
12. **transformers 5.x `max_pixels` rename** — vLLM pins `transformers<5,>=4.56.0`. Upgrading to 5.x for Qwen3.5/GLM-4.7 architecture recognition breaks `Qwen2VLImageProcessor.max_pixels` — the attribute was renamed to `size["longest_edge"]` (and `min_pixels` to `size["shortest_edge"]`). Intel's llm-scaler Docker image installs transformers from git HEAD and applies `vllm_for_multi_arc.patch` which adds full Qwen3.5 support. On a native install without the patch, apply the one-line fix below:
    ```python
    # In vllm/model_executor/models/qwen2_vl.py, line ~944
    # Change:
    max_pixels = image_processor.max_pixels or image_processor.size["longest_edge"]
    # To:
    max_pixels = getattr(image_processor, "max_pixels", None) or image_processor.size["longest_edge"]
    # Same for min_pixels on the next line:
    min_pixels = getattr(image_processor, "min_pixels", None) or image_processor.size["shortest_edge"]
    ```

## Benchmark Results (vLLM SYCL on Lunar Lake)

**Hardware:** MSI Claw 8 AI+ — Intel Core Ultra 7 258V, Arc 140V iGPU, 32GB LPDDR5x (136.5 GB/s)
**Model:** Intel/Qwen3-8B-int4-AutoRound (5.7GB on disk, 5.69 GiB loaded)
**Server config:** `--max-model-len 8192 --gpu-memory-utilization 0.8 --enforce-eager --allow-deprecated-quantization`
**KV cache:** 116,672 tokens (14.2x concurrency at 8K context)
**Tool:** `vllm bench serve` with random dataset

### Single-User Performance (`--max-concurrency 1`)

The most relevant benchmark for interactive chat (OpenClaw/Lyra). Requests run strictly one at a time.

#### Decode Speed (Token Generation)

| Context Length | TPOT (median) | Decode Speed | ITL P99 | TTFT (median) |
|---------------|--------------|-------------|---------|---------------|
| 128 in / 128 out | **53.5 ms** | **18.7 tok/s** | 59.7 ms | 168 ms |
| 1,024 in / 1,024 out | **72.8 ms** | **13.7 tok/s** | 73.4 ms | 1,510 ms |
| 2,048 in / 2,048 out | **75.4 ms** | **13.3 tok/s** | 76.3 ms | 1,347 ms |

Decode speed degrades with longer context due to growing KV cache attention computation and LPDDR5x bandwidth limits. TPOT is remarkably consistent across runs (verified twice).

### Batched Throughput (5 concurrent, `--request-rate inf`)

| Workload | Output tok/s | Peak tok/s | TPOT | TTFT (median) |
|----------|-------------|-----------|------|---------------|
| 128 in / 128 out | 18.9 | 90.0 | 56.4 ms | 26,693 ms |
| 1,024 in / 1,024 out | 48.6 | 80.0 | 81.7 ms | 21,829 ms |
| 2,048 in / 2,048 out | 45.3 | 75.0 | 97.9 ms | 25,591 ms |

> **High batched TTFT (21-27s)** matches the Qwen3.5-4B pattern at 0.8 util — the 116K token KV cache creates massive scheduling overhead when 5 requests arrive simultaneously.

### Analysis

- **Interactive chat (single-user):** 13-19 tok/s decode depending on context length — comfortable for real-time chat
- **TPOT scales linearly with context:** 53.5ms (128 tok) → 72.8ms (1K) → 75.4ms (2K) — pure attention overhead
- **Batched TPOT only ~5% worse than single-user:** 56.4ms vs 53.5ms at 128 tokens — iGPU already well-utilized for single requests
- **Concurrency kills TTFT:** 5 concurrent requests push TTFT from 168ms to 26,693ms due to KV cache management at 0.8 util
- **Long context penalty:** Decode at 2K context is ~1.4x slower than short context (75.4ms vs 53.5ms), but still usable at 13.3 tok/s
- **Memory efficient:** Model uses 5.71 GiB, KV cache peaks at ~18% with 5 concurrent 2K-context requests

### Server-Side Observations (10-prompt long-context run)

From the vLLM engine logs during `4096 in / 2048 out × 10 prompts`:

| Metric | Value | Notes |
|--------|-------|-------|
| **Generation throughput (5 concurrent)** | 55 → 35 tok/s | Declines as KV cache grows |
| **Generation throughput (3 concurrent)** | 48 → 27 tok/s | Better per-request latency |
| **Generation throughput (1 sequential)** | 12-13 tok/s | Single long request, consistent |
| **Prompt throughput (burst)** | 1,638 tok/s | Batched prefill for 5 requests |
| **KV cache peak (5 concurrent, 4K+2K)** | ~35% | Well within budget |
| **KV cache per single long request** | ~7% | Very efficient |
| **Prefix cache hit rate** | 0% → 62.5% | Improves across repeated runs at same input length |
| **Request queuing** | 2 running + 3 waiting | Memory-limited batching with chunked prefill |

**Key insight:** Generation speed degrades with context length — from 77 tok/s (short context, 5 concurrent) to 35 tok/s (long context, 5 concurrent) to 12-13 tok/s (single very long request). This is the attention computation overhead growing with sequence length on shared LPDDR5x.

### Comparison Notes

- For llama.cpp Vulkan comparison, use the same Qwen3-8B model in GGUF Q4_K_M format
- Expected Vulkan speed: similar decode (both LPDDR5x bandwidth-bound), but slower prefill
- vLLM advantage: much faster prefill (1K-11K vs ~300 tok/s), continuous batching, OpenAI-compatible API

## Benchmark Results: Qwen3.5-4B (Hybrid Mamba + Attention with Triton)

**Model:** Qwen3.5-4B-int4-AutoRound (3.68 GiB loaded) — first Triton-dependent model running on Lunar Lake
**Tool:** `vllm bench serve` with random dataset
**Prerequisites:** `triton-xpu==3.6.0` (clean install), `oneapi-level-zero-devel`, `torch.cuda→torch.xpu` patches from `vllm_for_multi_arc.patch`

### Single-User Performance (`--max-concurrency 1`) — Side-by-Side

| Context | Metric | 0.8 (8K ctx) | 0.42 (32K ctx) | 0.35 (32K ctx) |
|---------|--------|:------------:|:--------------:|:--------------:|
| **128/128** | Decode | **23.0 tok/s** | **23.2 tok/s** | **21.9 tok/s** |
| | TPOT | 43.7 ms | 43.1 ms | 44.1 ms |
| | TTFT | 220 ms | 207 ms | 204 ms |
| | ITL P99 | 51.1 ms | 44.2 ms | 47.9 ms |
| **1024/1024** | Decode | **16.4 tok/s** | **16.8 tok/s** | **16.2 tok/s** |
| | TPOT | 60.8 ms | 59.5 ms | 61.6 ms |
| | TTFT | 1,044 ms | 790 ms | 794 ms |
| | ITL P99 | 67.8 ms | 66.4 ms | 70.1 ms |
| **2048/2048** | Decode | **15.5 tok/s** | **16.4 tok/s** | **16.1 tok/s** |
| | TPOT | 62.0 ms | 61.1 ms | 61.7 ms |
| | TTFT | 1,885 ms | 1,510 ms | 1,496 ms |
| | ITL P99 | 68.1 ms | 67.3 ms | 70.1 ms |

Single-user decode is virtually identical across all three memory configs — expected since single requests don't benefit from larger KV cache. The ~1 tok/s variance at 0.35 is within run-to-run noise.

### Batched Throughput (5 concurrent, `--request-rate inf`) — Side-by-Side

| Context | Metric | 0.8 (8K ctx) | 0.42 (32K ctx) | 0.35 (32K ctx) |
|---------|--------|:------------:|:--------------:|:--------------:|
| **128/128** | Output tok/s | **22.7** | **59.3** | **100.3** |
| | Peak tok/s | **110.0** | **115.0** | **115.0** |
| | TPOT | 45.9 ms | 44.6 ms | 45.0 ms |
| | TTFT | 22,370 ms | 5,120 ms | 664 ms |
| **1024/1024** | Output tok/s | **50.2** | **71.7** | **76.4** |
| | Peak tok/s | **100.0** | **100.0** | **100.0** |
| | TPOT | 67.1 ms | 65.9 ms | 62.9 ms |
| | TTFT | 33,178 ms | 3,916 ms | 2,577 ms |
| **2048/2048** | Output tok/s | **46.9** | **50.5** | **52.6** |
| | Peak tok/s | **85.0** | **70.0** | **80.0** |
| | TPOT | 90.1 ms | 95.4 ms | 93.1 ms |
| | TTFT | 33,727 ms | 7,290 ms | 4,118 ms |

> **0.8 util has genuinely high batched TTFT (22-34s)** — confirmed across two runs (cold and warmed up). This is NOT JIT warmup; it's KV cache management overhead. The 141K token KV cache creates massive scheduling/prefill latency when 5 requests arrive simultaneously. TTFT scales with KV cache size: 0.35 (34K) → 664ms, 0.42 (51K) → 5,120ms, 0.8 (141K) → 22,370ms at 128 tokens. TPOT (decode speed) is identical across all configs.

All three configs use identical methodology (5 prompts, `--request-rate inf`).

### Configuration Summary

| Setting | KV Cache | Max Context | KV Headroom | Init Time | Single-User | Batched TTFT (128 tok) | Batched Peak | Best For |
|---------|----------|-------------|-------------|-----------|-------------|----------------------|-------------|----------|
| **0.8** | 141,120 tokens | 8,192 | 17x | ~47s | 23.0 tok/s | **22,370 ms** | 110 tok/s | Benchmark only |
| **0.42** | ~51,520 tokens | 32,768 | 57% over 32K | ~13s | 23.2 tok/s | **5,120 ms** | 115 tok/s | **OpenClaw (recommended)** |
| **0.35** | ~34,560 tokens | 32,768 | 5% over 32K | ~13s | 21.9 tok/s | **664 ms** | 115 tok/s | LLM + ASR + TTS |

> **Sweet spot analysis:** 0.35 has the best batched TTFT but only 5% KV headroom — risky for full 32K conversations. 0.42 trades 4.5s extra batched TTFT for 57% safety margin. Single-user TTFT is identical across all three configs (~204-220ms). For OpenClaw single-user chat, 0.42 gives the best balance of safety and performance.

### Qwen3.5-4B vs Qwen3-8B Comparison

| Metric | Qwen3-8B (5.69 GiB) | Qwen3.5-4B (3.68 GiB) | Delta |
|--------|---------------------|----------------------|-------|
| **Short decode (128 tok)** | 18.7 tok/s (53.5ms) | **23.2 tok/s** (43.1ms) | **+24% faster** |
| **Medium decode (1K tok)** | 13.7 tok/s (72.8ms) | **16.8 tok/s** (59.5ms) | **+23% faster** |
| **Long decode (2K tok)** | 13.3 tok/s (75.4ms) | **16.4 tok/s** (61.1ms) | **+23% faster** |
| **Peak single-user** | 19 tok/s | 23 tok/s | **+21% faster** |
| **Batched peak (0.8 util)** | 90 tok/s | **110 tok/s** (22s TTFT!) | **+22% higher** |
| **Batched peak (0.42 util)** | — | **115 tok/s** | 32K context mode |
| **TTFT (128 tok, single)** | 168 ms | 207 ms | Slightly slower (Triton JIT) |
| **KV cache (0.8 util)** | 116,672 tokens | 141,120 tokens | **+21% more** |
| **KV cache (0.42 util)** | — | ~51,520 tokens | 32K context mode |
| **Memory footprint** | 5.71 GiB | 3.68 GiB | **36% smaller** |

### Analysis: Qwen3.5-4B on Lunar Lake

- **Fastest single-user decode so far** — 23.0 tok/s at short context, staying at 15.5-16.4 tok/s even at 2K context (Qwen3-8B drops to 13.3 at 2K)
- **Decode speed plateau at 1K+** — TPOT stays flat at ~61ms from 1K to 2K context, suggesting the Mamba hybrid attention has a different scaling curve than pure attention
- **Massive batched throughput** — 115 tok/s peak across all configs, 159 tok/s in ad-hoc 8-concurrent post-warmup test
- **0.35 util penalty is minimal** — Single-user decode at 0.35 is identical to 0.42 and 0.8 (21.9→16.1 tok/s). Batched peak drops from 115→80 tok/s as context grows, but TPOT (45→93ms) matches 0.42 exactly. The only real difference: 35K KV cache leaves razor-thin headroom for 32K context
- **TTFT is higher** — 232ms single-user vs 1,710ms batched at 128 tokens. The Triton Intel backend JIT overhead plus chunked prefill queuing with 5 concurrent requests
- **Triton works on Xe2 iGPU** — This is the first confirmed Triton-dependent model running on Lunar Lake. The `fla/ops` linear attention kernels (Flash Linear Attention) execute correctly via triton-xpu's Intel backend
- **Server logs confirm stability** — Steady 16.1-16.8 tok/s generation over extended runs, no degradation or GPU faults

## Benchmark Results: Qwen3.5-9B Claude Distilled (BF16 — Too Slow)

**Model:** qwen3.5-9b-claude-4.6-opus-reasoning-distilled (17.66 GiB loaded, BF16, no quantization)
**Architecture:** `Qwen3_5ForConditionalGeneration` — hybrid Mamba + attention (same as 4B), multimodal with image processor
**Note:** Despite the "9B" name, the Mamba state-space layers add ~80% more weights — total is ~18B parameters at BF16.
**Server config:** `--max-model-len 32768 --gpu-memory-utilization 0.8 --enforce-eager`
**KV cache:** 26,240 tokens (0.8 util with 17.66 GiB model leaves minimal room)

### Batched Throughput (5 concurrent, `--request-rate inf`)

| Context | Output tok/s | TPOT | TTFT | Notes |
|---------|:----------:|:----:|:----:|-------|
| **128/128** | 6.0 | 205 ms | 81,417 ms | TTFT inflated by 404 errors (multimodal `/v1/completions` routing) |
| **1024/1024** | 16.3 | 268 ms | 40,104 ms | Steady ~16.5-18.5 tok/s total across 5 concurrent |
| **2048/2048** | 15.4 | 299 ms | 52,743 ms | Memory pressure evident |

### Analysis: Qwen3.5-9B Distilled on Lunar Lake

- **~4-5x slower than Qwen3.5-4B** — 205ms TPOT vs 43ms at 128 tokens, 299ms vs 61ms at 2K tokens
- **Not viable for interactive chat** — ~5 tok/s single-user (~200ms per token) feels sluggish. Qwen3.5-4B delivers 23 tok/s.
- **17.66 GiB at BF16 eats nearly all shared memory** — only 26K tokens KV cache at 0.8 util, vs 141K for the 4B model
- **Multimodal routing issue** — `/v1/completions` returns 404; must use `/v1/chat/completions` endpoint. Benchmark client eventually retried correctly.
- **Would need INT4 quantization to be practical** — `Intel/Qwen3.5-9B-int4-AutoRound` (~9 GiB) would free up memory, but the ~9B Mamba hybrid is still ~2x slower decode than the 4B variant
- **Verdict: Qwen3.5-4B remains the best model for Lunar Lake** — 3.68 GiB, 23 tok/s, room for 32K context + ASR + TTS

## Benchmark Results: Qwen3.5-9B Claude Distilled (FP8 Online Quantization)

**Model:** qwen3.5-9b-claude-4.6-opus-reasoning-distilled (11.22 GiB loaded, FP8 online quantization)
**Architecture:** `Qwen3_5ForConditionalGeneration` — hybrid Mamba + attention, multimodal
**Quantization:** `--quantization fp8` (online FP8 — the only working online quantization method on XPU native install)
**Memory savings:** 11.22 GiB vs 17.66 GiB BF16 = **37% reduction**
**Server config:** `--max-model-len 32768 --gpu-memory-utilization 0.8 --enforce-eager --quantization fp8`
**KV cache:** 79,040 tokens (0.8 util — 3x more than BF16's 26,240 tokens, fits 9.8x concurrent 8K contexts)
**Peak allocated:** 13.19 GB

### Batched Throughput (5 concurrent, `--request-rate inf`)

| Context | Output tok/s | TPOT | TTFT | Peak tok/s | Notes |
|---------|:----------:|:----:|:----:|:----------:|-------|
| **128/128** | 13.66 | 117 ms | 31,949 ms | 45 | ~1.7x faster than BF16 (205ms) |
| **1024/1024** | 28.97 | 149 ms | 24,099 ms | 45 | Steady 27-43.5 tok/s generation |
| **2048/2048** | 26.89 | 171 ms | 30,487 ms | 40 | Memory pressure but stable |

### Analysis: FP8 vs BF16 for Qwen3.5-9B Distilled

| Metric | BF16 | FP8 | Improvement |
|--------|------|-----|-------------|
| Model size | 17.66 GiB | 11.22 GiB | **37% smaller** |
| KV cache tokens | 26,240 | 79,040 | **3x more** |
| TPOT (128 batched) | 205 ms | 117 ms | **1.75x faster** |
| TPOT (1024 batched) | 268 ms | 149 ms | **1.80x faster** |
| TPOT (2048 batched) | 299 ms | 171 ms | **1.75x faster** |
| Est. single-user tok/s | ~5 | ~8.5 | **1.7x faster** |

- **FP8 significantly improves both speed and memory** — 37% less memory, 1.75x faster decode. The smaller model reads less data per forward pass, improving bandwidth utilization.
- **Still not fast enough for interactive chat** — estimated ~8.5 tok/s single-user (117ms TPOT) vs Qwen3.5-4B's 23 tok/s. Borderline usable but noticeably slower.
- **KV cache dramatically improved** — 79K tokens vs 26K at BF16. Can now handle 9.8x concurrent 8K contexts instead of 3.3x.
## Benchmark Results: Qwen3.5-9B Claude Distilled (sym_int4 Online Quantization)

**Model:** qwen3.5-9b-claude-4.6-opus-reasoning-distilled (8.11 GiB loaded, sym_int4 online quantization)
**Architecture:** `Qwen3_5ForConditionalGeneration` — hybrid Mamba + attention, multimodal
**Quantization:** `--quantization sym_int4` with `VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=1`
**Memory savings:** 8.11 GiB vs 17.66 GiB BF16 = **54% reduction** (vs FP8's 37%)
**Server config:** `--max-model-len 8192 --gpu-memory-utilization 0.8 --enforce-eager --quantization sym_int4 --dtype float16`
**KV cache:** 104,640 tokens (0.8 util — 4x more than BF16's 26,240, 1.3x more than FP8's 79,040)
**Peak allocated:** 10.08 GB
**Note:** Uses `vllm_int4_for_multi_arc.so` for CPU-side quantization via ctypes, then IPEX `IPEXWeightOnlyQuantizedLinear` for INT4 GEMM on XPU

### Single-User Throughput (`--max-concurrency 1`)

| Context | Output tok/s | TPOT (median) | TTFT (median) | Notes |
|---------|:----------:|:----:|:----:|-------|
| **128/128** | 14.7 | 68 ms | 298 ms | **Best single-user decode speed for 9B model** |
| **1024/1024** | 10.59 | 89 ms | 1,377 ms | Remarkably stable ~11 tok/s throughout 1024 output tokens |
| **2048/2048** | 10.49 | 92 ms | 2,519 ms | Minimal degradation even at 4K total context |

### Batched Throughput (5 concurrent, `--request-rate inf`)

| Context | Output tok/s | TPOT (median) | TTFT (median) | Peak tok/s | Notes |
|---------|:----------:|:----:|:----:|:----------:|-------|
| **128/128** | 18.61 | 82 ms | 24,010 ms | 70 | Good batched throughput |
| **1024/1024** | 37.15 | 100 ms | 35,065 ms | 65 | Strong aggregate throughput |
| **2048/2048** | 34.21 | 129 ms | 35,240 ms | 60 | KV cache grows to 6.1%, still stable |

### Analysis: sym_int4 vs FP8 vs BF16 for Qwen3.5-9B Distilled

| Metric | BF16 | FP8 | sym_int4 | Best |
|--------|------|-----|----------|------|
| Model size | 17.66 GiB | 11.22 GiB | 8.11 GiB | **sym_int4 (54% smaller)** |
| KV cache tokens | 26,240 | 79,040 | 104,640 | **sym_int4 (4x more)** |
| Single-user tok/s (128) | ~5 | ~8.5 | **14.7** | **sym_int4 (2.9x vs BF16)** |
| Single-user tok/s (1024) | ~4 | ~6.7 | **10.6** | **sym_int4 (2.7x vs BF16)** |
| Single-user tok/s (2048) | ~3.3 | ~5.8 | **10.5** | **sym_int4 (3.2x vs BF16)** |
| TPOT 128 batched | 205 ms | 117 ms | 82 ms | **sym_int4 (2.5x faster)** |
| TPOT 1024 batched | 268 ms | 149 ms | 100 ms | **sym_int4 (2.7x faster)** |
| TPOT 2048 batched | 299 ms | 171 ms | 129 ms | **sym_int4 (2.3x faster)** |

- **sym_int4 is the clear winner** — smallest memory footprint, fastest decode, most KV cache headroom
- **14.7 tok/s single-user at short context, 10.5 tok/s at 2K** — usable for chat with the 9B model. Still slower than Qwen3.5-4B's 23 tok/s, but much more capable model with 2.3x the parameters
- **Remarkably stable decode speed** — single-user TPOT stays 68-92ms across all context lengths (server logs show steady ~11 tok/s). KV cache usage stays under 1.2% single-user, under 6.1% batched
- **Both `.so` and env var are required** — `vllm_int4_for_multi_arc.so` at `/opt/lib/` provides the C quantization function (loaded via `ctypes.CDLL()`), and `VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=1` ensures weights are quantized on CPU to avoid GPU OOM.

- **Online quantization status on XPU native install:**
  - `sym_int4` — **NOW WORKS** ✓ (requires `vllm_int4_for_multi_arc.so` at `/opt/lib/` + `VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=1`, see below)
  - `fp8` — **works** ✓
  - `inc` — IPEX `varlen_attention` arg count mismatch in vision encoder (19 args given, max 18)
  - `rtn` — hardcoded `.cuda()` call in `rtn.py:147` (CUDA-only, deprecated)

## Building sym_int4 Support for Native XPU Install

The `sym_int4` online quantization works on native XPU install. It requires **two things**:

1. **`vllm_int4_for_multi_arc.so` at `/opt/lib/`** — The C library that performs the actual INT4 quantization. Intel's vLLM patch (`sym_int4.py`) loads it via `ctypes.CDLL()` and will **crash with RuntimeError** if not found. The default search path is hardcoded to `/opt/lib/vllm_int4_for_multi_arc.so` (configurable via `VLLM_QUANTIZE_Q40_LIB` env var).
2. **`VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=1`** — Loads BF16 weights to CPU first, then the .so quantizes them to INT4 on CPU, then sends INT4 weights to GPU. Without this flag on Lunar Lake, loading 17.66 GiB BF16 weights directly to GPU causes OOM.

**Source code reference:** The sym_int4 implementation lives in Intel's `vllm_for_multi_arc.patch` in the [intel/llm-scaler](https://github.com/intel/llm-scaler) repo at `vllm/patches/vllm_for_multi_arc.patch`, which adds `vllm/model_executor/layers/quantization/sym_int4.py` to vLLM. The quantization function `quantize_q4_0_to_qweight_and_scale()` is called via ctypes from the C library, then the quantized weights are handed to IPEX's `IPEXWeightOnlyQuantizedLinear` for INT4 GEMM on XPU.

**Note:** The .so is loaded transiently via `ctypes.CDLL()` during weight loading only, which is why it may not appear in `/proc/maps` after model loading completes.

### The 12KB File That Saved 10GB of Downloads

**The discovery:** Intel ships `vllm_int4_for_multi_arc.so` only inside their Docker images (`intel/llm-scaler-platform`, ~10+ GB). On a mobile device with limited data (8GB plan), downloading these images is impractical. Investigation revealed:

1. **The file is just a renamed `libquantize.so`** — built from `intel/BigDL-core` at `bigdl-core-xe/ggml/quantize.c`. It's a pure C library (~12KB compiled) that implements GGML Q4_0 quantization: `quantize_q4_0_to_qweight_and_scale()`.
2. **Intel's `vllm_for_multi_arc.patch`** hardcodes the default path as `/opt/lib/vllm_int4_for_multi_arc.so` in `vllm/envs.py`. The Docker image copies the file there during build.
3. **Building from source takes ~10 seconds** — just `cmake .. && cmake --build .` in the ggml directory, then rename the output. No GPU SDK, no SYCL, no special dependencies.
4. **The naming is misleading** — `vllm_int4_for_multi_arc` sounds like a complex vLLM-specific library, but it's literally just the BigDL-core GGML quantizer renamed. The "multi_arc" refers to multi-architecture Intel GPU support.

This saved downloading ~10 GB of Docker layers over mobile data to extract a 12KB file. A pre-built copy is available in this repo at `artifacts/vllm_int4_for_multi_arc.so` (x86_64 Linux only).

The .so and optional GPU GEMM kernel can be built from source from [intel/BigDL-core](https://github.com/intel/BigDL-core):

### 1. Build the CPU-side quantizer (required)

Converts FP16/BF16 weights to INT4 at model load time. Pure C, no GPU SDK needed.

```bash
git clone https://github.com/intel/BigDL-core.git
cd BigDL-core/bigdl-core-xe/ggml
mkdir build && cd build
cmake ..
cmake --build .
# Rename and install
sudo mkdir -p /opt/lib
sudo cp libquantize.so /opt/lib/vllm_int4_for_multi_arc.so
```

### 2. Build the GPU-side fused INT4 GEMM kernel (optional, for max performance)

Allows the Xe GPU to compute directly on packed INT4 weights without dequanting to FP16. Requires `icpx` (oneAPI DPC++ compiler).

```bash
cd BigDL-core/bigdl-core-xe/bigdl-core-xe-addons
CPLUS_INCLUDE_PATH="/opt/intel/oneapi/compiler/2025.3/include/sycl:$CPLUS_INCLUDE_PATH" \
  CXX=icpx CC=icx python3 setup.py build_ext --inplace
# Copy the .so to the venv
cp bigdl_core_llm.cpython-312-x86_64-linux-gnu.so \
  $(python3 -c "import site; print(site.getsitepackages()[0])")/
```

### 3. IPEX patch required for PyTorch 2.10+

PyTorch 2.10 removed `_PYBIND11_*` attributes that IPEX's build system expects. Patch two lines in `intel_extension_for_pytorch/xpu/cpp_extension.py`:

```python
# Line ~239: change
val = getattr(torch._C, f"_PYBIND11_{name}")
# to
val = getattr(torch._C, f"_PYBIND11_{name}", None)

# Line ~1213: same change
pval = getattr(torch._C, f"_PYBIND11_{pname}")
# to
pval = getattr(torch._C, f"_PYBIND11_{pname}", None)
```

This patch only affects the IPEX C++ extension build system, not inference. It will be overwritten if IPEX is upgraded.

### 4. Launch vLLM with sym_int4

```bash
export VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=1  # CPU-side quant, avoids GPU OOM during loading

vllm serve /shared/models/<model> \
    --quantization sym_int4 \
    --dtype float16 \
    --enforce-eager \
    --gpu-memory-utilization 0.8 \
    --host 127.0.0.1 --port 8090
```

**Note:** `sym_int4` requires `--dtype float16` (BF16 not supported). The `VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=1` flag enables CPU-side quantization, which avoids the FP16 unpack memory spike that causes OOM on large models.

## Running Recipes (MSI Claw 8 AI+)

All services run on the same machine. Use `127.0.0.1` since OpenClaw/Lyra accesses them locally.

### LLM — Qwen3-8B INT4 (port 8000)

```bash
vllm-activate
vllm serve /shared/models/qwen3-8b-int4-autoround \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.8 \
    --enforce-eager \
    --max-model-len 8192 \
    --allow-deprecated-quantization \
    --host 127.0.0.1 --port 8000
```

### LLM — Qwen3.5-4B INT4 (port 8000) — Fastest option, benchmark mode

```bash
vllm-activate
vllm serve /shared/models/qwen3.5-4b-int4-autoround \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.8 \
    --enforce-eager \
    --max-model-len 8192 \
    --allow-deprecated-quantization \
    --host 127.0.0.1 --port 8000
```

### LLM — Qwen3.5-4B INT4 (port 8082) — OpenClaw fallback model, 32K context + tool calling

```bash
vllm-activate
vllm serve /shared/models/qwen3.5-4b-int4-autoround \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.42 \
    --enforce-eager \
    --max-model-len 32768 \
    --allow-deprecated-quantization \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --host 127.0.0.1 --port 8082
```

**Flags explained:**
- `--reasoning-parser qwen3` — parses `<think>...</think>` reasoning blocks from Qwen3.5's hybrid thinking mode
- `--enable-auto-tool-choice` — lets the model decide when to call tools (required for OpenClaw agent)
- `--tool-call-parser qwen3_coder` — parses Qwen3.5's JSON tool call format into OpenAI-compatible `tool_calls` responses

> **Note:** If tool calls fail or return malformed JSON, try `--tool-call-parser qwen3_xml` as a fallback parser. A dedicated `qwen35_coder` parser is in development ([vllm-project/vllm#35347](https://github.com/vllm-project/vllm/pull/35347)).

**Memory budget at 0.42 util:** ~51,520 token KV cache — fits 1 full 32K conversation with comfortable headroom.

| `--gpu-memory-utilization` | KV Cache | 32K Concurrency | Use Case |
|---------------------------|----------|-----------------|----------|
| **0.8** | ~140K tokens | ~4 concurrent | LLM only / benchmarking |
| **0.5** | ~70K tokens | ~2 concurrent | LLM + ASR on same GPU |
| **0.45** | ~50K tokens | 1 (comfortable) | Backup chatbot (recommended) |
| **0.35** | ~34K tokens | 1 (tight) | LLM + ASR + TTS on same GPU |

**Requires:** correct `triton-xpu` install (see install script), `oneapi-level-zero-devel`, and `torch.cuda→torch.xpu` patches from `vllm_for_multi_arc.patch`.

### ASR — Qwen3-ASR-1.7B (port 8001)

```bash
vllm-activate
vllm serve /shared/models/qwen3-asr-1.7b \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.25 \
    --enforce-eager \
    --max-model-len 2048 \
    --trust-remote-code \
    --host 127.0.0.1 --port 8001
```

Test: `curl http://127.0.0.1:8001/v1/audio/transcriptions -F file=@audio.wav -F model=/shared/models/qwen3-asr-1.7b`

### TTS — Qwen3-TTS-1.7B (Python script)

```bash
source ~/qwen-tts-env/bin/activate
oneapi
python3 tts_generate.py
```

### Memory Budget (32GB LPDDR5x)

| Service | GPU Memory | Notes |
|---------|-----------|-------|
| LLM (Qwen3-8B INT4) | ~22.9 GB (0.8 × 28.6) | 5.7GB model + KV cache |
| ASR (Qwen3-ASR-1.7B) | ~7.2 GB (0.25 × 28.6) | 3.9GB model + KV cache |
| TTS (Qwen3-TTS-1.7B) | ~2 GB | Loaded on demand |
| **LLM + ASR together** | ~30 GB | Tight — reduce LLM to 0.7 |
| **ASR + TTS together** | ~9 GB | Comfortable |

## Alternative: llama.cpp with Vulkan

For simpler setup without the oneAPI stack, [llama.cpp with Vulkan](https://github.com/MegaStood/OpenClaw-on-MSI-Claw-8) is a proven alternative on Lunar Lake. The SYCL/vLLM path offers advantages for:
- Larger model support (vLLM handles model sharding and KV cache management)
- OpenAI-compatible API serving
- FP8/INT4 dynamic online quantization
- Multimodal model support

## Qwen3-ASR on Lunar Lake (vLLM XPU)

**Model:** Qwen/Qwen3-ASR-1.7B (~3.9 GiB loaded)
**VRAM:** ~7.2 GB with `--gpu-memory-utilization 0.25`
**API:** OpenAI Whisper-compatible `/v1/audio/transcriptions`

### Setup

```bash
# Download model
huggingface-cli download Qwen/Qwen3-ASR-1.7B --local-dir /shared/models/qwen3-asr-1.7b
```

No separate venv needed — runs directly via vLLM.

### Serve

```bash
vllm-activate
vllm serve /shared/models/qwen3-asr-1.7b \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.25 \
    --enforce-eager \
    --max-model-len 2048 \
    --trust-remote-code \
    --host 127.0.0.1 --port 8001
```

### Test

```bash
curl http://127.0.0.1:8001/v1/audio/transcriptions \
  -F file=@/path/to/audio.wav \
  -F model=/shared/models/qwen3-asr-1.7b
```

### Notes

- `0.25` GPU utilization is sufficient — allocates 30K+ tokens of KV cache (14x concurrency for 2048 token sequences)
- `--trust-remote-code` is required for the ASR architecture
- Init time: ~2.4 seconds (fast due to small model + low memory allocation)
- Can run alongside TTS (~9GB combined) or alongside LLM (reduce LLM to 0.7 utilization)

## Qwen3-TTS on Lunar Lake (XPU)

**Model:** Qwen/Qwen3-TTS-12Hz-1.7B-Base (~3.6GB) + Qwen/Qwen3-TTS-Tokenizer-12Hz (~651MB)
**VRAM:** ~2GB on XPU
**Use case:** Voice cloning from 3-second reference audio, 10 languages supported

### Setup

```bash
# 1. Create venv using Python 3.12 from vLLM install (Nobara ships 3.14, PyTorch XPU needs ≤3.12)
~/llm-scaler-vllm/venv/bin/python3.12 -m venv ~/qwen-tts-env --system-site-packages

# 2. Link XPU PyTorch from vLLM venv
echo "$HOME/llm-scaler-vllm/venv/lib64/python3.12/site-packages" > \
    ~/qwen-tts-env/lib64/python3.12/site-packages/vllm-xpu.pth

# 3. Install qwen-tts + deps (no CUDA torch)
source ~/qwen-tts-env/bin/activate
pip install qwen-tts --no-deps
pip install transformers==4.57.3 huggingface_hub
pip install librosa soundfile sox onnxruntime einops accelerate torchaudio --no-deps

# 4. Download models
huggingface-cli download Qwen/Qwen3-TTS-Tokenizer-12Hz --local-dir /shared/models/qwen3-tts-tokenizer-12hz
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-Base --local-dir /shared/models/qwen3-tts-12hz-1.7b-base
```

### Usage — Voice Cloning

```bash
source ~/qwen-tts-env/bin/activate
oneapi   # alias for: source /opt/intel/oneapi/setvars.sh --force
```

```python
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "/shared/models/qwen3-tts-12hz-1.7b-base",
    device_map="xpu:0",
    dtype=torch.bfloat16,
)

wavs, sr = model.generate_voice_clone(
    text="Hello, this is a test of Qwen3 text to speech on Intel Lunar Lake.",
    language="English",
    ref_audio="/path/to/reference.wav",   # 3-second voice sample
    ref_text="Transcript of the reference audio.",
)
sf.write("output.wav", wavs[0], sr)
```

### Notes

- Flash-attn warning is harmless — it falls back to PyTorch SDPA attention on XPU
- The Base model requires a reference audio for voice cloning; for preset voices use `Qwen3-TTS-12Hz-1.7B-CustomVoice` with `generate_custom_voice()`
- Can run simultaneously with vLLM — TTS uses ~2GB, leaving plenty for LLM serving
- `transformers==4.57.3` is required (newer versions break the `check_model_inputs` decorator)

## Meteor Lake / Arrow Lake Compatibility

The vLLM XPU stack also works on other Intel iGPU platforms with shared system memory. An install script is provided for these platforms.

### Supported Platforms

| Platform | Architecture | Example CPUs | iGPU | PCI Device IDs | Install Script |
|----------|-------------|-------------|------|----------------|---------------|
| **Lunar Lake** | Xe2 | Core Ultra 258V, 238V | Arc 140V | `64a0` | `install_lunar_lake.sh` |
| **Meteor Lake** | Xe-LPG | Core Ultra 155H, 135H | Arc Graphics | `7d55`, `7dd5`, `7d40`, `7d45` | `install_meteor_arrow_lake.sh` |
| **Arrow Lake-H** | Xe-LPG+ | Core Ultra 255H, 245H | Arc 130T/140T | `7d51`, `7dd1`, `7d41`, `7d67` | `install_meteor_arrow_lake.sh` |

### Meteor Lake Notes

- Meteor Lake uses **i915 driver by default**. For SYCL/oneAPI XPU support, switch to the `xe` driver:
  ```
  # Add to GRUB_CMDLINE_LINUX_DEFAULT in /etc/default/grub:
  i915.force_probe=!<device_id> xe.force_probe=<device_id>
  ```
- Some Meteor Lake laptops ship with **single-channel RAM** (16GB), which limits both iGPU performance and model size. Dual-channel is strongly recommended.
- Arc Graphics branding (device `7d55`) requires dual-channel memory + OEM enablement. Single-channel configs show as "Intel Graphics" (`7dd5`).

### Arrow Lake-H Notes

- Arrow Lake uses the `xe` driver by default on kernel 6.8+.
- Arrow Lake-H laptops support up to **96GB DDR5**, giving significantly more headroom for models compared to Lunar Lake's 32GB.
- The Arc 130T/140T iGPU has similar Xe-core count to Lunar Lake's Arc 140V.

### Install

```bash
cd llm-scaler/vllm/scripts
chmod +x install_meteor_arrow_lake.sh
./install_meteor_arrow_lake.sh
```

The script auto-detects your platform and adjusts memory recommendations accordingly.

---

*Updated: 2026-03-30*
