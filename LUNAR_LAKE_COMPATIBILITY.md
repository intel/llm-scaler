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
| **Marlin kernels are CUDA-only** | AWQ, GPTQ (compressed-tensors format) | `gptq_marlin_repack` is an NVIDIA CUDA kernel. AWQ/GPTQ MoE models route to `CompressedTensorsWNA16MarlinMoEMethod`. |
| **Pre-quantized 2x memory** | AutoRound, GPTQ (>14B) | Loading INT4→FP16 intermediate doubles peak memory. 35B models OOM on 32GB. |
| **transformers 5.x `max_pixels` rename** | Qwen3.5 multimodal models | vLLM pins `transformers<5`. Upgrading to 5.x enables `qwen3_5`/`glm4_moe_lite` architecture recognition but renames `image_processor.max_pixels` → `size["longest_edge"]`. Fix: `getattr()` fallback in `qwen2_vl.py` (see limitation #12). Intel's Docker image applies `vllm_for_multi_arc.patch` which includes full Qwen3.5 support. |

### What DOES Work

Only models meeting **all three** criteria work on Lunar Lake XPU:
1. **Standard attention** or **fla/linear attention with Triton** (requires correct `triton-xpu` install — see blocker fix above)
2. **FP16/BF16 base weights** with **online quantization** (`--quantization fp8` or `--quantization int4`), OR **AutoRound INT4** pre-quantized (for models ≤8B)
3. **Steady-state VRAM ≤ ~20GB** (leaving room for OS on 32GB shared memory)

### Recommended Models for 32GB Lunar Lake

| Model | Quantization | VRAM Needed | Context | Notes |
|-------|-------------|-------------|---------|-------|
| **Qwen3.5-4B-int4-AutoRound** | AutoRound INT4 | **3.68 GiB** | 4k | **Tested & working** — 23.4 tok/s single, 159 tok/s peak batched. Hybrid Mamba+attention (Triton fla/ops). |
| **Intel/Qwen3-8B-int4-AutoRound** | AutoRound INT4 | **5.69 GiB** | 8k | **Tested & working** — 18.6 tok/s single, 90 tok/s peak batched |
| Qwen3-8B | FP8 (online) | ~10GB | 32k | Standard attention, online quantization |
| DeepSeek-R1-Distill-Qwen-7B | FP8 (online) | ~8GB | 32k | Good reasoning |
| Qwen3-14B | INT4 (online) | ~10GB | 16k | Needs `--quantization int4` |
| Qwen3-8B | FP16 | ~18GB | 16k | No quantization loss |

### Models That Do NOT Work

| Model | Format | Failure Mode |
|-------|--------|-------------|
| ~~Qwen3.5-* (any size)~~ | Any | ~~Triton kernel crash~~ **RESOLVED & VERIFIED** — was a packaging bug (plain `triton` shadowing `triton-xpu`). Qwen3.5-4B now runs successfully with correct triton-xpu install + `torch.cuda→torch.xpu` patches + `oneapi-level-zero-devel`. See benchmark results below. |
| GLM-4.7-flash AWQ | AWQ 4-bit | Marlin CUDA kernel missing |
| GLM-4.7-flash AutoRound INT4 | AutoRound | 27B OOM (>24GB after FP16 unpack) |
| Qwen3.5-35B-A3B GPTQ | GPTQ INT4 | OOM + GPU DEVICE_LOST at 79% loading |
| Qwen3.5-35B-A3B AutoRound | AutoRound INT4 | OOM (14GB on disk → ~28GB peak) |
| Qwen3-30B-A3B GPTQ INT4 | GPTQ INT4 | Loads 15.7 GiB via IPEX, OOM during MoE expert weight shuffle → DEVICE_LOST |
| Qwen3-Coder-30B-A3B AWQ | AWQ 4-bit | Marlin CUDA kernel missing (same as GLM AWQ) |
| ~~Qwen3.5-4B AutoRound INT4~~ | AutoRound | **NOW WORKING** — 3.68 GiB, 23.4 tok/s single-user, 159 tok/s batched. Moved to recommended models. Required fixes: (1) `triton-xpu` clean install, (2) `oneapi-level-zero-devel`, (3) `torch.cuda→torch.xpu` patches, (4) `getattr()` fix for `max_pixels`. |
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

## Known Limitations

1. **Memory pressure** — GPU and CPU share the same LPDDR5x. Running a 20GB model leaves little room for the OS. Monitor with `free -h` or `nvtop`.
2. **No multi-GPU** — Single iGPU only. Tensor parallelism is not available.
3. **No PCIe P2P** — CCL collective operations run in USM mode, not P2P.
4. **Bandwidth-bound TG** — Token generation speed is limited by LPDDR5x bandwidth (136.5 GB/s), similar to the Vulkan path.
5. **Platform installer** — The B60 offline installer (BMG firmware, Xeon kernel) does not apply. Use native oneAPI install instead.
6. **Large models may OOM** — Pre-quantized models (AutoRound/GPTQ) require unpacking INT4 weights to FP16 during loading, doubling peak memory. The Qwen3.5-35B-A3B INT4 AutoRound model (~14GB on disk) OOMs during initialization on 32GB shared memory. Use online `--quantization int4` instead, or choose models ≤14B.
7. **GPU crash requires reboot** — If vLLM OOMs and the GPU enters `UR_RESULT_ERROR_DEVICE_LOST` state, a full system reboot is required to reset the GPU.
8. **Build time** — vllm-xpu-kernels compilation (933 SYCL files) takes **1.5-2 hours** on Lunar Lake. Ensure the device is plugged in and sleep is disabled (`systemctl mask sleep.target suspend.target`).
9. **~~Triton XPU broken on Xe2~~ RESOLVED — packaging bug** — The native install scripts installed `triton-xpu` without first removing the plain `triton` package. The `vllm-xpu-kernels` build pulls in plain `triton` as a transitive dependency, and its `libtriton.so` (which lacks the Intel backend) shadows `triton-xpu`'s version. This caused `ImportError: cannot import name 'intel' from 'triton._C.libtriton'`, which made `@triton.jit` kernels degrade to plain functions (`TypeError: 'function' object is not subscriptable`). **Fix:** `pip uninstall triton triton-xpu -y && pip install triton-xpu==3.6.0`. Additionally, `oneapi-level-zero-devel` must be installed for Triton's Intel backend to JIT-compile `driver.c → spirv_utils.so` (requires `level_zero/ze_api.h` header). Intel's Docker images always did the uninstall-first pattern — the bug was only in the native install scripts. The `triton-xpu` source code explicitly recognizes `lnl` (Lunar Lake) as a valid architecture — there is NO iGPU exclusion. Install scripts have been updated.
10. **Marlin kernels CUDA-only** — AWQ and GPTQ models using compressed-tensors format route to Marlin repack kernels (`_C.gptq_marlin_repack`), which are NVIDIA CUDA kernels with no XPU equivalent. Pre-quantized AWQ/GPTQ models cannot be used on XPU.
11. **Download FP16 base models** — The only reliable path on Lunar Lake is downloading FP16/BF16 base weights and using vLLM's online quantization (`--quantization fp8` or `--quantization int4`). This uses layer-by-layer loading without the 2x memory spike of pre-quantized formats.
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
**Tool:** `vllm bench serve` with random dataset

### Single-User Performance (`--max-concurrency 1`)

The most relevant benchmark for interactive chat (OpenClaw/Lyra). Requests run strictly one at a time.

#### Prefill Speed (Time to First Token)

| Input Length | TTFT (median) | Prefill Throughput |
|-------------|--------------|-------------------|
| 128 tokens | **120 ms** | **1,067 tok/s** |
| 1,024 tokens | **278 ms** | **3,683 tok/s** |
| 4,096 tokens | **382 ms** | **10,723 tok/s** |

Prefill throughput scales dramatically with prompt length — the Xe2 GPU is better utilized with larger matrix operations. Short prompts underutilize the GPU.

#### Decode Speed (Token Generation)

| Context Length | TPOT (median) | Decode Speed | ITL P99 |
|---------------|--------------|-------------|---------|
| 128 in / 128 out | **53.7 ms** | **18.6 tok/s** | 54.4 ms |
| 1,024 in / 1,024 out | **72.6 ms** | **13.8 tok/s** | 92.5 ms |
| 4,096 in / 4,096 out | **81.7 ms** | **12.2 tok/s** | 103.3 ms |

Decode speed degrades with longer context due to growing KV cache attention computation and LPDDR5x bandwidth limits.

### Batched Throughput (5 concurrent, `--request-rate inf`)

| Workload | Output tok/s | Peak tok/s | TPOT | Server Prefill tok/s |
|----------|-------------|-----------|------|---------------------|
| 128 in / 128 out | 23.0 | 90.0 | 56.7 ms | 64 |
| 1,024 in / 1,024 out | 48.6 | 80.0 | 82.6 ms | 512 |
| 4,096 in / 4,096 out | 37.3 | 60.0 | 126.5 ms | 1,638 (peak) |

### Analysis

- **Interactive chat (single-user):** 13-19 tok/s decode depending on context length — comfortable for real-time chat
- **Prefill is fast:** 1K-11K tok/s, far exceeding llama.cpp (~300 tok/s). Longer prompts prefill faster per-token due to better GPU utilization
- **Batched decode barely faster:** 5 concurrent requests only marginally increase per-request TPOT (56.7→53.7ms for short), showing the iGPU is already well-utilized for single requests
- **Concurrency kills TTFT:** 5 concurrent requests push TTFT from 120ms to 20,600ms due to prefill queuing with chunked prefill (`max_num_batched_tokens=2048`)
- **Long context penalty:** Decode at 4K context is ~1.5x slower than short context (81.7ms vs 53.7ms), but still usable at 12.2 tok/s
- **Memory efficient:** Model uses 5.69 GiB, KV cache peaks at ~35% with 5 concurrent long-context requests

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

| Context | Metric | 0.8 (4K ctx) | 0.42 (32K ctx) | 0.35 (32K ctx) |
|---------|--------|:------------:|:--------------:|:--------------:|
| **128/128** | Decode | **23.4 tok/s** | **23.2 tok/s** | **21.9 tok/s** |
| | TPOT | 42.7 ms | 43.1 ms | 44.1 ms |
| | TTFT | 232 ms | 207 ms | 204 ms |
| | ITL P99 | 44.5 ms | 44.2 ms | 47.9 ms |
| **1024/1024** | Decode | **16.3 tok/s** | **16.8 tok/s** | **16.2 tok/s** |
| | TPOT | 61.2 ms | 59.5 ms | 61.6 ms |
| | TTFT | 836 ms | 790 ms | 794 ms |
| | ITL P99 | 68.8 ms | 66.4 ms | 70.1 ms |
| **2048/2048** | Decode | **16.3 tok/s** | **16.4 tok/s** | **16.1 tok/s** |
| | TPOT | 61.2 ms | 61.1 ms | 61.7 ms |
| | TTFT | 1,541 ms | 1,510 ms | 1,496 ms |
| | ITL P99 | 70.3 ms | 67.3 ms | 70.1 ms |

Single-user decode is virtually identical across all three memory configs — expected since single requests don't benefit from larger KV cache. The ~1 tok/s variance at 0.35 is within run-to-run noise.

### Batched Throughput (5 concurrent, `--request-rate inf`) — Side-by-Side

| Context | Metric | 0.8 (4K ctx) | 0.42 (32K ctx) | 0.35 (32K ctx) |
|---------|--------|:------------:|:--------------:|:--------------:|
| **128/128** | Output tok/s | — | **59.3** | **100.3** |
| | Peak tok/s | **158.8** (8-conc) | **115.0** | **115.0** |
| | TPOT | — | 44.6 ms | 45.0 ms |
| | TTFT | — | 5,120 ms | 664 ms |
| **1024/1024** | Output tok/s | — | **71.7** | **76.4** |
| | Peak tok/s | — | **100.0** | **100.0** |
| | TPOT | — | 65.9 ms | 62.9 ms |
| | TTFT | — | 3,916 ms | 2,577 ms |
| **2048/2048** | Output tok/s | — | **50.5** | **52.6** |
| | Peak tok/s | — | **70.0** | **80.0** |
| | TPOT | — | 95.4 ms | 93.1 ms |
| | TTFT | — | 7,290 ms | 4,118 ms |

Note: 0.8 util batched tests were ad-hoc (8-concurrent, not standardized 5-concurrent). 0.35 and 0.42 use identical methodology (5 prompts, `--request-rate inf`). 0.35 batched data is from the same session as the 0.35 single-user tests above (fresh run).

### Configuration Summary

| Setting | KV Cache | Max Context | Init Time | Single-User | Batched Peak | Best For |
|---------|----------|-------------|-----------|-------------|-------------|----------|
| **0.8** | 140,160 tokens | 4,096 | ~172s | 23.4 tok/s | 159 tok/s | Max throughput benchmark |
| **0.42** | ~51,520 tokens | 32,768 | ~13s | 23.2 tok/s | 115 tok/s | **OpenClaw (recommended)** |
| **0.35** | ~34,560 tokens | 32,768 | ~13s | 21.9 tok/s | 115 tok/s | Tight memory budget |

### Qwen3.5-4B vs Qwen3-8B Comparison

| Metric | Qwen3-8B (5.69 GiB) | Qwen3.5-4B (3.68 GiB) | Delta |
|--------|---------------------|----------------------|-------|
| **Short decode (128 tok)** | 18.6 tok/s (53.7ms) | **23.2 tok/s** (43.1ms) | **+25% faster** |
| **Medium decode (1K tok)** | 13.8 tok/s (72.6ms) | **16.8 tok/s** (59.5ms) | **+22% faster** |
| **Long decode (2K/4K tok)** | 12.2 tok/s (81.7ms) | **16.4 tok/s** (61.1ms) | **+34% faster** |
| **Peak single-user** | 24 tok/s | 24 tok/s | Same ceiling |
| **Batched peak (0.8 util)** | 90 tok/s | **159 tok/s** | **+77% higher** |
| **Batched peak (0.42 util)** | — | **115 tok/s** | 32K context mode |
| **TTFT (128 tok, single)** | 120 ms | 207 ms | Slower (Triton JIT) |
| **KV cache (0.8 util)** | 62,720 tokens | 140,160 tokens | **2.2x more** |
| **KV cache (0.42 util)** | — | ~45,000 tokens | 32K context mode |
| **Memory footprint** | 5.69 GiB | 3.68 GiB | **35% smaller** |

### Analysis: Qwen3.5-4B on Lunar Lake

- **Fastest single-user decode so far** — 23.4 tok/s at short context, staying at 16.3 tok/s even at 2K context (Qwen3-8B drops to 12.2 at 4K)
- **Decode speed plateau at 1K+** — TPOT stays flat at ~61ms from 1K to 2K context, suggesting the Mamba hybrid attention has a different scaling curve than pure attention
- **Massive batched throughput** — 159 tok/s aggregate at 0.8 util, 115 tok/s even at 0.35 util with 32K context mode
- **0.35 util penalty is minimal** — Single-user decode at 0.35 is identical to 0.42 and 0.8 (21.9→16.1 tok/s). Batched peak drops from 115→80 tok/s as context grows, but TPOT (45→93ms) matches 0.42 exactly. The only real difference: 35K KV cache leaves razor-thin headroom for 32K context
- **TTFT is higher** — 232ms single-user vs 1,710ms batched at 128 tokens. The Triton Intel backend JIT overhead plus chunked prefill queuing with 5 concurrent requests
- **Triton works on Xe2 iGPU** — This is the first confirmed Triton-dependent model running on Lunar Lake. The `fla/ops` linear attention kernels (Flash Linear Attention) execute correctly via triton-xpu's Intel backend
- **Server logs confirm stability** — Steady 16.1-16.8 tok/s generation over extended runs, no degradation or GPU faults

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
    --max-model-len 4096 \
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
