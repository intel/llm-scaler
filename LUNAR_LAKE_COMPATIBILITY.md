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
| **Triton XPU backend broken on Xe2** | Qwen3.5 (all sizes) | `triton.backends` fails to import; `@triton.jit` kernels don't launch. Qwen3.5 uses fla/linear attention ops written entirely as Triton kernels. |
| **Marlin kernels are CUDA-only** | AWQ, GPTQ (compressed-tensors format) | `gptq_marlin_repack` is an NVIDIA CUDA kernel. AWQ/GPTQ MoE models route to `CompressedTensorsWNA16MarlinMoEMethod`. |
| **Pre-quantized 2x memory** | AutoRound, GPTQ (>14B) | Loading INT4→FP16 intermediate doubles peak memory. 35B models OOM on 32GB. |

### What DOES Work

Only models meeting **all three** criteria work on Lunar Lake XPU:
1. **Standard attention** (not fla/linear attention — avoids Triton dependency)
2. **FP16/BF16 base weights** with **online quantization** (`--quantization fp8` or `--quantization int4`)
3. **Steady-state VRAM ≤ ~20GB** (leaving room for OS on 32GB shared memory)

### Recommended Models for 32GB Lunar Lake

| Model | Quantization | VRAM Needed | Context | Notes |
|-------|-------------|-------------|---------|-------|
| Qwen3-8B | FP8 (online) | ~10GB | 32k | **Best choice** — standard attention, proven to load |
| DeepSeek-R1-Distill-Qwen-7B | FP8 (online) | ~8GB | 32k | Good reasoning |
| Qwen3-14B | INT4 (online) | ~10GB | 16k | Needs `--quantization int4` |
| Qwen3-8B | FP16 | ~18GB | 16k | No quantization loss |

### Models That Do NOT Work

| Model | Format | Failure Mode |
|-------|--------|-------------|
| Qwen3.5-* (any size) | Any | Triton kernel crash (`fla/ops` linear attention) |
| GLM-4.7-flash AWQ | AWQ 4-bit | Marlin CUDA kernel missing |
| GLM-4.7-flash AutoRound INT4 | AutoRound | 27B OOM (>24GB after FP16 unpack) |
| Qwen3.5-35B-A3B GPTQ | GPTQ INT4 | OOM + GPU DEVICE_LOST at 79% loading |
| Qwen3.5-35B-A3B AutoRound | AutoRound INT4 | OOM (14GB on disk → ~28GB peak) |
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
9. **Triton XPU broken on Xe2** — The `triton-xpu` package installs but `triton.backends` fails to import on Lunar Lake. Any model architecture using `@triton.jit` kernels (e.g., Qwen3.5's fla/linear attention) will crash with `TypeError: 'function' object is not subscriptable`. This is a triton-xpu packaging/driver issue, not a vLLM bug.
10. **Marlin kernels CUDA-only** — AWQ and GPTQ models using compressed-tensors format route to Marlin repack kernels (`_C.gptq_marlin_repack`), which are NVIDIA CUDA kernels with no XPU equivalent. Pre-quantized AWQ/GPTQ models cannot be used on XPU.
11. **Download FP16 base models** — The only reliable path on Lunar Lake is downloading FP16/BF16 base weights and using vLLM's online quantization (`--quantization fp8` or `--quantization int4`). This uses layer-by-layer loading without the 2x memory spike of pre-quantized formats.

## Alternative: llama.cpp with Vulkan

For simpler setup without the oneAPI stack, [llama.cpp with Vulkan](https://github.com/MegaStood/OpenClaw-on-MSI-Claw-8) is a proven alternative on Lunar Lake. The SYCL/vLLM path offers advantages for:
- Larger model support (vLLM handles model sharding and KV cache management)
- OpenAI-compatible API serving
- FP8/INT4 dynamic online quantization
- Multimodal model support

---

*Updated: 2026-03-26*
