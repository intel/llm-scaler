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
2. **FP16/BF16 base weights** with **online quantization** (`--quantization fp8` or `--quantization int4`), OR **AutoRound INT4** pre-quantized (for models ≤8B)
3. **Steady-state VRAM ≤ ~20GB** (leaving room for OS on 32GB shared memory)

### Recommended Models for 32GB Lunar Lake

| Model | Quantization | VRAM Needed | Context | Notes |
|-------|-------------|-------------|---------|-------|
| **Intel/Qwen3-8B-int4-AutoRound** | AutoRound INT4 | **5.69 GiB** | 8k | **Tested & working** — 17.6 tok/s single, 90 tok/s peak batched |
| Qwen3-8B | FP8 (online) | ~10GB | 32k | Standard attention, online quantization |
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

## Benchmark Results (vLLM SYCL on Lunar Lake)

**Hardware:** MSI Claw 8 AI+ — Intel Core Ultra 7 258V, Arc 140V iGPU, 32GB LPDDR5x (136.5 GB/s)
**Model:** Intel/Qwen3-8B-int4-AutoRound (5.7GB on disk, 5.69 GiB loaded)
**Server config:** `--max-model-len 8192 --gpu-memory-utilization 0.8 --enforce-eager --allow-deprecated-quantization`
**Tool:** `vllm bench serve` with random dataset, request rate 1.0

### Throughput

| Workload | Prompts | Output tok/s | Peak tok/s | Total tok/s |
|----------|---------|-------------|-----------|------------|
| 128 in / 128 out | 5 | 44.8 | 90.0 | 89.6 |
| 1024 in / 1024 out | 5 | 54.0 | 80.0 | 108.0 |
| 4096 in / 2048 out | 10 | 50.2 | 70.0 | 150.5 |

### Latency

| Workload | TTFT (mean) | TPOT (mean) | ITL (median) | P99 ITL |
|----------|------------|------------|-------------|---------|
| 128 in / 128 out | 3,846 ms | 56.7 ms | 56.1 ms | 61.4 ms |
| 1024 in / 1024 out | 5,764 ms | 83.9 ms | 91.2 ms | 110.6 ms |
| 4096 in / 2048 out | 20,978 ms | 186.4 ms | 173.8 ms | 233.5 ms |

### Analysis

- **Single-request generation:** ~17.6 tok/s (1000 / 56.7ms TPOT)
- **Batched throughput scales well:** 90 tok/s peak with 5 concurrent short requests
- **Bandwidth-bound:** TPOT scales ~linearly with KV cache size. Long context (4K+2K) is 3.3x slower than short (128+128) due to LPDDR5x read bandwidth limits
- **TTFT scales with input length:** 3.8s (128 tokens) → 21s (4096 tokens) — prefill throughput ~820 tok/s
- **Memory efficient:** Model uses 5.69 GiB, KV cache stays under 44% even with 10 concurrent long-context requests

### Comparison Notes

- For llama.cpp Vulkan comparison, use the same Qwen3-8B model in GGUF Q4_K_M format
- Expected Vulkan speed: similar ballpark (both are LPDDR5x bandwidth-bound for token generation)
- vLLM advantage: continuous batching, OpenAI-compatible API, higher aggregate throughput

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

---

*Updated: 2026-03-27*
