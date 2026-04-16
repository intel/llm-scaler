# GLM-4.7-Flash INT4 MoE: Marlin Shuffle DEVICE_LOST on Lunar Lake XPU

## Status: Bug E (shuffle OOM) FIXED — Bug H (Level Zero resource exhaustion) OPEN

## Problem

When serving `glm-4.7-flash-int4-autoround` via vLLM on Intel Arc 140V (Lunar Lake),
the IPEX Marlin weight shuffle crashes the Level Zero backend during model warmup.

```
RuntimeError: level_zero backend failed with error: 20 (UR_RESULT_ERROR_DEVICE_LOST)
```

The crash occurs in `_IPEXGatedMLPMOEXPU.__init__()` at:
```
intel_extension_for_pytorch/transformers/models/xpu/fusions/linear_fusion.py:152
  self.W13 = self.marlin_shuffle_weight(self.W13)
```

## Root Cause

The `marlin_shuffle_weight()` function reshuffles INT4 packed weights into Marlin
kernel layout. This happens **lazily at first forward** via `init_on_device()`,
at which point the model (16.52 GB) is already loaded on XPU. The iGPU shares
system memory (30 GB total), leaving very little headroom.

The function iterates over all 65 experts per layer (64 routed + 1 shared),
performing complex bit manipulation on XPU. With 46 MoE layers, this is ~3000
shuffle operations that overwhelm the iGPU.

## Approaches Tested (all failed)

### 1. CPU-side shuffle (whole tensor)
Move entire `qweight` to CPU, shuffle there, move back.
- **Result**: `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` - the `.cpu()` call itself
  needs staging memory that isn't available.

### 2. CPU-side shuffle (per-expert)
Move one expert at a time to CPU via `qweight[e].cpu()`.
- **Result**: `UR_RESULT_ERROR_DEVICE_LOST` - even a single expert transfer
  crashes the device under memory pressure.

### 3. Sync + empty_cache before transfer
Add `torch.xpu.synchronize()` and `torch.xpu.empty_cache()` before CPU transfer.
- **Result**: Still `DEVICE_LOST` - the driver instability isn't from pending ops.

### 4. Pre-shuffle in process_weights_after_loading (on CPU before XPU load)
Perform Marlin shuffle in vLLM's `process_weights_after_loading()` while weights
are still on CPU, then pass pre-shuffled weights to IPEX GatedMLPMOE.
- **Result**: `UR_RESULT_ERROR_OUT_OF_RESOURCES` - the pre-shuffled weights + model
  together exceed available memory when loaded to XPU. The shuffle creates a second
  copy of the weight tensor temporarily.

### 5. IPEX_MOE_GEMM_NATIVE=1 (bypass Marlin, use native GEMM)
Skip Marlin shuffle entirely, use IPEX native MoE GEMM path.
- **Result**: `expected self and mat2 to have the same dtype, but got: c10::BFloat16 != int`
  The native path doesn't dequantize INT4 packed weights - it expects FP16/BF16.

## Key Discovery: MXFP4 Bypasses the Reshuffle Entirely

gpt-oss-20b is also an MoE model (20B params, 3.6B activated per token) and loads
fine on the same 32GB Lunar Lake — even **two instances simultaneously**. The difference
is the quantization format:

| Format | Works on XPU? | Needs marlin_shuffle? | MoE on 32GB shared? |
|--------|--------------|----------------------|---------------------|
| **MXFP4** | Yes | **No** — weights are pre-formatted for Marlin | **Yes** (gpt-oss-20b proves it) |
| **INT4 AutoRound** | Yes | **Yes** — must reshuffle at runtime | Bug E fixed (CPU shuffle), Bug H open (40+ layers) |
| **GPTQ** | No | N/A — requires CUDA-only Marlin kernels | No |
| **AWQ** | No | N/A — requires CUDA-only Marlin kernels | No |
| **FP8** | Yes | No — uses `GatedMLPMOE(is_fp8=True)` | No — 30B model → ~30 GiB, won't fit |
| **GGUF** | No (vLLM) | N/A — vLLM GGUF is CUDA-only | Yes — via llama.cpp with SYCL backend |

**Two separate issues for INT4 MoE on shared-memory iGPU:**
1. **Bug E (FIXED)**: marlin_shuffle_weight OOM during init — solved by CPU shuffle
2. **Bug H (OPEN)**: xetla `group_mm_int4_out_marlin` kernel crashes on Xe2-LPG —
   the very first `torch.xpu.moe_gemm(is_int4=True)` causes DEVICE_LOST.
   Not a resource accumulation issue — the INT4 MoE GEMM kernel itself is
   unsupported on Lunar Lake's iGPU architecture.

MXFP4 avoids both issues: no shuffle needed (Bug E), and uses a different
xetla kernel (`group_mm_mxfp4_out_marlin`) that works on Xe2-LPG.

### Implication for shared-memory iGPUs
On shared-memory systems with 40+ MoE layers, **MXFP4 is the only vLLM-compatible
quantization format that works for MoE models** without Level Zero env var tuning.
INT4/GPTQ/AWQ all require either reshuffling (Bug E, now fixed) or CUDA-only kernels
(unsupported), and still hit Bug H on 40+ layer models. The alternative path is
llama.cpp GGUF via SYCL backend.

## Bug H: INT4 MoE GEMM Kernel Broken on Lunar Lake Xe2-LPG

**Status: OPEN — IPEX `group_mm_int4_out_marlin` xetla kernel unsupported on Xe2-LPG**

### Distinction from Bug E

Bug E (DEVICE_LOST during marlin_shuffle) was solved by the CPU shuffle workaround.
Bug H is a **separate** kernel compatibility issue: the IPEX INT4 MoE GEMM kernel
itself does not work on Lunar Lake's Xe2-LPG iGPU architecture.

### Key diagnostic finding

With per-layer `torch.xpu.synchronize()` instrumentation in `fused_moe_experts()`:

```
[MoE-DIAG] Layer call 1: W13 GEMM FAILED: DEVICE_LOST
```

**The very first `torch.xpu.moe_gemm(..., is_int4=True)` call crashes.**
It is not a resource accumulation problem from many layers — the xetla INT4
MoE GEMM kernel itself is broken on this architecture.

### Affected models

| Model | Layers | Quant | Kernel Path | Result |
|-------|--------|-------|-------------|--------|
| GPT-OSS-20B | 24 | MXFP4 | `group_mm_mxfp4_out_marlin` | **Works** |
| Qwen3.5-35B-A3B | 40 | INT4 | `group_mm_int4_out_marlin` | **DEVICE_LOST** |
| GLM-4.7-Flash | 47 | INT4 | `group_mm_int4_out_marlin` | **DEVICE_LOST** |

The "40+ layers" correlation was a red herring — the crash happens at layer 1.
ALL INT4 MoE models fail regardless of layer count. MXFP4 works because it uses
a different xetla kernel (`group_mm_mxfp4_out_marlin`).

### Root cause: xetla INT4 kernel not validated for Xe2-LPG

The IPEX `xetla_arch.h` maps Lunar Lake to `gpu_arch::XeHpc` (same as PVC/BMG
data center GPUs), with `gpu_arch::Xe2Lpg` commented out:

```cpp
// csrc/gpu/aten/operators/xetla/kernels/xetla_arch.h
// LNL mapped to XeHpc — Xe2Lpg alternative is commented out
```

This means the INT4 MoE GEMM kernel dispatches a tile configuration designed for
data center GPUs (PVC with HBM, many execution units) onto a mobile iGPU with
fundamentally different capabilities (shared LPDDR5x, fewer EUs, different cache
hierarchy). The kernel likely uses hardware features or tile sizes that Xe2-LPG
doesn't support, causing DEVICE_LOST on first dispatch.

The MXFP4 kernel has a different tile implementation that happens to be compatible
with Xe2-LPG — hence GPT-OSS-20B works.

### What was ruled out

1. **Resource pool exhaustion** — crash happens at layer 1, call 1 (not accumulation)
2. **Level Zero env vars** — `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1` +
   `COMMANDLISTS_CLEANUP_THRESHOLD=5` + `BATCH_SIZE=4` had no effect
3. **Memory pressure** — model loads fine, KV cache allocates, server starts
4. **Data corruption** — random synthetic tensors also crash (see test script)

### Minimal reproducer

```bash
python vllm/test/test_int4_moe_gemm_xpu.py
```

Tests `torch.xpu.moe_gemm()` with MXFP4 (expected: pass) and INT4 (expected:
DEVICE_LOST on Lunar Lake) using small synthetic tensors.

### Potential fixes

#### A. Fix xetla arch mapping for Lunar Lake (requires IPEX C++ rebuild)

Change the architecture mapping in `xetla_arch.h` from `XeHpc` to `Xe2Lpg` for
Lunar Lake device IDs, and ensure the INT4 MoE GEMM kernel has a valid tile
configuration for Xe2-LPG. **Not feasible** on the archived IPEX repo without
forking and rebuilding.

#### B. Use MXFP4 instead of INT4

Re-quantize models to MXFP4 format. MXFP4 uses `group_mm_mxfp4_out_marlin` which
works on Xe2-LPG. Requires MXFP4 quantization tooling or community-quantized models.

#### C. Use llama.cpp GGUF via SYCL backend

llama.cpp handles INT4 dequantization with its own SYCL kernels (not IPEX xetla),
which are validated on consumer GPUs including Lunar Lake.

#### D. Wait for vLLM v0.19+ XPU INT4 kernel

vllm-xpu-kernels v0.1.5 has oneDNN-backed INT4 GEMM (W4A16) that doesn't use
IPEX's xetla kernels. Once the `XPUExpertsInt4` class is wired up in vLLM
(tracked in RFC #33214), this would provide a working INT4 MoE path on XPU.

### IPEX code references

| File | What it does |
|------|-------------|
| `xetla_arch.h` | Architecture mapping — LNL → XeHpc (wrong?) |
| `moe_gemm.cpp:188` | C++ `fused_moe_gemm_persistent` — dispatches xetla kernel |
| `group_gemm_int4_marlin_impl.h` | INT4 xetla kernel template — tile policies |
| `group_gemm_mxfp4_marlin_impl.h` | MXFP4 xetla kernel template — works on Xe2 |
| `intrinsic/__init__.py:443` | Python `moe_gemm()` — routes to INT4 or MXFP4 |
| `linear_fusion.py:235-291` | `fused_moe_experts()` — calls moe_gemm twice |

---

## Ideas for Future Investigation

### A. Patch IPEX native path to dequantize INT4
The native `torch.xpu.moe_gemm` with `use_native=True, is_int4=True` falls back to
plain `torch.mm` which can't handle packed INT4. If IPEX added INT4 dequantization
in the native path (unpack INT4 -> BF16 -> GEMM), it would bypass Marlin entirely.
- **File**: `intel_extension_for_pytorch/xpu/intrinsic/__init__.py` line ~506
- **Effort**: Medium - need to add dequant logic per-expert in the moe_gemm native path

### B. Reduce model memory to make room for shuffle
If model memory could be reduced from 16.52 GB to ~14 GB, there would be enough
headroom for the Marlin shuffle. Options:
- Use a more aggressive quantization (e.g. 2-bit or mixed precision)
- Quantize the dense layer (layer 0) which is currently FP16
- Reduce the number of loaded experts (expert parallelism / offloading)

### C. Streaming shuffle during weight loading
Instead of loading all weights first then shuffling, interleave:
load layer N weights -> shuffle layer N -> move to XPU -> free CPU copy -> next layer.
This requires modifying the vLLM weight loader to call `process_weights_after_loading`
per-layer instead of at the end.
- **File**: `vllm/model_executor/model_loader/default_loader.py`

### D. Use GGUF via llama.cpp instead
The `GLM-4.7-Flash-Q4_K_M.gguf` (18 GB) in `/shared/models/gguf/` can be served
via llama.cpp with SYCL backend, which handles INT4 dequantization differently
and doesn't need Marlin shuffle.

### E. FP8 quantization (ruled out for 32GB)
Re-quantize the model to FP8 instead of INT4. The IPEX FP8 MoE path
(`XPUFp8MoEMethod`) doesn't need Marlin shuffle. Uses `GatedMLPMOE(..., is_fp8=True)`.
However, GLM-4.7-Flash has ~30B parameters; FP8 = 1 byte/param = ~30 GiB, which
exceeds the 30.9 GiB shared memory pool. Only viable on discrete GPUs with ≥32GB VRAM.

### F. MXFP4 re-quantization (most promising)
If GLM-4.7-Flash could be quantized to MXFP4 format, it would bypass marlin_shuffle
entirely — proven by gpt-oss-20b (also MoE) loading without issues. MXFP4 stores
weights pre-formatted for the Marlin kernel layout. This would need Intel's MXFP4
quantization tooling or a community-quantized model on HuggingFace.

## Related Files

| File | Role |
|------|------|
| `intel_extension_for_pytorch/transformers/models/xpu/fusions/linear_fusion.py:151-208` | Marlin shuffle init + function |
| `intel_extension_for_pytorch/xpu/intrinsic/__init__.py:~506` | Native MoE GEMM fallback |
| `vllm/model_executor/layers/quantization/ipex_quant.py:742-757` | vLLM creates IPEX GatedMLPMOE |
| `vllm/model_executor/layers/fused_moe/layer.py:1995` | MoE forward dispatch |

## GPU Memory and KV Cache Sizing on Shared Memory

### Usable GPU memory ≠ physical RAM
On shared-memory iGPUs, the advertised 32 GB LPDDR5x does not all go to the GPU.
The Level Zero driver, firmware, and display framebuffer reserve a portion. The actual
usable GPU memory can be queried at runtime:

```python
import torch
torch.xpu.get_device_properties(0).total_memory  # returns bytes
```

On our MSI Claw 8 AI+: **28.57 GiB usable** out of 32 GB physical. This number can
vary with different driver versions or runtime stacks.

### gpu-memory-utilization is a fraction of usable GPU memory
vLLM's `--gpu-memory-utilization` is applied to the **usable** GPU memory (28.57 GiB),
not the physical RAM. The KV cache budget is:

```
KV cache = (usable_gpu × gpu_util) - peak_model_memory
```

For gpt-oss-20b at 0.6 utilization:
```
28.57 × 0.6 = 17.14 GiB budget
17.14 - 13.95 = 3.19 GiB KV cache
KV per token ≈ 48 KB → 69,696 tokens capacity
```

### VLLM_SKIP_PROFILE_RUN overhead multiplier matters
The skip-profile patch estimates peak memory as `memory_allocated × multiplier`. The
multiplier directly affects KV cache capacity:

| Multiplier | Peak estimate | KV cache (0.6 util) | Tokens |
|-----------|--------------|--------------------:|-------:|
| 1.20 (default) | 15.94 GiB | 1.20 GiB | 26,176 |
| 1.05 (tuned) | 13.95 GiB | 3.19 GiB | 69,696 |

The 1.05x multiplier is safe for gpt-oss-20b (verified stable under load) and yields
**2.67× more KV cache**. For models with larger activation spikes during inference,
a higher multiplier may be needed.

### Multi-model memory planning
When running multiple models simultaneously, `gpu-memory-utilization` values must sum
to less than 1.0 (since each is a fraction of the same shared pool):

```
gpt-oss-20b:    0.55  →  15.7 GiB
Qwen3-ASR-1.7B: 0.15  →   4.3 GiB
Qwen3.5-4B:     0.20  →   5.7 GiB
Total:          0.90  →  25.7 GiB (leaves 2.9 GiB for OS)
```

## Build Parallelism on Shared-Memory Systems

When building vLLM, IPEX, or llama.cpp natively on shared-memory iGPUs, the
compiler can OOM if too many parallel jobs run simultaneously. Use RAM-based
`MAX_JOBS` auto-detection:

```bash
TOTAL_RAM_GB=$(free -g | awk '/^Mem:/{print $2}')
if [ "$TOTAL_RAM_GB" -le 16 ]; then
    export MAX_JOBS=3   # Claw A1M (16GB Meteor Lake)
elif [ "$TOTAL_RAM_GB" -le 32 ]; then
    export MAX_JOBS=6   # Claw 8 AI+ (32GB Lunar Lake)
else
    export MAX_JOBS=8   # Discrete GPU systems with more RAM
fi
cmake --build build -j$MAX_JOBS   # or: pip install with MAX_JOBS set
```

The `MAX_JOBS` environment variable is respected by both CMake and Python
setuptools/pip builds.

## Future Development: Multi-Device Architecture

### Planned Setup

| Device | Role | Hardware |
|--------|------|----------|
| 2x DGX Spark | LLM server (120B+ models) | Grace Blackwell, 128GB each |
| MSI Claw 8 AI+ | Portable LLM (20B MoE) + vLLM serving | Lunar Lake, 32GB |
| MSI Claw A1M | Edge client, voice I/O, lightweight tasks | Meteor Lake, 16GB |

### Claw A1M as Edge Client

The A1M (16GB) is too constrained to run large LLMs and ASR/TTS simultaneously.
Planned architecture splits workloads across devices:

```
A1M (OpenClaw UI) ──→ DGX Spark (120B LLM via vLLM)
    ├── ASR: Whisper on NPU (zero iGPU memory cost)
    ├── ASR: SenseVoice/Qwen2-Audio on CPU (fallback, no NPU op support)
    ├── TTS: local on CPU/iGPU
    └── If standalone: 4B LLM on iGPU + Whisper on NPU
```

### OpenVINO NPU Inference on Meteor Lake

The Intel NPU (Neural Processing Unit) on 155H can offload small models,
freeing iGPU memory for other tasks:

- **Runtime**: OpenVINO 2026.0+ (`pip install openvino`)
- **NPU driver**: intel/linux-npu-driver v1.30.0+
- **Supported on NPU**: Whisper (built-in GenAI pipeline), small vision models
- **CPU/GPU only**: Qwen2-Audio, SenseVoice, Paraformer (complex ops not on NPU)
- **Convert any PyTorch model**: `ov.convert_model()` → run on CPU/GPU/NPU

```python
import openvino_genai as ov_genai
# ASR on NPU — dedicated hardware, no iGPU memory competition
pipe = ov_genai.WhisperPipeline("whisper-base", device="NPU")
result = pipe.generate("audio.wav")
```

Key insight: NPU has its own dedicated compute — it does NOT share iGPU memory.
This makes it ideal for always-on ASR/TTS while the iGPU handles LLM inference.

## vLLM Native Build Steps (from llm-scaler Dockerfile)

### Build Order

The build must follow this order because `--no-build-isolation` means each
step relies on packages installed in the previous step:

```
Step 1: System deps        (oneAPI, compilers, Python 3.12)
Step 2: Set env vars        (VLLM_TARGET_DEVICE=xpu, CPATH for DPCPP)
Step 3: Clone vLLM v0.14.0 + apply vllm_for_multi_arc.patch
Step 4: pip install -r requirements/xpu.txt   ← PyTorch XPU, IPEX, etc.
        pip install arctic-inference==0.1.1
Step 5: pip install --no-build-isolation .     ← builds vLLM against Step 4
Step 6: Build vllm-xpu-kernels (commit 4c83144)
Step 7: Fix triton-xpu==3.6.0
Step 8: Configure production env vars in ~/.bashrc
```

`--no-build-isolation` tells pip to use the already-installed PyTorch XPU
and IPEX from Step 4, instead of creating a fresh isolated build environment
(which would pull CPU PyTorch and break the XPU build).

### Built .so Libraries

| Library | Source | Purpose |
|---------|--------|---------|
| `vllm_int4_for_multi_arc.so` | bigdl-core | INT4 quantization kernel for XPU |
| `vllm_xpu_kernels*.so` | vllm-xpu-kernels | XPU compute kernels (attention, etc.) |
| `libtriton_xpu*.so` | triton-xpu | Triton JIT compiler for XPU |
| IPEX `*.so` libraries | intel-extension-for-pytorch | Marlin kernel, MoE fusion, etc. |

The `vllm_for_multi_arc.patch` (466KB) modifies vLLM's Python source to add
XPU device support. It is NOT a separate .so — it patches the Python code
which then calls into the .so kernels above for actual GPU compute.

Referenced by env var:
```bash
export VLLM_QUANTIZE_Q40_LIB="<python-site>/vllm_int4_for_multi_arc.so"
```

### Docker Build vs Native Build

**Docker build** — everything happens inside the container:
```bash
cd ~/llm-scaler
docker build -f vllm/docker/Dockerfile -t vllm-xpu .
# Run with GPU access:
docker run --device /dev/dri -p 8000:8000 vllm-xpu \
    vllm serve /models/gpt-oss-20b --device xpu
```
- No need to build vLLM locally first — Docker does it all
- Pins to specific versions (reproducible)
- Adds ~15-20GB disk overhead for the image
- GPU access via `--device /dev/dri`
- Models mounted via `-v /shared/models:/models`

**Native build** — installs directly on the host:
```bash
sudo bash ~/llm-scaler/vllm/scripts/install_vllm_native.sh
# Or the post-install script:
sudo bash ~/OpenClaw-on-MSI-Claw-8/scripts/claw-post-install-vllm.sh
```
- No Docker overhead (saves RAM on 16GB machines)
- More flexible (can use any vLLM version)
- Dependencies may conflict with system packages
- Uses MAX_JOBS auto-detection for OOM prevention

**Recommendation:**
- Claw A1M (16GB): **native build** — Docker daemon eats precious RAM
- Claw 8 AI+ (32GB): either works, Docker is cleaner
- DGX Spark (server): **Docker** — reproducible, isolated, easy to update

## Environment
- Intel Core Ultra 7 258V (Lunar Lake), Arc 140V iGPU
- 32 GB LPDDR5x shared memory (28.57 GiB usable by GPU)
- Intel Core Ultra 7 155H (Meteor Lake), Xe-LPG iGPU
- 16 GB LPDDR5 shared memory
- 2x NVIDIA DGX Spark (Grace Blackwell) — home server for large models
- vLLM 0.14.1.dev0, IPEX XPU
- Model: glm-4.7-flash-int4-autoround (16.52 GB loaded)
