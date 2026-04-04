# GLM-4.7-Flash INT4 MoE: Marlin Shuffle DEVICE_LOST on Lunar Lake XPU

## Status: BLOCKED - XPU driver limitation

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
| **INT4 AutoRound** | Yes | **Yes** — must reshuffle at runtime | **No** — OOM during shuffle |
| **GPTQ** | No | N/A — requires CUDA-only Marlin kernels | No |
| **AWQ** | No | N/A — requires CUDA-only Marlin kernels | No |
| **FP8** | Yes | No — uses `GatedMLPMOE(is_fp8=True)` | No — 30B model → ~30 GiB, won't fit |
| **GGUF** | No (vLLM) | N/A — vLLM GGUF is CUDA-only | Yes — via llama.cpp with SYCL backend |

**The root cause is not MoE itself, but INT4 AutoRound → marlin_shuffle_weight.**
MXFP4 stores weights already in the Marlin kernel's expected layout, so no runtime
reshuffling is needed. If an MXFP4 version of GLM-4.7-Flash existed, it would likely
load without OOM, just like gpt-oss-20b.

### Implication for shared-memory iGPUs
On shared-memory systems, **MXFP4 is the only vLLM-compatible quantization format
that works for MoE models**. INT4/GPTQ/AWQ all require either reshuffling (OOM) or
CUDA-only kernels (unsupported). The alternative path is llama.cpp GGUF via SYCL.

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

## Environment
- Intel Core Ultra 7 258V (Lunar Lake), Arc 140V iGPU
- 32 GB LPDDR5x shared memory (28.57 GiB usable by GPU)
- Intel Core Ultra 7 155H (Meteor Lake), Xe-LPG iGPU
- 16 GB LPDDR5 shared memory
- vLLM 0.14.1.dev0, IPEX XPU
- Model: glm-4.7-flash-int4-autoround (16.52 GB loaded)
