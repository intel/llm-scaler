# GLM-4.7-Flash MoE MLA on Intel Lunar Lake XPU: Bugs and Fixes

**Model**: GLM-4.7-Flash (INT4 AutoRound, MoE + MLA architecture)
**Hardware**: Intel Lunar Lake shared-memory iGPU (28.6 GiB LPDDR5x)
**Software**: vLLM v0.14.1, IPEX 2.7.10+xpu
**Date**: 2026-04-15

## Model Configuration Reference

- **Architecture**: 47 layers, 64 MoE experts per layer
- **Attention**: MLA (Multi-head Latent Attention) with kv_lora_rank=512, qk_rope_head_dim=64
- **Disk size**: ~17 GiB INT4 quantized
- **MLA KV cache**: 1 head x 576 x 2 bytes = 1152 bytes/token/layer (11.1x more efficient than standard MHA)
- **Config key for top-k routing**: `top_k_experts` (non-standard)

---

## Bug A: AutoRound FusedMoE routing missing for XPU/IPEX

**File**: `vllm/model_executor/layers/quantization/auto_round.py`
**Function**: `apply_ipex_quant_layer()`

### Problem

`apply_ipex_quant_layer()` only handles `LinearBase` and `ParallelLMHead` instances. When a `FusedMoE` layer is passed, it falls through to `return None`, which causes the caller to fall back to CUDA's `GPTQMarlinMoEMethod`. That method calls `marlin_shuffle_weight` on XPU, which triggers a `DEVICE_LOST` error because the CUDA Marlin kernel is not available on XPU.

### Fix

Add an `elif isinstance(layer, FusedMoE)` branch after the `LinearBase` block that routes to `XPUGPTQMarlinMoEMethod` with a properly constructed `IPEXConfig`.

### Patch

[autoround_fusedmoe_ipex_routing.patch](../vllm/patches/autoround_fusedmoe_ipex_routing.patch)

### Upstream

vLLM

---

## Bug B: `top_k_experts` config key missing from MoE lookup

**File**: `vllm/model_executor/models/transformers/moe.py`, line 187
**Function**: `MoEMixin.recursive_replace()`

### Problem

GLM-4.7-Flash uses `top_k_experts` as the config key for the number of experts per token. The `getattr_iter` call only checks `["num_experts_per_tok", "top_k"]`, so it returns `None`, and the subsequent `assert top_k is not None` fails.

### Fix

Add `"top_k_experts"` to the lookup list: `["num_experts_per_tok", "top_k", "top_k_experts"]`.

### Patch

[gemma4_moe_top_k_experts.patch](../vllm/patches/gemma4_moe_top_k_experts.patch)

### Upstream

vLLM

---

## Bug C: GPTQ non-aligned dimensions cause ValueError

**File**: `vllm/model_executor/layers/quantization/gptq.py`, line 251
**Function**: `GPTQLinearMethod.create_weights()`

### Problem

The original code raises a strict `ValueError` when `input_size_per_partition % group_size != 0`. Models like Gemma 4 have non-aligned dimensions (e.g., 704 % 128 = 64, 2112 % 128 = 64), which triggers this error even though the quantization can work correctly with ceiling division.

### Fix

Replace the `raise ValueError` with `logger.warning` and use `math.ceil` for computing `scale_and_zero_size` and `qweight` sizes. This allows non-aligned dimensions to proceed with the correct padded sizes.

### Patch

[gptq_math_ceil_alignment.patch](../vllm/patches/gptq_math_ceil_alignment.patch)

### Upstream

vLLM

---

## Bug D: MLA `return_attn_probs` unsupported by IPEX flash_attn

**File**: `vllm/v1/attention/backends/mla/common.py`, lines 1384-1389
**Function**: `_flash_attn_varlen_diff_headdims()`

### Problem

When `is_vllm_fa` is `False` on XPU, the code falls into the ROCm path which passes `return_attn_probs` to the flash attention function. IPEX's `flash_attn` implementation does not support the `return_attn_probs` parameter, causing a `TypeError`.

### Fix

Add an `elif current_platform.is_xpu(): pass` branch before the ROCm branch to skip the unsupported parameter on XPU.

### Patch

[mla_xpu_return_attn_probs.patch](../vllm/patches/mla_xpu_return_attn_probs.patch)

### Upstream

vLLM

---

## Bug E: IPEX `marlin_shuffle_weight` causes DEVICE_LOST on shared-memory iGPU

**File (IPEX)**: `intel_extension_for_pytorch/transformers/models/xpu/fusions/linear_fusion.py`, lines 151-153, 169-208
**File (vLLM)**: `vllm/model_executor/layers/quantization/ipex_quant.py`, `XPUGPTQMarlinMoEMethod.process_weights_after_loading()`

### Problem

`GatedMLPMOE.__init__()` stores weights but defers the Marlin shuffle to `init_on_device()` (lazy, triggered on first forward pass). By that point, the KV cache has been allocated and the XPU Level Zero allocator has only ~0.9 GiB headroom. The per-expert fancy indexing in `marlin_shuffle_weight` creates ~96 MiB temporaries per expert times 64 experts, exhausting available memory and causing `DEVICE_LOST`.

### Fix (two parts)

**vLLM side** (`XPUGPTQMarlinMoEMethod.process_weights_after_loading()`):
1. Move weights to CPU
2. Free XPU originals and call `empty_cache()`
3. Create `GatedMLPMOE` on CPU (shuffle runs using the uncapped CPU allocator)
4. Transfer shuffled weights back to XPU
5. Set `_marlin_shuffled = True` flag on W13 and W2

**IPEX side** (`_MoEGEMMXpu.__init__()`):
1. Check `getattr(W13, '_marlin_shuffled', False)`
2. If True, skip `marlin_shuffle_weight` and `torch.xpu.empty_cache()`

### Patches

- [xpu_gptq_moe_cpu_shuffle.patch](../vllm/patches/xpu_gptq_moe_cpu_shuffle.patch) (vLLM side)
- [ipex_marlin_shuffle_skip_preshuffled.patch](../vllm/patches/ipex_marlin_shuffle_skip_preshuffled.patch) (IPEX side)

### Upstream

IPEX (root cause) + vLLM (workaround)

---

## Bug F: Warmup dummy run crashes on XPU

**File**: `vllm/v1/worker/gpu_worker.py`, line 541
**Function**: `compile_or_warm_up_model()`

### Problem

`compile_or_warm_up_model()` always runs `_dummy_run` for sampler preallocation, even when `enforce_eager` is set and `VLLM_SKIP_PROFILE_RUN=1`. The dummy forward pass triggers `init_on_device` in IPEX modules, which causes `OUT_OF_RESOURCES` on the memory-constrained XPU shared-memory iGPU.

### Fix

Override `compile_or_warm_up_model` in `XPUWorker` to skip the entire warmup sequence when `VLLM_SKIP_PROFILE_RUN=1` is set.

### Patch

[xpu_skip_warmup_dummy_run.patch](../vllm/patches/xpu_skip_warmup_dummy_run.patch)

### Upstream

vLLM (XPU platform)

---

## Bug G: XPU allocator budget vs shared memory (architectural insight)

### Problem

The XPU Level Zero allocator is capped at `gpu_memory_utilization * total_memory` (e.g., 0.8 x 28.6 GiB = 22.9 GiB), while the CPU allocator has no corresponding cap. Both allocators share the same physical LPDDR5x memory pool on Lunar Lake's shared-memory architecture.

### Insight

The CPU shuffle workaround for Bug E exploits this asymmetry: by shuffling weights on CPU (uncapped allocator), then transferring the result to XPU (within budget since the original XPU tensors were freed first), the operation succeeds without exceeding the XPU allocator's artificial cap.

This is not a bug per se, but an architectural constraint that must be understood when working with shared-memory iGPUs.

### Upstream

N/A (architectural insight)

---

## Status

| Bug | Status | Applied to |
|-----|--------|-----------|
| A | Fixed | site-packages |
| B | Fixed | site-packages |
| C | Fixed | site-packages |
| D | Fixed | site-packages |
| E | Fixed | site-packages |
| F | Fixed | site-packages |
| G | N/A | architectural insight |

**End-to-end test result**: Model loads (16.12 GiB), KV cache allocates (116K tokens), server starts. First inference pending clean reboot (swap thrashing from debug iterations).
