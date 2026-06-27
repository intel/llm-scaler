# [Bug] GLM-4.7-flash MLA not activated on XPU — 3 fixes needed (whitelist + backend routing + import)

## Summary

`vllm_for_multi_arc.patch` adds full `Glm4MoeLiteMLAAttention` support (subclassing `DeepseekV2MLAAttention`) but GLM-4.7-flash never actually uses MLA on XPU due to three bugs:

1. **`"glm4_moe_lite"` missing from `is_deepseek_mla()` whitelist** — The patch creates the attention class but never adds the model type to the whitelist, so `use_mla` is always `False`
2. **XPU platform doesn't route MLA to TRITON_MLA backend** — `get_attn_backend_cls()` in `platforms/xpu.py` has no `use_mla` check, so MLA requests fall through to `FLASH_ATTN` which doesn't accept MLA kwargs
3. **MLA common backend imports CUDA-only `flash_attn_varlen_func`** — `mla/common.py` imports from `vllm_flash_attn` (CUDA) but XPU needs to import from `fa_utils` (IPEX-backed)

All three fixes are trivial — one-line additions each. A pure Triton MLA backend (`v1/attention/backends/mla/triton_mla.py`) already exists with no CUDA dependencies, and `triton-xpu` is available on Intel GPUs.

**Verified working:** With all 3 fixes applied, MLA KV compression is confirmed on XPU — KV cache drops from 3.67 GiB to 0.21 GiB for 4096 tokens (17.5× reduction). See patch: `vllm/patches/glm4_moe_lite_mla_xpu.patch`.

## Environment

- **Hardware:** Intel Lunar Lake (Arc 140V Xe2 iGPU, 32GB shared LPDDR5x) and Intel Arc B60 (24GB)
- **vLLM:** 0.14.1.dev0+gb17039bcc.d20260326 (XPU backend)
- **oneAPI:** 2025.3 (compiler 2025.3.3)
- **Model:** `Intel/glm-4.7-flash-int4-autoround` (30B-A3B MoE, 47 layers, `kv_lora_rank=512`, `qk_rope_head_dim=64`)

## Bug 1: MLA Whitelist Omission

### Expected behavior
`vllm_for_multi_arc.patch` registers `Glm4MoeLiteForCausalLM` and creates `Glm4MoeLiteMLAAttention(DeepseekV2MLAAttention)`. Setting `VLLM_MLA_DISABLE=0` should enable MLA for GLM-4.7-flash.

### Actual behavior
`VLLM_MLA_DISABLE=0` has no effect. The `use_mla` flag is always `False` because `"glm4_moe_lite"` is not in the MLA model whitelist.

### Root cause
In `vllm/transformers_utils/model_arch_config_convertor.py`, the `is_deepseek_mla()` function checks if the model type is in a hardcoded list:
```python
def is_deepseek_mla(model_config) -> bool:
    return model_config.model_type in (
        "deepseek_v2",
        "deepseek_v3",
        "deepseek_v32",
        "deepseek_mtp",
        # "glm4_moe_lite" is MISSING here
        "kimi_k2",
        "kimi_linear",
        "longcat_flash",
        ...
    )
```
The patch registers the `Glm4MoeLiteForCausalLM` architecture and creates `Glm4MoeLiteMLAAttention` but never adds `"glm4_moe_lite"` to this list. Result: `model_config.use_mla` returns `False`, and the decoder layer always falls back to `Glm4MoeLiteAttention` (uncompressed KV).

### Fix
Add `"glm4_moe_lite"` to the `is_deepseek_mla()` model type list:
```python
        "deepseek_mtp",
        "glm4_moe_lite",  # <-- ADD THIS
        "kimi_k2",
```

## Bug 2: XPU Platform Missing MLA Backend Routing

### Expected behavior
When `use_mla=True`, the XPU platform should route to the `TRITON_MLA` backend (which exists at `vllm/v1/attention/backends/mla/triton_mla.py` and uses pure Triton kernels — no CUDA dependencies).

### Actual behavior
`get_attn_backend_cls()` in `vllm/platforms/xpu.py` has no `use_mla` check. MLA requests fall through to `FLASH_ATTN`, which crashes:
```
TypeError: FlashAttentionImpl.__init__() got an unexpected keyword argument 'q_lora_rank'
```

Note: The XPU platform already has MLA-aware code (line ~181: disables chunked prefill, adjusts batched tokens for MLA) — but the backend selector itself doesn't route to an MLA backend.

### Fix
Add MLA routing before the existing backend checks in `get_attn_backend_cls()`:
```python
        if attn_selector_config.use_sparse:
            raise NotImplementedError("Sparse Attention is not supported on XPU.")
        if attn_selector_config.use_mla:  # <-- ADD THIS
            logger.info_once("Using Triton MLA backend for MLA attention on XPU.")
            return AttentionBackendEnum.TRITON_MLA.get_path()
        if selected_backend == AttentionBackendEnum.TRITON_ATTN:
```

### Impact
Without MLA, KV cache per token is:
```
47 layers × 20 heads × 256 dim × 2 bytes = 940 KB/token (full expanded KV)
```

With MLA it would be:
```
47 layers × (512 kv_lora_rank + 64 qk_rope_head_dim) × 2 bytes = 53 KB/token (18× smaller)
```

| Metric | Without MLA | With MLA |
|--------|-------------|----------|
| KV cache per token | 940 KB | 53 KB |
| 32K context KV cache | ~29 GiB | ~1.66 GiB |
| Feasible on B60 (24GB)? | No | Yes |
| Feasible on Lunar Lake (32GB shared)? | No | Yes |

This makes GLM-4.7-flash (and any future MLA model like DeepSeek-V2/V3) impractical on **any** single Intel GPU without these two fixes. Even B60 with 24GB dedicated VRAM cannot fit a 32K context window without MLA compression.

## Bug 3: Duplicate `llama_4_scaling` in `mla.py`

The patch adds a duplicate `"llama_4_scaling"` → `"yarn_scaling"` entry in `mla.py`'s `_ROPE_DICT`. This causes a `SyntaxError` if both dict entries are evaluated. The second entry should likely be a different scaling type or removed.

## Suggested Fixes

1. Add `"glm4_moe_lite"` to `is_deepseek_mla()` in `vllm/transformers_utils/model_arch_config_convertor.py`
2. Add `use_mla` → `TRITON_MLA` routing in `vllm/platforms/xpu.py` `get_attn_backend_cls()`
3. Add `elif current_platform.is_xpu():` import of `flash_attn_varlen_func` from `fa_utils` in `vllm/v1/attention/backends/mla/common.py`
4. Remove the duplicate `llama_4_scaling` entry in `mla.py`

Fixes 1-3 are one-line additions each. The TRITON_MLA backend already exists and uses pure Triton kernels compatible with `triton-xpu`. A complete patch is available at `vllm/patches/glm4_moe_lite_mla_xpu.patch`.

## Reproduction Steps

```bash
# 1. Launch GLM-4.7-flash — MLA not activated (whitelist bug)
VLLM_MLA_DISABLE=0 vllm serve Intel/glm-4.7-flash-int4-autoround \
    --gpu-memory-utilization 0.7 \
    --max-model-len 4096 --enforce-eager --allow-deprecated-quantization
# Observe: "Using Flash Attention backend" (NOT Triton MLA), KV cache 3.67 GiB → OOM

# 2. Apply Fix 1 only (add to whitelist), re-launch
# Observe: "MLA is enabled on non-GPU platform" then crash:
# TypeError: FlashAttentionImpl.__init__() got an unexpected keyword argument 'q_lora_rank'

# 3. Apply Fix 1 + Fix 2, re-launch
# Observe: "Using Triton MLA backend" then crash:
# NameError: flash_attn_varlen_func is not defined (CUDA-only import)

# 4. Apply all 3 fixes, re-launch
# Observe: MLA activated, KV cache 0.21 GiB for 4096 tokens (17.5× reduction)
# NOTE: Model still crashes during MoE warmup (marlin_shuffle_weight OOM) —
# this is a separate MoE issue affecting all 30B-A3B models on memory-constrained devices.
```

## Related: MoE `marlin_shuffle_weight` OOM (Separate Issue)

Even with MLA working (KV cache reduced from 3.67 GiB to 0.21 GiB), GLM-4.7-flash crashes during the warmup forward pass due to `marlin_shuffle_weight` in IPEX's MoE expert reshuffling:
```
RuntimeError: level_zero backend failed with error: 39 (UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY)
```
at `intel_extension_for_pytorch/transformers/models/xpu/fusions/linear_fusion.py:181`.

### Attempted workarounds (all failed on 32GB shared memory):

| Approach | Result |
|----------|--------|
| **CPU-side shuffle** — move weights to CPU, shuffle, move back | Swap absorbs CPU temps (11 GiB swap), but `.to(xpu)` OOMs — old + new coexist briefly in GPU-mapped memory |
| **Per-expert processing** — shuffle one expert at a time | Reduces CPU peak but result tensor `[E, k//8, N]` still full-size on return to GPU |
| **`empty_cache()` before shuffle** | Frees ~0.5 GiB fragmented GPU memory — insufficient headroom |
| **`IPEX_MOE_GEMM_NATIVE=1`** — bypass Marlin kernel | Native path can't handle INT4 packed weights (dtype mismatch BFloat16 != int) |

### Root cause
The 16.55 GiB model leaves only ~4 GiB headroom in 32GB shared memory. The `marlin_shuffle_weight` function creates a full-size output tensor per weight matrix while the original still exists — briefly requiring 2× one layer's MoE weights on GPU. On discrete GPUs this is fine (separate VRAM pool), but on shared memory systems the model + shuffle overhead exceeds the single pool.

### Suggested fix for IPEX
1. **Pre-shuffle weights offline** — save already-shuffled INT4 weights to disk, skip runtime reshuffling
2. **True in-place shuffle** — overwrite the original weight tensor block-by-block instead of allocating a new one
3. **Fix `IPEX_MOE_GEMM_NATIVE=1`** — make the native MoE GEMM path handle INT4 packed weights properly

For full investigation details (5 approaches tested, 5 future ideas), see [`issues/glm4_moe_lite_int4_xpu_marlin_shuffle.md`](glm4_moe_lite_int4_xpu_marlin_shuffle.md).
