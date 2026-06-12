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

Fixed upstream in vLLM source tree. Original patch `autoround_fusedmoe_ipex_routing.patch` removed in commit 279a611.

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

Fixed upstream in vLLM source tree. Original patch `gemma4_moe_top_k_experts.patch` removed in commit 279a611.

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

Fixed upstream in vLLM source tree. Original patch `gptq_math_ceil_alignment.patch` removed in commit 279a611.

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

## Bug H: Level Zero OUT_OF_RESOURCES during inference (HARD BLOCKER — see definitive analysis below)

**Error**: `UR_RESULT_ERROR_OUT_OF_RESOURCES` (Level Zero error 40)
**Location**: Various — sampler, attention, any kernel after resource pool exhaustion

### Problem

The model loads successfully, KV cache allocates, and the server starts. But the first inference request triggers `init_on_device` for all MoE layers, each creating an `_IPEXGatedMLPMOEXPU` object with its own XPU kernel handles and command queue resources. This exhausts the Level Zero driver's internal resource pool for the Lunar Lake iGPU.

The error is **not** memory-related (error 39 = OOM, error 40 = resource exhaustion). It's a driver-level limit on the number of concurrent kernel objects, command lists, or event pools.

### Affected Models

| Model | Layers | Experts | Attention | Quant | Result |
|-------|--------|---------|-----------|-------|--------|
| GPT-OSS-20B | 24 | 32 | GQA | MXFP4 | **Works** |
| Qwen3.5-35B-A3B | 40 | 256 | GQA | INT4 AutoRound | **OUT_OF_RESOURCES** |
| GLM-4.7-Flash | 47 | 64 | MLA+Triton | INT4 AutoRound | **OUT_OF_RESOURCES** |
| Qwen3-VL-30B-A3B | 48 | 128 | GQA+Vision | INT4 AutoRound | **OUT_OF_RESOURCES** |

GPT-OSS-20B works because: (1) fewer layers, (2) MXFP4 uses a different kernel path (no `_IPEXGatedMLPMOEXPU`), (3) GQA uses simpler IPEX attention kernels. All INT4 AutoRound GPTQ MoE models with 40+ layers are blocked.

### Attempted Mitigations

- `ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE`: No effect
- `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1`: No effect — immediate command lists don't reduce kernel object count
- `IPEX_MOE_GEMM_NATIVE=1`: Fixes resource exhaustion but causes dtype mismatch (`BFloat16 != int`) — native GEMM path doesn't support INT4 packed weights
- Reducing `gpu-memory-utilization`: No effect (not a memory issue)
- Different model (Qwen3.5 40L vs GLM 47L): Both fail — even 40 layers exceeds the limit
- Different model sizes (40 vs 47 layers): Both exceed the resource limit

### Root Cause

IPEX's `_IPEXGatedMLPMOEXPU` creates per-layer XPU kernel objects via `init_on_device` (lazy init on first forward). Each layer's MoE fusion allocates Level Zero command queue resources that are never released. With 40+ layers, the cumulative resource usage exceeds the Lunar Lake iGPU's Level Zero resource pool.

GPT-OSS-20B avoids this because MXFP4 quantization uses pre-formatted weights that don't go through `_IPEXGatedMLPMOEXPU` — they use a different, more resource-efficient kernel dispatch.

### Upstream

Intel Level Zero driver / IPEX — needs either:
1. Kernel object reuse across layers (share one `_IPEXGatedMLPMOEXPU` instance)
2. Lazy resource release (free command queue resources after kernel dispatch)
3. Larger resource pool for iGPU in Level Zero driver

---

## Bug I: Attention query/key dtype mismatch (Qwen3.5-35B-A3B)

**Error**: `The datatype of key should be the same as query`
**File**: `intel_extension_for_pytorch/transformers/models/xpu/fusions/mha_fusion.py:599`

### Problem

INT4 GPTQ dequantization produces Float16 query projections, but the KV cache stores BFloat16. IPEX's `chunked_prefill` kernel requires matching dtypes.

Observed: `q=torch.float16, k=torch.bfloat16, v=torch.bfloat16`

### Fix

Cast k/v to match q dtype before calling IPEX flash attention:
```python
# In vllm/_ipex_ops.py, flash_attn_varlen_func (paged attention path):
if q.dtype != k.dtype:
    k = k.to(q.dtype)
    v = v.to(q.dtype)
```

### Upstream

vLLM / IPEX — the dequantization should produce the model's configured dtype (BFloat16), not Float16.

---

## Bug J: ESIMD decode kernel called for non-FP16 query

**File**: `vllm_xpu_kernels/eagle_ops.py`

### Problem

The `eagle_ops.page_attn_decode` ESIMD kernel only supports FP16 query tensors. INT4 GPTQ models produce BF16 queries, causing a kernel crash.

### Fix

Add a `query.dtype == torch.float16` guard before calling the ESIMD decode path:
```python
if query.dtype == torch.float16:
    eagle_ops.page_attn_decode(...)
```

### Patch

[eagle_ops_dtype_check.patch](../vllm/patches/eagle_ops_dtype_check.patch)

### Upstream

vllm-xpu-kernels

---

## Bug K: torch.ops.vllm custom op dispatch incompatible with XPU

**Files**: `vllm/attention/layer.py`, `vllm/model_executor/layers/fused_moe/layer.py`

### Problem

vLLM's `torch.ops.vllm.unified_attention_with_output` and `torch.ops.vllm.moe_forward` custom ops route through a torch-compile custom-op dispatch chain. On XPU with IPEX, this dispatch interacts poorly with the XPU allocator and Level Zero context, causing hangs or incorrect dispatch.

### Fix

Force XPU to bypass the custom ops and call `forward_impl` directly:

- `xpu_attention_direct_call.patch`: Routes attention directly to the backend implementation
- `xpu_moe_direct_call.patch`: Routes MoE directly to the backend implementation

### Patch

[xpu_attention_direct_call.patch](../vllm/patches/xpu_attention_direct_call.patch)
[xpu_moe_direct_call.patch](../vllm/patches/xpu_moe_direct_call.patch)

### Upstream

vLLM — XPU backend should register its own custom op implementations or bypass the dispatch chain.

---

## Status

| Bug | Status | Patch | Blocker? |
|-----|--------|-------|----------|
| A | Fixed upstream | (removed — in source tree) | No |
| B | Fixed upstream | (removed — in source tree) | No |
| C | Fixed upstream | (removed — in source tree) | No |
| D | Fixed | `mla_xpu_return_attn_probs.patch` | No (GLM-4.7 MLA only) |
| E | Fixed | `ipex_marlin_shuffle_skip_preshuffled.patch` | No |
| F | Fixed | `xpu_worker_skip_profile_and_warmup.patch` | No |
| G | N/A | (architectural insight) | No |
| **H** | **Partial** | `xpu_gptq_moe_int4_w4a16.patch` | **YES — perf ceiling 0.9 tok/s** |
| I | Fixed | `ipex_attention_dtype_cast.patch` | No |
| J | Fixed | `eagle_ops_dtype_check.patch` | No |
| K | Fixed | `xpu_attention_direct_call.patch` + `xpu_moe_direct_call.patch` | No |

Bug H detail: Crash is solved (oneDNN bypass of GatedMLPMOE). Correctness is solved (0.999997 corr). But throughput is 0.9 tok/s (88% Python overhead in sequential expert loop). Batched IPEX marlin kernel (`group_mm_int4_out_marlin`) returns zeros/NaN on Xe2 — blocked on Intel kernel fix.

### Next steps

1. **File IPEX upstream bug** — `group_mm_int4_out_marlin` returns all-zero output (BF16) or NaN (FP16) on Xe2-LPG. Include 4-layout test results. This blocks the batched path (~15 tok/s theoretical).
2. **Optimize oneDNN loop** — use IPEX C++ routing ops (`moe_rows_counts`, `moe_scatter`) to reduce Python overhead in the sequential `int4_gemm_w4a16` path. Target: 2-4 tok/s.
3. **Consider MXFP4 re-quantization** — convert INT4 AutoRound weights to MXFP4 format. MXFP4 batched path works (GPT-OSS-20B at 3 tok/s). Quality impact unknown.

### Environment

| Component | Version |
|-----------|---------|
| vLLM | 0.14.1.dev0+gb17039bcc.d20260414 |
| IPEX | 2.7.10+xpu |
| PyTorch | 2.10.0+xpu |
| compute-runtime | 25.48.36300.8 |
| Level Zero loader | 1.26.3 |
| Level Zero driver | 1.14.36300+8 |
| Kernel | 6.19.11-201.nobara.fc43.x86_64 (xe driver) |
| GPU | Intel Arc Graphics (Lunar Lake, Xe2, 28.6 GiB shared LPDDR5x) |

### End-to-end test results

#### GLM-4.7-Flash (INT4 AutoRound, 47 layers, 64 experts, MoE+MLA)

| Stage | Time | Cumulative |
|-------|------|-----------|
| Weight loading | 21s | 0:21 |
| CPU shuffle (47L × 64E) | 3m 12s | 3:33 |
| KV cache + engine init | 1s | 3:34 |
| Server ready | ~17s | **~3:51** |

- Model memory: 16.12 GiB, KV cache: 31,936 tokens (gpu-util=0.65)
- **Inference: FAIL — DEVICE_LOST (error 20)**

#### Qwen3.5-35B-A3B (INT4 AutoRound, 40 layers, 256 experts, MoE+GQA)

| Stage | Time | Cumulative |
|-------|------|-----------|
| Weight loading | 34s | 0:34 |
| CPU shuffle (40L × 256E) | 2m 52s | 3:26 |
| KV cache + engine init | 1s | 3:27 |
| Server ready | ~11s | **~3:38** |

- Model memory: 18.98 GiB, KV cache: 19,456 tokens (gpu-util=0.75)
- **Inference: FAIL — dtype mismatch (Bug I) then DEVICE_LOST (error 20)**

#### Qwen3-VL-30B-A3B (INT4 AutoRound, 48 layers, 128 experts, MoE+GQA+Vision)

| Stage | Time | Cumulative |
|-------|------|-----------|
| Weight loading | 22s | 0:22 |
| CPU shuffle (48L × 128E) | 2m 45s | 3:07 |
| KV cache + engine init | 3s | 3:10 |
| Server ready | ~9s | **~3:19** |

- Model memory: 16.00 GiB, KV cache: 50,432 tokens (gpu-util=0.75)
- **Inference: FAIL — OUT_OF_RESOURCES (error 40)**

#### GLM-4.7-Flash AWQ-4bit (compressed-tensors)

- Uses `CompressedTensorsWNA16MarlinMoEMethod` (Marlin MoE)
- **BLOCKED at load: `gptq_marlin_repack` is CUDA-only** (vllm._C missing on XPU)

#### Qwen3-30B-A3B (native GPTQ, not AutoRound)

- Uses `GPTQConfig` → `MoeWNA16Method` → Marlin MoE (CUDA-only)
- Different code path from AutoRound — our fixes (A-F) do NOT apply
- **Would be BLOCKED at load** by missing Marlin CUDA kernels (not tested)

### Memory profiles

#### GLM-4.7-Flash (gpu-memory-utilization=0.65, max-model-len=16384)

| Phase | RAM Used | Available | Swap | Note |
|-------|----------|-----------|------|------|
| Pre-load | 17.2 GiB | 14.4 GiB | 1.6 GiB | Clean state |
| Weights loaded | 20.6 GiB | 11.0 GiB | 5.7 GiB | 18.75s load |
| CPU shuffle peak | 22.6 GiB | 9.0 GiB | 5.5 GiB | 46 layers × 64 experts |
| Shuffle done | 20.1 GiB | 11.5 GiB | 6.1 GiB | Temps freed |
| Server ready | 23.1 GiB | 8.4 GiB | 5.9 GiB | KV cache allocated |

#### Qwen3.5-35B-A3B (gpu-memory-utilization=0.75, max-model-len=16384)

| Phase | RAM Used | Available | Swap | Note |
|-------|----------|-----------|------|------|
| Pre-load | 16.6 GiB | 15.0 GiB | 1.7 GiB | Clean state |
| Weights loaded | 24.8 GiB | 6.8 GiB | 5.5 GiB | 40s load (21 GiB model) |
| CPU shuffle peak | 25.7 GiB | 5.9 GiB | 5.8 GiB | 40 layers × 256 experts |
| Shuffle done | 24.1 GiB | 7.5 GiB | 6.0 GiB | Temps freed |
| Server ready | 26.4 GiB | 5.2 GiB | 5.9 GiB | KV cache allocated |

### Key insight: why GPT-OSS-20B works but INT4 GPTQ MoE models don't

GPT-OSS-20B uses **MXFP4 quantization** which stores weights pre-formatted for the XPU MoE kernel — no runtime shuffle, no `_IPEXGatedMLPMOEXPU` per-layer objects, no Level Zero resource accumulation. INT4 GPTQ models go through IPEX's `GatedMLPMOE` → `_IPEXGatedMLPMOEXPU` path which creates heavy per-layer kernel objects that exhaust the iGPU's resource pool.

The fix must come from Intel (IPEX or Level Zero driver) — either share kernel objects across layers or increase the resource pool for iGPU.

---

## Bug H Final Diagnosis: IPEX GatedMLPMOE INT4 broken on Lunar Lake (2026-04-16)

### Root cause identified

`GatedMLPMOE.forward()` with `is_int4=True` crashes with `DEVICE_LOST` (error 20) on Lunar Lake Xe2 iGPU. This is an **IPEX bug**, not a memory, resource pool, or shuffle issue.

### Proof

```python
# Direct kernel call — WORKS (even with 18 GiB allocated)
torch.xpu.moe_gemm(x, W13, rows, E, None, scale,
    is_int4=True, ...)  # → SUCCESS

# GatedMLPMOE wrapper — CRASHES (even with 0.28 GiB allocated)
moe = ipex.llm.modules.GatedMLPMOE(W13, W2,
    w1_scale_inv=s13, w2_scale_inv=s2, is_int4=True)
moe.W13 = moe.W13.to("xpu")
moe.W2 = moe.W2.to("xpu")
moe(x, router_logits=logits, ...)  # → DEVICE_LOST
```

Both tests use identical real model weights from safetensors, same shapes, same dtypes. The only difference is whether the call goes through `_IPEXGatedMLPMOEXPU.fused_moe_experts()` or directly to `torch.xpu.moe_gemm()`.

### Tested configurations (all crash via GatedMLPMOE)

| Config | XPU Allocated | Result |
|--------|--------------|--------|
| Pre-shuffled (_marlin_shuffled=True) | 0.28 GiB | DEVICE_LOST |
| Native IPEX shuffle (no flag) | 0.28 GiB | DEVICE_LOST |
| Full model + 0.75 util | 19.02 GiB | DEVICE_LOST |
| Full model + 0.85 util | 20.63 GiB | DEVICE_LOST |
| Full model + 0.65 util | 17.77 GiB | DEVICE_LOST |

### Tested configurations (all pass via direct moe_gemm)

| Config | XPU Allocated | Result |
|--------|--------------|--------|
| Zero weights, 2 experts | 0.01 GiB | PASS |
| Real weights, 128 experts | 0.19 GiB | PASS |
| Real weights + shuffle | 0.19 GiB | PASS |
| Real weights + 18 GiB dummy | 18.19 GiB | PASS |
| Real weights + KV cache blocks | 0.34 GiB | PASS |

### What this means

1. **Our CPU shuffle (Bug E) is correct** — the weights are in the right format
2. **Memory is not the issue** — crashes at 0.28 GiB allocated
3. **Level Zero resources are not the issue** — crashes with a single MoE layer
4. **The bug is in `_IPEXGatedMLPMOEXPU`** — specifically how it wraps `torch.xpu.moe_gemm` with routing, activation, and expert dispatch logic
5. The `fused_moe_experts()` method or the routing/gather/scatter around it produces invalid kernel state on Lunar Lake

### Upstream

**Intel IPEX** — `_IPEXGatedMLPMOEXPU.fused_moe_experts()` or its `forward()` method is broken on Xe2 iGPU with `is_int4=True`. The underlying `torch.xpu.moe_gemm` kernel works correctly when called directly.

### Workaround path

A potential workaround would be to bypass `GatedMLPMOE` entirely and call `torch.xpu.moe_gemm` directly in `XPUGPTQMarlinMoEMethod.apply()`, implementing the routing/gating logic in Python/PyTorch instead of relying on IPEX's fused implementation.

---

## Bug H Final Root Cause: Level Zero Context Pollution (2026-04-16) — SUPERSEDED

> **Note (2026-04-17)**: This "context pollution" theory was disproven. Bypassing GatedMLPMOE and calling `group_mm_int4_out_marlin` directly works after attention without crashing (14,400+ calls completed). The real issue is two-fold: (1) GatedMLPMOE/topk_softmax crash with 128 experts (SOLVED by bypass), (2) the INT4 marlin kernel itself returns zeros/NaN on Xe2 (UNSOLVED — kernel bug). See "Bug H Definitive" section below.

### Definitive proof chain

1. `torch.xpu.moe_gemm(is_int4=True)` **works in isolation** — any memory level, any tensor count, KV cache allocated, routing ops applied, real model weights
2. `GatedMLPMOE.forward()` **crashes** — even in isolation with 0.28 GiB allocated (IPEX wrapper bug)
3. Direct `torch.xpu.moe_gemm` bypass in `apply()` **also crashes** — but only inside vLLM's forward pass, after attention ran
4. **Cloned tensors crash too** — ruling out tensor state/strides as the cause

### Root cause

IPEX's attention kernels (`flash_attn_varlen_func` → `chunked_prefill` or `PagedAttention`) configure the Level Zero context/SYCL queue in a way that corrupts subsequent `torch.xpu.moe_gemm` INT4 kernel dispatch. Even `torch.xpu.synchronize()` + `empty_cache()` cannot reset this state. The corruption persists at the Level Zero driver level.

### Why GPT-OSS-20B works

GPT-OSS-20B uses MXFP4 (not INT4 GPTQ). The `moe_gemm` kernel with `is_mxfp4=True` may use a different internal code path that isn't affected by the attention context pollution. Or the simpler 24-layer architecture doesn't trigger the same attention kernel configuration.

### Fix required

This is an Intel issue in either:
1. **IPEX attention kernel** — leaves Level Zero context in a dirty state
2. **IPEX moe_gemm INT4 kernel** — doesn't properly initialize its Level Zero context, inheriting corrupted state from attention
3. **Level Zero driver (compute-runtime 25.48.36300)** — context isolation between different kernel types is broken on Xe2 iGPU

### Workaround attempted

Bypassing `GatedMLPMOE` and calling `torch.xpu.moe_gemm` directly doesn't help — the context pollution comes from the attention layer, not the MoE wrapper.

### Test 2+3 Results (2026-04-16)

| Test | What it bypasses | Result |
|------|-----------------|--------|
| Test 3: sync + empty_cache + gc | Stale allocator state | Still crashes |
| Test 2: Python routing (no IPEX scatter) + moe_gemm | IPEX routing ops | **DEVICE_LOST** |
| Test IPEX: IPEX routing + moe_gemm | (nothing — crashed before reaching) | N/A |

**Conclusion**: `torch.xpu.moe_gemm(is_int4=True)` cannot execute after IPEX attention in the same process on Lunar Lake Xe2 iGPU with compute-runtime 25.48.36300.8. The Level Zero context is fatally corrupted by the attention kernel dispatch. No userspace workaround exists — requires Intel driver or IPEX fix.

### Upstream Intel llm-scaler Issues (related)

| Issue | Model | GPU | INT4 Type | Result |
|-------|-------|-----|-----------|--------|
| [#324](https://github.com/intel/llm-scaler/issues/324) | Qwen3.5-27B-GPTQ-Int4 | Arc B60 (24GB) | Pre-quantized GPTQ | OOM (model too large for 24GB) |
| [#324 comment](https://github.com/intel/llm-scaler/issues/324) | Qwen3.5-27B-int4-autoround | Arc B60 | Pre-quantized AutoRound | **Works at 13 tok/s** |
| [#314](https://github.com/intel/llm-scaler/issues/314) | Qwen3-30B-A3B | Multi-GPU TP2/4 | Online INT4 (sym_int4) | Works (with OFFLOAD fix) |
| [#269](https://github.com/intel/llm-scaler/issues/269) | AWQ-Int4 model | Unknown | Pre-quantized AWQ | CUDA-only codepath failure |

**Key finding**: A user in #324 ran `Intel/Qwen3.5-27B-int4-autoround` on Arc B60 discrete GPU and got 13 tok/s — no crash reported. This suggests AutoRound INT4 MoE works on discrete GPU but not on Lunar Lake iGPU.

The difference: discrete GPU has dedicated VRAM with a separate Level Zero context, while Lunar Lake iGPU shares LPDDR5x memory with the CPU and may have a different Level Zero driver configuration for context management.

### Open question

The README's INT4 support refers to "Dynamic Online Int4" (quantize at load time), NOT pre-quantized GPTQ/AutoRound models. The two paths use completely different code:

| | Online INT4 | Pre-quantized AutoRound INT4 |
|---|---|---|
| Code path | `SymInt4LinearMethod` / `XPUGPTQInt4LinearMoEMethod` | `XPUGPTQMarlinMoEMethod` |
| MoE kernel | Direct INT4 GEMM | `GatedMLPMOE` + `marlin_shuffle_weight` |
| Tested by Intel | Yes (README ✅) | Unknown (no docs) |
| On iGPU | Unknown | Broken (this investigation) |

Testing Online INT4 on Lunar Lake iGPU would require FP16 base model weights (~60 GiB for Qwen3-30B) which don't fit in 32 GiB shared memory.

---

## Critical Finding: INT4 AutoRound MoE Never Tested by Intel (2026-04-16)

### The #324 user ran a DENSE model, not MoE

Issue [#324](https://github.com/intel/llm-scaler/issues/324) reported `Intel/Qwen3.5-27B-int4-autoround` working on Arc B60. But **Qwen3.5-27B is a dense model** (`Qwen3.5ForCausalLM`), not MoE. Dense models use `IPEXGPTQLinearMethod` for linear layers — no `FusedMoE`, no `GatedMLPMOE`, no `marlin_shuffle_weight`. This path works fine.

Our models are all MoE (`Qwen3MoeForCausalLM`, `Qwen3VLMoeForConditionalGeneration`, `Glm4MoeLiteForCausalLM`) which require `XPUGPTQMarlinMoEMethod` → `GatedMLPMOE`. This path is broken.

### IPEX Issue #838: Exact Same Bug

[intel/intel-extension-for-pytorch#838](https://github.com/intel/intel-extension-for-pytorch/issues/838) reports the identical `UR_RESULT_ERROR_OUT_OF_RESOURCES` (error 40) on `torch.ops.torch_ipex.topk_softmax()` when running `Qwen/Qwen3-30B-A3B` with `GatedMLPMOE` on Intel Data Center GPU Max 1550 (discrete GPU!).

Key details:
- **Filed**: 2025-06-14
- **Closed**: 2026-01-05 by ZhaoqiongZ (no linked PR, no public fix)
- **Hardware**: Intel Data Center GPU Max 1550 (discrete, NOT iGPU)
- **The bug affects discrete GPU too** — not just Lunar Lake iGPU
- Smaller MoE models (Llama-4-Scout-17B-16E, Falcon3-MoE-2x7B) work; Qwen3-30B-A3B (128 experts) fails
- Same IPEX version `2.10.10.post1+xpu` — no public fix available

### Docker image doesn't help

The Docker image `intel/llm-scaler-vllm:latest` (b8.1) has:
- Same compute-runtime `25.48.36300.8` as our native install
- Same IPEX `2.10.10.post1+xpu`
- Bug A unfixed (no FusedMoE routing in AutoRound)
- No CPU shuffle workaround (Bug E)
- Same `GatedMLPMOE.forward()` code path

### What Intel supports vs what we need

| | Dense INT4 AutoRound | MoE INT4 AutoRound | MoE FP8/FP16 | MoE MXFP4 |
|---|---|---|---|---|
| Code path | `IPEXGPTQLinearMethod` | `XPUGPTQMarlinMoEMethod` | `XPUGPTQInt4LinearMoEMethod` | Direct MoE kernel |
| Intel tested | ✅ | ❌ (IPEX #838 open) | ✅ | ✅ |
| Works on discrete | ✅ | ❌ (128 experts crashes) | ✅ | ✅ |
| Works on iGPU | ✅ (too big for 32GB) | ❌ | ❌ (FP16 too big) | ✅ (GPT-OSS-20B) |

### Conclusion

INT4 AutoRound MoE inference via `GatedMLPMOE` is broken on ALL Intel GPUs for models with 128+ experts. The fix is in Intel's internal IPEX but hasn't been released. Our patches (Bugs A-F) correctly route and prepare the weights, but the underlying IPEX `GatedMLPMOE.forward()` → `topk_softmax()` / `moe_gemm()` path has a known unfixed bug (#838).

---

## Bug H Deep Dive: vLLM Execution Path is the Trigger (2026-04-16)

### Kernel dispatch diagnostics

```
INT4 dispatch: total_m=72 n_experts=128 average_m=0 policy=GEMV
  input_dtype=torch.bfloat16 scale_dtype=torch.bfloat16
  k=2048 n=1536 group_size=128 has_2d_block=True has_xmx=True
```

- **GEMV policy** (8×64 tiles) — NOT the large 256×256 GEMM tiles
- **BF16 input** — not FP16 (C++ kernel uses `sycl::half` internally but accepts BF16)
- **Fused kernel path** taken (has_2d_block + has_xmx)

### Hypotheses tested and eliminated

| Hypothesis | Test | Result |
|---|---|---|
| A: Large tile size (256×256) | Kernel uses GEMV (8×64), average_m=0 | **Eliminated** |
| C: BF16→FP16 dtype mismatch | Both BF16 and FP16 work in isolation | **Eliminated** |
| Attention state pollution | Paged attention → moe_gemm in same process | **Works** |
| Full model memory pressure | 16.66 GiB + 56K tensors + attention → moe_gemm | **Works** |

### The remaining suspect: vLLM execution framework

All isolated tests pass — even with the full model loaded (16.66 GiB, 56,466 tensors), KV cache allocated, and paged attention executed. The crash ONLY occurs through vLLM's model runner.

The difference between our passing test and vLLM's failing forward:

| | Isolated test | vLLM forward |
|---|---|---|
| Weight loading | `load_file().to("xpu")` | vLLM `DefaultModelLoader` + `weight_loader` |
| Attention | Direct `PagedAttention.flash_attn_varlen_func` | Through `torch.ops.vllm.unified_attention_with_output` custom op |
| MoE dispatch | Direct `torch.xpu.moe_gemm` | Through `FusedMoE.forward_native` → `torch.ops.vllm.moe_forward_shared` |
| Compilation | None (eager) | `torch._dynamo` + custom ops registration |
| Tensor management | Simple Python references | vLLM's tensor pinning, input/output buffers, scheduler metadata |

### Next investigation

The crash vector is in vLLM's custom op dispatch chain, not the IPEX kernels themselves. Need to trace which vLLM layer wraps the call and what state it adds.

### Component-by-component test results (all PASS in isolation)

| Test | XPU Alloc | Result |
|------|-----------|--------|
| `torch.xpu.moe_gemm` alone | 0.2 GiB | PASS |
| + 18 GiB dummy tensors | 18.2 GiB | PASS |
| + 56K model tensors (16.66 GiB) | 16.7 GiB | PASS |
| + KV cache (paged blocks) | 16.7 GiB | PASS |
| + `reshape_and_cache_flash` | 0.3 GiB | PASS |
| + paged attention | 0.3 GiB | PASS |
| + IPEX routing ops (topk, scatter, gather) | 0.3 GiB | PASS |
| + shuffled weights (Bug E pipeline) | 0.3 GiB | PASS |
| + full model + attention + shuffled MoE | 17.0 GiB | PASS |
| + `GatedMLPMOE.forward()` with `init_on_device` | 0.3 GiB | PASS |
| vLLM custom op bypass (`forward_impl` direct) | 16.6 GiB | **FAIL** |
| vLLM attention bypass (`use_direct_call`) | 16.6 GiB | **FAIL** |
| vLLM full pipeline | 16.6 GiB | **FAIL** |

### Conclusion

Every IPEX kernel, every weight format, every memory configuration works correctly in standalone Python scripts — even with the full 17 GiB model loaded, KV cache allocated, and all operations chained. The crash occurs ONLY through vLLM's model runner execution path.

The root cause is in vLLM's model graph execution infrastructure — likely the `ForwardContext`, `no_compile_layers` registry, tensor buffer pinning, or scheduler metadata management that runs during the full model forward but not in isolated tests.

This requires either:
1. Systematic binary search through vLLM's model runner components
2. Intel investigation with the reproduction evidence provided here
3. Testing with a future vLLM version that may change the execution path

---

## Final Verdict: INT4 AutoRound MoE Never Worked in IPEX (2026-04-16)

### IPEX repo was archived with the bug unfixed

The IPEX repo was archived on **2026-03-30** (issue #867). The INT4 MoE bug (#838) was closed on 2026-01-05 with **no public fix, no linked PR, no commit**. The closure was either "closed as stale" or "fixed internally" in a branch that was never released.

### Complete IPEX issue analysis

| Issue | Problem | Status | Fix |
|---|---|---|---|
| [#838](https://github.com/intel/intel-extension-for-pytorch/issues/838) | `topk_softmax` OUT_OF_RESOURCES with 128 experts | Closed (no fix) | **None** |
| [#864](https://github.com/intel/intel-extension-for-pytorch/issues/864) | `weight_only_qlinear_prepack_int4` missing for AutoRound | Open | None |
| [#869](https://github.com/intel/intel-extension-for-pytorch/issues/869) | CPU offload crash with INT4 AutoRound MoE | Open | None |
| [#867](https://github.com/intel/intel-extension-for-pytorch/issues/867) | IPEX end-of-life — GatedMLPMOE NOT upstreamed | Closed | N/A |

### Key facts

- **Zero PRs** exist fixing topk_softmax, moe_gemm, or GatedMLPMOE
- **#838 is the ONLY issue** mentioning GatedMLPMOE in the entire repo (198 issues)
- INT4 AutoRound MoE was **never working** for models with 128+ experts on ANY Intel GPU
- GatedMLPMOE, marlin_shuffle_weight, and MoE fusion layers were **NOT upstreamed** to PyTorch — they're frozen in the archived IPEX
- Intel committed to "critical bug fixes for two quarters" after v2.8, but #838 was closed without a fix

### Successor: vllm-xpu-kernels

INT4 MoE is listed as future work in [vllm-project/vllm#33214](https://github.com/vllm-project/vllm/issues/33214) (vllm-xpu-kernels RFC). The new path would use **oneDNN INT4 GEMM** instead of IPEX's GatedMLPMOE.

### Paths forward for Lunar Lake iGPU

1. **MXFP4 quantization** — works today (GPT-OSS-20B proven), uses a different kernel path
2. **vllm-xpu-kernels** — future Intel INT4 MoE implementation (not yet available)
3. **llama.cpp GGUF** — alternative inference engine with SYCL backend, known to work on Lunar Lake
4. **Wait for oneDNN INT4 MoE** — Intel's next-gen kernel library

### What our patches achieved

Despite Bug H being unfixable (IPEX limitation), our patches (A-G) solved every OTHER blocker:

| Bug | What it fixed | Value |
|---|---|---|
| A | FusedMoE routing to IPEX | Model loads (was: CUDA fallback crash) |
| B | top_k_experts config | GLM-4.7 architecture support |
| C | GPTQ non-aligned dims | Gemma 4 model support |
| D | MLA return_attn_probs | GLM-4.7 MLA attention |
| E | CPU shuffle for iGPU | Weights load correctly (was: DEVICE_LOST) |
| F | Warmup skip | Server starts (was: OUT_OF_RESOURCES) |
| G | Shared memory insight | Correct memory budgeting |

These patches will be needed when Intel releases a working INT4 MoE kernel (via vllm-xpu-kernels or oneDNN).

---

## vLLM 0.19 / vllm-xpu-kernels Analysis (2026-04-16)

### New XPU MoE path (bypasses IPEX entirely)

vLLM 0.16+ migrated from IPEX to `vllm-xpu-kernels` ([RFC #33214](https://github.com/vllm-project/vllm/issues/33214)). The new `XPUExperts` class uses CUTLASS grouped GEMM instead of IPEX's GatedMLPMOE:

| Component | Old (IPEX) | New (vllm-xpu-kernels) |
|---|---|---|
| MoE routing | `torch.ops.torch_ipex.topk_softmax` | `torch.ops._moe_C.remap_hidden_states` |
| MoE GEMM | `torch.xpu.moe_gemm` / `GatedMLPMOE` | `torch.ops._xpu_C.cutlass_grouped_gemm_interface` |
| Weight shuffle | `marlin_shuffle_weight` | `implement_zp()` (uint8 conversion) |
| Weight format | INT32-packed GPTQ (8 nibbles/int32) | UINT8-packed (2 nibbles/byte, signed) |

### RFC migration status

- [x] Unquantized MoE — Done
- [x] FP8 MoE — Done
- [x] MXFP4 MoE — Done
- [ ] **INT4 MoE** — NOT done (no PR)
- [ ] INT4 GEMM (AWQ/GPTQ linear) — WIP ([PR #33662](https://github.com/vllm-project/vllm/pull/33662))

### INT4 MoE kernel exists but needs format bridging

`xpu_fused_moe(..., is_int4=True)` is implemented in vllm-xpu-kernels and uses CUTLASS with `is_B_int4=True`. However:

1. It expects **uint8-packed** weights (via `implement_zp()`)
2. GPTQ/AutoRound models have **int32-packed** weights with group scales
3. A format conversion layer is needed (this is what PR #33662 is building for linear layers)
4. Once PR #33662 lands, extending it to MoE should be straightforward

### Can we backport to vLLM 0.14?

Not directly — `vllm-xpu-kernels` requires vLLM 0.16+ infrastructure (modular kernel API, `FusedMoEExpertsModular`, `_moe_C` ops). The APIs don't exist in 0.14.

### Recommendation

1. **Short term**: Use MXFP4 on vLLM 0.14 (works today)
2. **Medium term**: Upgrade to vLLM 0.19 when INT4 MoE support lands
3. **Long term**: Contribute the GPTQ→uint8 format bridge for INT4 MoE to vllm-xpu-kernels

---

## Future Plan: INT4 MoE on vLLM 0.19 + vllm-xpu-kernels

### Prerequisites

1. Check vLLM 0.19 compatibility with current stack (torch 2.10.0+xpu, oneAPI 2025.2)
2. Build vLLM 0.19 for XPU (fresh compile required — 0.14 and 0.19 are incompatible)
3. Install vllm-xpu-kernels (pre-built wheel, no compile needed)

### Contribution: INT4 MoE format bridge

**What exists**: `cutlass_grouped_gemm_interface(is_B_int4=True)` works with uint8-packed weights

**What's needed**: GPTQ int32-packed → uint8-packed weight conversion for MoE

**Files to create/modify**:
- `vllm/model_executor/layers/fused_moe/xpu_fused_moe.py` — add `XPUExpertsInt4` class
- `vllm_xpu_kernels/fused_moe_interface.py` — weight format conversion (or in vLLM side)
- Oracle/quant selection — register INT4 quant key for XPU MoE

**Estimated scope**: ~200 lines Python, no C++/SYCL changes needed

### Steps

1. Verify vLLM 0.19 builds and runs on Lunar Lake (unquantized MoE first)
2. Port GPTQ INT4 weight unpacking from PR #33662 (linear layers) to MoE
3. Add `XPUExpertsInt4` with `is_int4=True` and GPTQ weight conversion in `process_weights_after_loading`
4. Test with Qwen3-VL-30B-A3B INT4 AutoRound
5. Submit PR to vllm-project/vllm referencing RFC #33214 checklist item "int4 moe support"

### Dependencies to watch

- [PR #33662](https://github.com/vllm-project/vllm/pull/33662) — INT4 GEMM for linear layers (WIP, same format bridge needed)
- [RFC #33214](https://github.com/vllm-project/vllm/issues/33214) — XPU kernel migration checklist
- vllm-xpu-kernels releases — check for INT4 MoE additions

---

## BREAKTHROUGH: INT4 MoE Inference Working via CUTLASS (2026-04-16)

### The fix: bypass IPEX entirely, use CUTLASS sequential expert loop

Replaced the IPEX `GatedMLPMOE` → `moe_gemm` path with:
1. **Weight conversion**: GPTQ int32-packed → uint8 + `implement_zp()` (no marlin_shuffle needed)
2. **Sequential expert loop**: process one expert at a time via `torch.ops._xpu_C.cutlass_grouped_gemm_interface(is_B_int4=True)`
3. **Python routing**: topk + manual scatter/gather

### Why sequential works but batched doesn't

The CUTLASS `cutlass_grouped_gemm_interface` with `is_B_int4=True` crashes with DEVICE_LOST when dispatching to 50+ active expert groups simultaneously on Xe2 iGPU. Individual expert GEMMs (1 expert at a time) work perfectly — same as how llama.cpp handles MoE.

| Dispatch | Expert groups | Xe2 iGPU | Result |
|---|---|---|---|
| Batched (IPEX moe_gemm) | 50-128 simultaneous | Crash | DEVICE_LOST |
| Batched (CUTLASS grouped) | 50-128 simultaneous | Crash | DEVICE_LOST |
| Sequential (1 expert/call) | 1 at a time | Works | **SUCCESS** |

### Test result

```
Model: Qwen3-VL-30B-A3B (INT4 AutoRound, 48 layers, 128 experts)
Server: UP (16.85 GiB model, 22,336 tokens KV cache)
Inference: 83s for 30 tokens (sequential, unoptimized)
Output: Garbled (weight conversion needs tuning)
GPU: No crash — full pipeline completes
```

### Additional fix: Bug J — eagle_ops FP16 assertion

ESIMD eagle_ops page_attn_decode expects FP16 query but GPTQ dequant produces BF16. Fixed by skipping eagle_ops when `query.dtype != torch.float16`.

### What changed from the IPEX path

| | IPEX path (old, crashes) | CUTLASS path (new, works) |
|---|---|---|
| Weight prep | marlin_shuffle_weight (CPU) | int32→uint8 + implement_zp |
| MoE GEMM | `torch.xpu.moe_gemm` batched | `cutlass_grouped_gemm_interface` sequential |
| Routing | IPEX topk_softmax + moe_scatter | Python topk + manual scatter |
| IPEX dependency | GatedMLPMOE, _IPEXGatedMLPMOEXPU | None |
| vllm-xpu-kernels | Not used | `_xpu_C.cutlass_grouped_gemm_interface` |

### Remaining work

1. **Fix weight conversion**: GPTQ nibble ordering → CUTLASS uint8 format (output is garbled)
2. **Optimize speed**: Sequential loop is 83s — small-batch grouped GEMM (4-8 experts/batch) or expert-parallel could be 10-20x faster
3. **Test on other models**: GLM-4.7-Flash, Qwen3.5-35B-A3B
4. **Commit patches and push**

### Weight conversion findings (2026-04-16)

The CUTLASS INT4 kernel uses a **column-interleaved** nibble packing, not row-sequential:
- Each uint8 byte packs nibbles from **two different output columns** (N dimension)
- Low nibble non-zero values produce NaN → low nibble is used differently
- High nibble maps linearly: hi=4→0.75/elem, hi=8→1.5/elem, hi=9→2.25/elem

GPTQ packs 8 nibbles along the **K dimension** (input) per int32. CUTLASS interleaves across **N dimension** (output). This requires a full layout transformation, not just int32→uint8 repacking.

Next step: study the CUTLASS INT4 packed format from vllm-xpu-kernels source to implement correct GPTQ→CUTLASS conversion.

---

## SOLVED: Correct INT4 MoE Inference on Lunar Lake (2026-04-17)

### The final fix: int4_gemm_w4a16 with oneDNN format

Replaced CUTLASS `cutlass_grouped_gemm_interface(is_B_int4=True)` (which uses FP4 internally, not INT4) with `int4_gemm_w4a16` — a proper GPTQ INT4 W4A16 GEMM kernel.

### Weight format: oneDNN (no shuffle needed!)

```python
# oneDNN format: K-contiguous strides, original GPTQ int32 packing
w = qweight.transpose(0, 1).contiguous().transpose(0, 1)  # [K//8, N]
scale = scales.contiguous()                                 # [K//gs, N]
zp = torch.tensor([8], dtype=torch.int8)                   # scalar for symmetric GPTQ
```

No marlin_shuffle, no implement_zp, no uint8 conversion. The kernel reads GPTQ int32 directly.

### Verified: 0.999997 correlation with CPU reference

```
CUTLASS int4_gemm_w4a16: mean=0.0161, std=0.8711
CPU float dequant:       mean=0.0162, std=0.8716
Correlation: 0.999997
```

### Inference result

```
Model: Qwen3-VL-30B-A3B (INT4 AutoRound, 48 layers, 128 experts)
Prompt: "Hello! What are you? Reply in one sentence."
Output: Chinese text (real language tokens, not garbled)
Tokens: 50 generated from 19 prompt tokens
Time: 315s (~0.16 tok/s, sequential expert loop)
HTTP: 200 OK
```

### Performance: sequential bottleneck

Current implementation loops over 128 experts, calling `int4_gemm_w4a16` for each active expert (~50 per layer). Each call is M=1-2 tokens — severe GPU underutilization.

Optimization paths:
1. **Batch experts**: group token-expert pairs into fewer, larger GEMM calls
2. **Threshold test**: find max expert groups before DEVICE_LOST for grouped GEMM
3. **Hybrid**: use grouped GEMM for small expert batches (≤8), sequential for rest

### Complete solution stack

| Component | Fix | No IPEX needed |
|-----------|-----|---------------|
| Weight format | oneDNN: transpose strides + scalar zp=8 | ✓ |
| MoE GEMM | `int4_gemm_w4a16` per expert | ✓ |
| Routing | Python topk + manual scatter/gather | ✓ |
| Attention dtype | Cast k/v + skip eagle_ops for BF16 | ✓ |
| Profile/warmup | Skip both on XPU | ✓ |

### Performance optimization (2026-04-17)

Optimized `apply()` method — 7.7x speedup:

| Change | Before | After |
|--------|--------|-------|
| Expert discovery | Loop all 128, check `.any()` | `.unique()` on active only |
| Accumulation | `output[idx] += w * g2` | `index_add_()` in-place |
| Python overhead/layer | ~128ms | ~17ms |
| **Generation throughput** | **0.16 tok/s** | **0.9 tok/s** |
| **30 tokens** | **315s** | **41s** |

### Kernel benchmarks (Xe2 iGPU, K=2048, N=768)

| Kernel | M=1 | M=4 | Notes |
|--------|-----|-----|-------|
| `int4_gemm_w4a16` (oneDNN) | 33μs | 28μs | Our path — fastest |
| `cutlass_grouped_gemm` (FP4) | 72μs | 45μs | Wrong format for INT4 |
| `torch.mm` BF16 | 1209μs | 62μs | Reference |

Theoretical max: 30μs × 50 experts × 2 GEMMs × 48 layers = 144ms/token = ~7 tok/s.
Current: ~1.1s/token = 0.9 tok/s. Python overhead is ~88% of total time.

### GPT-OSS-20B benchmark (2026-04-17)

GPT-OSS-20B uses **IPEX marlin backend** with `GatedMLPMOE(is_mxfp4=True)`, NOT CUTLASS:

```
Using ipex marlin backend on XPU
→ GatedMLPMOE(is_mxfp4=True) → torch.xpu.moe_gemm(is_mxfp4=True)
```

The same `GatedMLPMOE` wrapper that crashes with `is_int4=True` **works fine with `is_mxfp4=True`** on Xe2 iGPU. The bug is in IPEX's INT4 kernel path, not the MoE wrapper.

| Model | Quant | Kernel | Gen tok/s | Layers | Experts |
|-------|-------|--------|-----------|--------|---------|
| GPT-OSS-20B | MXFP4 | IPEX moe_gemm | **3.0** | 24 | 32 |
| Qwen3-VL-30B-A3B | INT4 GPTQ | oneDNN int4_w4a16 | **0.9** | 48 | 128 |

Normalized per-layer: GPT-OSS gets 3.0/24=0.125 tok/s/layer, Qwen3-VL gets 0.9/48=0.019 tok/s/layer. MXFP4 is **~6.5x faster per layer** due to batched grouped GEMM vs sequential expert loop.

### CUTLASS vs IPEX grouped GEMM comparison (2026-04-17)

CUTLASS `cutlass_grouped_gemm_interface(is_B_mxfp4=True)` with GPT-OSS-20B MXFP4 weights:

| Active experts | CUTLASS MXFP4 | IPEX moe_gemm |
|---------------|---------------|---------------|
| 4 | 1.41ms ✓ | Works (GPT-OSS runs fine) |
| 8 | 2.58ms ✓ | Works |
| 16 | **DEVICE_LOST** ✗ | Works |
| 32 | Not tested | Works |

**CUTLASS grouped GEMM crashes at 16+ active experts on Xe2** — same threshold for both MXFP4 and INT4. The bug is in the CUTLASS kernel dispatch, NOT in the data format.

**IPEX's `torch.xpu.moe_gemm`** handles 32 experts without crashing. The difference is in how they dispatch to the GPU — IPEX uses a different SYCL kernel template that's compatible with Xe2's Level Zero resource limits.

This means: if we could use IPEX's `moe_gemm` with proper INT4 support, we'd get the batched performance. The IPEX INT4 path crashes for a different reason (Bug H: kernel bug in INT4 dequant, not resource limits).

---

## Bug H Definitive: Two Independent Failures (2026-04-17)

Bug H is actually **two separate bugs**, not one:

### Failure 1: GatedMLPMOE crash (SOLVED)

`GatedMLPMOE(is_int4=True)` crashes with DEVICE_LOST on Xe2. Likely causes:
- `torch.ops.torch_ipex.topk_softmax` with 128 experts (documented in IPEX #838)
- The wrapper's INT4 code path internally (not the kernel)

**Fix**: Bypass GatedMLPMOE entirely. Use `torch.topk` for routing + direct IPEX C++ ops (`moe_rows_counts`, `moe_scatter`, `group_mm_int4_out_marlin`, `silu_and_mul`, `moe_gather`). With `VLLM_INT4_MOE_MARLIN=1`, the full vLLM pipeline completes 14,400+ INT4 GEMM calls across 48 layers × 150 decodes without DEVICE_LOST or OUT_OF_RESOURCES.

**Correction**: The earlier "Level Zero context pollution from attention" theory (lines 434-497) was wrong. The crash was in the GatedMLPMOE wrapper / topk_softmax, not in attention corrupting the Level Zero context. Direct `group_mm_int4_out_marlin` calls work fine after attention.

### Failure 2: `group_mm_int4_out_marlin` returns wrong output on Xe2 (UNSOLVED)

The kernel runs without crashing but produces **numerically incorrect results**. Tested with 4 weight layout candidates against a CPU float32 reference (1 expert, M=4, K=2048, N=768):

| Layout candidate | BF16 input | FP16 input |
|---|---|---|
| A: raw `[E, K/8, N]` | all zero | corr=0.019 (near zero) |
| B: permute `[E, N, K/8]` | all zero | NaN |
| C: permute + marlin shuffle `[E, N, K/8]` | all zero | NaN |
| D: raw + marlin shuffle `[E, K/8, N]` | all zero | NaN |

- **BF16 input**: always returns all-zero output regardless of weight layout
- **FP16 input**: returns near-zero correlation (layout A) or NaN (all others)
- **No weight layout produces correct output** — the kernel itself is broken on Xe2

The "unprecedented 15.4 tok/s" run with `VLLM_INT4_MOE_MARLIN=1` was actually the kernel returning zeros from every MoE layer. All-zero MoE contribution → deterministic downstream logits → greedy argmax picks token ID 0 (`!`). Not real computation.

### Why MXFP4 works but INT4 doesn't

`group_mm_mxfp4_out_marlin` (used by GPT-OSS-20B at 3 tok/s) works correctly on Xe2. The INT4 variant uses different SYCL kernel templates:
- Different dequantization logic (integer vs microscaling float)
- Different tile configurations (`dequant_s=128` vs `dequant_s=32`)
- Possibly untested on Xe2-LPG by Intel (discrete GPU focus)

### Current status

| Path | Correct | Speed | Status |
|---|---|---|---|
| oneDNN `int4_gemm_w4a16` sequential | ✓ (0.999997 corr) | 0.9 tok/s | **Working** — only correct INT4 MoE path |
| IPEX `group_mm_int4_out_marlin` batched | ✗ (zeros/NaN) | 15.4 tok/s* | **Blocked** — kernel broken on Xe2 |
| IPEX `moe_gemm(is_mxfp4)` | ✓ | 3.0 tok/s | Working (different quant, GPT-OSS-20B) |
| CUTLASS `grouped_gemm` | ✗ (crashes ≥16 experts) | N/A | **Broken** on Xe2 |

\*not real computation — kernel returns zeros

### Upstream action needed

File bug on `intel/intel-extension-for-pytorch`: `group_mm_int4_out_marlin` returns all-zero output (BF16) or NaN (FP16) on Xe2-LPG (Lunar Lake Arc 140V). Include 4-layout test script and results. The MXFP4 variant works correctly on the same hardware, so this is specific to the INT4 kernel path.
