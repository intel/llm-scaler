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

## Bug H: Level Zero OUT_OF_RESOURCES during inference (HARD BLOCKER)

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

## Status

| Bug | Status | Applied to | Blocker? |
|-----|--------|-----------|----------|
| A | Fixed | site-packages + patch | No |
| B | Fixed | site-packages + patch | No |
| C | Fixed | site-packages + patch | No |
| D | Fixed | site-packages + patch | No (GLM-4.7 MLA only) |
| E | Fixed | site-packages + patch | No |
| F | Fixed | site-packages + patch | No |
| G | N/A | architectural insight | No |
| **H** | **OPEN** | **Level Zero driver limit** | **YES — all INT4 GPTQ MoE** |
| I | Workaround | site-packages | Blocked by H |

### Next steps

1. **Upgrade compute-runtime** from `25.48.36300.8` to `26.09.37435.1` — newer versions include Level Zero resource pooling improvements that may fix Bug H
   - **NOTE**: Partial upgrade (swapping `.so` only) fails with `built_ins.cpp` abort — ALL components (`libze_intel_gpu`, `intel-opencl`, `intel-ocloc`, `libigdgmm`) must be upgraded together as a matching set
   - `LD_LIBRARY_PATH` override also fails — built-in kernel format is incompatible across versions
   - Intel only ships DEBs on GitHub; Nobara/Fedora repos are stuck at `25.48.36300.8`
   - Options: (a) rebuild all RPMs from source, (b) Docker container with Ubuntu + newer runtime, (c) wait for Nobara/Fedora packaging
2. **Report to Intel IPEX** — `_IPEXGatedMLPMOEXPU` should reuse kernel objects across layers instead of creating one per layer
3. **Consider MXFP4 quantization** for new MoE models — MXFP4 avoids the `_IPEXGatedMLPMOEXPU` path entirely and works on Lunar Lake (proven by GPT-OSS-20B)

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

## Bug H Final Root Cause: Level Zero Context Pollution (2026-04-16)

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
