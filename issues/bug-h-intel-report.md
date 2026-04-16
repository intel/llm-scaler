# Intel Bug Report: IPEX attention kernel corrupts Level Zero SYCL context for INT4 MoE GEMM on Lunar Lake Xe2 iGPU

## Title

`flash_attn_varlen_func` corrupts SYCL context — subsequent `torch.xpu.moe_gemm(is_int4=True)` crashes with DEVICE_LOST on Lunar Lake Xe2-LPG

## Environment

- **Hardware**: Intel Core Ultra 7 258V (Lunar Lake), Arc 140V Xe2-LPG iGPU
- **Memory**: 32 GB LPDDR5x shared (28.57 GiB usable by GPU)
- **OS**: Nobara 42 (Fedora-based), kernel 6.18.5
- **PyTorch**: 2.7.0a0+xpu (from intel-extension-for-pytorch)
- **IPEX**: 2.7.10+xpu (intel-extension-for-pytorch)
- **oneAPI**: dpcpp-ct 2025.2.0-517
- **Level Zero**: compute-runtime (check `dpkg -l | grep level-zero` for version)
- **vLLM**: 0.14.1.dev0 (Intel llm-scaler fork)

## Summary

When running INT4 AutoRound MoE models via vLLM on Lunar Lake Xe2-LPG iGPU, `torch.xpu.moe_gemm(is_int4=True)` crashes with `UR_RESULT_ERROR_DEVICE_LOST` (Level Zero error 20) immediately after IPEX attention kernels (`flash_attn_varlen_func` / `chunked_prefill`) execute in the same forward pass.

**The INT4 MoE GEMM kernel works perfectly in isolation.** The crash only occurs when attention kernels have previously executed in the same SYCL queue/context.

MXFP4 MoE models (using `group_mm_mxfp4_out_marlin` xetla kernel) are NOT affected — they work correctly after attention in the same pipeline.

## Reproduction

### 1. INT4 MoE GEMM works in isolation (PASSES)

```python
import torch
import intel_extension_for_pytorch as ipex

num_experts, hidden_dim, intermediate_dim, group_size = 4, 256, 512, 128
K_packed = intermediate_dim // 8

W_int4 = torch.randint(0, 2**31, (num_experts, K_packed, hidden_dim),
                        dtype=torch.int32, device="xpu")
hidden = torch.randn(8, hidden_dim, dtype=torch.bfloat16, device="xpu")
rows = torch.tensor([2, 2, 2, 2], dtype=torch.int64, device="xpu")
scale = torch.ones(num_experts, intermediate_dim // group_size, hidden_dim,
                   dtype=torch.bfloat16, device="xpu")

result = torch.xpu.moe_gemm(
    hidden, W_int4, rows, num_experts,
    weight_scale_inv=scale,
    is_mxfp4=False, is_fp8=False, is_int4=True,
    use_native=False,
)
torch.xpu.synchronize()
print(f"PASSED — output shape: {result.shape}")  # Works!
```

### 2. INT4 MoE GEMM crashes after attention in vLLM (FAILS)

```bash
# Serve any INT4 AutoRound MoE model
vllm serve /path/to/Qwen3.5-35B-A3B-INT4 --device xpu \
    --gpu-memory-utilization 0.90 --max-model-len 1024

# Send any prompt
curl -X POST http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "Qwen3.5-35B-A3B-INT4", "prompt": "Hello", "max_tokens": 1}'
```

**Result**: `RuntimeError: level_zero backend failed with error: 20 (UR_RESULT_ERROR_DEVICE_LOST)` at the first `torch.xpu.moe_gemm(is_int4=True)` call in Layer 1's MoE block, immediately after Layer 1's attention completes.

### 3. MXFP4 MoE GEMM works after attention (PASSES)

```bash
# Same pipeline, MXFP4 model
vllm serve /path/to/gpt-oss-20b --device xpu \
    --gpu-memory-utilization 0.55 --max-model-len 1024
# Works — attention + MXFP4 MoE GEMM in same pipeline, no crash
```

## Diagnostic evidence

We performed exhaustive elimination testing to isolate the root cause:

### What was eliminated

| # | Hypothesis | Test | Result |
|---|-----------|------|--------|
| 1 | INT4 xetla kernel broken on Xe2-LPG | Standalone moe_gemm test (above) | **Eliminated** — works with synthetic + real model weights |
| 2 | Memory pressure | 18 GiB dummy allocation + moe_gemm passes; 4.4 GiB free in pipeline still crashes | **Eliminated** |
| 3 | Resource pool exhaustion (40+ layers) | Per-layer synchronize: crash at Layer 1, call 1 | **Eliminated** — not accumulation |
| 4 | Tensor data corruption from weight shuffle | CPU shuffle vs native IPEX shuffle | **Eliminated** — both crash in pipeline |
| 5 | GatedMLPMOE wrapper bug | Direct `torch.xpu.moe_gemm` bypass (skip GatedMLPMOE entirely) | **Eliminated** — bypass also crashes after attention |
| 6 | Corrupted tensor backing memory | Clone all inputs (W13, scale, input, rows) before moe_gemm | **Eliminated** — freshly cloned tensors also crash |
| 7 | IPEX routing ops (moe_scatter, moe_rows_counts) | Pure Python routing (repeat_interleave + manual indexing, zero IPEX ops) then moe_gemm | **Eliminated** — still crashes |
| 8 | Stale allocator state | `torch.xpu.synchronize()` + `torch.xpu.empty_cache()` + `gc.collect()` before moe_gemm | **Eliminated** — full cleanup doesn't help |
| 9 | Level Zero env vars | `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1`, `COMMANDLISTS_CLEANUP_THRESHOLD=5`, `BATCH_SIZE=4` | **Eliminated** — no effect |

### What was confirmed

After eliminating all other causes, the only remaining variable is: **IPEX attention kernels (`flash_attn_varlen_func` / `chunked_prefill`) executed before `torch.xpu.moe_gemm(is_int4=True)` in the same process.**

The critical test: inside the vLLM forward pass, after Layer 1's attention completes:
1. Full SYCL cleanup: `synchronize()` + `empty_cache()` + `gc.collect()`
2. Pure Python routing: `x.repeat_interleave()`, manual row counting — zero IPEX ops
3. `torch.xpu.moe_gemm(..., is_int4=True)` with these Python-constructed tensors
4. **Result: DEVICE_LOST**

This proves the Level Zero SYCL context is irrecoverably corrupted by the attention kernel dispatch on Xe2-LPG. No user-space cleanup can recover it.

## Affected models

| Model | Layers | Experts | Quant | xetla kernel | Isolated | After attention |
|-------|--------|---------|-------|-------------|----------|-----------------|
| GPT-OSS-20B | 24 | 36 | MXFP4 | `group_mm_mxfp4_out_marlin` | PASS | **PASS** |
| Qwen3.5-35B-A3B | 40 | 256 | INT4 | `group_mm_int4_out_marlin` | PASS | **DEVICE_LOST** |
| GLM-4.7-Flash | 47 | 64 | INT4 | `group_mm_int4_out_marlin` | PASS | **DEVICE_LOST** |

## Key observation

MXFP4 (`group_mm_mxfp4_out_marlin`) is resilient to whatever state the attention kernels leave in the SYCL context, while INT4 (`group_mm_int4_out_marlin`) is not. This suggests the INT4 xetla kernel makes assumptions about the SYCL queue/context state that are violated after attention kernel execution on Xe2-LPG.

This does NOT happen on discrete GPUs (e.g. B580 Xe2-HPG) — only on the Xe2-LPG iGPU with shared memory architecture.

## Expected behavior

`torch.xpu.moe_gemm(is_int4=True)` should work correctly after `flash_attn_varlen_func` / `chunked_prefill` in the same forward pass, just as `torch.xpu.moe_gemm(is_mxfp4=True)` does.

## Possible root causes (for Intel investigation)

1. **Attention kernel leaves dirty SYCL queue state** — barriers, events, or command list state that the INT4 xetla kernel template doesn't handle
2. **Level Zero iGPU driver bug** — the shared-memory UMA driver path doesn't properly reset context between heterogeneous kernel dispatches (attention → MoE GEMM)
3. **INT4 xetla kernel assumes clean context** — `group_gemm_int4_marlin_impl.h` may assume a pristine queue state that the MXFP4 kernel doesn't require

## IPEX code references

| File | Relevance |
|------|-----------|
| `linear_fusion.py:235-291` | `fused_moe_experts()` — calls moe_gemm (crash site) |
| `intrinsic/__init__.py:443` | Python `moe_gemm()` dispatch |
| `moe_gemm.cpp:188` | C++ `fused_moe_gemm_persistent` — xetla dispatch |
| `group_gemm_int4_marlin_impl.h` | INT4 xetla kernel — crashes after attention |
| `group_gemm_mxfp4_marlin_impl.h` | MXFP4 xetla kernel — survives after attention |
| IPEX flash attention | `flash_attn_varlen_func` — source of context pollution |
