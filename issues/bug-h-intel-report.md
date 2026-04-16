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

MXFP4 (`group_mm_mxfp4_out_marlin`) is resilient to whatever state the attention kernels leave in the SYCL context, while INT4 (`group_mm_int4_out_marlin`) is not.

This does NOT happen on discrete GPUs (e.g. B580 Xe2-HPG) — only on the Xe2-LPG iGPU with shared memory architecture.

## Source code analysis: INT4 vs MXFP4 kernel differences

Reading the IPEX xetla kernel source reveals these are fundamentally different kernels:

| Aspect | INT4 (`XEGEMM_INT4_marlin`) | MXFP4 (`XEGEMM_MXFP4_marlin`) |
|--------|------|-------|
| **Kernel class** | `int4_group_gemm_universal_t` | `persistent_mxfp4_group_gemm_universal_t` |
| **Dispatch model** | **Non-persistent** (one WG per tile, relies on SYCL scheduler) | **Persistent** (atomic self-scheduling via `atomic_buffer`) |
| **Compute policy** | `compute_policy_int4_dequantize_v2` | `compute_policy_mxfp4_dequantize` |
| **Input dtype** | `sycl::half` (FP16) | `sycl::ext::oneapi::bfloat16` (BF16) |
| **Scale dtype** | `sycl::half` (FP16) | `uint8_t` |
| **GEMM tile (wg_m×wg_n)** | **256×256** | **128×128** |
| **GEMV tile** | 8×64 (same) | 8×64 (same) |
| **Zero-point param** | `weight_zp` (always nullptr) | N/A |

### Primary suspect: 4× larger workgroup tiles

The INT4 GEMM policy uses 256×256 workgroup tiles vs MXFP4's 128×128. On Xe2-LPG (64 EUs), this demands significantly more SLM, registers, and synchronization barriers. After attention kernels leave residual SYCL state, the INT4 kernel's larger resource demands may push past a threshold.

**Tile policy is selected by `average_m = total_tokens / num_experts`:**
- `average_m ≤ 4` → GEMV (8×64) — **same tiles for both INT4 and MXFP4**
- `average_m > 128` → GEMM (256×256 INT4 vs 128×128 MXFP4)

If sending a 1-token prompt (GEMV policy) works after attention, the tile size is confirmed as root cause.

### Secondary suspect: Non-persistent vs persistent dispatch

Persistent kernels self-schedule via atomic counters — they don't depend on the SYCL runtime scheduler. Non-persistent kernels (INT4) rely on the runtime. If attention corrupts scheduler state, non-persistent dispatch fails.

### Third suspect: FP16 vs BF16 input type mismatch

The INT4 C++ hardcodes `using dtype_a = sycl::half` but vLLM default is BF16. If BF16 tensors reach the INT4 kernel without explicit cast, this could cause issues on Xe2-LPG.

## Expected behavior

`torch.xpu.moe_gemm(is_int4=True)` should work correctly after `flash_attn_varlen_func` / `chunked_prefill` in the same forward pass, just as `torch.xpu.moe_gemm(is_mxfp4=True)` does.

## Suggested fixes (for Intel investigation)

1. **Reduce INT4 GEMM tile to 128×128** (matching MXFP4) in `GEMM_INT4_marlin.h` — if the 256×256 tile exceeds Xe2-LPG resources after attention
2. **Switch INT4 to persistent dispatch** — use `persistent_int4_group_gemm_universal_t` (if it exists or can be templated from the MXFP4 version)
3. **Fix attention kernel SYCL cleanup** — ensure `flash_attn_varlen_func` releases all SLM/barriers before returning
4. **Level Zero iGPU driver hardening** — prevent residual state from one kernel affecting subsequent dispatches
5. **Add Xe2-LPG-specific tile policies** — detect iGPU at runtime and use smaller tiles

## IPEX source files (analyzed)

| File | Relevance |
|------|-----------|
| `XEGEMM_INT4_marlin.cpp:61-115` | INT4 C++ dispatch: `group_mm_int4_out_marlin()` |
| `XEGEMM_MXFP4_marlin.cpp:54-96` | MXFP4 C++ dispatch: `group_mm_mxfp4_out_marlin()` |
| `XEGEMM_INT4_marlin.h:80-178` | INT4 launch: tile policy switch (GEMV→GEMM) |
| `XEGEMM_MXFP4_marlin.h:56-142` | MXFP4 launch: tile policy switch |
| `xetla/GEMM_INT4_marlin.h:52-62` | INT4 GEMM policy: **wg 256×256** |
| `xetla/GEMM_MXFP4_marlin.h:53-63` | MXFP4 GEMM policy: **wg 128×128** |
| `kernels/GEMM/group_gemm_int4_marlin_impl.h` | INT4 xetla kernel: `int4_group_gemm_universal_t` |
| `kernels/GEMM/group_gemm_mxfp4_marlin_impl.h` | MXFP4 xetla kernel: `persistent_mxfp4_group_gemm_universal_t` |
| `linear_fusion.py:235-291` | `fused_moe_experts()` — calls moe_gemm (crash site) |
| `intrinsic/__init__.py:412-533` | Python `moe_gemm()` dispatch |
| `group_gemm_mxfp4_marlin_impl.h` | MXFP4 xetla kernel — survives after attention |
| IPEX flash attention | `flash_attn_varlen_func` — source of context pollution |
