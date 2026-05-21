# ESIMD Prefill Flash Attention Kernel Development

## Overview

Custom ESIMD Flash Attention kernel for Intel XPU, targeting Qwen3.5-27B prefill.
Designed to replace xetla FMHA which is being deprecated.

**Target model**: Qwen3.5-27B (HD=256, 24Q/4KV heads, 64 layers: 16 full_attention + 48 GDN)
**Target hardware**: Intel XPU (e223, BMG architecture)
**Compilation**: doubleGRF (256 registers), AOT for bmg-g21

---

## Task History & File Mapping

| Task | File | Description | Status |
|------|------|-------------|--------|
| Task 0 | `test_dpas_hd256.sycl` | Verify DPAS works for HD=256 dot product | PASS |
| Task 1 | `flash_attn_minimal.sycl` | Single-thread flash attention with online softmax | PASS |
| Task 2 | `flash_attn_parallel_q.sycl` | Multi-thread parallel Q (8 threads, Q_GROUP=32) | PASS |
| Task 3 | `flash_attn_slm.sycl` | SLM cooperative V load | PASS |
| Task 4 | `flash_attn_vec.sycl` | Vectorized PV + V double buffer | PASS |
| Task 4B | `flash_attn_dpas.sycl` | DPAS for QK^T and PV | PASS |
| Task 5 | `flash_attn_mhead.sycl` | Multi-Head + GQA (12Q/2KV) | PASS |
| Task 6 | `flash_attn_causal.sycl` | Causal mask + chunked prefill (q_start_pos) | PASS |
| Task 7 | `bench_task7.py` | Benchmark vs SDPA/xetla (GO/NO-GO) | 31x → 9x |
| Task 8 | `flash_attn_paged.sycl` | Paged KV cache (block_table) | PASS |
| Task 9 | `flash_attn_batch.sycl` | Variable-length batched prefill | PASS |
| Task 10 | (inline test) | 128K long sequence validation | PASS |
| Task 11 | vLLM integration | Patch flash_attn.py + prefill_attn_ops.py | IN PROGRESS |

**Final production kernel**: `flash_attn_batch.sycl` (combines all features)
**Optimized but single-head**: `flash_attn_opt.sycl` (best per-tile performance, used for perf development)

---

## Performance History

| Version | Per-tile | vs SDPA | Key Change |
|---------|---------|---------|------------|
| Task 6 baseline (causal) | 52us | 31x | First correct causal kernel |
| + K cooperative load (SLM) | ~45us | ~27x | Eliminated 8x redundant K reads |
| + Full K tile in SLM | ~45us | ~27x | Reduced K barriers from 32→4 |
| + VNNI-on-store | 33.4us | 21x | K stored pre-packed in SLM, no runtime transpose for QK |
| + Vectorized softmax/P | 21.8us | 13x | Merged 2 exp passes into 1, SIMD control flow |
| + slm_scatter for K store | **14.1us** | **~9x** | Scatter write replaces 512 scalar stores |

**Final: ~9x slower than SDPA (xetla)** at typical workloads (Q=256, KV=4096).

---

## Architecture & Design Decisions

### Kernel Parameters (flash_attn_batch)
```
HD = 256           (Qwen3.5-27B head dimension)
KV_TILE = 32       (KV rows per outer loop iteration)
Q_PER_THREAD = 4   (Q rows processed per thread)
WG_SIZE = 8        (threads per workgroup)
Q_GROUP = 32       (Q rows per workgroup = WG_SIZE * Q_PER_THREAD)
DPAS_M = 8, DPAS_K = 16, DPAS_N = 16  (DPAS tile dimensions)
```

### SLM Layout (48KB total, limit 64KB)
```
[0, 32KB):    V double buffer (2 × KV_TILE × HD × fp16 = 2 × 16KB)
[32KB, 48KB): K VNNI tiles (K_ITERS × N_GROUPS × 512B = 16 × 2 × 512B)
```

### Compilation
```bash
# Build single kernel for development:
KERNEL=flash_attn_batch USE_DOUBLE_GRF=1 TORCH_XPU_ARCH_LIST=bmg-g21 \
  python3 setup_test.py build_ext --inplace

# Requires doubleGRF (256 GRF) due to:
#   accum[4,256] fp32 = 64 GRF
#   Q_data[4,256] fp16 = 32 GRF
#   qk_acc[2] × 128 fp32 = 16 GRF
#   + temporaries
```

---

## Why ~9x Gap vs SDPA (xetla)

### Root cause: VNNI transpose is fundamentally expensive

The DPAS B operand requires VNNI format (row-pair interleaving). This means:
1. **K data**: loaded row-major from global → must be transposed + pair-packed → stored to SLM via scatter
2. **V data**: loaded row-major to SLM → read pairs + interleaved at PV compute time

xetla avoids this entirely by using **hardware 2D block load with transpose** (`lsc_load_2d`), which produces VNNI-ready data directly from memory with no software swizzle.

### Breakdown of remaining per-tile cost (~14us)
| Phase | Est. Cost | Notes |
|-------|-----------|-------|
| K VNNI scatter store | ~4us | 64 slm_scatter<u32,8> calls per tile |
| PV V VNNI pack | ~4us | 32 iterations of pair-load + interleave from SLM |
| Softmax/P exp | ~2us | Vectorized but still 32 exp per Q row |
| DPAS compute (QK+PV) | ~2us | 64 DPAS calls — very fast |
| Barriers + misc | ~2us | 1 K barrier + 2 V barriers + score extract |

### What would close the gap further
- **2D block load with transpose**: eliminates VNNI swizzle entirely (~3-5x improvement)
  - BUT: incompatible with paged KV cache (requires contiguous memory surface)
  - Could be a "fast path" for non-paged prefill
- **V VNNI pre-store**: blocked by pair-conflict (two threads own the two rows of a VNNI pair)
- **Larger KV_TILE**: reduces overhead amortization but SLM limit (64KB) prevents >32

---

## Paths That Don't Work

| Approach | Why It Failed |
|----------|--------------|
| K from global memory (no SLM) | 8 threads all load same K → 8x bandwidth waste → 53us/tile (worse) |
| V VNNI pre-store | VNNI pairs require data from 2 rows owned by different threads → need atomic/extra barrier |
| Fast polynomial exp (vectorized) | Integer truncation + merge direction bugs with NEG_INF → NaN. Scalar exp fallback works correctly |
| Reducing barriers to 2 per tile | Minimal improvement (6→4→2 barriers), barrier is not dominant cost |
| singleGRF (128 registers) | Spills with DPAS (accum + qk_acc + Q_data > 128 GRF) |

---

## vLLM Integration (Task 11)

### Files Modified
1. **`/llm/shaojun/code/llm-scaler-vllm-xpu/vllm/v1/attention/backends/flash_attn.py`**
   - Added ESIMD prefill branch before `flash_attn_varlen_func` call
   - Conditions: XPU + HD=256 + prefill + causal + kv_len>=32
   - Falls back to xetla if import fails or conditions not met

2. **`/llm/shaojun/code/llm-scaler/vllm/custom-esimd-kernels-vllm/python/custom_esimd_kernels_vllm/prefill_attn_ops.py`** (new)
   - Kernel .so loader module

### Known Issues
- **KV < 32 hang**: kernel requires kv_len >= KV_TILE (32). Guarded in vLLM integration.
- **KV=1 hang**: VNNI pairing reads out-of-bounds row. Guarded by kv_len>=32.
- **barrier + early return**: if q_start >= q_len, thread returns before barrier → potential deadlock for partial last WG. Not triggered in practice (Q always >= 32 in prefill).

### TODO for production
- [ ] Fix KV < 32 edge case in kernel (not just vLLM guard)
- [ ] Fix barrier early-return issue
- [ ] Merge flash_attn_batch.sycl into main setup.py
- [ ] Performance optimization: 2D load fast path for contiguous KV
- [ ] Test with actual Qwen3.5-27B end-to-end generation quality
- [ ] Benchmark TTFT improvement vs xetla baseline
