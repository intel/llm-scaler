# Comment for vLLM Issue #30359 (QeRL RFC)

**Post at:** https://github.com/vllm-project/vllm/issues/30359

---

## Real-world impact on Intel Lunar Lake (32GB shared memory)

We've been benchmarking LLMs on an **MSI Claw 8 AI+** (Intel Core Ultra 7 258V, Arc 140V Xe2 iGPU, 32GB LPDDR5x shared between CPU and GPU) using vLLM's XPU backend. The pre-quantized model loading peak memory problem described in this RFC is a key blocker for running larger models on shared-memory iGPU platforms.

### The problem in practice

On shared-memory iGPU, CPU and GPU share the same 32GB physical memory pool. Both `process_weights_after_loading` (layer-by-layer repacking) and initial model loading contribute to peak memory. While the layer-by-layer processing itself adds minimal overhead (~one layer's worth), the total memory pressure comes from the initial load plus runtime allocations (KV cache, IPEX kernel buffers, MoE expert shuffling).

**Models that OOM:**
- Qwen3.5-35B-A3B AutoRound INT4: ~18 GiB on disk → OOM + GPU DEVICE_LOST on 32GB
- GLM-4.7-flash AutoRound INT4: 27B model → OOM → DEVICE_LOST
- Qwen3-30B-A3B GPTQ INT4: Loads 15.7 GiB weights, OOMs during MoE expert weight shuffle

### Loading peak analysis (corrected)

Both AutoRound INT4 and sym_int4 online use **layer-by-layer** `process_weights_after_loading` — each layer is processed and the old format is freed before the next layer. So the peak is NOT "old format + new format for the entire model" but rather **initial load size + ~one layer overhead**:

| Quant Method | Initial Load | Loading Peak (shared mem) | Final Model | Single-user tok/s |
|---|---|---|---|---|
| BF16 (none) | 17.66 GiB | ~18 GiB | 17.66 GiB | ~5 |
| FP8 (online) | 17.66 GiB | ~18 GiB | 11.22 GiB | ~8.5 |
| **sym_int4 (online)** | 17.66 GiB (BF16) | **~18 GiB** | **8.11 GiB** | **14.7** |
| **AutoRound INT4** | ~9 GiB | **~9 GiB** | ~9 GiB | ~14 (est.) |

**AutoRound INT4 is the preferred quantization** when it fits — smaller on disk, lower initial load (~9 vs ~18 GiB), better quality (calibration-based), and same inference speed. sym_int4 online quantization is the fallback when no pre-quantized INT4 model exists.

### Where the OOM actually happens

For larger models like Qwen3-30B-A3B (~18 GiB INT4 on disk), the layer-by-layer weight processing itself isn't the problem — the initial load fits in 32GB. The OOM likely occurs during:
1. **MoE expert weight shuffle** — reshaping/permuting expert weights may temporarily duplicate large tensors
2. **KV cache pre-allocation** — vLLM pre-allocates KV cache blocks after model loading
3. **IPEX kernel buffer allocation** — internal buffers for quantized inference kernels
4. **CUDA graph / warmup profiling** — additional transient memory during engine startup

### What would help

1. **Memory-efficient MoE weight loading** — process MoE experts one at a time rather than shuffling all experts simultaneously
2. **Lazy KV cache allocation** — defer KV cache pre-allocation or reduce default block count on memory-constrained platforms
3. **The layerwise approach in this RFC** — while current `process_weights_after_loading` is already layer-by-layer, extending this to cover the full load-process-allocate pipeline would help memory-constrained platforms

This is especially important for **integrated GPU platforms** (Lunar Lake, Meteor Lake, Arrow Lake) where CPU and GPU share the same memory pool — every byte of transient allocation during startup competes with the model weights and KV cache.

Full findings documented at: https://github.com/MegaStood/llm-scaler/blob/claude/check-lunar-lake-compatibility-CB5w6/LUNAR_LAKE_COMPATIBILITY.md
