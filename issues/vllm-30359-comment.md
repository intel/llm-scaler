# Comment for vLLM Issue #30359 (QeRL RFC)

**Post at:** https://github.com/vllm-project/vllm/issues/30359

---

## Real-world impact on Intel Lunar Lake (32GB shared memory)

We've been benchmarking LLMs on an **MSI Claw 8 AI+** (Intel Core Ultra 7 258V, Arc 140V Xe2 iGPU, 32GB LPDDR5x shared between CPU and GPU) using vLLM's XPU backend. The pre-quantized model loading peak memory problem described in this RFC is the single biggest blocker for us.

### The problem in practice

**AutoRound INT4 models OOM during loading** because `process_weights_after_loading` unpacks INT4→FP16, doubling peak memory:
- Qwen3.5-35B-A3B AutoRound INT4: 18 GiB on disk → ~36 GiB peak during FP16 unpack → OOM + GPU DEVICE_LOST on 32GB
- GLM-4.7-flash AutoRound INT4: 27B model peaks >24GB during unpack → OOM → DEVICE_LOST
- Qwen3-30B-A3B GPTQ INT4: Loads 15.7 GiB, OOMs during MoE expert weight shuffle

### Our workaround: sym_int4 online quantization

We use Intel's `--quantization sym_int4` (from `vllm_for_multi_arc.patch`) with `VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=1` to quantize BF16 models on CPU before sending INT4 weights to GPU. This avoids the GPU-side loading peak but still requires the full BF16 model in system memory during quantization:

| Quant Method | On Disk | System Peak (CPU+GPU) | Final GPU | Single-user tok/s |
|---|---|---|---|---|
| BF16 (none) | 17.66 GiB | 17.66 GiB | 17.66 GiB | ~5 |
| FP8 (online) | 17.66 GiB | ~24 GiB (BF16 + FP8 coexist) | 11.22 GiB | ~8.5 |
| **sym_int4 (online)** | 17.66 GiB | **~26 GiB** (BF16 + INT4 coexist) | **8.11 GiB** | **14.7** |
| AutoRound INT4 | ~9 GiB | **~18 GiB** (INT4→FP16 unpack, 2x!) | ~9 GiB | ~14 (est.) |

**AutoRound INT4 is the preferred quantization** when it fits — smaller on disk, lower system peak, better quality (calibration-based), and same inference speed. sym_int4 online quantization is only the fallback when no pre-quantized INT4 model exists or when AutoRound's 2x loading peak still causes OOM.

On Lunar Lake, CPU and GPU share the same 32GB memory pool. sym_int4 online quantization peaks at ~26 GiB system memory (BF16 weights + INT4 output coexist on CPU during quantization), leaving just enough for OS + KV cache. This limits sym_int4 to **≤ ~10B dense models** on 32GB.

For larger models like the 35B-A3B MoE (~70 GiB BF16), sym_int4 is impossible — the BF16 weights alone don't fit in 32GB RAM. AutoRound INT4 (~18 GiB on disk) is the right format for these models, but the 2x loading peak (INT4→FP16 unpack during `process_weights_after_loading`) pushes it to ~36 GiB and OOMs.

### What would help

The **layerwise weight processing** proposed in this RFC would directly solve our problem. If `CompressedTensorsWNA16` could process and repack weights layer-by-layer (loading INT4 → repacking for IPEX kernel → freeing the original before loading next layer), the 35B-A3B model at ~18 GiB INT4 would fit easily in 32GB with no spike.

This is especially important for **integrated GPU platforms** (Lunar Lake, Meteor Lake, Arrow Lake) where CPU and GPU share the same memory pool — there's no separate VRAM to absorb the loading spike.

Full findings documented at: https://github.com/MegaStood/llm-scaler/blob/claude/check-lunar-lake-compatibility-CB5w6/LUNAR_LAKE_COMPATIBILITY.md
