# GLM-4.7-Flash INT4 MoE: Marlin Shuffle DEVICE_LOST on Lunar Lake XPU

## Status: Bug E (shuffle OOM) FIXED — Bug H (attention→MoE SYCL state pollution) OPEN

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
| **INT4 AutoRound** | Yes | **Yes** — must reshuffle at runtime | Bug E fixed (CPU shuffle), Bug H open (40+ layers) |
| **GPTQ** | No | N/A — requires CUDA-only Marlin kernels | No |
| **AWQ** | No | N/A — requires CUDA-only Marlin kernels | No |
| **FP8** | Yes | No — uses `GatedMLPMOE(is_fp8=True)` | No — 30B model → ~30 GiB, won't fit |
| **GGUF** | No (vLLM) | N/A — vLLM GGUF is CUDA-only | Yes — via llama.cpp with SYCL backend |

**Two separate issues for INT4 MoE on shared-memory iGPU:**
1. **Bug E (FIXED)**: marlin_shuffle_weight OOM during init — solved by CPU shuffle
2. **Bug H (OPEN)**: Level Zero SYCL runtime state pollution from IPEX attention
   kernels (flash_attn_varlen_func / chunked_prefill) corrupts the SYCL context,
   causing subsequent `torch.xpu.moe_gemm(is_int4=True)` to crash with DEVICE_LOST.
   The INT4 kernel works perfectly in isolation — the crash only occurs after
   attention ops have executed in the same SYCL queue.

MXFP4 avoids both issues: no shuffle needed (Bug E), and its xetla kernel
(`group_mm_mxfp4_out_marlin`) appears resilient to the attention state pollution.

### Implication for shared-memory iGPUs
On shared-memory systems, **MXFP4 is the only vLLM-compatible quantization format
that works for INT4-class MoE models** on Lunar Lake Xe2 iGPU. INT4 AutoRound models
load successfully (Bug E fixed) but crash at first inference due to Bug H (attention
→ MoE SYCL state pollution). The alternative path is llama.cpp GGUF via SYCL backend.

## Bug H: Level Zero SYCL State Pollution — Attention Kernels Break INT4 MoE GEMM

**Status: OPEN — Level Zero SYCL context pollution from IPEX attention kernels
(flash_attn_varlen_func / chunked_prefill) causes subsequent
`torch.xpu.moe_gemm(is_int4=True)` to crash with DEVICE_LOST on Lunar Lake Xe2 iGPU.**

**The fix must come from Intel — either IPEX attention kernel fix or Level Zero driver fix.**

### Distinction from Bug E

Bug E (DEVICE_LOST during marlin_shuffle) was solved by the CPU shuffle workaround.
Bug H is a **separate** runtime issue: the SYCL queue state becomes corrupted after
IPEX attention kernels execute, causing the next INT4 MoE GEMM dispatch to crash.
The INT4 MoE GEMM kernel itself works perfectly in isolation.

### Definitive root cause: SYCL runtime state pollution

Through exhaustive elimination testing, we determined:

1. `torch.xpu.moe_gemm(is_int4=True)` **works in isolation** — with synthetic
   tensors, real model weights (16+ GiB allocated), 8093 dummy tensors, KV cache
   tensors, and all IPEX routing ops (topk_softmax, moe_rows_counts, moe_scatter,
   moe_gather). No crash.

2. The **same call crashes** inside the full vLLM forward pass, where IPEX attention
   kernels (`flash_attn_varlen_func` → `chunked_prefill`) execute before the MoE
   layer. The crash occurs at the very first MoE GEMM call (Layer 1, W13).

3. The corruption is in the **Level Zero SYCL runtime state**, not in tensor data:
   - Cloning all input tensors (reordered, W13, scale, rows_for_experts) before
     calling moe_gemm → still crashes
   - `torch.xpu.synchronize()` between attention and MoE → does NOT clear it
   - `torch.xpu.empty_cache()` between attention and MoE → does NOT clear it

4. **Memory is NOT the issue**: 18 GiB dummy allocation + moe_gemm passes in
   isolation. Inside vLLM with 4.4 GiB free, it still crashes. The crash is not
   caused by insufficient memory.

### Evidence chain

```
# PASSES — INT4 MoE GEMM in isolation (no attention ops)
python test_int4_moe_gemm_xpu.py
→ Test 1 (MXFP4): PASSED
→ Test 2 (INT4 fused): PASSED
→ Test 3 (INT4 native): PASSED

# PASSES — INT4 MoE GEMM with real model weights + 8093 dummy tensors + routing
python stress_test.py  # 16 GiB allocated, routing + scatter + moe_gemm
→ moe_gemm(is_int4=True): PASSED

# CRASHES — INT4 MoE GEMM after attention in vLLM forward pass
vllm serve Qwen3.5-35B-A3B-INT4 --device xpu
→ L1 pre-attn: OK (0.39 GiB free)
→ L1 post-attn: OK
→ L1 pre-mlp: OK
→ L1 W13 GEMM: DEVICE_LOST  ← crash at first moe_gemm after attention

# CRASHES — Even with cloned tensors (proves it's not tensor state)
# Inside vLLM forward, after attention:
reordered_clone = reordered.clone()
W13_clone = fusion.W13.clone()
scale_clone = fusion.w1_scale_inv.clone()
rows_clone = rows_for_experts.clone()
torch.xpu.synchronize()
torch.xpu.moe_gemm(reordered_clone, W13_clone, rows_clone, ...)
→ DEVICE_LOST  ← freshly cloned tensors still crash

# CRASHES — Pure Python routing (no IPEX ops at all) after attention
# Inside vLLM forward apply(), after L1 attention completes:
torch.xpu.synchronize()      # flush all pending ops
torch.xpu.empty_cache()      # free unused memory
gc.collect()                  # Python garbage collection
# Then pure Python routing: x.repeat_interleave(), manual row counting
# No moe_scatter, no moe_rows_counts — zero IPEX routing ops
torch.xpu.moe_gemm(repeated_input, W13, manual_rows, ..., is_int4=True)
→ DEVICE_LOST  ← attention alone poisons the context

# NEVER REACHED — IPEX routing test
# Device already dead after pure Python test above
```

**Final elimination**: The pure Python routing test (Test 2) removes ALL IPEX
ops between attention and moe_gemm — no `moe_scatter()`, no `moe_rows_counts()`,
no `topk_softmax()`. Only standard PyTorch ops (`repeat_interleave`, tensor
indexing) and then `torch.xpu.moe_gemm`. Still crashes. This proves the Level
Zero context is irrecoverably poisoned by the attention kernel alone.

### GatedMLPMOE bypass (works in isolation, crashes in pipeline)

A bypass was developed that calls `torch.xpu.moe_gemm` directly instead of going
through `GatedMLPMOE.forward()` → `_IPEXGatedMLPMOEXPU` → `fused_moe_experts()`.
The bypass uses IPEX routing ops directly:

```python
# In ipex_quant.py XPUGPTQMarlinMoEMethod.apply():
# 1. torch.ops.torch_ipex.topk_softmax() — routing
# 2. torch.ops.torch_ipex.moe_rows_counts() — expert assignment
# 3. torch.ops.torch_ipex.moe_scatter() — input reordering
# 4. torch.xpu.moe_gemm(is_int4=True) — W13 GEMM (gate+up)
# 5. SiLU activation + element-wise multiply
# 6. torch.xpu.moe_gemm(is_int4=True) — W2 GEMM (down)
# 7. torch.ops.torch_ipex.moe_gather() — output gathering
```

This bypass works perfectly in isolation but still crashes in the full vLLM
pipeline due to the attention state pollution.

### Affected models

| Model | Layers | Quant | Kernel Path | Isolated Test | vLLM Pipeline |
|-------|--------|-------|-------------|---------------|---------------|
| GPT-OSS-20B | 24 | MXFP4 | `group_mm_mxfp4_out_marlin` | **Works** | **Works** |
| Qwen3.5-35B-A3B | 40 | INT4 | `group_mm_int4_out_marlin` | **Works** | **DEVICE_LOST** |
| GLM-4.7-Flash | 47 | INT4 | `group_mm_int4_out_marlin` | **Works** | **DEVICE_LOST** |

The INT4 kernel works on Xe2-LPG — earlier diagnosis of "kernel broken on Xe2-LPG"
was incorrect. MXFP4 models work in the pipeline because their xetla kernel
(`group_mm_mxfp4_out_marlin`) appears resilient to the attention state pollution.

### Hypotheses tested and eliminated

| # | Hypothesis | Test | Result |
|---|-----------|------|--------|
| 1 | Level Zero resource pool exhaustion from 40+ layers | Per-layer synchronize instrumentation | **Eliminated** — crash at Layer 1, call 1 |
| 2 | INT4 xetla kernel broken on Xe2-LPG architecture | Standalone test_int4_moe_gemm_xpu.py | **Eliminated** — kernel works with synthetic + real weights |
| 3 | Memory pressure (model too large) | 18 GiB dummy + moe_gemm; 4.4 GiB free | **Eliminated** — passes in isolation, crashes with headroom |
| 4 | Tensor data corruption from shuffle | CPU shuffle vs native shuffle | **Eliminated** — both crash in pipeline |
| 5 | GatedMLPMOE wrapper bug | Direct moe_gemm bypass | **Eliminated** — bypass also crashes in pipeline |
| 6 | Tensor backing memory state | Clone all inputs before moe_gemm | **Eliminated** — cloned tensors also crash |
| 7 | Level Zero env var tuning | IMMEDIATE_COMMANDLISTS + CLEANUP_THRESHOLD + BATCH_SIZE | **Eliminated** — no effect |
| 8 | IPEX routing ops (moe_scatter etc.) poison queue | Pure Python routing + moe_gemm (no IPEX routing ops) | **Eliminated** — crashes without any IPEX routing ops |
| 9 | Stale allocator state | synchronize + empty_cache + gc.collect before moe_gemm | **Eliminated** — full cleanup doesn't help |
| 10 | SYCL context pollution from attention ops | All above combined | **CONFIRMED** — attention alone poisons context |

### Why MXFP4 is not affected

The MXFP4 kernel (`group_mm_mxfp4_out_marlin`) uses a different xetla template
implementation than the INT4 kernel (`group_mm_int4_out_marlin`). Source code
analysis of the IPEX xetla kernels reveals **5 critical differences**:

#### Kernel-level differences (from IPEX source)

| Aspect | INT4 (`XEGEMM_INT4_marlin.cpp`) | MXFP4 (`XEGEMM_MXFP4_marlin.cpp`) |
|--------|------|-------|
| **Kernel class** | `int4_group_gemm_universal_t` | `persistent_mxfp4_group_gemm_universal_t` |
| **Dispatch model** | Non-persistent (one WG per tile) | **Persistent** (atomic coordination) |
| **Compute policy** | `compute_policy_int4_dequantize_v2` | `compute_policy_mxfp4_dequantize` |
| **Input dtype** | `sycl::half` (FP16) | `sycl::ext::oneapi::bfloat16` (BF16) |
| **Scale dtype** | `sycl::half` (FP16) | `uint8_t` |
| **Accumulator** | `fp16` → `float` | `bf16` → `float` |
| **GEMM tile (wg_m×wg_n)** | **256×256** | **128×128** |
| **GEMV tile** | 8×64 (same) | 8×64 (same) |
| **Zero-point param** | `weight_zp` (always None) | None |

#### Primary suspect: 4× larger workgroup tiles (256×256 vs 128×128)

The INT4 GEMM policy uses 256×256 workgroup tiles — 4× the area of MXFP4's
128×128. On Xe2-LPG (64 EUs), this demands significantly more:
- Shared Local Memory (SLM) per workgroup
- Register file per subgroup
- SYCL barrier synchronization points

After attention kernels leave residual SYCL state (incomplete barriers, unreleased
SLM, stale command lists), the INT4 kernel's larger resource demands may push past
a threshold that causes DEVICE_LOST. MXFP4's smaller 128×128 tiles fit within
available resources even with residual state.

**Tile policy selection is based on `average_m`** (`total_tokens / num_experts`):
- `average_m ≤ 4` → GEMV (8×64) — same for both
- `average_m ≤ 32` → GEMV_16 (16×64) — same for both
- `average_m ≤ 128` → GEMV_32 (32×64) — same for both
- `average_m > 128` → GEMM (256×256 INT4 vs 128×128 MXFP4)

**Testable**: If sending a 1-token prompt (forcing GEMV policy, same tiles as
MXFP4) survives after attention → tile size is the root cause.

#### Secondary suspect: Non-persistent vs persistent dispatch

Persistent kernels manage their own workgroup scheduling via atomic counters
(pre-allocated `atomic_buffer`). Non-persistent kernels rely on the SYCL runtime
scheduler for workgroup dispatch. If attention corrupts the SYCL runtime scheduler
state, non-persistent dispatch (INT4) would fail while persistent dispatch (MXFP4)
remains functional because it self-schedules.

#### Third suspect: FP16 vs BF16 input type

The INT4 C++ kernel hardcodes `using dtype_a = sycl::half` (FP16). vLLM default
is BF16. If the INT4 path receives BF16 input without explicit conversion, this
could cause a type mismatch that only manifests after attention (which outputs BF16).
The MXFP4 kernel natively expects BF16, matching vLLM's default.

#### Source files examined

| File | Purpose |
|------|---------|
| `csrc/gpu/aten/operators/XEGEMM_INT4_marlin.cpp` | INT4 C++: `group_mm_int4_out_marlin()` |
| `csrc/gpu/aten/operators/XEGEMM_MXFP4_marlin.cpp` | MXFP4 C++: `group_mm_mxfp4_out_marlin()` |
| `csrc/gpu/aten/operators/XEGEMM_INT4_marlin.h` | INT4 tile policy switch |
| `csrc/gpu/aten/operators/XEGEMM_MXFP4_marlin.h` | MXFP4 tile policy switch |
| `csrc/gpu/aten/operators/xetla/GEMM_INT4_marlin.h` | INT4 policies: GEMM=256×256 |
| `csrc/gpu/aten/operators/xetla/GEMM_MXFP4_marlin.h` | MXFP4 policies: GEMM=128×128 |
| `csrc/gpu/aten/operators/xetla/kernels/GEMM/group_gemm_int4_marlin_impl.h` | INT4 xetla kernel |
| `csrc/gpu/aten/operators/xetla/kernels/GEMM/group_gemm_mxfp4_marlin_impl.h` | MXFP4 xetla kernel |

#### Note: Marlin shuffle is identical for both

Both MXFP4 and INT4 go through the same `marlin_shuffle_weight()` in
`__init__()` (line 151-153 of `linear_fusion.py`). The "CPU shuffle" in the
SymInt4 path is CPU *quantization* (`ggml_quantize_tensor`), not CPU Marlin
shuffling. The Marlin nibble rearrangement always runs on XPU inside `__init__()`.

### Minimal reproducer

```bash
# Kernel works in isolation:
python vllm/test/test_int4_moe_gemm_xpu.py

# Kernel crashes after attention ops in vLLM pipeline:
vllm serve /path/to/Qwen3.5-35B-A3B-INT4 --device xpu \
    --gpu-memory-utilization 0.90 --max-model-len 1024
# Send any prompt → DEVICE_LOST at Layer 1 MoE GEMM
```

### Required fix (Intel action needed)

This bug is in the interaction between IPEX attention kernels and the Level Zero
SYCL runtime on Xe2-LPG. The fix must come from Intel:

1. **IPEX attention kernel fix** — ensure `flash_attn_varlen_func` / `chunked_prefill`
   leaves the SYCL queue in a clean state for subsequent kernel dispatches
2. **Level Zero driver fix** — harden the L0 runtime against queue state leakage
   between kernel dispatches
3. **vLLM v0.19+ oneDNN INT4 path** — bypass IPEX entirely via `vllm-xpu-kernels`
   oneDNN-backed INT4 GEMM (`XPUExpertsInt4`), tracked in RFC vllm-project/vllm#33214

### Workarounds (current)

- **Use MXFP4 models** — works on Lunar Lake (proven with GPT-OSS-20B)
- **Use llama.cpp GGUF** — own SYCL kernels, validated on consumer GPUs
- **Wait for Intel llm-scaler v0.19** — oneDNN INT4 path avoids IPEX xetla entirely

### IPEX code references

| File | What it does |
|------|-------------|
| `linear_fusion.py:235-291` | `fused_moe_experts()` — calls moe_gemm twice (W13, W2) |
| `intrinsic/__init__.py:443` | Python `moe_gemm()` — routes to INT4 or MXFP4 |
| `moe_gemm.cpp:188` | C++ `fused_moe_gemm_persistent` — dispatches xetla kernel |
| `group_gemm_int4_marlin_impl.h` | INT4 xetla kernel template — crashes after attention |
| `group_gemm_mxfp4_marlin_impl.h` | MXFP4 xetla kernel template — resilient to pollution |
| `xetla_arch.h` | Architecture mapping — LNL → XeHpc |
| IPEX attention kernels | `flash_attn_varlen_func` — source of SYCL state pollution |

---

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

## Future Development: Multi-Device Architecture

### Planned Setup

| Device | Role | Hardware |
|--------|------|----------|
| 2x DGX Spark | LLM server (120B+ models) | Grace Blackwell, 128GB each |
| MSI Claw 8 AI+ | Portable LLM (20B MoE) + vLLM serving | Lunar Lake, 32GB |
| MSI Claw A1M | Edge client, voice I/O, lightweight tasks | Meteor Lake, 16GB |

### Claw A1M as Edge Client

The A1M (16GB) is too constrained to run large LLMs and ASR/TTS simultaneously.
Planned architecture splits workloads across devices:

```
A1M (OpenClaw UI) ──→ DGX Spark (120B LLM via vLLM)
    ├── ASR: Whisper on NPU (zero iGPU memory cost)
    ├── ASR: SenseVoice/Qwen2-Audio on CPU (fallback, no NPU op support)
    ├── TTS: local on CPU/iGPU
    └── If standalone: 4B LLM on iGPU + Whisper on NPU
```

### OpenVINO NPU Inference on Meteor Lake

The Intel NPU (Neural Processing Unit) on 155H can offload small models,
freeing iGPU memory for other tasks:

- **Runtime**: OpenVINO 2026.0+ (`pip install openvino`)
- **NPU driver**: intel/linux-npu-driver v1.30.0+
- **Supported on NPU**: Whisper (built-in GenAI pipeline), small vision models
- **CPU/GPU only**: Qwen2-Audio, SenseVoice, Paraformer (complex ops not on NPU)
- **Convert any PyTorch model**: `ov.convert_model()` → run on CPU/GPU/NPU

```python
import openvino_genai as ov_genai
# ASR on NPU — dedicated hardware, no iGPU memory competition
pipe = ov_genai.WhisperPipeline("whisper-base", device="NPU")
result = pipe.generate("audio.wav")
```

Key insight: NPU has its own dedicated compute — it does NOT share iGPU memory.
This makes it ideal for always-on ASR/TTS while the iGPU handles LLM inference.

## vLLM Native Build Steps (from llm-scaler Dockerfile)

### Build Order

The build must follow this order because `--no-build-isolation` means each
step relies on packages installed in the previous step:

```
Step 1: System deps        (oneAPI, compilers, Python 3.12)
Step 2: Set env vars        (VLLM_TARGET_DEVICE=xpu, CPATH for DPCPP)
Step 3: Clone vLLM v0.14.0 + apply vllm_for_multi_arc.patch
Step 4: pip install -r requirements/xpu.txt   ← PyTorch XPU, IPEX, etc.
        pip install arctic-inference==0.1.1
Step 5: pip install --no-build-isolation .     ← builds vLLM against Step 4
Step 6: Build vllm-xpu-kernels (commit 4c83144)
Step 7: Fix triton-xpu==3.6.0
Step 8: Configure production env vars in ~/.bashrc
```

`--no-build-isolation` tells pip to use the already-installed PyTorch XPU
and IPEX from Step 4, instead of creating a fresh isolated build environment
(which would pull CPU PyTorch and break the XPU build).

### Built .so Libraries

| Library | Source | Purpose |
|---------|--------|---------|
| `vllm_int4_for_multi_arc.so` | bigdl-core | INT4 quantization kernel for XPU |
| `vllm_xpu_kernels*.so` | vllm-xpu-kernels | XPU compute kernels (attention, etc.) |
| `libtriton_xpu*.so` | triton-xpu | Triton JIT compiler for XPU |
| IPEX `*.so` libraries | intel-extension-for-pytorch | Marlin kernel, MoE fusion, etc. |

The `vllm_for_multi_arc.patch` (466KB) modifies vLLM's Python source to add
XPU device support. It is NOT a separate .so — it patches the Python code
which then calls into the .so kernels above for actual GPU compute.

Referenced by env var:
```bash
export VLLM_QUANTIZE_Q40_LIB="<python-site>/vllm_int4_for_multi_arc.so"
```

### Docker Build vs Native Build

**Docker build** — everything happens inside the container:
```bash
cd ~/llm-scaler
docker build -f vllm/docker/Dockerfile -t vllm-xpu .
# Run with GPU access:
docker run --device /dev/dri -p 8000:8000 vllm-xpu \
    vllm serve /models/gpt-oss-20b --device xpu
```
- No need to build vLLM locally first — Docker does it all
- Pins to specific versions (reproducible)
- Adds ~15-20GB disk overhead for the image
- GPU access via `--device /dev/dri`
- Models mounted via `-v /shared/models:/models`

**Native build** — installs directly on the host:
```bash
sudo bash ~/llm-scaler/vllm/scripts/install_vllm_native.sh
# Or the post-install script:
sudo bash ~/OpenClaw-on-MSI-Claw-8/scripts/claw-post-install-vllm.sh
```
- No Docker overhead (saves RAM on 16GB machines)
- More flexible (can use any vLLM version)
- Dependencies may conflict with system packages
- Uses MAX_JOBS auto-detection for OOM prevention

**Recommendation:**
- Claw A1M (16GB): **native build** — Docker daemon eats precious RAM
- Claw 8 AI+ (32GB): either works, Docker is cleaner
- DGX Spark (server): **Docker** — reproducible, isolated, easy to update

## Environment
- Intel Core Ultra 7 258V (Lunar Lake), Arc 140V iGPU
- 32 GB LPDDR5x shared memory (28.57 GiB usable by GPU)
- Intel Core Ultra 7 155H (Meteor Lake), Xe-LPG iGPU
- 16 GB LPDDR5 shared memory
- 2x NVIDIA DGX Spark (Grace Blackwell) — home server for large models
- vLLM 0.14.1.dev0, IPEX XPU
- Model: glm-4.7-flash-int4-autoround (16.52 GB loaded)
