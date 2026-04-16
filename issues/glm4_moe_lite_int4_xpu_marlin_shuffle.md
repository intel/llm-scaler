# GLM-4.7-Flash INT4 MoE: Marlin Shuffle DEVICE_LOST on Lunar Lake XPU

## Status: Bug E (shuffle OOM) FIXED — Bug H (GatedMLPMOE broken for 64+ expert INT4 MoE) OPEN — Same as [IPEX #838](https://github.com/intel/intel-extension-for-pytorch/issues/838)

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
2. **Bug H (OPEN)**: `GatedMLPMOE` crashes with `OUT_OF_RESOURCES` / `DEVICE_LOST`
   on INT4 MoE models with 64+ experts. Same bug as
   [IPEX #838](https://github.com/intel/intel-extension-for-pytorch/issues/838)
   which crashes on discrete GPU Max 1550 with 128 experts. Intel closed #838
   (2026-01-05) with no public fix. Root cause: IPEX `GatedMLPMOE` exhausts
   Level Zero kernel resources (command lists, SLM, barriers) at scale.

MXFP4 avoids both issues: no shuffle needed (Bug E), and GPT-OSS-20B only has
36 experts (below the crash threshold). Dense INT4 models also work — they use
`IPEXGPTQLinearMethod` instead of `GatedMLPMOE`.

### Implication for shared-memory iGPUs
On shared-memory systems, **MXFP4 is the only vLLM-compatible quantization format
that works for INT4-class MoE models** on Lunar Lake Xe2 iGPU. INT4 AutoRound models
load successfully (Bug E fixed) but crash at first inference due to Bug H (attention
→ MoE SYCL state pollution). The alternative path is llama.cpp GGUF via SYCL backend.

## Bug H: GatedMLPMOE Broken for INT4 MoE with 64+ Experts — Known IPEX Bug

**Status: OPEN — Same bug as [IPEX #838](https://github.com/intel/intel-extension-for-pytorch/issues/838).
`GatedMLPMOE` crashes with `OUT_OF_RESOURCES` / `DEVICE_LOST` on INT4 AutoRound
MoE models with 64+ experts. Affects BOTH discrete and integrated GPUs.
Closed by Intel (2026-01-05) with no public fix.**

**Intel never fixed GatedMLPMOE. They replaced it with CUTLASS XE kernels in
[vllm-xpu-kernels](https://github.com/intel/vllm-xpu-kernels).** The IPEX repo
was [archived on March 30, 2026](https://github.com/intel/intel-extension-for-pytorch)
with Bug H unfixed. `GatedMLPMOE` and `marlin_shuffle_weight` were never
upstreamed to PyTorch.

### IPEX #838: The same bug on discrete GPU

[intel/intel-extension-for-pytorch#838](https://github.com/intel/intel-extension-for-pytorch/issues/838)
reports the identical `UR_RESULT_ERROR_OUT_OF_RESOURCES` (error 40) when running
`Qwen/Qwen3-30B-A3B` (128 experts) with `GatedMLPMOE` on Intel Data Center GPU
Max 1550 — a **discrete** GPU with dedicated HBM. This proves Bug H is NOT
specific to Lunar Lake iGPU or shared memory.

- **Filed**: 2025-06-14
- **Closed**: 2026-01-05 (no linked PR, no public fix — **zero PRs** found in IPEX)
- **Crash point**: `torch.ops.torch_ipex.topk_softmax()` (routing, before moe_gemm)
- Smaller MoE models (Falcon3-MoE-2x7B, Llama-4-Scout-17B-16E with 16 experts) work
- Same IPEX version `2.10.10.post1+xpu` as our install
- **IPEX archived March 30, 2026** — no further fixes will be released

### Other unfixed IPEX issues (related)

| Issue | Model | Problem | Status |
|-------|-------|---------|--------|
| [#838](https://github.com/intel/intel-extension-for-pytorch/issues/838) | Qwen3-30B-A3B (128 experts) | OUT_OF_RESOURCES on Max 1550 | Closed, no fix |
| [#864](https://github.com/intel/intel-extension-for-pytorch/issues/864) | GPT-OSS-20B-Int4 | INT4 variant of GPT-OSS-20B fails | Open (archived) |
| [#869](https://github.com/intel/intel-extension-for-pytorch/issues/869) | CPU offload | CPU offload broken for XPU models | Open (archived) |

All three remain unfixed in the archived IPEX repo.

### The #324 dense model red herring

Issue [intel/llm-scaler#324](https://github.com/intel/llm-scaler/issues/324)
reported INT4 AutoRound working on Arc B60 with `Qwen3.5-27B`. But Qwen3.5-27B
is a **dense model** (`Qwen3.5ForCausalLM`), not MoE. Dense models use
`IPEXGPTQLinearMethod` — no `FusedMoE`, no `GatedMLPMOE`, no `marlin_shuffle_weight`.
This code path works fine. It's irrelevant to our MoE models.

### Revised understanding

| Expert count | Discrete GPU (Max 1550) | iGPU (Lunar Lake) | Cross-platform |
|---|---|---|---|
| 2-16 experts | Works | Works | Works |
| 16 experts (Llama-4-Scout-17B-16E) | Works (IPEX #838 reporter) | Likely works | Works |
| 36 experts (GPT-OSS-20B MXFP4) | Works | **Works** | N/A (Intel-only model) |
| 64 experts (GLM-4.7-Flash INT4) | Unknown | **DEVICE_LOST** | Unknown |
| 128 experts (Qwen3-30B-A3B) | **OUT_OF_RESOURCES** (#838) | **DEVICE_LOST** | **Crashes on NVIDIA too** ([vLLM #35922](https://github.com/vllm-project/vllm/issues/35922), [SGLang #9872](https://github.com/sgl-project/sglang/issues/9872)) |
| 256 experts (Qwen3.5-35B-A3B) | Unknown | **DEVICE_LOST** | Unknown |

The crash correlates with expert count. `GatedMLPMOE` has internal resource
scaling that exceeds Level Zero limits above a threshold (around 64 experts on
iGPU, 128 on discrete). GPT-OSS-20B (36 experts, MXFP4) works because it's
below this threshold, not because MXFP4 is inherently more resilient.

**Cross-platform note**: 128-expert Qwen3-30B-A3B also crashes on NVIDIA A100
([vLLM #35922](https://github.com/vllm-project/vllm/issues/35922)) and RTX 4090
([SGLang #9872](https://github.com/sgl-project/sglang/issues/9872)), suggesting
the 128-expert MoE architecture is problematic across all inference frameworks.

### Distinction from Bug E

Bug E (DEVICE_LOST during marlin_shuffle) was solved by the CPU shuffle workaround.
Bug H is a **separate** `GatedMLPMOE` scaling bug: the IPEX MoE fusion layer
exhausts Level Zero resources when the number of experts is too high.

### Our diagnostic evidence (still valid)

Our elimination testing showed:

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

### Path forward: vllm-xpu-kernels replaces GatedMLPMOE

Intel never fixed `GatedMLPMOE` in IPEX. Instead, they built replacement kernels
in [vllm-project/vllm-xpu-kernels](https://github.com/vllm-project/vllm-xpu-kernels)
using CUTLASS XE + SYCL-TLA (not xetla). The repo was originally at
`intel/vllm-xpu-kernels` but has moved to `vllm-project/vllm-xpu-kernels` as an
official vLLM component. IPEX was archived March 30, 2026.

#### Two layers: kernel library vs vLLM integration

The INT4 MoE situation has two independent layers:

| Layer | INT4 MoE Status | Details |
|-------|----------------|---------|
| **vllm-xpu-kernels** (CUTLASS kernel library) | **Done since v0.1.0** (Jan 29, 2026) | PR #88 (INT4 grouped GEMM), PR #98 (fused MoE INT4), PR #114 (expert parallelism) |
| **vLLM** (model layer integration) | **Not done** | INT4 GEMM dense is WIP ([vllm#33662](https://github.com/vllm-project/vllm/pull/33662)), INT4 MoE is "Planned" — no PR yet |

**The CUTLASS INT4 MoE kernels work.** The blocker is vLLM model-layer
integration — someone needs to write the Python glue that routes INT4 MoE
models through vllm-xpu-kernels instead of the dead IPEX `GatedMLPMOE`.

#### vllm-xpu-kernels INT4 MoE kernel PRs (all at vllm-project/vllm-xpu-kernels)

| PR | Description | Status | Merged |
|----|-------------|--------|--------|
| [#88](https://github.com/vllm-project/vllm-xpu-kernels/pull/88) | CUTLASS INT4/FP8/MXFP4 grouped GEMM | Merged | 2025-12-10 |
| [#98](https://github.com/vllm-project/vllm-xpu-kernels/pull/98) | Fused MoE with INT4/FP8/FP4 quantized weights | Merged | 2025-12-15 |
| [#114](https://github.com/vllm-project/vllm-xpu-kernels/pull/114) | Expert parallelism (ep_rank/ep_size) for fused MoE | Merged | 2026-01-06 |
| [#252](https://github.com/vllm-project/vllm-xpu-kernels/pull/252) | Fused softmax/topk for 1024 experts | Open | — |
| [#253](https://github.com/vllm-project/vllm-xpu-kernels/pull/253) | Optimized fused_grouped_topk SYCL kernel | Open | — |

PR #88 benchmarks (E=16, B60 hardware):
- Prefill (M=8192, N=5120, K=8192): 50.8–81.1 TFLOPS
- Decode (M=32, N=5120, K=8192): 406–478 GB/s bandwidth

#### vLLM-side integration status (RFC [#33214](https://github.com/vllm-project/vllm/issues/33214))

| Feature | Kernel ready? | vLLM integrated? | Works in v0.19? |
|---------|:------------:|:----------------:|:---------------:|
| Unquantized MoE | Yes | Yes | **Yes** |
| FP8 MoE | Yes | Yes | **Yes** |
| MXFP4 MoE | Yes | Yes | **Yes** |
| INT4 GEMM (dense AWQ/GPTQ) | Yes | WIP ([#33662](https://github.com/vllm-project/vllm/pull/33662)) | **No** |
| **INT4 MoE** | **Yes** | **Planned (no PR)** | **No** |

#### vllm-xpu-kernels release history

| Version | Date | Key additions |
|---------|------|---------------|
| v0.1.0 | 2026-01-29 | Initial release: CUTLASS grouped GEMM (INT4/MXFP4/FP8), fused MoE, oneDNN |
| v0.1.1 | 2026-02-03 | FP8 KV cache, fused MoE interface fixes |
| v0.1.2 | 2026-02-11 | MoE align_block_size, oneDNN 3.11, FP8 block quant |
| v0.1.3 | 2026-03-04 | topk_sigmoid for MoE, layernorm, chunk GDN attention |
| **v0.1.4** | **2026-03-20** | **FP8 block quant, sliding window decode, swap_blocks** (pinned by vLLM 0.19) |
| v0.1.5 | 2026-04-03 | MLA kernels, FP8 w8a16 GEMM, MXFP4 block quant |

vLLM v0.19.0 pins `vllm_xpu_kernels==0.1.4` in `requirements/xpu.txt`.

#### Migration options (ranked)

1. **Write INT4 MoE integration for vLLM v0.19** — The CUTLASS kernels exist
   in vllm-xpu-kernels v0.1.4. The missing piece is a vLLM-side
   `XPUInt4MoEMethod` class that calls `vllm_xpu_kernels.fused_moe()` with
   `is_int4=True` instead of IPEX's `GatedMLPMOE`. This is primarily Python
   glue code. Could be contributed upstream as a PR.

2. **Wait for upstream INT4 MoE integration** — Intel is working on it
   (RFC #33214 tracks it as "Planned"). INT4 dense GEMM ([#33662](https://github.com/vllm-project/vllm/pull/33662))
   is the prerequisite and is under review.

3. **Use MXFP4 models on vLLM v0.19** — MXFP4 MoE is fully integrated and
   works (GPT-OSS-20B, 36 experts). Limited model selection.

4. **Use llama.cpp GGUF** — own SYCL kernels, no GatedMLPMOE dependency.
   Works for GLM-4.7-Flash-Q4_K_M.gguf today.

#### Why vLLM v0.14 + IPEX is a dead end

- IPEX archived March 30, 2026 with `GatedMLPMOE` broken for 64+ experts
- `GatedMLPMOE` and `marlin_shuffle_weight` were NEVER upstreamed to PyTorch
- No public fix exists in any IPEX release
- vllm-xpu-kernels is the actively maintained replacement
- vLLM v0.16+ dropped IPEX dependency for XPU compute

### What Intel supports vs what we need

| | Dense INT4 | MoE INT4 (IPEX) | MoE INT4 (vllm-xpu-kernels) | MoE MXFP4 |
|---|---|---|---|---|
| Code path | `compressed-tensors` w4a16 | `GatedMLPMOE` | CUTLASS XE fused MoE | CUTLASS XE fused MoE |
| Kernel exists? | Yes | Yes (broken) | **Yes** (v0.1.0+) | **Yes** |
| vLLM wired up? | Yes ([#33973](https://github.com/vllm-project/vllm/pull/33973)) | Yes (v0.14) | **No** (planned) | **Yes** |
| Works in v0.19? | **Yes** | N/A (IPEX removed) | **No** (needs integration PR) | **Yes** |
| Works on discrete? | Yes | No (128+ crash) | Unknown (needs testing) | Yes |
| Works on iGPU? | Yes (too big) | No (64+ crash) | Unknown (needs testing) | Yes (GPT-OSS-20B) |
| Status | Stable | **Dead** (archived) | **Kernel ready, integration pending** | Stable |

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
