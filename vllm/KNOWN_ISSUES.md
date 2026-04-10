
# 01. System Hang During Ubuntu 25.04 Installation with B60 Card Plugged In
The issue is caused by an outdated GPU GuC firmware bundled in the official Ubuntu 25.04 Desktop ISO image.

Workaround: Remove the B60 card before starting the Ubuntu installation, and plug it back in once the installation is complete.
We are also working with the Ubuntu team to address this issue upstream.

# 02. Limited 33 GB/s Bi-Directional P2P Bandwidth with 1x GPU Card
When using a single GPU card over a x16 PCIe connection without a PCIe switch, the observed bi-directional P2P bandwidth is limited to 33 GB/s.

Workaround: Change the PCIe slot configuration in BIOS from Auto/x16 to x8/x8.
With this change, over 40 GB/s bi-directional P2P bandwidth can be achieved.
Root cause analysis is still in progress.

# 03. Container OOM killed (and vllm performance drop) when starting container not by /bin/bash and not run `source /opt/intel/oneapi/setvars.sh`

When using `--enable-auto-tool-choice` and deploy container by docker-compose without `source /opt/intel/oneapi/setvars.sh`, the LD_LIBRARY_PATH will be different and cause the container OOM (or performance drop). It can be reproduced by this two command:

```bash
docker run --rm  --entrypoint "/bin/bash" --name=test intel/llm-scaler-vllm:latest -c env | grep LD_LIBRARY_PATH
 
docker run --rm --entrypoint "/bin/bash" --name=test intel/llm-scaler-vllm:latest -c "source /opt/intel/oneapi/setvars.sh --force && env | grep LD_LIBRARY_PATH"
```

So we need to run `source /opt/intel/oneapi/setvars.sh --force` to ensure some configurations are consistent.

# 04. xccl/oneCCL Distributed Backend Hangs on Meteor Lake iGPU

**Affected:** Intel Core Ultra 1xx (Meteor Lake) — Xe-LPG integrated GPU  
**Not affected:** Intel Core Ultra 2xx (Lunar Lake) — Xe2-LPG integrated GPU  
**Not affected:** Discrete Arc GPUs (A770, B580, etc.)

## Symptom

vLLM hangs during startup on Meteor Lake systems. The process blocks indefinitely
at `torch.distributed.init_process_group()` when the backend is `xccl` or `ccl`.
No error message — it simply freezes.

## Root Cause

vLLM's XPU platform auto-detects the distributed backend via `supports_xccl()`.
On PyTorch 2.10+xpu, this returns `True` (the API exists), so vLLM selects `xccl`.
However, xccl depends on oneCCL's Level Zero communication layer to initialize,
and this initialization hangs on Meteor Lake's Xe-LPG iGPU.

Falling back to `ccl` (oneCCL directly) also fails — same Level Zero init hang.

Per [PyTorch's XCCL announcement](https://pytorch.org/blog/pytorch-2-8-brings-native-xccl-support-to-intel-gpus-case-studies-from-argonne-national-laboratory/),
native XCCL support in PyTorch 2.8 was validated on **Intel Data Center GPUs**
(Max Series / Ponte Vecchio). Consumer iGPUs are not officially listed as supported.
Lunar Lake (Xe2-LPG) happens to work — likely due to its newer Level Zero driver
stack — but this is not guaranteed.

## Architecture Context

| Device | GPU | Architecture | Generation | xccl |
|---|---|---|---|---|
| Arc A770 / B580 | DG2 / BMG | Xe-HPG / Xe2-HPG | Alchemist / Battlemage | Works |
| Claw 8 AI+ (Lunar Lake) | Arc 140V | Xe2-LPG | Battlemage (2nd gen) | Works |
| Claw A1M (Meteor Lake) | Arc Graphics | Xe-LPG | Alchemist-derived (1st gen) | **Hangs** |
| Data Center Max (PVC) | Max 1550 | Xe-HPC | Ponte Vecchio | Works (primary target) |

Architecture references:
- [Intel Xe architecture family — Wikipedia](https://en.wikipedia.org/wiki/Intel_Xe)
- [Lunar Lake Xe2-LPG vs Meteor Lake Xe-LPG — NotebookCheck](https://www.notebookcheck.net/Intel-Lunar-Lake-Xe2-LPG-iGPU-appears-faster-and-more-efficient-than-Xe-LPG-of-Meteor-Lake-as-Core-Ultra-200V-chip-surfaces.832317.0.html)
- [Lunar Lake iGPU: Debut of Xe2 — Chips and Cheese](https://chipsandcheese.com/p/lunar-lakes-igpu-debut-of-intels)
- [Intel details Xe-LPG Meteor Lake GPU — VideoCardz](https://videocardz.com/newz/intel-details-arc-xe-lpg-meteor-lake-gpu-architecture-up-to-8-xe-cores-and-ray-tracing-acceleration)

## Workaround

Override the distributed backend to `gloo` via environment variable:

```bash
export VLLM_XPU_DIST_BACKEND=gloo
```

This requires patching `vllm/platforms/__init__.py` to read the env var
(included in `vllm_for_multi_arc.patch`):

```python
dist_backend = os.environ.get("VLLM_XPU_DIST_BACKEND", "")
if not dist_backend:
    if supports_xccl():
        dist_backend = "xccl"
    else:
        dist_backend = "ccl"
        import oneccl_bindings_for_pytorch  # noqa: F401
```

`gloo` is a CPU-based collective operations backend. For single-GPU inference
(which is the only mode on iGPU handhelds), distributed communication is never
actually used — vLLM just needs the backend to initialize without hanging.

## Notes

- On single-GPU systems, the distributed backend choice has **zero performance impact**
  — no collective ops (all-reduce, broadcast, etc.) are ever called during inference.
- The `gloo` workaround is safe for both Meteor Lake and Lunar Lake. Lunar Lake
  users can continue using `xccl` (the default) since it works, but `gloo` is
  also fine.
- This issue does not affect discrete Arc GPUs (A770, B580) or Data Center GPUs.

# 05. vLLM Cannot Run Inference on Meteor Lake iGPU (Missing XMX)

**Affected:** Intel Core Ultra 1xx (Meteor Lake) — Xe-LPG integrated GPU  
**Not affected:** Intel Core Ultra 2xx (Lunar Lake) — Xe2-LPG integrated GPU  
**Not affected:** Discrete Arc GPUs (A770, B580, etc.)

## Symptom

vLLM loads the model successfully on Meteor Lake, but crashes on the first inference
request with:

```
RuntimeError: SDP kernel requires XMX, but the current platform has no XMX
  ...intel_extension_for_pytorch/transformers/models/xpu/fusions/mha_fusion.py
  ...torch.ops.torch_ipex.chunked_prefill()
```

## Root Cause

Meteor Lake's Xe-LPG architecture is derived from Xe-HPG (Arc A770) but **deliberately
omits XMX (Xe Matrix Extensions)** — the matrix multiplication units on execution Port 2.
This was confirmed by [Chips and Cheese's Meteor Lake deep-dive](https://chipsandcheese.com/p/intels-ambitious-meteor-lake-igpu):

> "It's basically the same idea as Xe-HPG, just without the matrix multiplication units
> that would have been present on Port 2."

vLLM's XPU attention path calls IPEX's `chunked_prefill()` and `paged_attention_v1()`
ops, which invoke XMX-accelerated SDP (Scaled Dot-Product) kernels. When XMX hardware
is absent, IPEX throws a fatal error with no fallback.

This is a **hardware limitation**, not a driver or software configuration issue.

## What Does NOT Work

| Approach | Result |
|---|---|
| `VLLM_ATTENTION_BACKEND=TORCH_SDPA` | Not supported for XPU main attention path (only CPU) |
| `ENABLE_SDP_FUSION=0` | Controls fusion, not XMX usage — does not help |
| `BIGDL_LLM_XMX_DISABLED=1` | Only affects IPEX-LLM (llama.cpp), not vLLM's IPEX |
| `vllm_for_multi_arc.patch` | Does not add non-XMX attention for LLM path |
| Triton attention backend | Triton XPU also leverages XMX for dot products |
| vllm-xpu-kernels flash attention | Uses SYCL `joint_matrix` — Intel docs say "no emulation or fall back strategy" for non-XMX |
| Any IPEX env var to disable XMX | Does not exist — no `IPEX_XPU_ONEDNN` or `IPEX_SDPA_MATH_MODE` variable |

## Hardware Comparison

| Device | GPU | Architecture | XMX | vLLM |
|---|---|---|---|---|
| Claw A1M (Meteor Lake) | Arc Graphics | Xe-LPG (1st gen) | **No** | **Blocked** |
| Claw 8 AI+ (Lunar Lake) | Arc 140V | Xe2-LPG (2nd gen) | Yes | Works |
| Arrow Lake (desktop) | Arc Graphics | Xe-LPG Plus | Yes | Should work |
| Arc A770 (discrete) | DG2 | Xe-HPG | Yes | Works |
| Arc B580 (discrete) | BMG | Xe2-HPG | Yes | Works |
| Data Center Max (PVC) | Max 1550 | Xe-HPC | Yes | Works (primary target) |

Note: Arrow Lake added XMX back via "Xe-LPG Plus", indicating Intel recognized the
omission in Meteor Lake was a limitation.

## Upstream Status

- [vllm-project/vllm#28362](https://github.com/vllm-project/vllm/issues/28362):
  "Can't get vLLM to run on Intel 125H with XPU" — marked **NOT_PLANNED**
- No plans from vLLM project to support non-XMX Intel GPUs
- Post Feb 2026, vLLM switched from IPEX to `vllm-xpu-kernels` (designed for PVC/BMG
  only), which also doesn't support Xe-LPG

## Potential Future Path: PyTorch >= 2.9 Math SDP Backend

[PyTorch PR #156669](https://github.com/pytorch/pytorch/pull/156669) (merged July 2025)
added `SDPBackend.MATH` support on XPU. This enables a math-based SDP fallback that
does not require XMX. In theory, vLLM could be patched to use this:

```python
from torch.nn.attention import sdpa_kernel, SDPBackend
with sdpa_kernel(SDPBackend.MATH):
    # attention operations bypass XMX-dependent oneDNN kernel
```

**Status: Unvalidated.** This would require:
1. PyTorch >= 2.9 with XPU support
2. Patching vLLM's `TORCH_SDPA` backend to wrap attention in `sdpa_kernel(MATH)`
3. Ensuring paged attention / KV cache ops also work without XMX
4. Significant performance testing (math SDP is much slower than XMX-accelerated SDP)

Note: IPEX is being retired after version 2.8. Intel is directing users to PyTorch
native XPU support. Future vLLM versions using PyTorch >= 2.9 native XPU may
eventually support non-XMX platforms through this path.

## Alternatives for LLM Inference on Meteor Lake

vLLM is blocked, but other frameworks work because they use standard SYCL parallel
patterns (compiled to SPIR-V, JIT-compiled at runtime) and DP4a instructions instead
of DPAS/XMX systolic array operations.

### Option 1: IPEX-LLM + Ollama

[IPEX-LLM](https://github.com/intel/ipex-llm) (formerly BigDL-LLM) explicitly supports
Meteor Lake — it detects the device as `"mtl"` and applies architecture-specific
optimizations (automatic KV cache compression, tuned batch forwarding).

```bash
export BIGDL_LLM_XMX_DISABLED=1
export SYCL_CACHE_PERSISTENT=1

# Run via IPEX-LLM's patched Ollama
docker run -d --device /dev/dri --name ollama \
  -e BIGDL_LLM_XMX_DISABLED=1 \
  -v ollama:/root/.ollama \
  -p 11434:11434 \
  intelanalytics/ipex-llm-inference-cpp-xpu:latest
```

Notes:
- First-run JIT compilation takes **5-10 minutes** on Meteor Lake iGPU (SPIR-V → native ISA)
- Set `SYCL_CACHE_PERSISTENT=1` to cache compiled kernels across restarts
- Practical for models up to **~7-8B parameters in Q4 quantization** given 16GB shared memory
- IPEX-LLM's SDP kernels do NOT check for XMX — they use standard SYCL patterns

**Limitation:** Based on llama.cpp (GGUF format only). New model architectures need to
be added to llama.cpp first — there is always a lag between model release and support.
Novel architectures may never be supported.

### Option 2: OpenVINO GenAI (Best for New Model Support)

[OpenVINO GenAI](https://docs.openvino.ai/2025/openvino-workflow-generative/inference-with-genai.html)
supports Meteor Lake iGPU and can auto-convert HuggingFace models via
[Optimum-Intel](https://huggingface.co/docs/optimum/intel/index). This is the best
option for running **new models that aren't yet in llama.cpp/GGUF**.

```python
from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer

model = OVModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    export=True,        # auto-converts from HuggingFace
    device="GPU",       # uses iGPU
    load_in_4bit=True,  # INT4 weight compression
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
```

For serving, [OpenVINO Model Server](https://docs.openvino.ai/2025/ovms_what_is_openvino_model_server.html)
provides a REST/gRPC API (similar to vLLM's OpenAI-compatible endpoint).

Advantages over IPEX-LLM/Ollama:
- Supports most HuggingFace models directly (auto-conversion)
- No GGUF dependency — works with any model that Optimum-Intel can export
- Intel maintains it with dedicated iGPU optimizations (no XMX dependency)
- OpenVINO 2026.0 adds MoE models, multimodal, hybrid CPU/GPU/NPU execution

### Option 3: HuggingFace Transformers with Eager Attention on XPU

For maximum model compatibility with minimal setup, use HuggingFace Transformers
directly with `attn_implementation="eager"`, which bypasses SDPA entirely and uses
plain `torch.matmul + softmax` (no XMX needed):

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    attn_implementation="eager",  # plain matmul, no XMX
    torch_dtype=torch.float16,
    device_map="xpu",
)
```

**Any HuggingFace model works immediately**, but this is significantly slower than
optimized backends — no flash attention, no paged KV cache, no continuous batching.
Useful for testing/prototyping, not production serving.

### Option 4: llama.cpp (SYCL or Vulkan)

[llama.cpp's SYCL backend](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md)
lists "built-in Arc GPU in Meteor Lake" as verified hardware. The Vulkan backend also
works and requires no oneAPI installation.

```bash
# SYCL build (needs oneAPI)
cmake -B build -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx
cmake --build build --config Release -j

# Vulkan build (simpler, just needs GPU drivers)
cmake -B build -DGGML_VULKAN=ON
cmake --build build --config Release -j
```

Note: SYCL is faster for prompt processing (prefill), but Vulkan can be faster for
token generation — especially on MoE models. Both are limited to GGUF format.

### ~~Option 5: llama.cpp OpenVINO Backend~~ (BROKEN — Do Not Use)

> **Status: BROKEN** (tested April 2026 on MSI Claw A1M, Meteor Lake iGPU)
>
> Despite being [merged March 2026](https://github.com/ggml-org/llama.cpp/pull/15307)
> and having official release binaries, the OpenVINO backend **does not work** with
> K-quant models (Q4_K_M, Q4_K_XL, Q5_K_M) — which are the standard quantizations
> used in practice.

**Tested failures:**

1. **GPU with `-fa 1`**: Crashes immediately with `CPY operation not supported` —
   the backend doesn't implement the KV cache copy operation needed for flash attention
2. **GPU without `-fa`**: `"failed to decode prompt batch, res = -3"` — Q4_K_M
   quantization type not supported by the OpenVINO GGML backend
3. **CPU fallback**: Same `res = -3` failure — the issue is the quant format, not the device
4. **GPU device init**: `"Failed to get OpenCL device: -1"` even though `clinfo` shows
   the device correctly — OpenVINO's internal initialization fails

The backend only supports basic quantizations (Q4_0, Q8_0, FP16) which are rarely used
in practice. K-quants (Q4_K_M, Q5_K_M, Q6_K) provide better quality-per-bit and are
the standard recommendation, but none of them work.

**Use SYCL instead** — see Option 4 above. SYCL works with all quantization formats,
provides similar long-context scaling advantages over Vulkan, and is actively maintained.

**Backend comparison on Intel Meteor Lake iGPU** (llama-bench, real measurements on MSI Claw A1M, 5 runs averaged):

Gemma 4 E4B Q4_K_M (7.5B params, 4.62 GiB):

| Backend | pp512 | pp1024 | pp2048 | pp4096 | pp8192 | pp16384 | pp32768 | tg128 | tg256 | tg512 | tg1024 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **SYCL (JIT)** | 311 | 317 | 318 | 314 | 305 | 290 | 257 | 15.0 | 15.6 | 15.5 | 15.2 |
| **SYCL AOT (mtl_h)** | 309 | 319 | 318 | 314 | 306 | 290 | 259 | 15.0 | 15.0 | 14.9 | 14.6 |
| **Vulkan** | 292 | 278 | 281 | 286 | 267 | 237 | 188 | 14.6 | 14.6 | 14.4 | 14.1 |
| **OpenVINO** | — | — | — | — | — | — | — | — | — | — | — |

SYCL AOT = compiled with `-DGGML_SYCL_DEVICE_ARCH=mtl_h` (ahead-of-time for Meteor Lake-H).
SYCL JIT = default build without device arch (just-in-time compilation at runtime).

Key findings:
- **SYCL scales much better at long context** — only 17% drop from pp512→pp32768
  vs 36% drop for Vulkan (1.4× faster at 32K)
- **SYCL is faster at all context lengths**, including short context
- **OpenVINO is completely non-functional** with K-quant models
- **Token generation**: SYCL JIT 15.0-15.6 vs Vulkan 14.1-14.6 (~7% faster)
- **AOT does NOT improve performance** — prefill identical, TG ~4% slower than JIT
- **SYCL TG improves after warmup** (tg256 > tg128) — more pronounced with JIT build
- **Recommendation: use default SYCL build (JIT)** — AOT adds build complexity with no benefit

Sources: [llama.cpp #10879](https://github.com/ggml-org/llama.cpp/discussions/10879),
[ipex-llm #12318](https://github.com/intel-analytics/ipex-llm/issues/12318),
[llama.cpp SYCL MUL_MAT PR #12035](https://github.com/ggml-org/llama.cpp/pull/12035)

**Meteor Lake + Vulkan prefill degrades at long context** because Xe-LPG lacks
`VK_KHR_cooperative_matrix`. SYCL avoids this by routing through oneMKL's hand-tuned
GEMM kernels. SYCL does NOT require XMX — oneMKL dispatches to the best available
instruction path (DP4a on Meteor Lake, DPAS/XMX on Lunar Lake).

## Choosing an Alternative

| Need | Best Option |
|---|---|
| Run latest HuggingFace models on Meteor Lake | **OpenVINO GenAI** |
| Production serving with API endpoint (32K context) | **OpenVINO GenAI** (serve_ov.py) |
| Quick setup, any GGUF model, best long-context | **llama.cpp SYCL backend** |
| Quick setup, established models (Llama, Qwen, etc.) | **IPEX-LLM + Ollama** |
| Testing/prototyping any HF model | **HF Transformers + eager attention** |
| Minimal dependencies, no oneAPI | **llama.cpp Vulkan** |
| NPU inference | **OpenVINO GenAI** (llama.cpp OpenVINO backend broken) |

### Performance Expectations

Without XMX, Meteor Lake uses **DP4a** (Dot-Product of 4 elements and Accumulate)
instructions for INT8/INT4 operations — the older, scalar path vs. the DPAS systolic
array path that XMX enables.

**Actual benchmark** (Qwen3-4B INT4 on MSI Claw A1M, Meteor Lake iGPU, OpenVINO GenAI):

| Metric | Short prompt (~6 words) | Long prompt (~450 words) |
|---|---|---|
| TTFT (prefill) | **176 ms** | **993 ms** |
| Decode speed | **26.4 tokens/s** | **24.5 tokens/s** |
| Total time (100 tokens) | 3.85s | 4.86s |

This is **comparable to vLLM on Lunar Lake** (which has XMX) for the same model class.
The bottleneck for 4B-class models on iGPU is memory bandwidth (LPDDR5), not compute,
so the XMX absence has less impact than expected. OpenVINO's graph-level optimizations
(kernel fusion, layout optimization) compensate for the missing hardware acceleration.

## References

- [Chips and Cheese: Intel's Ambitious Meteor Lake iGPU](https://chipsandcheese.com/p/intels-ambitious-meteor-lake-igpu)
  — confirms XMX omission on Xe-LPG Port 2

# 06. OpenVINO GenAI Prefill Crashes at 8K+ Tokens on Meteor Lake iGPU (OOM)

**Affected:** Intel Core Ultra 1xx (Meteor Lake) — 16 GB LPDDR5 shared memory  
**Likely affected:** Any iGPU system with ≤ 16 GB shared memory  
**Not affected:** Systems with 32 GB+ RAM, discrete GPUs

## Symptom

OpenVINO GenAI's `LLMPipeline` crashes during prefill (prompt processing) when the
input exceeds ~8K tokens on Meteor Lake iGPU:

```
[GPU] CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST
```

Observed with Qwen3-4B INT4 on MSI Claw A1M (16 GB LPDDR5). Prefill works up to 4K
tokens but shows significant performance degradation at 4K (throughput drops from
674 → 378 tokens/s) before crashing at 8K.

## Root Cause: Quadratic Attention Memory

During prefill, the self-attention operation computes attention scores for the entire
prompt at once. For standard SDPA (Scaled Dot-Product Attention), this requires
materializing an attention score matrix of size:

```
[batch, num_query_heads, seq_len, seq_len] × sizeof(float16)
```

This grows **quadratically** with sequence length. For Qwen3-4B (32 Q-heads, 8 KV-heads,
128 head_dim, 36 layers):

| Context Length | KV Cache (FP16) | Attention Peak (1 layer, all heads) | Total Est. GPU Memory |
|---|---|---|---|
| 512 | 72 MB | 32 MB | ~2.6 GB |
| 1K | 144 MB | 128 MB | ~2.8 GB |
| 2K | 288 MB | 512 MB | ~3.3 GB |
| 4K | 576 MB | **2 GB** | ~5.1 GB |
| 8K | 1,152 MB | **8 GB** | **~11.7 GB** ← crashes |
| 16K | 2,304 MB | **32 GB** | impossible |
| 32K | 4,608 MB | **128 GB** | impossible |

**KV cache formula:** `2 × 36 layers × 8 KV-heads × 128 dim × 2 bytes = 147,456 bytes/token`

**Attention score formula:** `32 Q-heads × seq_len² × 2 bytes` per layer

Note: The actual peak depends on how OpenVINO's GPU plugin tiles the attention
computation. If it processes heads in groups rather than all at once, the peak is
lower — but the quadratic scaling remains.

On 16 GB shared memory, the GPU typically gets ~8 GB (configurable in BIOS). At 8K
tokens the combined model weights (2.5 GB) + KV cache (1.1 GB) + attention intermediates
push past this limit.

## Why 4K Shows a Performance Cliff

The 4K benchmark data shows prefill throughput dropping from 674 tokens/s (at 2K) to
378 tokens/s (at 4K). This is because at 4K, the attention scores reach ~2 GB per layer,
which approaches the GPU allocator's comfort zone. The GPU starts swapping or fragmenting
memory, causing massive slowdown before the hard crash at 8K.

**Pre-2025.4 OpenVINO had a 4.2 GB single-allocation limit** for GPU buffers. If the
installed version is older, this limit alone could cause the crash. OpenVINO 2025.4+
removed this limit via `GPU_ENABLE_LARGE_ALLOCATIONS`.

## Benchmark Data

Prefill benchmark on MSI Claw A1M (Meteor Lake, 16 GB), Qwen3-4B INT4, OpenVINO GenAI:

| Context | TTFT | Prefill Throughput | Status |
|---|---|---|---|
| 512 tokens | 773 ms | 663 tokens/s | OK |
| 1K tokens | 1,695 ms | 604 tokens/s | OK |
| 2K tokens | 3,040 ms | 674 tokens/s | OK |
| 4K tokens | 10,841 ms | 378 tokens/s | Slow (memory pressure) |
| 8K tokens | — | — | **CRASH** (CL_EXEC_STATUS_ERROR) |

## Workarounds

### 1. Use ContinuousBatchingPipeline with SchedulerConfig

The default `LLMPipeline` (StatefulLLMPipeline / SDPA backend) processes the entire
prompt in one pass with a contiguous KV cache — it has **no chunked prefill, no paged
KV cache, and no eviction**. This is what crashes at 8K tokens.

Passing a `SchedulerConfig` activates the `ContinuousBatchingPipeline` backend, which
uses PagedAttention and `dynamic_split_fuse` to chunk the prefill:

```python
import openvino_genai as ov_genai

scheduler_config = ov_genai.SchedulerConfig()
scheduler_config.cache_size = 3              # limit KV cache to 3 GB
scheduler_config.max_num_batched_tokens = 512  # chunk prefill into 512-token blocks
scheduler_config.dynamic_split_fuse = True   # enable chunked prefill (default: True)
scheduler_config.enable_prefix_caching = False  # CRITICAL: avoid memory explosion bug

pipe = ov_genai.LLMPipeline(
    model_path, "GPU",
    scheduler_config=scheduler_config,
    KV_CACHE_PRECISION="u8",   # INT8 KV cache (default on iGPU), or "u4" for INT4
)
```

`dynamic_split_fuse` splits long prompts into chunks of `max_num_batched_tokens`
(default: 256), processing them across multiple iterations and building the KV cache
incrementally. A 32K prompt with 512-token chunks = ~63 iterations instead of one
monolithic pass. This avoids the quadratic attention memory explosion.

**KV cache precision options:**

| `KV_CACHE_PRECISION` | Memory vs FP16 | Quality | Notes |
|---|---|---|---|
| `f16` / `bf16` | 1× (baseline) | Best | May OOM at 32K on 16GB |
| `u8` | 0.5× | Good | **Default on iGPU** (auto-detected) |
| `u4` | 0.25× | Acceptable | Best for 32K on 16GB — reduces 4.5 GB → ~1.1 GB KV |

You can also set `KEY_CACHE_PRECISION` and `VALUE_CACHE_PRECISION` separately for
fine-grained control.

**Important:** `enable_prefix_caching = False` is critical. There is a known bug
([openvino.genai#2406](https://github.com/openvinotoolkit/openvino.genai/issues/2406))
where prefix caching with Qwen3-4B causes an integer overflow in memory calculation
(requesting 2.3 exabytes), leading to immediate crash.

### 2. Use CPU for Long-Context Prefill

CPU has access to full system RAM (16 GB) without the GPU allocation limit. CPU prefill
is slower but avoids the OOM entirely:

```python
# CPU pipeline — no GPU memory limits
pipe = ov_genai.LLMPipeline(model_path, "CPU")

# Or use ContinuousBatchingPipeline on CPU for long-context
scheduler_config = ov_genai.SchedulerConfig()
scheduler_config.cache_size = 4  # 4 GB KV cache on system RAM
pipe = ov_genai.LLMPipeline(model_path, "CPU", scheduler_config=scheduler_config)
```

For OpenClaw's 32K context requirement, CPU prefill may be the only viable option on
16 GB systems. Expected TTFT: 30-60 seconds for 32K tokens on Meteor Lake CPU
(slower than GPU but doesn't crash).

### 3. Use a Model with Sliding Window Attention

Models with **hybrid attention** (sliding window + sparse full attention) use far less
memory for long contexts because most layers only attend to a local window:

- **Qwen3.5-4B**: Only 8 of 36 layers use full attention; 28 use sliding window.
  At 32K tokens, KV cache is ~8× smaller than Qwen3-4B's, and attention intermediates
  are bounded by the window size for most layers.
- **Gemma 4 E4B**: MoE architecture (2.3B active / 5.1B total) with 128K context
  support. Likely designed for efficient long-context inference.

### 4. Upgrade to OpenVINO 2025.4+

If running an older version, the 4.2 GB single-allocation limit may be the immediate
cause. OpenVINO 2025.4 removed this limit:

```bash
pip install --upgrade openvino openvino-genai
```

### 5. Increase GPU Memory Allocation in BIOS

MSI Claw BIOS may have a "GPU Shared Memory" or "DVMT Pre-Allocated" setting.
Increasing from the default (typically 256 MB pre-allocated + up to 50% dynamic) to
the maximum can help. Check:
- BIOS → Advanced → Graphics Configuration → DVMT Pre-Allocated
- Set to maximum available (e.g., 8 GB or "MAX")

### 6. Use KV Cache Eviction for Bounded Memory

OpenVINO GenAI supports KV cache eviction — automatically dropping least-important
tokens from the cache when memory is full:

```python
cache_eviction_config = ov_genai.CacheEvictionConfig()
cache_eviction_config.max_cache_size = 4096   # max tokens in KV cache (must be multiple of block_size)
cache_eviction_config.start_size = 64         # initial tokens always kept
cache_eviction_config.recent_size = 512       # recent tokens always kept
# Evictable region: max_cache_size - start_size - recent_size = 3520 tokens
# Uses SnapKV-based attention scoring to evict least-important tokens
```

This allows processing arbitrarily long contexts within a fixed memory budget, at the
cost of some accuracy for very distant tokens. For OpenClaw's large-context use case,
this may be acceptable since the most relevant context is usually recent.

### 7. Use NPU for Prefill (Experimental)

Meteor Lake includes a dedicated NPU (Neural Processing Unit) with its own memory
management. OpenVINO 2025.3+ supports Qwen3-4B on NPU. The NPU has a separate
memory pool, avoiding the shared GPU/CPU memory contention:

```python
# NPU with chunked prefill
pipe = ov_genai.LLMPipeline(model_path, "NPU")
# NPUW_LLM_PREFILL_CHUNK_SIZE controls prefill chunk size (default: 1024)
# Set to 256 for lower peak memory
```

**Status: Experimental.** NPU inference for LLMs is new and may have performance or
compatibility issues.

## Path to 32K Context on 16 GB Meteor Lake

For OpenClaw's requirement of 32K token input context, the recommended configuration:

```python
import openvino_genai as ov_genai

# --- Recommended 32K config for Claw A1M (16 GB) ---
scheduler_config = ov_genai.SchedulerConfig()
scheduler_config.cache_size = 3                    # 3 GB KV cache budget
scheduler_config.max_num_batched_tokens = 512      # chunked prefill (512 tokens/iter)
scheduler_config.dynamic_split_fuse = True         # enable chunked prefill
scheduler_config.enable_prefix_caching = False     # avoid bug #2406

pipe = ov_genai.LLMPipeline(
    "model_path",  # Qwen3.5-4B or Qwen3-4B INT4
    "GPU",
    scheduler_config=scheduler_config,
    KV_CACHE_PRECISION="u4",    # INT4 KV cache: 4.5 GB → ~1.1 GB at 32K
)
```

**Memory budget at 32K with this config (Qwen3-4B INT4):**
- Model weights: ~2.5 GB
- KV cache (U4): ~1.1 GB (vs 4.5 GB at FP16)
- Attention intermediates: bounded by 512-token chunks (~small)
- **Total: ~4-5 GB** — fits within 8 GB GPU allocation

**Recommended steps** (in order of priority):

1. **Use ContinuousBatchingPipeline** (pass `SchedulerConfig`) — the default
   StatefulLLMPipeline has no chunked prefill and will always crash at 8K+
2. **Set `KV_CACHE_PRECISION="u4"`** — reduces KV cache 4× (critical for 32K)
3. **Disable prefix caching** (`enable_prefix_caching = False`) — avoids memory explosion bug
4. **Set `max_num_batched_tokens=512`** — chunks 32K prompt into ~63 iterations
5. **Switch to Qwen3.5-4B** if available — hybrid sliding-window attention uses less memory
6. **Enable KV cache eviction** (`CacheEvictionConfig`) as a safety net
7. If still OOM: fall back to **CPU** for prompts > 4K tokens

**Note:** CPU prefill + GPU decode is **not supported** in OpenVINO GenAI — there is
no mechanism to transfer KV cache state between device pipelines. Use a single device.

Alternatively, on the **Claw 8 AI+ (Lunar Lake, 32 GB)**, the larger RAM budget
makes 32K prefill feasible with less aggressive settings — 32 GB provides ~16 GB for
GPU, enough even with FP16 KV cache.

## References

- [OpenVINO GPU Plugin Memory Allocation](https://github.com/openvinotoolkit/openvino/wiki/Memory-allocation-in-GPU-plugin)
  — GPU memory allocation types and limits (CL_DEVICE_GLOBAL_MEM_SIZE)
- [OpenVINO 2025.4 Release Notes](https://www.intel.com/content/www/us/en/developer/articles/release-notes/openvino/2025-4.html)
  — removed 4.2 GB single-allocation limit via GPU_ENABLE_LARGE_ALLOCATIONS
- [openvinotoolkit/openvino.genai#2406](https://github.com/openvinotoolkit/openvino.genai/issues/2406)
  — exceed_allocatable_mem_size GPU memory allocation failure
- [openvinotoolkit/openvino#34416](https://github.com/openvinotoolkit/openvino/issues/34416)
  — OOM on iGPU with 20B model (Core Ultra 7 265)
- [OpenVINO SchedulerConfig API](https://docs.openvino.ai/2025/api/genai_api/_autosummary/openvino_genai.SchedulerConfig.html)
  — cache_size, num_kv_blocks, dynamic_split_fuse, max_num_batched_tokens
- [OpenVINO KV Cache Eviction Algorithm](https://github.com/openvinotoolkit/openvino.genai/blob/master/site/docs/concepts/optimization-techniques/kvcache-eviction-algorithm.md)
  — automatic token eviction for bounded memory
- [LMCache KV Cache Calculator](https://docs.lmcache.ai/getting_started/kv_cache_calculator.html)
  — KV cache size estimation tool
- [openvinotoolkit/openvino#31781](https://github.com/openvinotoolkit/openvino/issues/31781)
  — CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST on Core Ultra 5 125H (Meteor Lake)
- [openvinotoolkit/openvino#32665](https://github.com/openvinotoolkit/openvino/issues/32665)
  — GPU memory leak with xe driver during inference
- [openvinotoolkit/openvino_notebooks#2632](https://github.com/openvinotoolkit/openvino_notebooks/issues/2632)
  — Qwen2 GPU memory error (CL_OUT_OF_RESOURCES) on iGPU
- [OpenVINO CacheEvictionConfig API](https://docs.openvino.ai/2025/api/genai_api/_autosummary/openvino_genai.CacheEvictionConfig.html)
  — max_cache_size, start_size, recent_size parameters
  — KV cache size estimation tool

---

## References (Issue #05 — XMX Blocker)

- [Chips and Cheese: Intel's Ambitious Meteor Lake iGPU](https://chipsandcheese.com/p/intels-ambitious-meteor-lake-igpu)
  — confirms XMX omission on Xe-LPG Port 2
- [Tom's Hardware: Arrow Lake Xe-LPG Plus with XMX](https://www.tomshardware.com/desktops/intels-next-gen-arrow-lake-gpu-will-have-new-xe-lpg-plus-architecture-with-xmx)
  — Arrow Lake restores XMX
- [Intel XMX documentation](https://www.intel.com/content/www/us/en/support/articles/000091112/graphics.html)
- [vllm-project/vllm#28362](https://github.com/vllm-project/vllm/issues/28362) — Meteor Lake vLLM issue (NOT_PLANNED)
- [IPEX-LLM FAQ](https://github.com/intel/ipex-llm/blob/main/docs/mddocs/Overview/FAQ/faq.md)
  — `BIGDL_LLM_XMX_DISABLED=1` documentation
- [PyTorch PR #156669](https://github.com/pytorch/pytorch/pull/156669)
  — adds `SDPBackend.MATH` on XPU (merged Jul 2025, available in PyTorch >= 2.9)
- [torch-xpu-ops #1724](https://github.com/intel/torch-xpu-ops/issues/1724)
  — original issue for missing SDPA backend selection on XPU
- [Intel oneAPI: Programming XMX with joint_matrix](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2023-2/programming-intel-xmx-using-sycl-joint-matrix.html)
  — confirms "no emulation or fall back strategy" for joint_matrix on non-XMX hardware
- [llama.cpp SYCL Backend](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md)
  — lists Meteor Lake iGPU as verified hardware
- [Intel: Run LLMs on GPUs Using llama.cpp](https://www.intel.com/content/www/us/en/developer/articles/technical/run-llms-on-gpus-using-llama-cpp.html)
- [HuggingFace: Phi-2 on Meteor Lake](https://huggingface.co/blog/phi2-intel-meteor-lake)
  — demonstrates LLM inference on Meteor Lake via BigDL-LLM/IPEX-LLM
- [IPEX-LLM Linux GPU Install Guide](https://github.com/intel/ipex-llm/blob/main/docs/mddocs/Quickstart/install_linux_gpu.md)
  — lists Meteor Lake as verified, with driver/kernel instructions
- [OpenVINO GenAI](https://docs.openvino.ai/2025/openvino-workflow-generative/inference-with-genai.html)
  — HuggingFace model auto-conversion and iGPU inference
- [OpenVINO 2026.0 Release](https://medium.com/openvino-toolkit/openvino-2026-0-new-models-enhanced-genai-and-smarter-compression-bf846a59cda8)
  — MoE models, multimodal, hybrid CPU/GPU/NPU execution
- [Optimum-Intel](https://huggingface.co/docs/optimum/intel/index)
  — HuggingFace integration for OpenVINO model conversion
- [llama.cpp SYCL vs Vulkan on MoE models](https://github.com/ggml-org/llama.cpp/issues/19918)
  — Vulkan faster for token generation in some cases
- [llama.cpp OpenVINO Backend](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/OPENVINO.md)
  — official docs for the OpenVINO GGML backend
- [llama.cpp PR #15307](https://github.com/ggml-org/llama.cpp/pull/15307)
  — "Add OpenVINO backend" (merged March 14, 2026)
- [OpenVINO 2026.1 Release](https://www.phoronix.com/news/OpenVINO-2026.1-Released)
  — llama.cpp OpenVINO backend benchmarks (8B INT4: 12.8 tok/s on Core Ultra 7 iGPU)
