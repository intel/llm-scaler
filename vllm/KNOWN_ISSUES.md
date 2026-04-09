
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

## Alternative: Use IPEX-LLM (Ollama) for Meteor Lake

[IPEX-LLM](https://github.com/intel/ipex-llm) (formerly BigDL-LLM) **does work on
Meteor Lake** because it uses a llama.cpp backend with a software SDP fallback that
does not require XMX:

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

This is currently the **only viable path** for LLM inference on Meteor Lake iGPU.
Performance will be lower than on XMX-equipped GPUs, but it works.

## References

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
