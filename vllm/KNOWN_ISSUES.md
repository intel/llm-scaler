
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
