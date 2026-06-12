# PR body: [Platform][XPU] Opt-in integrated-GPU override for unified memory

Target: `vllm-project/vllm` main
Branch: `MegaStood:xpu-integrated-gpu-mem` → https://github.com/MegaStood/vllm/tree/xpu-integrated-gpu-mem

Not posted yet. Review, tweak as needed, then create PR from laptop.

---

## Summary

Intel integrated GPUs on Lunar Lake / Arrow Lake / Panther Lake SoCs share LPDDR with the CPU. `torch.xpu.mem_get_info()`'s "free" figure excludes reclaimable Linux page cache, so after vLLM reads the model safetensors during startup (which populates page cache), `MemorySnapshot.measure()`'s free-memory check fails and the engine aborts with:

```
ValueError: Free memory on device xpu:0 (18.79/28.57 GiB) on startup
is less than desired GPU memory utilization (0.7, 20.0 GiB).
```

This happens even on a freshly-booted machine with no other XPU processes, because the measured "free" drops by multiple GiB as vLLM itself reads the model file during startup.

## Why this fix

`vllm/utils/mem_utils.py::MemorySnapshot.measure()` already routes through `current_platform.is_integrated_gpu(device.index)` and swaps in `psutil.virtual_memory().available` when True — the correct, page-cache-aware number. CUDA implements this via `torch.cuda.get_device_properties().is_integrated`. XPU doesn't, so integrated Intel GPUs fall back to the default `False` from `interface.py` and hit the miscounted path.

This PR adds the XPU override, gated on a new `VLLM_XPU_UNIFIED_MEMORY` env var. When set (`=1`), `is_integrated_gpu()` returns True and the existing UMA-handling code in `mem_utils.py` activates.

## Why not auto-detect

A heuristic comparing device `total_memory` to host RAM (ratio >= 0.8 implies unified) was considered and rejected. It misclassifies PVC-class deployments where dedicated HBM happens to match host RAM, causing false positives and broken memory accounting on discrete datacenter GPUs. Until PyTorch exposes an `is_integrated` flag on `XpuDeviceProperties` (parallel to CUDA's), explicit opt-in is the safer default. Zero regression risk for discrete Arc / Battlemage / Ponte Vecchio users.

## Changes

- `vllm/envs.py`: register `VLLM_XPU_UNIFIED_MEMORY` (default False)
- `vllm/platforms/xpu.py`: override `is_integrated_gpu()` to return `envs.VLLM_XPU_UNIFIED_MEMORY`

Total: +25 lines, 2 files.

## Duplicate-PR check (per AGENTS.md)

```bash
gh pr list --repo vllm-project/vllm --state all --search "xpu unified memory mem_get_info"
gh issue list --repo vllm-project/vllm --state all --search "xpu integrated gpu memory"
```

Closest hit: issue #37828 ("Intel ARC 140v not supported as XE2 cutlass kernel") — different issue (XE2 cutlass kernel arch allowlist, fixed in vllm-xpu-kernels v0.1.5). No open PR addresses the unified-memory accounting gap.

## Testing

Hardware: Intel Core Ultra 7 258V with Arc 140V iGPU (Lunar Lake, `intel_gpu_lnl_m`, Xe2-LPG), 30.86 GiB system RAM (28.57 GiB visible to XPU via unified LPDDR5x).

Backported the patch to vLLM 0.19.0 via equivalent direct `elif current_platform.is_xpu()` branch in `mem_utils.py` (v0.19.0 predates the `is_integrated_gpu` refactor). Verified the semantics:

**Before** (stock v0.19.0):
```
ValueError: Free memory on device xpu:0 (18.79/28.57 GiB) on startup
is less than desired GPU memory utilization (0.7, 20.0 GiB).
```
Occurs for any model ≥17 GiB, even with `drop_caches` run seconds earlier — the mere act of vLLM reading safetensors during startup repopulates cache below the check threshold.

**After** (patch applied, `VLLM_XPU_UNIFIED_MEMORY=1`):
- `GLM-4.7-Flash-AWQ-4bit` (19 GiB model): loads successfully; `vllm bench serve` 1024/1024/3 @ concurrency=1 → 6.58 tok/s.
- `GLM-4.7-Flash-int4-autoround` (17 GiB): loads successfully; 4.71 tok/s.
- `Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit` (17 GiB): loads successfully; 18.22 tok/s.

Bench recipe:
```bash
vllm bench serve \
    --backend openai \
    --base-url http://localhost:8080 \
    --model <served-name> \
    --tokenizer /shared/models/<model-dir> \
    --dataset-name random \
    --random-input-len 1024 \
    --random-output-len 1024 \
    --num-prompts 3 \
    --max-concurrency 1
```

Also verified `psutil.virtual_memory().available` returns a realistic free-memory estimate that accounts for reclaimable page cache: right after `drop_caches`, `mem_get_info.free = 24.89 GiB` and `psutil.available = 26.95 GiB` (consistent with 30.86 GiB system RAM minus ~4 GiB used by desktop/browser). Without the patch, as vLLM loads safetensors this drops to `mem_get_info.free = 18.79 GiB` (cache-polluted) vs `psutil.available = 22.26 GiB` (correctly page-cache-aware).

## AI assistance disclosure

This change was drafted with AI assistance (Claude). Every line was reviewed by the human submitter, the behavior was validated end-to-end on the hardware described above, and the duplicate-PR check was run per AGENTS.md.
