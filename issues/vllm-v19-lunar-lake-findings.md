# vLLM v0.19.0 on Lunar Lake Xe2 iGPU — Findings (2026-04-18)

**Hardware**: Intel Core Ultra 7 258V + Arc 140V (Xe2-LPG, Lunar Lake), 32 GiB LPDDR5x (28.57 GiB visible to XPU)
**Software**: vLLM 0.19.0, torch 2.10.0+xpu, triton-xpu 3.6.0, vllm-xpu-kernels (see below), Python 3.12

## TL;DR

vLLM 0.19.0 runs on Lunar Lake Arc 140V, but **the default `requirements/xpu.txt` pins `vllm-xpu-kernels==0.1.4` which does NOT recognise Lunar Lake as XE2**. Fix: upgrade to v0.1.5 (pre-built wheel, no source build). Plus one config flag to work around v0.19's memory-profile allocator budget on shared LPDDR5x.

Our `install_lunar_lake_v19.sh` now does both automatically.

## Finding 1: `vllm-xpu-kernels v0.1.4` rejects Lunar Lake

### Symptom

First inference request (any model) kills the engine:
```
RuntimeError: Only XE2 cutlass kernel is supported currently.
  at vllm_xpu_kernels.flash_attn_interface.flash_attn_varlen_func
  at _vllm_fa2_C.varlen_fwd
```

Server startup succeeds (weights load, Uvicorn comes up on port 8080). The error hits the first forward pass — `HTTP 500 InternalServerError` + `EngineDeadError`.

### Root cause

`is_xe2_arch()` in `csrc/utils.h` of `vllm-xpu-kernels`:

| Version | Arches returning `true` from `is_xe2_arch()` |
|---|---|
| v0.1.4 (pinned by vLLM 0.19.0) | `intel_gpu_bmg_g21`, `intel_gpu_bmg_g31`, `intel_gpu_pvc` |
| v0.1.5 | + `intel_gpu_lnl_m` ← **Lunar Lake** |

Lunar Lake's SYCL arch ID is `intel_gpu_lnl_m` (Lunar Lake M — "mobile" / xe2-lpg). v0.1.4 only recognised Battlemage discrete GPUs (`bmg_g21`/`g31`) and Ponte Vecchio datacentre GPUs (`pvc`). The check fires inside every attention call (`cutlass_chunk_prefill_interface` and `cutlass_paged_decode_interface` in `csrc/xpu/attn/attn_interface.cpp`) and raises `TORCH_CHECK(false, "Only XE2/XE3 cutlass kernel is supported currently.")` when it returns false.

### Fix

Upgrade wheel (no source build needed — Intel ships a pre-built v0.1.5 wheel):
```bash
pip install --force-reinstall --no-deps \
    https://github.com/vllm-project/vllm-xpu-kernels/releases/download/v0.1.5/vllm_xpu_kernels-0.1.5-cp38-abi3-manylinux_2_28_x86_64.whl
```

Automated in `install_lunar_lake_v19.sh` Phase 4 (right after `pip install -r requirements/xpu.txt`).

### Upstream commit

[vllm-project/vllm-xpu-kernels @ v0.1.5 — `csrc/utils.h`](https://github.com/vllm-project/vllm-xpu-kernels/blob/v0.1.5/csrc/utils.h)

## Finding 2: v0.19 memory profile too pessimistic on 28 GiB shared memory

### Symptom

Model loads fine, but KV cache allocation fails:
```
Available KV cache memory: -1.01 GiB
ValueError: This number can not be negative ...
```

Happens even with `--max-num-seqs 4 --max-model-len 2048 --gpu-memory-utilization 0.7`. Lowering further doesn't help.

### Root cause

`vllm/v1/worker/gpu_worker.py::determine_available_memory` runs `profile_run()` with a worst-case batch, measures peak XPU allocated memory, and subtracts from the `gpu_memory_utilization * total_memory` budget. On Lunar Lake the new v0.19 **MXFP4 `MoEPrepareAndFinalizeNoDPEPModular`** backend pre-allocates a large scratch buffer for expert dispatch, plus the GPT-OSS vocab-logits buffer (201,088 tokens × bf16), giving a profile peak ≈ 8 GiB regardless of how low `max_num_seqs`/`max_model_len` are pushed. On the 28 GiB iGPU budget, after 13 GiB of MoE weights, that leaves too little for any KV cache.

### Fix

v0.19 added a built-in opt-out. Pass `--kv-cache-memory-bytes N` → vLLM skips the peak-measurement profile and reserves exactly N bytes for KV cache (from `gpu_worker.py::determine_available_memory` line 344).

```bash
vllm serve /shared/models/gpt-oss-20b \
    --gpu-memory-utilization 0.6 \
    --max-model-len 2048 --max-num-seqs 2 \
    --kv-cache-memory-bytes 2147483648    # 2 GiB for KV cache
```

KV cache sizing rule of thumb — for GPT-OSS-20B (24 layers, GQA 8×64):
- 2 KB per token per layer × 24 = 48 KB/token
- 2 GiB = ~42k tokens across all concurrent sequences → ample for decode at `max_num_seqs=2, max_model_len=2048` (4k tokens max).

### v0.14 equivalent

Our v0.14 patch used `VLLM_SKIP_PROFILE_RUN=1` + `VLLM_SKIP_PROFILE_MULTIPLIER=1.15`:
```python
# Skip profile, estimate peak = torch.xpu.memory_allocated() * 1.15
peak_allocated = int(used_memory * multiplier)
```
`--kv-cache-memory-bytes` in v0.19 is stricter (you specify the exact number) but upstream.

## Finding 3: XPU shared-memory leaks on crashed launches

### Symptom

After 3-4 failed `vllm serve` launches, startup preflight fails:
```
ValueError: Free memory on device xpu:0 (0.52/28.57 GiB) on startup is less
than desired GPU memory utilization (0.6, 17.14 GiB).
```

### Cause

`SIGKILL`-ed `EngineCore` / `APIServer` processes don't release the Level Zero shared-memory mappings, and the XPU allocator's `mem_get_info` doesn't see that memory as free. `free -h` shows it as `shared: 3.7 GiB`. Not reclaimable by `echo 3 > drop_caches`.

### Workaround

- Drop page cache before each launch: `sync && echo 3 | sudo tee /proc/sys/vm/drop_caches`
  (gets back the buff/cache but not the leaked shared mappings)
- If preflight fails with < 3 GiB free after clean launches: **reboot**.

### Upstream

Likely an IPEX / Level Zero driver issue. Worth filing upstream when a minimal repro is possible.

## Finding 4: Attention backend selection

`VLLM_ATTENTION_BACKEND` env var from v0.14 is unknown in v0.19. Use the CLI flag:
```bash
vllm serve ... --attention-backend FLASH_ATTN   # default, uses vllm_xpu_kernels
vllm serve ... --attention-backend TRITON_ATTN  # triton fallback
vllm serve ... --attention-backend TORCH_SDPA   # native pytorch
```

With `vllm_xpu_kernels v0.1.5`, `FLASH_ATTN` works on Lunar Lake — no backend change needed.

## What's unchanged from v0.14 (i.e. still works)

- Python 3.12 required
- `torch 2.10.0+xpu` from the PyTorch XPU wheel index
- `triton-xpu 3.6.0` (shipped as a transitive dep of torch 2.10.0+xpu)
- Intel oneAPI base toolkit (runtime libs only — source build doesn't need it for v0.19)
- `render` group membership for `/dev/dri/renderD128` access

## What's new in v0.19 that helps

- Native Intel XPU support (no `vllm_for_multi_arc.patch` needed)
- Native Gemma 4 support (requires `transformers>=5.5.0`)
- `--kv-cache-memory-bytes` flag (replaces our v0.14 `VLLM_SKIP_PROFILE_RUN` patch)
- Pre-built `vllm_xpu_kernels` wheels (no 1.5-2 hour source build needed for most users)
- `requirements/xpu.txt` pins the right torch + triton-xpu versions

## What's not yet tested on v0.19 + Lunar Lake

- INT4 AutoRound MoE models (Qwen3-VL-30B-A3B, GLM-4.7-Flash). Expected to work via vLLM 0.19's new `XPUExperts` + vllm-xpu-kernels modular MoE — but we didn't get that far.
- Gemma 4 models (would need the transformers v5 path)

## End-to-end verification on v0.19 + v0.1.5 wheel (2026-04-18, post-reboot)

GPT-OSS-20B MXFP4 with `--kv-cache-memory-bytes 2147483648` (2 GiB KV), `max_num_seqs=2`, `max_model_len=2048`, `enforce_eager=True`.

### Model loading

| Stage | Time |
|---|---:|
| Safetensors load (3 shards, warm cache) | 19.1 s |
| Model init total | 23.8 s |
| Memory | 12.87 GiB |

### Decode benchmarks (`vllm bench serve --dataset-name random --num-prompt 5 --max-concurrency 1 --ignore-eos`)

| Config | Output throughput (avg) | Peak | Median TPOT | Median TTFT |
|---|---:|---:|---:|---:|
| 128 in / 128 out | 19.1 tok/s | 24.0 tok/s | 43.1 ms | 214 ms |
| 1024 in / 1024 out | 15.2 tok/s | 24.0 tok/s | 67.5 ms | 1066 ms |

Steady-state decode is ~23 tok/s — **on par with v0.14's IPEX `GatedMLPMOE(is_mxfp4=True)` baseline of ~22 tok/s**, so MXFP4 models see no meaningful speedup on v0.19 (same kernel underneath; only the routing wrapper changed). The real v0.19 payoff is expected to be INT4 AutoRound MoE, where v0.14 was stuck at 0.9 tok/s due to our Python per-expert loop.

### Serving config for openclaw integration

`~/bin/vllm-gptoss` (one-shot launcher, no systemd needed):
```bash
vllm serve /shared/models/gpt-oss-20b \
    --served-model-name gpt-oss-20b \        # openclaw sends this id
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.70 \
    --enforce-eager \
    --port 8080 \                             # matches openclaw's local provider
    --max-model-len 65536 \                   # 64k context
    --max-num-seqs 1 \                        # single concurrent seq (fits 3 GiB KV)
    --kv-cache-memory-bytes 3221225472 \      # 3 GiB; 65536 × 48 KB = 3.07 GiB
    --enable-auto-tool-choice \
    --tool-call-parser openai \
    --reasoning-parser openai_gptoss
```

**Important**: `--served-model-name` is required when `--model` is a filesystem path. Without it, the OpenAI API only accepts the full path as the `model` field and returns 404 for short ids like `"gpt-oss-20b"`.

**Memory budget math at 64k context:**
- KV per token (GPT-OSS-20B, GQA 8×64, bf16) = 2 × 8 × 64 × 2 B × 24 layers = 48 KB/token
- 1 seq × 65536 tokens = 3.07 GiB
- 2 seqs × 65536 tokens = 6.14 GiB (needs 7 GiB `--kv-cache-memory-bytes` for margin + util=0.82)
- Single-seq config (above) is the safer default.

## Finding 3 (expanded): XPU shared-memory leak is worse than first thought

The leak is not confined to `SIGKILL` — even graceful shutdowns (Ctrl+C on `exec vllm serve`, SIGTERM to the engine) leak shared-memory mappings on Lunar Lake iGPU. Observed pattern across this session:

| Event | XPU free after |
|---|---:|
| Post-reboot, clean | 28 GiB |
| After 1st run (loaded model, bench, Ctrl+C) | 7-8 GiB |
| `sync; echo 3 > drop_caches` | 21-27 GiB ← often recovers |
| After 2nd crashed launch | 0.5-3 GiB |
| After 3rd crashed launch | 0.5 GiB — stuck |
| Reboot only reliable reset | 28 GiB |

**Workaround for restart loop:**
Before each `vllm-gptoss` relaunch:
```bash
sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
```
Recovers enough free XPU for a fresh launch in most cases. If free stays below ~15 GiB afterwards, reboot.

**Why cache drop helps:** On Lunar Lake's unified memory, the XPU allocator's `mem_get_info()` counts the Linux page cache as "in use" even though it's reclaimable. Dropping caches makes the XPU allocator see the real free pool. This is separate from actual L0 shared-memory leaks, which are not reclaimable without reboot.

**Upstream bug territory**: L0 driver should free mappings on process exit (any exit path) — Linux normally does this automatically for anon mmap, but L0's shared-mem pool has its own lifetime tracking that misses cleanup on some exit paths.

## Summary of commits this session (CB5w6)

| Commit | Purpose |
|---|---|
| `cbd1c2f` | Address critical issues in v19 install script (sudo guard, render check, etc.) |
| `e254ae4` | Revert MAX_JOBS to 6 with rationale |
| `43e284e` | Align install script with official vLLM 0.19 XPU recipe |
| `43145e1` | Upgrade vllm-xpu-kernels → v0.1.5 (Lunar Lake XE2 fix) + this findings doc |
| (next) | End-to-end verification + launcher script + memory-leak docs |

## Files

- `vllm/scripts/install_lunar_lake_v19.sh` — full installer (v0.1.5 fix included)
- `~/bin/vllm-gptoss` — one-shot launcher (not committed, install manually via the snippet above)
- `~/.openclaw/openclaw.json` — openclaw config (`providers.local.baseUrl = http://127.0.0.1:8080/v1`, default model set to `local/gpt-oss-20b`)
- `issues/vllm-v19-lunar-lake-findings.md` — this doc

## Install script changes

`vllm/scripts/install_lunar_lake_v19.sh`:

1. Phase 4 now force-reinstalls `vllm_xpu_kernels v0.1.5` after `pip install -r requirements/xpu.txt` pulls in v0.1.4.
2. Quick-start example updated to show the `--kv-cache-memory-bytes` workaround and the drop-caches / reboot hints.
3. (From earlier session) Phase 5 source build of vllm-xpu-kernels is now opt-in via `VLLM_BUILD_XPU_KERNELS=1` — pre-built wheel is default, saves ~2h.
