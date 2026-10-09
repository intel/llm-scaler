# XPU graph mode on vLLM — status for Lunar Lake / Arc 140V (2026-10-09)

Scope: should XPU graph mode be enabled on the MSI Claw 8 (Core Ultra 7 258V, Arc 140V, 32 GB
unified memory) for Gemma 4 26B-A4B INT4 MoE + MTP assistant, or Qwen3-Coder-30B-A3B INT4?
Method: 5 research angles, each critical claim attacked by two independent skeptics; claims below
are tagged **[confirmed]**, **[corrected]** (skeptic found an overstatement, corrected text shown),
or **[unverified]**.

## 1. TL;DR

- **Keep eager on this box.** The only eager-vs-graph A/B on the same hardware (MSI Claw 8 AI+, 258V,
  32 GB, vLLM 0.30 / torch 2.13 / kernels 0.1.14.1) measured **+3 % decode without MTP and 0 % with
  MTP k=2**, at a cost of 0.7–1.0 GiB device memory and 1.2–1.4 GiB host `MemAvailable`; the author
  rejected graphs on memory grounds. Decode here is bandwidth-bound, so launch-overhead removal buys
  almost nothing. [confirmed]
- **v0.31.0 turns graphs ON by default** (PR #51600, merged Sep 24) with no gating for MoE, INT4, MTP,
  structured outputs or multimodal; `VLLM_XPU_ENABLE_XPU_GRAPH` is gone and silently ignored. When
  upgrading, opt out explicitly. [confirmed]
- **Your env var breaks graphs on torch 2.14:** `SYCL_UR_USE_LEVEL_ZERO_V2=0` forces the legacy
  Level Zero adapter, which stubs the whole graph record/replay API; torch 2.14's native-recording
  XPUGraph has no fallback, so capture throws. Graphs on v0.31 require unsetting it (V2 is the
  default on Lunar Lake). [confirmed]
- **The grouped-GEMM race is fixed upstream** — not by PR #524/#457 (still open) but by
  vllm-xpu-kernels **#586** (merged Sep 10), shipped in 0.1.15 / PyPI 0.1.15.1–0.1.15.4. Your
  0.1.14.1 does **not** have it: never enable graphs on the current stack for INT4 MoE. [confirmed]
- Open correctness bugs that touch this workload class: #54785 (wrong logits, graphs + MTP k=4,
  GDN model, SYCL-graph era; maintainer asked for a torch-2.14 retest Sep 28) and #54698 (TP=1
  `replay()` hang under concurrent load, Qwen3-30B-A3B-class GPTQ INT4, torch 2.13). Neither has a
  fix or a retest on the Level Zero path. No Gemma 4 graph failure is on record. [corrected: both
  issues have maintainer replies; they are not "zero comments"]
- Intel's own vLLM fork (llm-scaler, base v0.26 / torch 2.12) is still eager-only in every serve
  example; its SGLang track states "XPU graph capture cannot express the speculative control flow,
  so decode stays eager". [confirmed]
- If you do experiment on v0.31: unset the V2 env var, compute-runtime ≥ 26.27, kernels 0.1.15.4,
  oneAPI 2026.1.x pip runtime in the venv, `cudagraph_mode=FULL_DECODE_ONLY`, capture sizes `[1]`
  (plus `[1+k]` for MTP), greedy parity vs eager, and watch host `MemAvailable`.

## 2. What changed since Sep 24

| Date | Change | Impact |
|---|---|---|
| Jul 31 | pytorch #188874 "Enable XPUGraph native recording mode" merged → torch 2.14.0 (Sep 2) | XPU graphs move from SYCL command buffers to Level Zero record/replay on every non-PVC GPU, incl. Arc 140V; requires oneAPI 2026.1 runtime and L0 V2 adapter |
| Sep 10 | vllm-xpu-kernels #586 `at::empty`→`at::zeros` for the Xe2 grouped-GEMM atomic counter | Closes the replay race #524 described; in 0.1.15+ |
| Sep 23 | vLLM #56013 XPU → torch 2.14 | Prerequisite for native-recording graphs |
| Sep 24 | vLLM #51600 graphs default-on, env var removed, XPU now runs the CUDA-graph memory profiler | Graph memory is estimated before KV sizing (`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1`) |
| Oct 2 | **v0.31.0** (torch 2.14.0, triton 3.8.0+xpu, kernels 0.1.15.4, auto_round_lib 0.15.0) | First release with graphs on by default |
| Oct 4 | vLLM #59159 DeepSeek-V4 FP8 sparse decode made capturable (in v0.31.1rc0) | Not relevant to Gemma/Qwen |
| Oct 8 | vLLM #60529 MiMo MTP drafter D2H sync → `torch.where` (main only) | Shows the class of bug: any host sync in a drafter breaks capture with `UR_RESULT_ERROR_UNSUPPORTED_FEATURE` |
| Oct 1–7 | compute-runtime master: graph replay synchronization fixes (cross-command-list patch preamble; synchronize previous graph before patching) | Driver-side replay bugs still being fixed; not in any released driver yet |
| Oct 8 | kernels maintainer branch `kunshang/torch-2.15` (torch 2.15 + oneAPI 2026.1.3); vLLM draft #60539 ports llm-scaler eager decode fusions | Churn continues; torch 2.15 XPUGraph.cpp is byte-identical to 2.14 |

Still open / unchanged: `Gemma4Config` forces TRITON_ATTN on XPU (your `--attention-backend FLASH_ATTN`
workaround stays); no `is_integrated_gpu` in `xpu.py`.

## 3. How it works in v0.31

- **Default:** `-O2` → `CompilationMode.VLLM_COMPILE` + `cudagraph_mode=FULL_AND_PIECEWISE`.
  FLASH_ATTN on XPU reports `UNIFORM_BATCH`, so uniform decode batches (including MTP batches with
  query_len = 1+k) replay **FULL** graphs; prefill/mixed batches use **PIECEWISE** graphs around
  attention. [confirmed]
- **Wrapper:** generic `CUDAGraphWrapper` with `torch.cuda.*` monkey-patched to `torch.xpu.*`
  process-wide (`torch.cuda.graph → torch.xpu.graph`, `CUDAGraph → XPUGraph`). Every capture first
  calls `torch.xpu.synchronize()` + `torch.xpu.empty_cache()`. [confirmed]
- **Auto-disable on XPU:** only torch < 2.11, `--enable-sleep-mode`, and (rc0+) V2 runner with
  `-cc.mode=stock_torch_compile`. Generic rules also apply: `--enforce-eager`, pooling models,
  encoder-decoder, KV connectors. Nothing for MoE/INT4/MTP. [corrected: list completed]
- **Disable:** `--enforce-eager` (also disables torch.compile and JIT warmup), `-O0`, or
  `--compilation-config '{"cudagraph_mode":"NONE"}'` (keeps compile). [confirmed]
- **Tune:** `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1]}'`
  gives decode graphs without torch.compile; `max_cudagraph_capture_size` and `--max-num-seqs` bound
  the size grid (default grid is sized for a 256-request server). [confirmed]
- **Memory:** profiler captures the two largest sizes per mode into a temporary pool, extrapolates,
  subtracts from the KV budget. Measured pool footprint: 0.74 GiB (Gemma 4 26B int4, B70, v0.31.0);
  0.68–0.97 GiB device + 1.2–1.4 GiB host on the Claw (v0.30). torch 2.14 has an open pool-handle
  bug (#198794: pool unusable after all graphs released; ~10 MiB leaked) that does not affect
  capture-once/replay-forever serving. [confirmed]
- **Runtime prerequisites for native recording:** Level Zero **V2** adapter (V1 stubs `urGraph*Exp` →
  `UR_RESULT_ERROR_UNSUPPORTED_FEATURE`); compute-runtime advertising
  `ZE_extension_record_replay_graph` (≥ 26.27.39122.11; experimental name since 25.40) and L0 loader
  ≥ 1.32; oneAPI 2026.1.x runtime pip packages (`intel-cmplr-lib-ur`, `intel-sycl-rt`, … ==2026.1.1 per
  vLLM docs; 2026.1.2 preferred — 2026.1.1 + driver 26.27 segfaulted on first submit on Battlemage,
  intel/llvm #22958). [corrected: driver minimum is 25.40 for the experimental name]
- Native recording limits: in-order queue only, no host tasks, no async malloc/free inside capture,
  no graph update — any custom kernel patch that does a D2H read or allocation during decode will
  fail capture with the `UR_RESULT_ERROR_UNSUPPORTED_FEATURE` / "wait method cannot be used for an
  event associated with a command graph" class of error. [confirmed]

## 4. Correctness status

| Bug | Component | Status (Oct 9) | Affects this user? |
|---|---|---|---|
| Grouped-GEMM counter race under replay (#524/#457) | kernels, Xe2 INT4 MoE | **Fixed by #586**, in 0.1.15+; absent from 0.1.14.1 | Yes — INT4 compressed-tensors MoE dispatches to this kernel on `intel_gpu_lnl_m` |
| #54785 wrong logits, graphs + MTP k=4 | vLLM, GDN hybrid (Qwen3.8-27B), torch 2.11, TP=2 | Open; maintainer asked for 2.14 retest Sep 28; no PR | Indirect: Gemma 4 is not GDN; k≤3 was bit-stable in the report |
| #54698 `replay()` hang, TP=1, GPTQ INT4 MoE, concurrent load | vLLM/torch 2.13 | Open; analysis posted, no fix | Closest to Qwen3-Coder; single-stream use may not trigger it |
| #53993 V2 runner + graphs + structured outputs | vLLM | Issue closed Sep 24 as linked to #51600; fix PR #53997 unmerged, code unchanged | Yes if OpenClaw uses guided/JSON decoding with graphs on — `VLLM_USE_V2_MODEL_RUNNER=0` or graphs off |
| #58388 / #60379 / #56917 capture crashes | oneCCL, TP≥2 | Open; fix #58415 closed unmerged | No (single GPU) |
| #60529 drafter D2H sync breaks capture | MiMo MTP | Fixed on main | Gemma 4 drafter has no such sync in upstream code [unverified on hardware] |
| kernels #389 GDN spec metadata under graph padding | GDN models | Open | No |
| llm-scaler #596/#698 NaN logits under graphs, INT4 MoE, long context | Intel fork, old images | Closed/unresolved | Same model class; different stack |
| Driver replay-sync fixes (Oct 1/7) | compute-runtime master | Unreleased | Yes — back-to-back replays are the decode loop |

The kernels repo has **zero** graph-capture/replay tests. [confirmed]

## 5. Performance evidence

| Hardware | Model | Stack | Eager → graph decode | Source |
|---|---|---|---|---|
| **MSI Claw 8 AI+ (258V / Arc 140V / 32 GB)** | Qwen3.6-35B-A3B MXFP4, batch 1 | vLLM 0.30, torch 2.13, kernels 0.1.14.1, sizes [1,2,4] | 19.27 → 19.84 tok/s (**+3 %**); compile alone −0.4 % | alpha2beta/vllm.xpu status.md |
| same | same + MTP k=2 | sizes [1] | 27.69 → 27.54 (**0 %**) | same |
| Arc Pro B70 | gemma-4-26B-A4B int4 g32 + draft | v0.31.0 image, graphs default | 147 → 149 short (+1 %); 119 → 128 after 12.9k prompt (+8 %); 0.74 GiB | epsilonagentx/intel_arc_gpu_llm |
| Arc Pro B60 | gemma-4-26B int4 | v0.31.0 | eager 52 → compiled 56 (+7 %) | same |
| Arc Pro B70 | Nemotron 3.5 30B-A3B GPTQ | 0.26-era, env var | 21.8 → 87–93 (launch-bound case) | SergiioB cookbook |
| Arc Pro B70 | INT4 MoE + MTP | cookbook configs | ~+9–10 % | SergiioB cookbook |
| Arc Pro B70 | Qwen3-0.6B dense | PR #51600 benchmark | 494 → 2060 (launch-bound) | PR #51600 comment |

Reading for the 140V: at ~89 GB/s a 3.8B-active INT4 step is ~30 ms of pure weight traffic;
launch overhead is a small fraction, which is why the same-hardware A/B lands at +3 % / 0 %. The
B70 multiples come from launch-bound cases (0.6B dense, Nemotron's Mamba path) and do not transfer.
Realistic expectation on the Claw: **0–5 % single-stream, possibly slightly more at long context**,
against ~1 GiB device + ~1.3 GiB host memory on a box whose weight ceiling is already ~21 GiB.

## 6. Lunar Lake specifics

- Only one public iGPU data point exists (the Claw A/B above); no Lunar Lake report on the torch 2.14
  native-recording path. Intel CI tests LNL in torch-xpu-ops, but Fedora/Nobara + kernel 6.18 is
  outside the validated matrix. [confirmed]
- `SYCL_UR_USE_LEVEL_ZERO_V2=0` must be **unset** for graphs on torch 2.14 (see §1). If V2 was
  disabled because of an eager-mode bug on this box, that bug must be re-evaluated on oneAPI 2026.1
  first; graphs and the V1 adapter are mutually exclusive. [confirmed]
- Check driver: `ls /usr/lib64/libze_intel_gpu.so*` / `sycl-ls --verbose` → need ≥ 26.27 (stable
  extension) and L0 loader ≥ 1.32. [confirmed]
- Memory on UMA: graph pools are device allocations from the shared pool and `empty_cache()` runs
  before every capture; the Claw A/B saw host `MemAvailable` fall to 0.7 GiB with capture size [1]
  on top of a ~20 GiB model. On this box the memory cost, not correctness, is the binding constraint.

## 7. Recommended procedure

**Now (v0.28 + kernels 0.1.14.1):** nothing to do — graphs are off and must stay off (no #586).

**On v0.31.x (control run, recommended default):**
```bash
# keep torch.compile, no graphs
vllm serve <gemma-4-26b-int4> --attention-backend FLASH_ATTN --language-model-only --block-size 64 \
  --compilation-config '{"cudagraph_mode":"NONE"}' \
  --speculative-config '{"method":"mtp","model":"<assistant>","num_speculative_tokens":2}' ...
# or simply --enforce-eager (also disables compile; the Claw A/B showed compile alone is -0.4 %)
```

**Experiment (only if curious; expect ≤ 5 %):**
```bash
unset SYCL_UR_USE_LEVEL_ZERO_V2            # V1 adapter cannot capture
# prerequisites: compute-runtime >= 26.27, kernels 0.1.15.4, oneAPI 2026.1.x pip runtime in venv
vllm serve <gemma-4-26b-int4> --attention-backend FLASH_ATTN --language-model-only --block-size 64 \
  --max-num-seqs 2 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,3]}' \
  --speculative-config '{"method":"mtp","model":"<assistant>","num_speculative_tokens":2}' ...
```
Checks: (1) startup log shows capture completing and the graph-memory estimate; (2) greedy output
identical to the control for 3 prompts incl. one 4–8K-token generation; (3) `free -g` host
`MemAvailable` ≥ 2.5 GiB at steady state; (4) 30-minute soak with two concurrent requests (the
#54698 pattern); (5) bench 1024/1024/3 vs control. Fall back to `cudagraph_mode=NONE` on any
divergence, capture error (`UR_RESULT_ERROR_UNSUPPORTED_FEATURE`, "command graph" event errors), or
if host headroom drops below ~2 GiB.

## 8. Watch list

- kernels: #524 / #457 (superseded by #586; close expected), #389 / #391 (GDN), graph-capture tests
  (none exist).
- vLLM: #54785, #54698 (retest on 2.14 pending), #53997 (structured outputs + V2 runner), #58415
  (TP oneCCL), #60539 (eager decode fusions from llm-scaler — may make eager faster, not graphs).
- pytorch: #198794 (graph pool reuse), #198615 (device-wide sync regression 4–8 %), #190988
  (per-capture RNG / pool tracking, open).
- compute-runtime: next release carrying the Oct 1/7 graph replay sync fixes (26.40-ish).
- Gemma4Config TRITON_ATTN forcing on XPU — unchanged; still needs the explicit backend flag.

## 9. Sources

- https://github.com/vllm-project/vllm/pull/51600 · /pull/56013 · /pull/59159 · /pull/60529 · /pull/60539 · /pull/58415 · /pull/53997 · /pull/50038
- https://github.com/vllm-project/vllm/issues/54785 · /issues/54698 · /issues/53993 · /issues/58388 · /issues/60379 · /issues/56917 · /issues/58912
- https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/platforms/xpu.py · /vllm/config/vllm.py · /vllm/config/compilation.py · /vllm/v1/worker/gpu_worker.py · /vllm/v1/worker/xpu_model_runner.py · /docs/getting_started/installation/gpu.xpu.inc.md · /requirements/xpu.txt
- https://github.com/vllm-project/vllm-xpu-kernels/pull/586 · /pull/524 · /pull/457 · /issues/389 · /issues/567 · /issues/509 · /releases/tag/v0.1.15 · https://pypi.org/project/vllm-xpu-kernels/#history
- https://github.com/pytorch/pytorch/pull/188874 · /issues/193756 · /issues/198794 · /pull/198615 · /pull/190988 · /issues/187277 · /releases/tag/v2.14.0 · https://raw.githubusercontent.com/pytorch/pytorch/v2.14.0/torch/xpu/graphs.py
- https://github.com/intel/llvm/pull/21626 · /pull/22489 · /issues/22958 · /issues/23073 · https://raw.githubusercontent.com/intel/llvm/sycl/unified-runtime/source/adapters/level_zero/graph.cpp · …/sycl-rel-7_1/unified-runtime/source/adapters/level_zero/device.cpp · …/adapter.cpp · sycl/doc/extensions/experimental/sycl_ext_oneapi_graph.asciidoc
- https://github.com/intel/compute-runtime/blob/26.27.39122.11/level_zero/core/source/driver/driver_handle_helper.cpp · https://github.com/intel/compute-runtime/commits/master/level_zero/experimental/source/graph
- https://github.com/alpha2beta/vllm.xpu/blob/main/status.md (MSI Claw 8 AI+ eager-vs-graph A/B)
- https://github.com/epsilonagentx/intel_arc_gpu_llm/blob/developer/vllm_openai_xpu/README.md (gemma-4-26B int4 on B70, v0.31.0)
- https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook (Nemotron/Qwen graph numbers)
- https://github.com/intel/llm-scaler/blob/main/vllm/README.md · /sglang/README.md · /issues/596 · /issues/698 · /issues/725 · /issues/640
