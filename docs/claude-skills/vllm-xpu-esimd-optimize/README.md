# vllm-xpu-esimd-optimize

Skill for optimizing vLLM models on Intel XPU (BMG / Arc B-series) via ESIMD kernels in llm-scaler.

## When Claude picks this up

Any prompt that asks to make a vLLM model faster on XPU and provides (or implies) a container with editable vllm-xpu + a llm-scaler checkout. Examples:

- *"在容器 `wj-test-new-021` 里给 gemma-4-26B-A4B-it 写 esimd kernel 把 decode ITL 压到 17ms"*
- *"qwen3-next-80B-A3B 跑得太慢，能不能 fuse RMSNorm + GEMV"*
- *"phi-4 dense MLP 跑 fp8 比 bf16 还慢，看下能不能用 esimd 替代"*
- *"31B 输出空，跟 26B 对比 bisect 一下"*

## What you get

A repeatable 8-phase loop the agent will follow rather than reinvent:

1. **Smoke test** — 16-token offline reproducer + token-id fingerprint (`assets/` has no template — built ad-hoc, see SKILL.md Phase 1).
2. **Accuracy gate** — `assets/offline_chat_reqs.py` replays 5 GSM8K few-shot chat requests offline; 5/5 = 1.000 is the bar.
3. **ITL bench** — `assets/bench_itl.py`, median of 5, kv_cache=fp8.
4. **Profile** — three lenses: (A) wall-time per layer subcomponent (`assets/profile_attn.py` env-gated `PROFILE_ATTN=1`), (B) kernel-level micro-bench, and (C) **unitrace whole-run kernel timeline — the bottleneck killer; start here when you don't yet know where the time goes**. One trace gives device-busy% (launch-bound vs compute-bound verdict), kernel-family self-time, and the gap profile; `assets/unitrace_agg.py` does the aggregation. It is prebuilt in the container — `cp` it onto the path, don't rebuild.
5. **Write ESIMD kernel** — author header, watch GRF spill (>12 KB triggers `UR_RESULT_ERROR_OUT_OF_RESOURCES` under load), handle K-divisibility, decide whether the fuse actually wins for the workload's N.
6. **Wire into model** — bind, re-export, env-gate, cache buffers on the module.
7. **Verify** — Phase 1 + Phase 2 + Phase 3 every time.
8. **Bisect** — env-disable → baseline diff → vllm file diff → git bisect, fix by gating not reverting.

## Required inputs

The agent will ask for whichever of these are missing:

| Variable | Example |
|---|---|
| `CONTAINER` | `wj-test-new-021` |
| `VLLM_PATH` | `/workspace/llm-scaler-vllm-xpu` |
| `LLMSCALER_PATH` | `/llm/models/test/llm-scaler/vllm/custom-esimd-kernels-vllm` |
| `MODEL_PATH` | `/llm/models/weights/gemma-4-26B-A4B-it` |
| `MODEL_FAMILY` | `gemma4` |
| `TP` + `ZE_AFFINITY_MASK` | `2` + `6,7` |
| `BASELINE_CONTAINER` (optional) | `wj-test-new-021-1` |

## Read order

1. **`SKILL.md`** — full recipe, anti-patterns, bisect priority list, rebase guidance.
2. **`assets/`** — copy-paste templates; only `MODEL_PATH` / `TP` / `PROMPT_IDS` change between models.

## Past wins this skill encodes

- **gemma-4-26B-A4B-it (TP=2 fp8)**: ITL 46.8 ms → 21.6 ms via 4 fused kernels (norm_gemv_norm_fp16 / norm_add_norm × 2 / BMG fp8 GEMV redirect).
- **gemma-4-31B-it accuracy fix**: bisected `28d462ed5` (GeluAndMul XPU esimd) → root cause `d=10752` overflow → fix by `d <= 4096` gate, no perf regression on 26B.
- **chat-prefill MoE NaN fix**: bisected to esimd MoE kernel called with M>1 chunked-prefill chunks → fix by `x.size(0) != 1` gate (decode-only).
- **Header-mismatch silent rebuild trap**: documented the `rm build/temp.../esimd_kernel.o` step ninja header-only edits otherwise miss.
- **MiniCPM-V 4.6 (single-card fp16, multi-image) — online unitrace + the windowing trap**: the deployed bottleneck for multi-slice TTFT was **CPU-side image preprocessing pinned to 1 thread** (vLLM's `set_default_torch_num_threads(OMP_NUM_THREADS=1)`); setting `OMP_NUM_THREADS=8` cut multi-slice TTFT **19–26%** end-to-end (zero code, zero accuracy risk). Profiling lesson now baked into Lens C: an **offline** serial trace read **GPU busy ~13% → "launch-bound, skip kernels"**, but that was inter-request idle; the **online server windowed to one in-flight request read 88.5% → genuinely compute-bound**. Same trace, opposite verdict — window to a single request and prefer the online server. Reversed the old blanket "never use unitrace."
- **MiniCPM-V 4.6 — two textbook "size-the-prize" negatives (now anti-patterns)**: (1) a `SKIP_GELU=1` **no-op** showed 65 ms, but that's the gelu's *total* cost (mostly unavoidable 11 MB×2 activation traffic); a 22%-faster ESIMD gelu bought ~2% end-to-end. No-op measures op-vs-absent, not the replaceable cost — use current-vs-roofline instead. (2) tried fp8 to break the compute bound on the ViT encoder GEMM; `torch._scaled_mm` fp8 measured **74 TFLOPS vs fp16's 131** on this XPU stack (no XMX fp8 path) — fp8 is *slower* for large-M prefill. Both caught by a 3-line micro-bench before any kernel was written.
