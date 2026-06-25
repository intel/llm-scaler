---
name: vllm-xpu-esimd-optimize
description: End-to-end loop for optimizing a vLLM model on Intel XPU (BMG / Arc B-series) by writing/接入 ESIMD kernels in llm-scaler, while keeping accuracy verifiable. Use whenever the user asks to "optimize / speed up / fuse / write esimd kernel for <some vllm model> on XPU" and supplies (1) a container name with editable vllm-xpu installed and (2) a path to a llm-scaler custom-esimd-kernels checkout that can be rebuilt in-place. Covers offline reproducer setup, accuracy gating, decode-ITL profiling, kernel-level profiling, and the kernel-write → bind → wire-up → verify cycle.
---

# vLLM XPU ESIMD optimization loop

This skill is a recipe — **always invoke it at the start of any session whose goal is to make a specific vLLM model faster on XPU by writing or接入 ESIMD kernels.** It encodes hard-won lessons from the gemma-4-26B / 31B work; do not reinvent these steps.

## Required inputs (ask if missing)

Before doing anything, confirm with the user:

1. **`CONTAINER`** — docker container name (host network), e.g. `wj-test-new-021`. vllm-xpu must already be installed editable inside this container. Verify with `docker exec $CONTAINER bash -c 'cd /workspace/llm-scaler-vllm-xpu && git log --oneline -1'` (path may differ — ask).
2. **`VLLM_PATH`** — editable vllm-xpu path inside the container, e.g. `/workspace/llm-scaler-vllm-xpu`. Changes to `vllm/...` files are picked up immediately.
3. **`LLMSCALER_PATH`** — llm-scaler custom-esimd-kernels checkout, e.g. `/llm/models/test/llm-scaler/vllm/custom-esimd-kernels-vllm`. New kernels must be built here and the resulting `.so` files copied into the python package directory.
4. **`MODEL_PATH`** — weights, e.g. `/llm/models/weights/gemma-4-26B-A4B-it`.
5. **`MODEL_FAMILY`** — model class name in vllm (so we know which `vllm/model_executor/models/<name>.py` to wire kernels into), e.g. `gemma4`, `qwen3_next`.
6. **`TP`** — tensor_parallel_size (typically 2). And `ZE_AFFINITY_MASK` for which two devices, e.g. `6,7`.
7. **`BASELINE_CONTAINER`** *(optional but strongly recommended)* — a sibling container with stock upstream vllm-xpu (no optimizations) on the same model. Used as ground truth for accuracy regression. Same model weights, same TP, same `MODEL_PATH`.

If any of these are not supplied, ask 2–3 short clarifying questions before proceeding.

## Standing assumptions

- `host` network for both containers ⇒ they cannot run a server on the same port simultaneously. Always use **offline** flows; only spin up an HTTP server when the user explicitly asks.
- `enforce_eager=True` everywhere (XPU graph compilation is incomplete; this also matches all numbers we ever quote).
- `dtype="float16"` + `quantization="fp8"` is the default benchmarking config.
- `kv_cache_dtype="fp8"` for ITL benchmarking (matches release config); leave default for accuracy gating.
- All env vars listed below are gates the existing optimization stack already understands. Use them rather than ripping code out:
  - `DISABLE_ESIMD_GEMV`, `DISABLE_ESIMD_NORM`, `DISABLE_ESIMD_QKV_FUSED`, `DISABLE_ESIMD_PAGE_ATTN`, `DISABLE_ESIMD_ROUTER_GEMV`, `DISABLE_ESIMD_MOE_GELU`
  - `DISABLE_GEMMA4_FUSED_ROUTER`, `DISABLE_GEMMA4_FUSED_H2`, `DISABLE_GEMMA4_FUSED_ATTN_OUT`, `DISABLE_GEMMA4_XFUSE`, `DISABLE_GEMMA4_GELU_ESIMD`, `DISABLE_BMG_GEMV`
  - `DISABLE_ESIMD_MOE_PREFILL`, `MAX_ESIMD_MOE_TOKENS`
- Standard launch envs: `ZE_AFFINITY_MASK=$ZE_DEVS TORCH_LLM_ALLREDUCE=1 CCL_ZE_IPC_EXCHANGE=pidfd VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_MLA_DISABLE=1 VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=0`.

## The loop

The phases below are mostly sequential, but you can re-enter Profile → Write Kernel → Verify many times. Always close every kernel change with the **Verify** phase before claiming it works.

### Phase 1 — Smoke test: does the model generate sensible output offline?

Goal: produce a 16-token prompt → 64 token decode reproducer that prints a token id list. Two-purpose: (a) lets you eyeball that the model isn't NaN-ing, (b) the printed `Token ids:` is a stable A/B beacon for later kernel changes (token-by-token equality with baseline ⇒ optimization is functionally equivalent).

Skeleton script (write to `/tmp/run_${MODEL_FAMILY}_chat.py` inside `$CONTAINER`):

```python
import torch
from vllm import LLM, SamplingParams

# 16-token prompt that exercises the chat template / instruction path.
# For gemma4: PROMPT_IDS = [2, 105, 2364, 107, 12553, 54847, 236881, 106, 107, 105, 4368, 107, 100, 45518, 107, 101]
# For other models, get this once via apply_chat_template on a real chat message.
PROMPT_IDS = [...]

def main():
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=TP,
        max_model_len=8192,
        enforce_eager=True,
        gpu_memory_utilization=0.90,
        dtype='float16',
        quantization='fp8',
        trust_remote_code=True,
        max_num_seqs=8,
    )
    out = llm.generate(prompts=[{'prompt_token_ids': PROMPT_IDS}],
                       sampling_params=SamplingParams(max_tokens=64, temperature=0))
    print('=' * 60)
    print(f'Output: {repr(out[0].outputs[0].text)}')
    print(f'Token ids: {list(out[0].outputs[0].token_ids)}')
    print('=' * 60)

if __name__ == '__main__':
    main()
```

Run with the standard env block. **Save the token-id list** as the per-model functional fingerprint — every later kernel change must reproduce it (or document the deviation).

If output text is empty / `Token ids: [0, 0, 0, ...]` / `finish_reason=length` with `chars=0`: this is the **NaN-logits-→-pad-token failure mode**. Bisect (Phase 5) to find the offending kernel before doing any new perf work.

### Phase 2 — Accuracy gating with offline gsm8k chat replay

Single 16-token reproducer is not enough — long prompts, batched prefill, and chat-template-applied paths exercise different kernel shapes. Use this for actual accuracy regression checking.

1. **Extract the requests once** (do this in `$BASELINE_CONTAINER` or any container that has the gsm8k jsonl already cached at `/tmp/{train,test}.jsonl`):

```python
# /tmp/extract_chat_reqs.py
import json
test = [json.loads(l) for l in open("/tmp/test.jsonl")]
train = [json.loads(l) for l in open("/tmp/train.jsonl")]
NUM_SHOTS, NUM_Q = 5, 5
few = []
for i in range(NUM_SHOTS):
    few += [{"role": "user", "content": train[i]["question"]},
            {"role": "assistant", "content": train[i]["answer"]}]
reqs = []
for i in range(NUM_Q):
    msgs = [{"role": "system", "content":
             "Solve the math problem step by step. End your answer with a line in the form: #### <number>."},
            *few, {"role": "user", "content": test[i]["question"]}]
    reqs.append({"messages": msgs, "model": SERVED_MODEL_NAME,
                 "temperature": 0, "max_tokens": 512, "seed": 42,
                 "_label": test[i]["answer"].split("####")[-1].strip().replace(",", "")})
json.dump(reqs, open("/tmp/chat_reqs.json", "w"), indent=2)
```

`docker cp` the result into `$CONTAINER:/tmp/chat_reqs.json`.

2. **Offline replay**: apply chat template locally, hand prompt token ids to `llm.generate`. Critical — `apply_chat_template` returns a `BatchEncoding` when `tokenize=True`; use the two-step form to get a list[int]:

```python
prompt_str = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
ids = tok(prompt_str, add_special_tokens=False).input_ids
```

The full template for `/tmp/offline_chat_reqs.py` is in `assets/offline_chat_reqs.py` of this skill — adapt `MODEL_PATH` and TP only.

Expected baseline: **5/5 accuracy** on the 5 GSM8K questions for gemma-4-26B/31B-fp8/TP=2. **A failed acc on this script is a hard stop**.

A `pred=-9999999, finish=length, chars=0` triple means NaN logits sampled token id 0 every step — model is broken, not a sampling fluke.

### Phase 3 — Performance baseline: ITL bench

`/tmp/bench_itl.py` template:

```python
import os, time
def main():
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=MODEL_PATH, tensor_parallel_size=TP, max_model_len=8192,
        enforce_eager=True, gpu_memory_utilization=0.90,
        dtype='float16', quantization='fp8', kv_cache_dtype='fp8',
        trust_remote_code=True, max_num_seqs=8,
    )
    PROMPT_IDS = [...]   # same 16-token prompt as Phase 1
    sp = SamplingParams(max_tokens=128, temperature=0)
    llm.generate(prompts=[{'prompt_token_ids': PROMPT_IDS}], sampling_params=sp)  # warmup
    tpots = []
    for i in range(5):
        t0 = time.perf_counter()
        out = llm.generate(prompts=[{'prompt_token_ids': PROMPT_IDS}], sampling_params=sp)
        dt = time.perf_counter() - t0
        n = len(out[0].outputs[0].token_ids)
        itl = dt * 1000 / n
        print(f"  run {i+1}: {n} tokens in {dt*1000:.1f} ms  → {n/dt:.2f} tok/s  avg_itl={itl:.2f} ms", flush=True)
        tpots.append(itl)
    median = sorted(tpots)[len(tpots)//2]
    print(f"median ITL = {median:.2f} ms ({1000/median:.2f} tok/s)")

if __name__ == "__main__":
    main()
```

Always quote **median of 5** (not best, not mean). Note 5-run spread — if it's > 0.3 ms the result is noisy and you need to re-run. The most stable comparisons are A/B with an env-gate flip (e.g. `DISABLE_FOO=1 vs unset`) on the same binary, in the same minute, on the same EUs.

### Phase 4 — Profile to find the bottleneck

Two complementary lenses:

**Lens A — wall-time of model layer subcomponents** (decide where to spend kernel time). Lightweight: monkeypatch the layer's `forward` with `torch.xpu.synchronize() ; perf_counter` brackets. Stats accumulate in module attrs and dump every N calls. Template lives in `assets/profile_attn.py` (gated by `PROFILE_ATTN=1`, zero overhead at 0).

For attention specifically, also bucket by `(gqa_ratio, is_sliding)` so you can see whether sliding vs full layers behave the same — that's how we discovered gemma4 sliding-window attn is already at the kernel limit.

**Lens B — kernel-level micro-bench** (decide how fast a *replacement* kernel needs to be). Standalone `/tmp/bench_gemv.py` skeleton:

```python
import torch, time
from custom_esimd_kernels_vllm import esimd_<your_op>
device = "xpu:0"; torch.xpu.set_device(0)

def bench(op, *args, n_iter=200):
    for _ in range(20): op(*args)
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter): op(*args)
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) * 1e6 / n_iter

# loop over (K, N) tuples for every shape the model exercises
```

Always report the **HBM bandwidth utilization** alongside time: `bw_GBps = (bytes_in + bytes_w + bytes_out) / dt_us / 1e3`. BMG ~456 GB/s peak; if a kernel is already > 700 GB/s it's L3-cache-fed and you're at a hard bound.

**Lens C — unitrace whole-run kernel timeline (the bottleneck killer; START HERE when you don't yet know where time goes).** Lens A/B require you to already guess *what* to bracket; unitrace needs no guess — one trace gives you, for the whole forward, (1) **device-busy%** = the launch-bound vs compute-bound verdict, (2) **kernel family self-time breakdown** = which op to even consider writing, and (3) the **gap profile** = whether the wall time is host-side dead air the GPU never sees. **Use it to size the prize before writing any kernel** — §4.2 of the unitrace doc killed a multi-day attention-kernel plan by showing attention was 0.2% of device time.

⚠️ **Profile the ONLINE server, and window to a single in-flight request — not an offline serial loop.** This is the single biggest correctness trap and it bit MiniCPM-V 4.6 hard: the *same* model traced two ways gave opposite verdicts. An **offline** reproducer that fires requests serially, aggregated over a wide tail window, showed **GPU busy ~13%** → "launch-bound, no kernel worth writing." But that 87% idle was just the dead time *between* my hand-fired requests, not the model waiting. Re-running on the **online server** (real overlap, real scheduler) and phase-slicing to the tail **4%** — one request's actual execution — showed **GPU busy 88.5% → genuinely compute-bound**. Rule: offline + wide window *under*-counts busy% and fabricates a launch-bound story; only the online server with the window tightened to a single request's execution reflects the deployed bottleneck. Sweep `tail_frac` (e.g. 0.5 → 0.1 → 0.04) and watch busy% climb as the inter-request idle is squeezed out — the tightest stable window is your answer.

Install once per container (it's prebuilt in the llm-scaler tools checkout — do NOT rebuild):
```bash
B=/llm/models/test/tools/pti-gpu/tools/unitrace/build   # path may differ — `find / -name libunitrace_tool.so`
docker exec $CONTAINER bash -c "cp $B/unitrace /usr/local/bin/ && cp $B/libunitrace_tool.so /usr/local/lib/ && ldconfig"
# the ldconfig "not a symbolic link" spam about oneAPI .so files is harmless
docker exec $CONTAINER unitrace --version    # expect 2.x.x
```
The full field guide (install, every flush gotcha, ready-to-copy aggregator scripts) lives at `/llm/models/test/unitrace.md` inside the container — read it before driving a server-mode trace.

**Which mode:** for a **TTFT/serving** question (multimodal, prefill, batching) trace the **online server** — it's the only thing that shows real overlap/scheduling and the deployed busy%. For a pure **decode-kernel** question the offline reproducer is simpler and fine. Both below.

**(A) Online server (preferred for serving/TTFT).** Foreground unitrace + `vllm serve`; flush by SIGINT to the EngineCore. `docker exec -d` detaches on the *host* side only, so the container's bash stays foreground and the spawned EngineCore inherits the default SIGINT handler (the §2.4 SIG_IGN trap is avoided):
```bash
# launch (foreground inside container; NEVER exec/&/nohup reaching `vllm serve`)
docker exec -d $CONTAINER bash -c '
  cd /tmp/utr_online && \
  NEOReadDebugKeys=1 EnableImplicitConvertionToCounterBasedEvents=0 \
  ZE_AFFINITY_MASK=$ZE_DEVS VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_V1=1 OMP_NUM_THREADS=8 \
  unitrace --chrome-device-logging vllm serve $MODEL_PATH ... --enforce-eager 2>&1 | tee serve.log'
# wait for "Application startup complete"; head -1 serve.log must show NO "ld.so ... libunitrace_tool.so" error
# drive the REAL workload (e.g. the multi-image request you care about) ×N to build a steady window
# flush — SIGINT the EngineCore that the live api-server actually forked, NOT a stale one:
EP=$(docker exec $CONTAINER bash -c 'tr "\0" "\n" </proc/$(pgrep -f "vllm serve"|head -1)/task/*/children 2>/dev/null')   # or: pick the EngineCore child of the running api-server
docker exec $CONTAINER kill -INT $EP
# poll python3.$EP.json until >1MB and stable (≈26MB for a few multimodal reqs)
```
⚠️ **Zombie-EngineCore trap (cost me a 130-byte empty flush):** prior killed runs leave defunct `VLLM::EngineCore` processes around; `pgrep -f VLLM::EngineCore | head -1` happily returns a *dead* one and your SIGINT hits nothing → 130-byte trace. Always resolve the EngineCore as the child of the **currently-listening** api-server pid, and verify the json grows past 1MB after the signal. If 130 bytes: you signaled the wrong/instr-less pid (or §2.4/§2.6) — re-check, don't re-theorize. (`docker exec -d` + `tee` is the host-side `&`; do not add a container-side `&`.)

**(B) Offline reproducer (simpler; fine for decode-kernel work).** Single process, exits cleanly so the destructor flushes with no SIGINT:
```bash
docker exec $CONTAINER bash -c '
  rm -rf /tmp/utr && mkdir -p /tmp/utr && cd /tmp/utr
  NEOReadDebugKeys=1 EnableImplicitConvertionToCounterBasedEvents=0 \
  ZE_AFFINITY_MASK=$ZE_DEVS TORCH_LLM_ALLREDUCE=1 CCL_ZE_IPC_EXCHANGE=pidfd \
  VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_MLA_DISABLE=1 VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=0 \
  timeout 600 unitrace --chrome-device-logging python3 /tmp/run_${MODEL_FAMILY}.py > serve.log 2>&1'
# output: python3.<pid>.json per process (TP=2 → two ~10-15MB worker traces)
```
⚠️ Offline fires requests serially, so the wide-window busy% is dominated by *inter-request* idle and reads falsely launch-bound (see the boxed warning above). Run warmup + a burst of N identical steps and **window tight to one step's execution** (sweep `tail_frac` down) before trusting busy%.

Analyze (aggregator skeleton — auto-detects device pid as the pid with the largest total `ph=="X"` dur; full versions in unitrace doc §5.5/§5.6):
```python
import sys, ijson, collections          # /tmp/utr_agg.py <trace.json> [tail_frac]
pd = collections.defaultdict(float)
for ev in ijson.items(open(sys.argv[1],'rb'),'traceEvents.item'):
    if ev.get('ph')=='X': pd[ev['pid']] += float(ev.get('dur',0))
DP = max(pd, key=pd.get)               # device pid (host-API events live on a different pid)
evs = sorted((float(e['ts']),float(e.get('dur',0)),e['name'])
             for e in ijson.items(open(sys.argv[1],'rb'),'traceEvents.item')
             if e.get('ph')=='X' and e['pid']==DP)
t0,t1 = evs[0][0], evs[-1][0]+evs[-1][1]
cut = t1-(t1-t0)*float(sys.argv[2] if len(sys.argv)>2 else 0.06)   # tail window, skips load/warmup
win = [e for e in evs if e[0]>=cut]; wall=(t1-cut)/1e3
busy = sum(d for _,d,_ in win); fam=collections.defaultdict(float)
for _,d,n in win: fam[n.split('<')[0].split('[')[0][:50]] += d
print(f"busy%={100*busy/1e3/wall:.1f}  (low=launch-bound, high=compute-bound)")
for k in sorted(fam,key=fam.get,reverse=True)[:20]: print(f"{fam[k]/1e3:8.2f}ms  {k}")
# gap profile: sort the >1ms inter-op gaps with their before/after kernel — a recurring
# multi-ms gap between a D2H (read token) and the next M2D (next request) == host dead air,
# not a kernel you can fix. (see /tmp/utr_biggap.py pattern in the doc)
```

**Reading the verdict (only on a single-request window — see the boxed warning):** `busy% low (≈15%)` → launch-bound; the lever is **fewer launches** (graph/fusion) or it's host-bound (preprocess/IPC) — a faster single kernel buys ~nothing. `busy% high (≈80%)` → compute/BW-bound; now Lens B roofline tells you if a kernel rewrite is worth it. Before trusting a *low* busy%, confirm the window isn't padded with inter-request idle (tighten `tail_frac`; if busy% climbs toward 80% it was a windowing artifact, not launch-bound). Two anti-artifact rules carry over: unitrace inflates absolute device-time (trust **relative** %, not the seconds), and `--chrome-call-logging` floods tiny-op counts (cross-check a suspicious "94% elementwise" with `torch.profiler(record_shapes=True)`).

**When unitrace genuinely hangs / won't flush** (it can, see below): fall back to Lens A monkeypatch/source-instrument timing.

### Phase 5 — Write a new ESIMD kernel

Workflow:

1. **Author kernel header** in `${LLMSCALER_PATH}/csrc/xpu/esimd_kernels/<name>.h`. Copy a similar-shaped existing kernel as the template (`fp8_GEMV_v2.h`, `fused_add_rms_norm.h`, `norm_gemv_norm_fp16.h`, `norm_add_norm.h`). Use `simd<float, VL>` accumulators, `block_load<fp16, VL>` / `block_load<uint8_t, VL>` for fp16 / fp8 weight reads; never recreate the fp8 dequant logic — copy `fp8_dequant_*` from a sibling kernel.

2. **Resource budget — register cache trap.** Avoid `simd<float, VL>[MAX_CHUNKS]` register caches once `VL * MAX_CHUNKS * 4 bytes > 12 KB`. BMG single-thread GRF is ~12 KB; spilling silently runs but accumulates Level Zero state and **eventually triggers `UR_RESULT_ERROR_OUT_OF_RESOURCES`** under server load. Stream loads twice instead — L3 caches the second pass at ~zero cost. (See `norm_add_norm.h` final form.)

3. **K-divisibility trap.** If `K % 64 != 0` or `K` not a clean power-of-2 multiple, the existing `select_vl_ks` in `fp8_GEMV_v2.h` falls back to `vl=32 ks=1` and runs at ~300 GB/s. Use the `fp8_GEMV_bmg.h` pattern: pick `(VL_BIG, VL_TAIL)` where `VL_BIG` chunks fit cleanly and one final `VL_TAIL` chunk handles the remainder.

4. **K_SPLIT for large-N matmuls — but watch redundant compute.** When fusing norm + GEMV, every WG re-computes `sum_sq` over the same K elements. With N WGs that's N× redundant memory traffic. The fuse is a **net win only if N is small** (~hundreds). For N=4096 (qkv_proj), shipping the kernel but **not wiring it up** is the right call (this is what `scaled_resadd_norm_gemv_fp8.h` is — kept as future material for a persistent-kernel rewrite).

5. **Bind the op:**
   - `csrc/xpu/esimd_kernel.sycl`: `#include` the header, write a thin `at::Tensor esimd_<name>(...)` wrapper.
   - `csrc/xpu/torch_extension.cc`: `m.def(...)` schema + `m.impl("...", torch::kXPU, &...)`.
   - `include/kernel_ops.h`: forward declaration.
   - `python/custom_esimd_kernels_vllm/ops.py`: thin python wrapper that calls `_ops.esimd_<name>`.
   - `python/custom_esimd_kernels_vllm/__init__.py`: re-export.

6. **Build:**
   ```bash
   docker exec $CONTAINER bash -c 'cd $LLMSCALER_PATH && \
     rm -f build/temp.linux-x86_64-cpython-312/csrc/xpu/esimd_kernel.o && \
     TORCH_XPU_ARCH_LIST=bmg python3 setup_gemv_only.py build_ext --inplace 2>&1 | tail -8'
   ```
   The explicit `rm` of the `.o` is required when a header changed but the `.sycl` file did not — ninja's dep tracking misses header-only edits. Look for `Build succeeded.` four times and `copying ... .so → python/custom_esimd_kernels_vllm`. If you only edit kernels in `csrc/moe_batch/`, also run `setup.py build_ext --inplace` (the moe ops are a separate extension).

7. **Unit-test the kernel against a reference:**
   ```python
   # /tmp/test_<name>.py
   ... call esimd op + compute torch reference + assert max abs diff < threshold
   ```
   Always check both numerical (max abs diff < ~0.01 fp16, < 0.5 for cumulative GEMV) and shape-edge cases (K not divisible by VL, K_SPLIT > 1, etc).

### Phase 5b — MoE decode: the fused-full-op pattern (preferred end state)

For **MoE models**, the decode bottleneck is usually NOT the expert GEMM — it's the **routing segment**: the topk/softmax kernel launch plus the Python-side glue (`.to(fp16)`, `.contiguous()`, scale-fold gather-mul, separate expert dispatch). On gemma-4-26B this segment was 0.133 ms/layer (×30 = 4 ms) — *larger* than the expert GEMM (0.087 ms/layer). **Always seg-profile MoE forward into `routing | expert-kernel | all_reduce` before optimizing — don't assume the GEMM dominates.** (Patch the model's MoE `forward` with `torch.xpu.synchronize()+perf_counter` brackets around each segment, env-gated.)

The proven end state mirrors `qwen3_next.py`'s decode path: **`router_op(x) → one fused full op(x, logits, ...) → all_reduce`** — a single op that does topk + (any scale fold) + up + activation + down + accumulate internally, so the only Python-visible launches are the router and the fused op.

How to get there (gemma-4 worked example, commits in `optimize_gemma4_continue`):

1. **First fix the expert GEMV load width (decode M==1).** The shared MoE kernel uses `lsc_load_2d<uint8_t,16,16,1>` (DPAS GEMM, tuned for prefill M>1). On decode that 16-byte-wide 2D load fills only 1/4 of BMG's 64B cacheline → ~315 GB/s. Expert weights are plain row-major inside each expert (`gate_up [E,2*inter,hidden]`, `down [E,hidden,inter]`, K contiguous), so a **1D `block_load<uint8_t,256>` along K** (the `fp8_GEMV_bmg` pattern) restores ~528 GB/s. Write decode-only `MoeUpDecode*`/`MoeDownDecode*` kernel structs in `csrc/moe_batch/<name>.h`. Gate `x.size(0)==1`; prefill keeps DPAS.

2. **Fuse routing INTO the op — but topk MUST be fp32-internal.** This is the single most important correctness rule. A fp16-internal softmax+topk (e.g. `esimd_moe_topk`) **diverges from the triton routing on near-tie top-k boundaries and silently drops gsm8k 5/5 → 4/5**, even when a single-shot unit test matched bit-for-bit. The production `dispatch_moe_topk_forward`/`moe_topk` op IS fp32-internal — verify it matches the reference routing (200-trial id-set match, weight diff < 2e-4) AND holds 5/5, then reuse it inside your fused op. Never reverse-engineer a new topk.

3. **Fold any model-specific routing scale on-device.** gemma's `per_expert_scale` (learnable, per-expert) can't be expressed by the generic topk's `norm` flag, so a tiny `MoeFoldExpertScale` kernel multiplies it into the topk weights between topk and the down kernel — no round-trip to Python. This per-model scale is exactly why you **cannot** reuse another model's `moe_forward_full` verbatim (qwen's has no scale fold, uses silu not gelu_tanh, and assumes a shared expert).

4. **Assemble the fused op** `moe_forward_full_<act>_decode(x, logits, w13, s13, w2, s2, <scale...>, top_k, n_experts)`: internal `dispatch_moe_topk_forward(norm=true)` → scale-fold kernel → `MoeUpDecode` → `MoeDownDecode` → `moe_accumulate_kernel`. Register + python-wrap as in Phase 5.5; wire into the model's MoE `forward` for decode, return early before the old routing/topk/expert sequence; prefill path unchanged.

5. **Prefill stays fp32-routing + DPAS.** Prefill (M>1) is precision-sensitive to routing-weight dtype (fp16 routing weights there also drop 5/5) and DPAS is the right kernel for M>1. Keep the two paths split by `x.size(0)==1`.

gemma-4-26B result: decode ITL 21.86 → 18.72 ms (−14%), gsm8k 5/5, token fingerprint identical. Per-op env gates (`DISABLE_MOE_DECODE_GEMV`, `DISABLE_MOE_PROD_TOPK`, `DISABLE_MOE_FULL_FUSED`) layer the steps so each is independently A/B-able and revertible.

### Phase 6 — Wire into the model

**Always add an env gate that disables the new path 1:1**, e.g. `DISABLE_<MODEL>_FUSED_<NAME>=1` defaulting to OFF. This is required to make subsequent A/B comparisons trivial and to give the user an emergency disable when a corner case breaks accuracy in production. Pattern:

```python
_fused_path = (
    ESIMD_AVAILABLE
    and not disable_esimd_norm()           # global gate
    and tensor.shape[0] == 1                # decode-only
    and tensor.is_contiguous()
    and tensor.dtype == torch.float16
    and weight.dtype == torch.float16
    and os.environ.get("DISABLE_<NAME>", "0") != "1"
)
if _fused_path:
    if not hasattr(self, "_buf"):
        self._buf = torch.empty_like(tensor)   # cache on the module
    from custom_esimd_kernels_vllm import esimd_<name>
    esimd_<name>(...)
    return self._buf
else:
    # original path, unchanged
    ...
```

### Phase 7 — Verify (mandatory after every kernel change)

Always run **both** before claiming any kernel change works:

1. **Phase 1 reproducer**: token-id list must match the recorded baseline. If it deviates, the optimization is **not** functionally equivalent and you must explain why before continuing.
2. **Phase 2 chat reqs replay**: `Accuracy: 5/5 = 1.000`. Less than that means a different shape (long prompts, prefill chunk M>1) is broken — common failure mode for kernels that tested fine in isolation.
3. **Phase 3 ITL bench**: median of 5 runs improved (or at least did not regress beyond noise). If your change doesn't improve ITL, either disable the wire-up by default and ship the kernel as future material (like `scaled_resadd_norm_gemv_fp8`), or revert.

### Phase 8 — Bisect when something breaks

If accuracy regresses anywhere (Phase 1 token mismatch / Phase 2 acc < 5/5), **stop adding things and bisect**.

The bisection priority list, in order:

1. **First, env-gate disable everything** — set every `DISABLE_*` env var listed in "Standing assumptions". If accuracy is restored, the bug is in our optimization stack; otherwise it's elsewhere (config, attention backend, vllm-xpu base).
2. **Compare against `$BASELINE_CONTAINER`** running the exact same offline script on the same prompt. If baseline is also broken, this isn't your bug.
3. **Diff `vllm/...` files between the two containers.** Frequent suspects: `vllm/v1/attention/backends/flash_attn.py` (`supports_head_size`), `vllm/model_executor/models/config.py` (forced backend selection), `vllm/model_executor/models/<family>.py` itself.
4. **Git bisect within `$VLLM_PATH`** — checkout an older commit, rerun Phase 2. Bisect interval halves each step. We have caught bugs this way at three distinct levels: (a) MoE kernel called with `M>1` prefill chunk produces NaN (`f7c0693e9`), (b) `GeluAndMul` esimd path NaN at large `d` (`28d462ed5`), (c) attention backend selector chose FLASH_ATTN where head_size=512 silently NaN'd (`config.py` XPU exception).
5. When you do find the offending commit, **fix by gating, not reverting** — narrow the activation condition (`d <= 4096`, `M == 1`, etc.) so the optimization keeps working in its safe regime. Always add the new gate as both a hard condition and an env override.

## Anti-patterns (don't waste time on these)

- **Don't dismiss unitrace — but scope it right (this reverses the old blanket "never use unitrace").** It is the *fastest* way to get the launch-bound-vs-compute-bound verdict and the kernel-family breakdown (Phase 4 Lens C) — reach for it FIRST when you don't yet know where the time goes. The real failure mode is narrow: a **server-mode** trace with `--chrome-call-logging` on a heavy **ESIMD decode** loop can hang the worker or refuse to flush (it needs SIGINT to the EngineCore, not the api-server — full gotcha list in `/llm/models/test/unitrace.md`). Mitigations that make it reliable: run the **offline** reproducer (exits clean → auto-flush, no SIGINT), start with `--chrome-device-logging` only (add `--chrome-call-logging` only when you specifically need host submit-gaps), and wrap in `timeout`. If it still won't flush after a couple tries, **don't chase it** — fall back to Lens A manual `torch.xpu.synchronize() + perf_counter` instrumentation.
- **Don't bypass `vllm` linear with hand-rolled `esimd_gemv` + manual `tensor_model_parallel_all_reduce`.** It's what we tried first; it's measurably slower because vllm linear's internal allocator and dispatcher are already cheap. Only intercept inside the linear method (e.g. `XPUFP8ScaledMMLinearKernel.apply_weights`).
- **Don't write fuses where every WG redundantly recomputes a global reduction.** Counter-example: gating `(h+r) → norm → fp8 GEMV` with 4096 work-groups each redoing K=2816 sum_sq is a net loss. If you must fuse, write a persistent kernel with a global atomic counter. Most of the time a simple two-launch sequence with cached buffers is faster.
- **Don't enable `enforce_eager=False`.** The model definition is not torch.compile-clean; you'll spend hours chasing meta-tensor errors. The whole optimization narrative assumes eager-mode dispatch.
- **Don't try to make `M > 1` prefill chunks share the decode kernel.** All MoE-style ESIMD kernels are tuned for `M == 1`. The prefill-chunk fallback to Triton is the right behavior. Set the gate as `x.size(0) != 1` (decode only) rather than `x.size(0) <= 8`.
- **Don't assume `apply_chat_template(..., tokenize=True)` returns a list.** It returns a `BatchEncoding`. Use `tokenize=False` then run the tokenizer directly to get a clean `list[int]`.
- **Don't optimize the MoE expert GEMM before seg-profiling the routing.** The expert kernel is the *obvious* target but usually not the bottleneck — routing + Python glue dominated on gemma-4 (4 ms vs 2.6 ms). Measure `routing | expert | all_reduce` first. (And: a 1D-load expert GEMV that micro-benches 1.68× faster may only buy ~0.15 ms end-to-end — micro-bench overstates kernel weight; always confirm in-model.)
- **Don't size the prize with a no-op (`SKIP_X=1`) substitution — it measures the op's *total* cost, not the *replaceable* cost.** On MiniCPM-V 4.6 an env-gated `SKIP_GELU=1` (gelu → identity) cut TTFT 65 ms, so the gelu looked worth a kernel. But that 65 ms is overwhelmingly the **unavoidable memory traffic** of reading+writing the (M, 4304) ≈ 11 MB×2 activation — *any* gelu kernel pays it. A hand-written ESIMD gelu that micro-benched 22% faster on the **compute** part bought ~2% end-to-end (lost in noise). A no-op deletes the memory traffic too, so it flatters every memory-bound elementwise op. The honest prize for *replacing* a kernel is **current-kernel time vs its roofline floor** (`max(bytes/BW, FLOPs/peak)`), not op-vs-absent. A no-op only reflects real prize when you can genuinely *delete* the op (fuse it into a neighbor's epilogue so the round-trip disappears, or remove it algorithmically) — and even then the win is the saved round-trip, which on a compute-bound layer largely overlaps the GEMM and shrinks again (resadd+LayerNorm fuse: ~0.4 ms/call isolated → ~3% end-to-end, not worth it).
- **Don't assume fp8 beats fp16 for large-M GEMM on XPU — measure `_scaled_mm` first; it can be 2× *slower*.** Expectation: fp8 doubles XMX throughput. Reality on BMG / this torch-xpu stack for a vision-tower-sized GEMM (M=30240, K=1152, N=4304): `torch._scaled_mm` fp8 = **74 TFLOPS vs fp16 `F.linear` = 131 TFLOPS** — fp8 never reaches the XMX fp8 path and runs a software/dequant route. The existing `esimd_gemm_fp8_pert` only covers M≤64, so it doesn't help large-M prefill either. Upshot: for a **compute-bound** large-M layer (e.g. a ViT encoder), fp16 oneDNN GEMM at ~55% of peak is already the best available path; you cannot break the compute bound by dropping to fp8 here. (fp8 *is* the win for **decode** M=1 GEMV, which is BW-bound — different regime.) Always run the 3-line `_scaled_mm`-vs-`linear` TFLOPS micro-bench before committing to an fp8 rewrite.
- **Don't replace a routing/topk with a fp16-internal kernel.** Topk is tie-break-sensitive: fp16-internal softmax+topk diverges on near-ties and drops accuracy (gsm8k 4/5) while passing single-shot unit tests. Only swap in an fp32-internal topk (the production `moe_topk`), and gate it behind the full Phase-2 gsm8k 5/5, never a unit test alone.
- **Don't reuse another model's `moe_forward_full` verbatim.** Per-model routing semantics differ — gemma folds a learnable `per_expert_scale` (qwen doesn't), uses gelu_tanh not silu, and has no shared expert. Write a model-specific fused op; share only the sub-kernels (topk, GEMV, accumulate).

## Final commit & rebase guidance

When the user is happy with a series of optimization commits, **expect to rebase onto the upstream release branch** (`origin/downstream/release/v0.21.0` typically). Two commits in our history overlap with later upstream cherry-picks of the same PR; `git rebase` will:
- Auto-`drop` commits whose patch is already upstream.
- Throw a small conflict on `vllm/model_executor/layers/esimd_utils.py` for the very first ESIMD infrastructure commit; resolve by `git rebase --skip` (upstream's version is functionally equivalent — your later commits will re-add the additional re-exports they need).

After rebase, recheck Phase 1 and Phase 2 (a successful textual rebase doesn't guarantee semantic correctness).

## Reference assets

The following helper scripts in `assets/` are templates — each one needs only `MODEL_PATH`, `TP`, and a per-model PROMPT_IDS update:

- `offline_chat_reqs.py` — Phase 2 accuracy replay
- `bench_itl.py` — Phase 3 ITL benchmark
- `profile_attn.py` — Phase 4 Lens A attention timing harness (PROFILE_ATTN=1)
- `unitrace_agg.py` — Phase 4 Lens C: aggregate a unitrace chrome trace into the device-busy% verdict + kernel-family breakdown + gap profile (auto device-pid, tail phase-slice)

Read them before writing your own. The full unitrace field guide (install, flush gotchas, server-mode driving, more parser variants) is `/llm/models/test/unitrace.md` inside the container.
