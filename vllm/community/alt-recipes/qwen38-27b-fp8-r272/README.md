# Unified-Runtime vLLM 0.27.2rc1 XPU stack for Qwen3.8-27B-FP8 on dual Arc B70 (BMG)

**Status: DRAFT / EXPERIMENTAL — pending a week-long soak test on live stable
traffic. Soak has NOT completed. Do not treat any number below as promotion
evidence until this PR is un-drafted with soak results.**

## What this is

This is a community-reported **alternative runtime recipe** for Qwen3.8-27B
(native block-FP8 checkpoint, Qwen official HF release) on 2x Intel Arc Pro
B70, built from **vLLM 0.27.2rc1.dev77+gac7509e2b.xpu (torch 2.13.0+xpu)** —
a newer fork point than the packaged `intel/llm-scaler-vllm:0.26.0-b2`
image (vLLM `0.26.1.dev0+g568afb3a1.d20260907.xpu`, torch `2.12.0+xpu`).

It is **not** a replacement for the Intel images, and it is **not** upstreaming
the pointer-fix from #683 (that fix is already merged on `main` here). It is a
different assembly that we run in production on this host, submitted so the
maintainers and other BMG users can:

1. see the delta that a newer fork point requires over `vllm/patches/`;
2. reuse our reproducible matched A/A benchmark methodology and results;
3. decide whether anything here should inform the next Intel release.

## Stack delta vs the 0.26.0-b2 image

| Component | This stack | intel/llm-scaler-vllm:0.26.0-b2 |
|---|---|---|
| vLLM | 0.27.2rc1.dev77+gac7509e2b.xpu | 0.26.1.dev0+g568afb3a1.d20260907.xpu |
| PyTorch | 2.13.0+xpu | 2.12.0+xpu |
| vllm-xpu-kernels | 0.1.14.dev544 (upstream #544 sliding-window conv state + #552 mixed spec/non-spec GDN batch, rebuilt `_xpu_C`/`libgdn`/`libmhc`; stock `.so` retained for untouched families) | Intel ESIMD GDN ENSEMBLE (separate integration) |
| Scheduler | `_is_uniform_decode` gating on per-request `num_computed_tokens >= num_prompt_tokens` (upstream vLLM PR #53059 lineage, commit 3462586) — see `vllm/community/alt-recipes/qwen38-27b-fp8-r272/gpu_model_runner_53059.diff` | shape-only `max_num_scheduled_tokens == uniform_decode_query_len` (prefills whose token count equals `1+num_spec_tokens` can alias uniform decode; with GDN state this can corrupt output — vllm-project #53051) |
| Mamba pointer metadata | `_reinterpret_u64_as_i64` at state-base + block-table assignment (same content as #677/#683, regenerated as a standalone test patch against the 0.27.2rc1 tree) | already in this repo's patches |

Why the newer fork: the 0.26.x fork does not contain the #53059-lineage
prefill/decode classification fix, and on our checkpoint the 0.26.0-b2 image
produced deterministic first-token corruption (multi-language degenerate token-1
output) across MTP on/off, mamba f16/bf16, and explicit/auto FP8. We did NOT
re-run that evaluation for this PR; it is recorded from our 2026-09-08 session
and is reproduced by the qualification probes in the checklist below if so
desired.

## Measured results (B70 pairs, TP2, host 10.71.71.112 removed per policy)

Hardware: 2x ASRock Arc B70 (BMG, 32 GiB each), `xe` driver, pytorch 2.13.0+xpu.
Model: Qwen3.8-27B block-FP8 official checkpoint, 66 shards, 1606 tensors, 407
`weight_scale_inv`, 882 excluded-layer tensors, 30.87 GiB on disk.

### Matched A/A direct comparison (pointer patch vs no pointer patch)

Both sides: identical prompt bytes (8,063 prompt tokens → 589 output tokens,
`finish_reason=stop`), temperature 1.0, top_p 0.95, `seed=42`, reasoning
effort low, MTP3 (`qwen3_next_mtp`, `num_speculative_tokens: 3`), TP=2, FP8
E4M3 KV cache, BF16 mamba SSM cache, chunked prefill + prefix caching on,
`max-model-len 200000`, `max-num-seqs 4`, `max-num-batched-tokens 8192`,
block size 64, PI Lizewise python 3.12. Every measured request verified
`cached_tokens=0` (request `cache_salt` randomized per request — salt varies
cache identity only, never model input). Isolation verified per request: the
server's `vllm:request_success_total` counter advanced by exactly one and a
250 ms sampler confirmed `num_requests_running == 1` and
`num_requests_waiting == 0` for the whole request. One identical warm-up
request (fresh server, cache-miss) preceded the three measured runs on each
side. All six measured outputs have the same SHA-256 hash.

| Metric (median of 3) | Without pointer fix | With pointer fix | Delta |
|---|---:|---:|---:|
| Server prefill tok/s | 2419.88 | 2417.88 | -0.08% |
| Streamed decode tok/s (T+TTFT) | 50.02 | 50.08 | +0.11% |
| TTFT | 3.344 s | 3.346 s | +0.05% |

The pointer patch is a correctness/robustness change only regarding GPU
pointer high-bits; measured performance is unchanged within run-to-run noise
(~3% back-to-back control drift on this host, ~10% across a session, same
as our earlier lab measurements).

### MTP acceptance with this stack (NOT the comparison workload)

On the pointer-patched image, same server, separate low-effort smoke (3
requests, prefix cache warm): 198 draft tokens, 153 accepted tokens counted
across speculative positions 0/1/2 = 63/48/42. This is a deployment smoke
check, not a controlled acceptance benchmark; do not compare it to the
counting or capital-bursts numbers in other packets without the matched
methodology.

## Known-good image chain

```
neural-download/vllm-openai-xpu:qwen38-fp8-gdn544         # R50 base + 0.1.14.dev544 kernel package
neural-download/vllm-openai-xpu:qwen38-fp8-gdn544-full    # + mamba_utils pointer patch and regression tests
neural-download/vllm-openai-xpu:qwen38-fp8-gdn544-ptr683  # current (identical to -full plus test-only diffs)
```

Rollback chain preserved one tag per meaningful step.

## Reproduce

```bash
docker run -d --name vllm-xpu-test \
  -v "$(pwd)"/model:/model \
  -v ~/.cache/vllm:/root/.cache/vllm \
  intel-container-args...   # see launcher diff for full arg list
```

(Exact launcher flags are in
`vllm/community/alt-recipes/qwen38-27b-fp8-r272/launcher.sh`; all flags are
captured verbatim from the live production command line via
`docker inspect .Config.Cmd` — audit before you claim an upstream image has
the same flags.)

### Launch flags (production-identical, captured live)

```bash
vllm serve /model \
  --served-model-name Qwen38-27b-0 \
  --host 0.0.0.0 --port 8001 \
  --dtype float16 \
  --tensor-parallel-size 2 \
  --quantization fp8 \
  --kv-cache-dtype fp8_e4m3 \
  --mamba-ssm-cache-dtype bfloat16 \
  --max-model-len 200000 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 8192 \
  --block-size 64 \
  --gpu-memory-utilization 0.95 \
  --cpu-offload-gb 0 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --disable-sliding-window \
  --enable-prompt-tokens-details \
  --language-model-only \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --compilation-config '{"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[1],"max_cudagraph_capture_size":1,"splitting_ops":[],"inductor_compile_config":{"combo_kernels":false,"benchmark_combo_kernel":false,"deterministic":true,"triton.autotune_pointwise":false,"benchmark_epilogue_fusion":false}}' \
  --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":3}' \
  --trust-remote-code
```

## Qualification checklist (re-run any of this to review)

| Gate | Result |
|---|---|
| Platform detect (XPUPlatform) | PASS |
| `_is_uniform_decode` RED/GREEN (14 synthetic cases: aliased prefill shapes rejected, 7 genuine uniform decode preserved) | PASS |
| MTP conv-state regression (stock `47.7%` mismatch recorded upstream; patched build `0/128` mismatch on `T%64==2`, sliding-window conv state) | PASS |
| fp32-reference comparison on speculative attention (`max_abs_diff == 0.0` vs fp32 oracle, same length) | PASS |
| 2-token alias probe (raw count, sensitivity, ladder) | PASS, output exact |
| 43K/93K chunked prefill + T%64∈{1,2,3,4} ladder | PASS, `finish_reason=stop` |
| 6 rounds of concurrent mixed (4-way) probes at descriptions ≤ 198K context | PASS |
| 200K context metric parity | KV pool 766,406 tokens = 3.83x max at 200K, unchanged vs baseline |
| Matched A/A kernelpatch-vs-kernelpatch speed | see table above |

Outstanding (NOT RUN / NOT COLLECTED):

- [ ] **Week-long live-traffic soak with the corruption watcher** — the
      purpose of this PR being kept in draft. No `!!!!!` corruption (or any
      other degenerate output) observed so far, but the soak client was
      intentionally paused to run the isolated A/A comparison and has not yet
      accumulated 168 hours since pointer-patch deployment.
- [ ] Vision (multimodal) qualification on this checkpoint.
- [ ] MTP acceptance-rate sensitivity at matched workloads (the numbers in
      "NOT the comparison workload" above are counts, not a rate study).
- [ ] Cross-checks on non-BMG hardware (B60, Arc Pro B-series).

## Pointers

- Upstream stack lineage and why MTP1 (`uniform_decode_query_len=2`) matters:
  vllm-project/vllm#53051, #53059, vllm-xpu-kernels#544, #548, #552.
- The pointer helper's origin in this repository: #683 (merged). Our
  standalone test patch exists for verification independence, not for novelty.

Signed-off-by: Dominick Pescetto <dominick253@gmail.com>
