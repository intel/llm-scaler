# OMIX 0.4.0 / Ubuntu 26.04 — vLLM v0.26.0 regression snapshot

Updated 2026-10-16 UTC from archived results. This documentation refresh does
not change the shipped runtime or rerun models. PR #764 remains Draft:
**scheduled testing completed, release acceptance FAIL**.

## Identity and artifacts

Build/source commit `b4bd2671048ea412c4837f78a4379fd953c2e888`, based on
main `49c94a32bed69fbabd824debc4b136a6d15cae6d` (remote reverified).
The documentation-only follow-up does not rebuild the image.
vLLM source tag v0.26.0, HEAD `568afb3a13806beb53bb2e6bd518269357b237c0`.
The patched wheel label `0.26.1.dev0+g568afb3a1.d20261015.xpu` is generated
SCM metadata, not a source upgrade to v0.26.1 or v0.31.

- Base: `intel/omix:0.4.0-devel-ubuntu26.04@sha256:59cef13f8e87275846cc08e01d32b49082694a5c01d842d2e845552cbc8c199f`.
- Runtime: `llm-scaler-vllm:omix040-v026-functional-r2`,
  `sha256:1882a2d8911107e282a0ebb392e13e10bae5cca4093ee8064d0d93c591dde468`.
- Saved builder: `llm-scaler-vllm:omix040-v026-builder-functional-r2`,
  `sha256:e70e642f73de071e753d87e4bfe7792b91b7e517c63ce80f77cfcd648eca1d78`.
- Baseline: `llm-scaler-vllm:pr707-review-fix`,
  `sha256:1091b5fc32eb50ec0b58cb00cbde3c1ab5e30deee2a6fc021fd79f13a66f8fc0`,
  **Python-only overlay**, not a clean matched full build.
- Local archive: `/home/intel/guancheng/omix040-v026-20261015`;
  wheels/source/build logs/manifests under `compatibility-attempt03/`.
  Images and complete raw evidence remain local, not uploaded to a registry/PR.
- Python 3.12.14, Torch 2.14.0+xpu, torchvision 0.29.0+xpu, torchaudio
  2.11.0+xpu, Triton 3.8.0+xpu shim with implementation 3.8.0,
  Intel runtimes 2026.1.1, oneCCL/oneCCL-devel 2022.1.2, PTI 1.1.0.
  Dockerfile.dev is unchanged.

All five native wheel SHA256 values:

| Family | SHA256 |
|---|---|
| vLLM | `4a3a6621e9cfe6bd2d0a1ef4553afae8ce70db3c473079a0d4e3f69a14472c8c` |
| custom ESIMD | `1b52ac9dc2133f5f953e805fe98dcb9f34ebd46cd939b716cec826f89463b5e5` |
| vllm-xpu-kernels | `f004038463814ad72b16e700b148e7a5f91282d4940b7aaf19589f36a133b695` |
| KVWeave | `9a829c88d70b92e6f10e73049d7c38834541cdf932993545c1fc7691d91d3785` |
| LMCache | `b4a08145379d1481b2f92f85db60dab30c1d3d03784c666feb2e3c459e7b435e` |

## Completed final-runtime checks

GPU tests use only physical devices 6,7, isolated containers/ports and serial
model loading. Baseline/candidate use the same weights and frozen test assets.
Config/template hashes and 136 file path/size/mtime identities over seven
models were verified; full weight shard contents were not rehashed.

| Check | Result | Evidence relative to local archive |
|---|---|---|
| Rust and five native builds; builder/wheel/source/log archive | PASS | `compatibility-attempt03/` |
| Exhaustive 29-DSO imports or RTLD_NOW, installed hashes/ldd | PASS | `logs/final-r2-native-census-2120.log` and 2124 retry |
| Installed wheel identity, dependencies, connector, observability, runtime ownership, audio/vision, contracts | PASS | `compatibility-attempt03/logs/cpu-strict/` |
| Two-XPU math, XCCL, changed-input native graph | PASS | `logs/candidate-functional-r2-basic-xpu.log` |
| Compiled opaque collective: both ranks, three changed-input replays each | PASS | `logs/functional-r2-compiled-allreduce-xccl.log` |
| Default Inductor + FULL graph, Qwen3.8/Gemma26 × MTP OFF/ON | PASS 4/4 | `regression/candidate-functional-r2-acceptance-graph/` |
| Native mode0 + FULL graph, same four groups | PASS 4/4 | `regression/candidate-functional-r2-native-graph/` |
| Five-model × MTP OFF/ON eager, 200×2 / concurrency32 / 64K | Minimum-threshold PASS 10/10 | `regression/candidate-functional-r2-eager/` |
| LMCache store/worker/Mamba/RAW/rollback | PASS 100 tests | `logs/candidate-functional-r2-store-tests.log` |
| LMCache no-cache/default RAW/explicit RAW | PASS three cases each; actual external hits for RAW | `regression/candidate-functional-r2-lmcache/` |
| Real MQ/SHM abort and RAW retry | PASS | `default/rpc-abort.log` in the LMCache matrix |
| Filesystem L2 eviction/reload | PASS; 180 evictions, eight prefetch/load, 1024 external hits, same output | `l2/l2.json` in the LMCache matrix |
| Attention8 third simple replay | FAIL original and retest | `regression/candidate-functional-r2-lmcache-attention8-retest/` |
| Qwen35 MTP ON matched review | BLOCKED causal review; runs complete | `regression/{baseline-final-retest01,candidate-functional-r2-retest01}-eager/` |

Both graph suites record actual capture/replay on both workers and five stable,
correct repeated responses. Eager results are not counted as graph acceptance.
Raw per-question, worker, device and runtime records are retained.

## Final original eager10 counts

All rows pass the accuracy >= 0.8 floor, health/warmup and 64K recall.
Counts are not a no-regression verdict and are not replaced by retest scores.

| Model | MTP | Thinking correct/200 (invalid) | Non-thinking correct/200 (invalid) | Actual 64K prompt tokens / recall / seconds |
|---|---|---|---|---|
| gemma-4-26B-A4B-it | OFF | 196 (1) | 196 (0) | 65524 / PASS / 11.159 |
| gemma-4-26B-A4B-it | ON | 197 (1) | 196 (0) | 65524 / PASS / 12.737 |
| gemma-4-31B-it | OFF | 198 (0) | 196 (0) | 65524 / PASS / 42.406 |
| gemma-4-31B-it | ON | 198 (0) | 196 (0) | 65524 / PASS / 44.789 |
| Qwen3.6-27B | OFF | 195 (0) | 196 (0) | 65525 / PASS / 25.812 |
| Qwen3.6-27B | ON | 193 (1) | 193 (0) | 65525 / PASS / 26.418 |
| Qwen3.6-35B-A3B | OFF | 195 (0) | 193 (0) | 65525 / PASS / 7.123 |
| Qwen3.6-35B-A3B | ON | 194 (1) | 187 (0) | 65525 / PASS / 7.488 |
| Qwen3.8-27B | OFF | 198 (0) | 197 (0) | 65525 / PASS / 25.806 |
| Qwen3.8-27B | ON | 198 (0) | 195 (0) | 65525 / PASS / 26.384 |

Single-request recall times are observations, not performance thresholds.

## Unresolved Qwen3.6-35B-A3B MTP ON

| Pair | Image | Thinking correct/invalid | Non-thinking correct/invalid |
|---|---|---|---|
| Original | baseline | 194/0 | 187/0 |
| Original | candidate r2 | 194/1 | 187/0 |
| Matched retest01 | baseline | 196/0 | 191/0 |
| Matched retest01 | candidate r2 | 191/2 | 194/0 |

Every correct count is out of 200. Correct −5/200 and invalid +2 remain
investigation triggers. Candidate first-run index54 has an empty final after
16384 tokens (length). Matched index115 has an empty final after 172 tokens
(stop), reasoning 708 characters; index186 has an empty final after 16384
tokens (length), reasoning 16875 characters. No HTTP errors accompany these
records. Actual MTP execution, health and 64K recall pass.

This is **eager without LMCache/Attention8**. Temperature 0.6 and no fixed
request seed mean one pair is not statistical proof; repeated empty finals
are not dismissed. The original harness lacks full reasoning/API/token output,
so the generation/parser/stop root cause remains unestablished.

See the [independent reproduction handbook](QWEN36_35B_MTP_ON_REPRO_HANDBOOK.md).
The local package `/home/intel/guancheng/ima-tests/qwen36-35b-mtp-on-repro`
and sibling tar.gz contain four raw groups, frozen assets, full API capture and
a matched runner. Package SHA256:
`33bfb2f105fb9e79e09fc45367ddd1bd886797eb528d9c5b994fca3490f1ace1`.
Six offline tests pass, including a 200-case/32-thread mock comparison of
original/capture payloads, scores and exit codes. New GPU reproduction is
NOT RUN. Only the handbook/report are published; the package and images/weights
are not uploaded.

## Unresolved Attention8 and evidence limits

Original and unchanged formal retest: third-case cold output is
`Answer:\n\n42`, six tokens/stop; replay changes to multiple-choice/reasoning,
48 tokens/length. Default/explicit RAW retain LMCache and pass all three cases
with actual external hits. Attention8 is lossy KV quantization: small logit
errors can change greedy argmax. Text inequality alone does not prove an
implementation bug, but the predefined strict gate is unchanged and the
incomplete reply remains FAIL.

Original/formal matched quality20: 18/20 cold/replay correct, 20/20 numeric,
19/20 text agreement; index5 has repeated markers and an unfinished tail.
Historical quality JSON lacks full API usage/finish metadata, so its exact
termination cannot be claimed. A later full-API baseline/candidate pair each
has 18/20 correct, 20/20 numeric and 18/20 text agreement; all 40 captured
responses per image finish with stop, without repeated markers. That run did
not reproduce the historical quality anomaly and does not override the
original/formal functional failures.

Diagnostics: 14 decoded attention H2D/gather roundtrips are bitwise equal and
finite; 32 same-input cross-runtime CPU codec chunks match. Native weight
quantization and eight native GDN output/state records match baseline/candidate.
FP8-linear and separate GDN-projection controls still reproduce the failure.
Triton prefill probe output hashes match; FP32 final-state hashes differ with
small reference-error changes, and this probe is not the actual native GDN
serving core. No causal kernel bug is established. Disabling graph, MTP,
LMCache or quantization is not substituted for the original acceptance tests.

## Audit, cleanup and scope

Read-only checklist rerun: **36 PASS / 2 FAIL / 2 BLOCKED / 0 NOT RUN**,
acceptance **FAIL**. Counts cover that checklist, not every delivery obligation
or future diagnostic. Evidence: `provenance/*-pr-refresh-20261016.json`.
Scheduled pipelines ended with `POST_R2_GRAPH_PIPELINE_FINISHED: lmcache=1`
and `FINAL_TRIGGER_RETESTS_FINISHED: 1 matched pairs`.

Explicitly authorized cleanup removed eight obsolete images and 61 archived
stopped old test containers. Final runtime/builder, overlay baseline, negative
control, all build/source/wheel/first-failure archives, unrelated resources and
build cache were preserved. Old builder tar archives remain; not all old runtime
images are directly recoverable without rebuilding. Exact cleanup plans,
inspect and logs are local.

Overall migration work remains paused except for this requested documentation
update. No new build/GPU test, registry upload or merge. No performance,
multimodal, BFCL, full IMA UI or all-pass release claim. Human review is required
before merge.
