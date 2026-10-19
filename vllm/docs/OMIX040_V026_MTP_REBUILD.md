# OMIX v0.26 MTP rebuild — 2026-10-19 UTC

Snapshot at 02:12 UTC. PR764 remains Draft. Build and targeted checks are
complete; the new full model matrix and graph/LMCache model suites are still
pending. Historical failures are retained, not replaced by diagnostic scores.

## Exact merged fixes

- [Kernel PR21](https://github.com/analytics-zoo/vllm-xpu-kernels/pull/21):
  source commit `488ad801e0f5b32b56235243d21e3ec57a16a54f`, merged as
  `4c3747023c73af0dfe9d0035442a6c0eb4c553cc`.
  Preserve packed GDN MTP convolution history and writeback without corrupting
  checkpoint blocks. Source diff SHA256:
  `bd110cf7de73674caf0edb7dc3bca78a9a3ab958070aca82f536348e0e4b9aea`.
- [Runner PR50](https://github.com/intel-innersource/applications.ai.gpu.llm-scaler-vllm/pull/50):
  source commit `932726f4cb86b818bf8fbc2566905f53b49eb77d`, merged as
  `12b8b9b09b12235499b03b3331f7c12dc6018966`; upstream
  `f725046204878c7de7c7dae10db6885d02d311a1`.
  Isolate async accepted-count D2H ownership and synchronize/remap at the correct
  point; sync mode completes DMA before InputBatch row changes. Diff SHA256:
  `788a5ea0be0dc4a2218075525af06de2002493693e27537ab17a92855755ee97`.

Both patches are byte-for-byte source-commit diffs and apply on the pinned
release sources. Reverse-apply checks also pass on the actual built sources.
No independent competing fix PR is introduced.

Implementation commit: `80af1fb34ff9d6e6aedc0c78dad20c4f8c508275`, including main
at `49d2f8d074397c91f138f933e7d7d6ff033f1581`. vLLM source remains v0.26.0 at
`568afb3a13806beb53bb2e6bd518269357b237c0`; generated wheel label
`0.26.1.dev0+g568afb3a1.d20261019.xpu` is SCM metadata, not a source upgrade.
OMIX digest, Python3.12.14, Torch2.14 XPU, Intel2026.1.1/oneCCL2022.1.2/PTI1.1.0,
kernel base and oneDNN pins remain unchanged. Dockerfile.dev is untouched.

## Build and complete wheel archive

- Runtime: `llm-scaler-vllm:omix040-v026-mtpfix-r1`,
  `sha256:9096e420a27adc1e69e2aac1cb806e9c28be53c8a3afdc3fae9e2a09ffc1a114`.
- Builder: `llm-scaler-vllm:omix040-v026-builder-mtpfix-r1`,
  `sha256:5d65224f376eea4d04dcf8a8da6e3468ffc4f2807d3dd22dec03380b68080f87`.
- Archive: `/home/intel/guancheng/omix040-v026-mtpfix-20261018`.
  Complete `/wheels` copied to `wheels/` before runtime construction, with all
  sources, input/applied diffs, source hashes, compiler/dependency versions,
  image inspect and build logs. Initial vLLM wheel retained in `initial-vllm/`;
  runtime installs only the final root wheel. All six wheels verify against
  the relative-path `wheels/SHA256SUMS` manifest.
- Saved builder: `builder-image.tar.gz`, SHA256
  `5e927cbe8aacbec66f3cb11365ba478bcd422e68e1ed68d4d9bc156a86faf68e`.
- Fresh Rust and all five native families compiled. A changed Rust Ubuntu
  digest invalidated those stages; shared Python/dependency setup reused cache.

| Final wheel family | SHA256 |
|---|---|
| vLLM | `c63d29c5f01101f9e1863d29960448a2a722b3e21e977b001a29fdb03a26ec45` |
| custom ESIMD | `16d3084f42a36299d609ccd2260ed4b80739f9f6dd3c3a20f1d72d9c8251b611` |
| vllm-xpu-kernels | `b752863bde87b1f52e06cc984c227a495b5ea203bb3c82039d1fe827f8455687` |
| KVWeave | `add024cf80a0b170a25ce3ec8274e1101b3dbf5921629e06d844d2bf2a89668e` |
| LMCache | `10d800176ad9954891266cf88e7fc21988eb35d171a1efa0c9f5ce3217c51454` |

## Actual new-image verification

GPU6,7 only; shared lock; fresh containers and ports; serial model loading.
No production wheel, model implementation or runtime dependency was changed
inside tested containers. Only read-only test assets/instrumentation mounted.

| Check | Measured result | Archive evidence |
|---|---|---|
| Full build; wheel/source export; saved builder | PASS | `logs/{builder,runtime}-r1.log`, `provenance/`, `wheels/` |
| Installed five-wheel identity; all29 native imports/DSOs/ldd | PASS | `runtime-native-basics-r1.log`, `runtime-library-and-native-census.log` |
| Dependency/connector/runtime ownership/actual OTLP export | PASS | runtime build log and `provenance/runtime-versions.txt` |
| Two-device math/native capture-replay/XCCL | PASS | `logs/runtime-native-basics-r1.log` |
| PR21 packed16 + compact8 using installed native op | PASS24/24 on each GPU | `logs/pr21-native-gpu{6,7}-r1.log` |
| Original PR50 pytest class | FAIL1/PASS7 (test fixture) | `logs/pr50-full-file-r1.log`, exact-class fallback log |
| PR50 fixture adapted to v0.26 | PASS8/8 | `logs/pr50-v026-fixture-r3.log` |
| LMCache store/worker/Mamba/RAW/SHM regressions | PASS100 | `logs/lmcache-store-r1.log` |
| New five-model MTP OFF/ON eager10 | NOT COMPLETE | Qwen35 ON runs below; other9 groups queued |
| New Qwen38/Gemma26 TP2 Inductor+FULL and native+FULL × OFF/ON | NOT RUN | queued; basic XPU graph is not model graph acceptance |
| New LMCache default/explicit RAW, Attention8, L2 and matched20 quality | NOT RUN | queued; historical Attention8 failure remains |

Original PR50 tests collect successfully. The last test leaves
`scheduler_output.free_encoder_mm_hashes` as an uniterable Mock; v0.26 runner
reads this field. Adding exactly `free_encoder_mm_hashes=[]` to a separate
test-only copy passes all8. Assertions and installed runner are unchanged;
original failure and fixture-only diff are archived. This is a test harness
compatibility result, not an image fix or a claim that the untouched8 tests pass.

Commands actually run include `docker build --progress plain --target builder`,
the final runtime build, `verify_installed_wheels.py`, `basic_xpu.py`,
`native_wheel_census.py`, direct exact PR21 test functions against the installed
op, `python -m pytest ...::TestSyncNumAcceptedTokens` (original and adapted
fixture), and the five LMCache test files with production runtime imports.
Static `git diff --check` and Docker build check passed. Exact argv/exit codes
and full output are retained locally.

## Qwen35 MTP ON matched runs

Frozen 5shot GSM8K, thinking200/nonthinking200,32threads,64K; FP8/FP16,TP2,eager,
same GPU/weights, no LMCache/Attention8. Thinking temperature0.6, no request seed.
Baseline `pr707-review-fix` (`1091b5fc32eb...`) is a Python-only overlay.
Full request/response/usage/finish capture is client-only and does not change
sampling, extraction or scores. This does not establish statistical significance.

| Round | Image | Thinking correct/200 (invalid) | Nonthinking correct/200 (invalid) | 64K actual prompt tokens/recall |
|---|---|---|---|---|
|1|new candidate|194 (0)|194 (0)|65525/PASS|
|1|overlay baseline|191 (1)|189 (0)|65525/PASS|
|2|new candidate|193 (0)|193 (0)|65525/PASS|
|2|overlay baseline|RUNNING|NOT RUN|NOT RUN|

Both new-candidate rounds have zero empty finals and zero invalid across all
800 responses. Thinking has no length-limit responses. Nonthinking has one
nonempty length-limit wrong answer in round1 (idx114), two in round2 (idx82,114).
Those remain anomalies for matched review, not hidden by accuracy-floor PASS.
Historical empty-final indices54,115,186 did not recur in either thinking round.
Round1 independent contract audit passes and candidate does not trigger correct
drop>=4 or increased invalid. Round2 comparison is not complete.

Evidence: `regression/mtpfix-matched-round{01,02}-{candidate,baseline}-eager/`;
`matched-round01-audit.json`, `matrix-audit-progress01.json`. The frozen port
probe briefly found EADDRINUSE after a stopped group with no listener; original
failure preserved. Wrapper now waits/resumes completed groups without killing
listeners or changing tested images/request scripts.

## Remaining boundaries

Do not treat prior successful graph/full-model/LMCache results as new-image
passes. The new remaining suites and matched review are still running/queued.
Attention8 third replay was not fixed by these two patches and remains an open
historical strict-gate failure pending new-image retest. Release acceptance is
not all-pass. No graph/MTP/LMCache feature disable substitutes for acceptance.

Images stay local; no registry upload or merge. Old release evidence and
worktree preserved. No image cleanup performed during this rebuild as of the
snapshot. AI assistance used for backport/build/test orchestration; human
review required before merge.
