# MiniCPM-V 4.7 DSpark on Intel XPU

This focused development recipe extends
[Intel llm-scaler](https://github.com/intel/llm-scaler) using the upstream
`vllm/vllm-openai-xpu` v0.31.0 runtime. It packages the MiniCPM-V 4.7 model,
XPU compiler fusions with BF16 activation support, and the DSpark noncausal
attention specialization. The default serving configuration uses seven
speculative tokens, online FP8 target weights, BF16 target activations, and a
BF16 DSpark draft. Draft Markov/vocabulary FP8 conversion is not enabled.

## Build

From the repository's `vllm/` directory:

```bash
docker build --progress=plain -f docker/Dockerfile.minicpmv47 .
```

The Dockerfile pins the upstream v0.31.0 image by digest, adds the oneAPI
2026.1.1 compiler, prepares exact source revisions, and compiles the additional
native operators. It preserves the upstream GPU runtime, Torch ABI, kernel
wheel and Rust binaries. Patched Python sources are selected with `PYTHONPATH`;
native overlays supply the additional fusion operators and attention route.
This is a development source/native overlay, not a new vLLM package release.

Sources are retained under `/opt/minicpmv47/src/`; the recipe and build outputs
are under `/opt/minicpmv47/recipe/`, `/opt/minicpmv47/build/` and
`/opt/minicpmv47/native/`. `prepare_sources.py` verifies every patch checksum
and the complete resulting Git tree. Existing developer checkouts are never
reset. The native build uses only Xe2 kernels and retains the unmodified
upstream sources, including other device backends.

The Python constraints preserve Torch `2.14.0+xpu`, Triton `3.8.0+xpu`, the
stock vLLM `0.31.0+xpu` metadata and kernels `0.1.15.4`. Transformers is
`5.17.0`; tokenizers is `0.23.2` to satisfy the pinned main source's
`tokenizers>=0.23.2` requirement. This differs from the base image's
tokenizers `0.23.1`. XGrammar is aligned to the source's `0.2.8` requirement.
The build aligns direct source requirements with `--no-deps` and checks them
after installation. It preserves the base's transitive runtime packages;
resolving Torch's dependencies again would downgrade the base's Intel runtime
and oneCCL packages. It does not replace those coordinated runtime libraries.
The retained v0.31.0 wheel metadata still declares XGrammar `0.2.7`, so a
distribution-wide `pip check` reports that metadata mismatch. The patched
source requires `0.2.8`; `check_environment.py` checks that source contract.

## Patch sources

The checksummed source manifest is
[`sources.json`](../patches/minicpmv47/sources.json).

| Patch | Base and scope |
| --- | --- |
| `0001-vllm-minicpmv47-xpu.patch` | vLLM main `95307ac6d4cd8be2c1e38f236e9104c569af9025`; MiniCPM-V 4.7, XPU compiler fusions, BF16 eligibility, QKV registration, speculative warmup and hybrid-state bounds. |
| `0002-kernels-pr622.patch` | kernels main `bdf9ac02f55293b6217a1a47e0a497d580474e23`; PR #622 through `bee39d9299538ce6e82a947443790450aa570f3c`. |
| `0003-kernels-bf16.patch` | PR #622 head; BF16 activation dispatch, accumulation, stores and wrappers. |

Upstream contributions are
[MiniCPM-V 4.7 PR #6](https://github.com/wyc55069407/vllm/pull/6),
[vLLM PR #60539](https://github.com/vllm-project/vllm/pull/60539),
[Qwen DSpark PR #58674](https://github.com/vllm-project/vllm/pull/58674), and
[XPU kernels PR #622](https://github.com/vllm-project/vllm-xpu-kernels/pull/622).
Qwen DSpark is already in the pinned vLLM main revision and does not require a
second patch. The manifest pins this recipe's source; moving PR heads are not
fetched during the build.

## Serve K7

Mount locally available target and draft model directories read-only at
`/models/target` and `/models/draft`. Use the Docker image ID printed by the
build. Select the intended Intel GPUs according to the host's actual device
mapping; the example assumes a verified two-device selection.

```bash
docker run --rm --device /dev/dri --shm-size 32g --network host \
  --mount type=bind,source="${TARGET_MODEL_DIR}",target=/models/target,readonly \
  --mount type=bind,source="${DRAFT_MODEL_DIR}",target=/models/draft,readonly \
  --env ZE_AFFINITY_MASK="${XPU_AFFINITY}" \
  "${IMAGE_ID}" /opt/venv/bin/python /opt/minicpmv47/recipe/serve_k7.py \
  --model /models/target --draft-model /models/draft
```

The launcher uses TP=2, one concurrent sequence, maximum model length 8704,
and `FULL_DECODE_ONLY` graphs with Inductor and the XPU fusion passes enabled.
It sets `num_speculative_tokens=7` explicitly, regardless of a checkpoint's
training block size. The compiled fusion range retains its default eight-row
limit; experimental M16 and draft-head quantization overlays are excluded.
KV cache page size remains determined by vLLM's hybrid cache manager. The
`64` in the native build config selects a compiled kernel variant, not the
runtime KV cache page size.

`serve_k7.py --dry-run` prints the invocation without starting inference.
Override `--tensor-parallel-size`, `--max-model-len`, `--host` and `--port`
only as needed for the deployment. Images/video inputs are disabled by this
text-only launcher. Runtime startup fails if a required overlay cannot load;
unmatched attention shapes retain the stock dispatch path.

This recipe requires a clean image build and GPU validation before use as a
validated deployment. It does not resolve or claim validation of long-context
DSpark draft behavior. Model weights are not downloaded or included in the
build context.
