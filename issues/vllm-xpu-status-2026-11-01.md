# vLLM on Intel Arc 140V (Lunar Lake) — status and idea uptake, 2026-11-01

Scope: what changed upstream after 2026-10-17 for the owner's baseline (vLLM v0.32.0 + vllm-xpu-kernels 0.1.15.4 + torch 2.14.0+xpu + triton 3.8.0+xpu + compute-runtime ~26.27, eager, Arc 140V `lnl_m`, 32 GB UMA, ~21 GiB practical ceiling), and which ideas in `issues/stack-work-plan-2026-10-29.md` upstream has already done, is doing, or has not touched. The report date is 2026-11-01. The skeptic passes ran through 2026-11-03, so a few facts dated Nov 2–3 are included and marked with their dates.

Legend:
- **[confirmed]**: a skeptic re-checked it against the primary source, or I re-checked it on 2026-11-03.
- **[corrected]**: the researcher's claim was overstated or wrong. The corrected text is shown.
- **[unverified]**: only the researcher checked it, or the source could not be fetched.
- Claims a skeptic refuted have been dropped or are labelled as refuted.
- *est.* marks an estimate. No number in this report was measured on the Claw.

---

## 1. TL;DR

- **Newest vLLM is v0.33.0.** It is an annotated tag dated 2026-11-01, on commit 4ba787e (the same commit as rc3, Oct 31). PyPI got it on 2026-11-02 10:13 UTC and Docker Hub got `vllm/vllm-openai-xpu:v0.33.0` (= `latest`, 5.21 GB) on 2026-11-02 02:49 UTC. I re-checked both on 11-03. There is no v0.32.1 final and no v0.33.1. **[confirmed]** ([tags](https://github.com/vllm-project/vllm/tags), [PyPI](https://pypi.org/pypi/vllm/json), [Docker Hub](https://hub.docker.com/v2/repositories/vllm/vllm-openai-xpu/tags?page_size=12&ordering=last_updated)). An A1 claim that "v0.33.0 is not distributed" was true on Nov 1 and was overtaken on Nov 2.
- **v0.33.0 uses the owner's pins unchanged:** torch 2.14.0, triton 3.8.0+xpu, vllm_xpu_kernels 0.1.15.4, auto_round_lib 0.16.0, numba 0.65.0. Its `requirements/xpu.txt` is byte-identical to v0.32.0's, and the driver stack in Dockerfile.xpu is unchanged (compute-runtime 26.27.39122.11, IGC 2.38.2, L0 1.32.0). **[confirmed]** ([v0.33.0 xpu.txt](https://raw.githubusercontent.com/vllm-project/vllm/v0.33.0/requirements/xpu.txt))
- **main is in flux. Do not use it.** Kernels moved to 0.1.16.2 (#61956, 10-28) and then to `==0.1.16` from a new GitHub-pages index (#64633, 10-30), which appears to be the 520 MB "full attention config" build. main also has an XPU spec-decode regression (#57770, 10-30; fix #64897 still open) and rewrites `gpu_worker.py` (#56492, 11-01). **[confirmed]**
- **Kernels 0.1.16.3, 0.1.16.4 and the coming v0.1.17 (cut 11-04, release 11-06) need torch 2.15.** The last torch-2.14 kernels are 0.1.16.2 (PyPI) and GH v0.1.16. torch 2.15.0 shipped 10-28, but vLLM's torch-2.15 PR #63977 is still a draft. Pin the kernels exactly. **[confirmed]**
- **Upgrade recommendation: move to v0.33.0 at the end of week 1, not today, and not to main.** The pins are identical, so the container risk is low. The gains are real:
  - Gemma 4 no longer forced to TRITON_ATTN on XPU, and FA accepts head ≤512 (#62014).
  - A Triton/SYCL GDN toggle (#56835).
  - EAGLE/MTP lookahead block hashes (#61589).
  - A DFlash prefix-block fix (#54163).
  - `--sleep-mode-backend` (#63138).

  The cost is a patch rebase. The Gemma4Config hunks become obsolete, `unify_kv_cache_spec_page_size` moved to `kv_cache_planning.py` (#61183), INC MoE now routes to ARK by default (#50396), and XPU FLASH_ATTN now declares FULL-graph support for mixed batches (#62244), so eager must be pinned explicitly. Run the week-1 baselines on v0.32.0, then do the S45 dry run against the official v0.33.0 image. **[confirmed]**
- **Biggest correction this round:** the #54740 GDN n-gram assert fix (kernels #600) is **already in the owner's 0.1.15.4**. It was cherry-picked as f4b729a onto release/0.1.15.4. I re-confirmed it on 11-03 by grepping the PyPI wheel's `_xpu_C.abi3.so`, which contains `spec_token <= num_spec_decodes * (num_speculative_tokens + 1)`. The plan's "never n-gram on GDN" cut no longer has its reason. Whether it actually removes the crash on hardware is untested. **[confirmed]**
- **Second correction:** on v0.32.0, `method:"ngram"` runs under Model Runner V2 as a **GPU Triton speculator** (#40704). It is not the CPU numba proposer. The CPU path exists only with `VLLM_USE_V2_MODEL_RUNNER=0`, and draft #63373 would remove V1 entirely. **[confirmed]**
- **Idea uptake: done upstream.** The #40128 hybrid page-size cherry-pick was superseded by #39931 + #45181 (in v0.24.0+), and the PR itself closed unmerged on 2026-04-21. Delete the patch and do not post the comment draft. **[confirmed]**
- **Idea uptake: partly upstream.**
  - S02/S04: the Gemma 4 backend selection (#62014) is upstream, but the S02 probe itself is not.
  - S47: the attention-tuple half probably exists in the GH 0.1.16 full-config wheel; LNL AOT and the `is_bmg()` gate do not.
  - S07: XPU sleep exists, but is untested on iGPU and loses MRV2 model state at level 2.
  - S13/S37: an ARK INT4 MoE path exists, and llm-compressor 0.15.0 is out.
  - S16: generic DFlash fixes only.
- **Idea uptake: in progress upstream.** The kernel pin churn (S47) and the torch 2.15 move (#63977). Several XPU decode and quant PRs are open: #644, #670/#64687, #61630, #58131 and #61731.
- **Idea uptake: not touched.**
  - S10: XPU memory accounting is unchanged; there is still no `is_integrated_gpu` in xpu.py.
  - S12a: no FP8-KV change.
  - S06: no load-time work and no cache variables in Dockerfile.xpu.
  - S51, S52, and S21 (no OVMS, OpenVINO or NPU-driver release since Oct 1).
  - No DFlash or EAGLE result exists on any Arc iGPU. **[confirmed]**
- **Act today:** the owner's PR **#40295** has carried the stale label since 10-04, so its 30-day auto-close window ends about 11-03. Either post a holding comment now or let it close. Also, **#41370 was written by arnokamphuis, not the owner**, so the plan's "close #41370" is not an action the owner can take. **[confirmed]** ([#40295](https://github.com/vllm-project/vllm/pull/40295), [#41370](https://github.com/vllm-project/vllm/pull/41370))

---

## 2. Versions and pins

### 2.1 vLLM tags after v0.32.0

| Tag | Commit / date | PyPI | XPU image (Docker Hub) | XPU wheel (wheels.vllm.ai) | Tag |
|---|---|---|---|---|---|
| v0.32.0 (baseline) | 5df63a7, commit 2026-10-17; GH release 2026-10-18 | 0.32.0, 2026-10-18 10:03 UTC | `v0.32.0`, 2026-10-18 10:35 UTC, 5.18 GB | named in release notes at `wheels.vllm.ai/0.32.0/xpu`; index not fetchable | [confirmed] (wheel [unverified]) |
| v0.32.1rc0 | 2365177, 2026-10-18; plain tag on main, never promoted; does **not** contain #62014 | none | none seen | — | [confirmed] |
| v0.33.0rc0 | 4135590, 2026-10-26 (main cut, branch releases/v0.33.0) | none | nightlies only | — | [confirmed] |
| v0.33.0rc1 / rc2 / rc3 | cbafc46 10-28 / 61cbf54 10-30 / 4ba787e 10-31 | none | nightlies only | — | [confirmed] |
| **v0.33.0** | annotated tag 3a4d82f → 4ba787e, 2026-11-01 01:21 PST. rc0→final is 12 cherry-picks, none XPU | 0.33.0, 2026-11-02 10:13 UTC | `v0.33.0` = `latest` = `v0.33.0-x86_64`, 2026-11-02 02:49 UTC, 5.21 GB | [unverified] (host blocked) | [confirmed] |
| GitHub release page v0.33.0 | A1's skeptic (11-03) saw no release notes and no Latest badge. B3/B8 researchers reported a release listed 11-02 with an XPU wheel line | — | — | — | [unverified] (conflicting readings) |
| main | 254a2a3 (2026-11-03 09:34 +08) at last skeptic check | — | `nightly-76c6f3f` 2026-11-02 06:34 UTC, 5.61 GB | — | [confirmed] |

The nightly image grew from 5.22 GB (Oct 30) to 5.62 GB (Oct 31) right after #64633. That matches swapping the ~150 MB PyPI kernels wheel for the 520 MB GitHub-release wheel. **[confirmed]**

### 2.2 `requirements/xpu.txt`

| Ref | torch | triton | vllm_xpu_kernels | auto_round_lib | numba | Tag |
|---|---|---|---|---|---|---|
| v0.32.0, v0.32.1rc0, v0.33.0rc0–rc3, **v0.33.0** | 2.14.0 (download.pytorch.org/whl/xpu) | 3.8.0+xpu (wheels.vllm.ai/xpu shim) | **0.1.15.4** | 0.16.0 | 0.65.0 | [confirmed] |
| main (re-checked 2026-11-03) | 2.14.0 | 3.8.0+xpu | **0.1.16** via `--extra-index-url=https://vllm-project.github.io/vllm-xpu-kernels/whl` | 0.16.0 | 0.65.0 | [confirmed] |
| main history | #61956 → 0.1.16.2 (merged 2026-10-28); #64633 → ==0.1.16 from the GH index (2026-10-30; commit body is title only) | | | | | [confirmed] |
| draft #63977 (torch 2.15) | 2.15.0+xpu from download.pytorch.org/whl/test/xpu | shim dropped; `triton-xpu==3.9.0` installed in Dockerfile only | 0.1.16.3 from the GH index (commit d7f72d00 "use full config", 10-31, switched it from PyPI 0.1.16.4) | — | — | [confirmed] ([#63977](https://github.com/vllm-project/vllm/pull/63977)) |

### 2.3 vllm-xpu-kernels

| Version | PyPI upload | Git tag / commit | torch | Notes | Tag |
|---|---|---|---|---|---|
| 0.1.15.4 (baseline) | 2026-09-22 08:39 | release/0.1.15.4 branch (no tag) | 2.14.0+xpu | contains #586 (grouped-GEMM race), #499 (getMemoryInfo fallback, first in v0.1.15), **#600** (GDN ragged spec, cherry-pick f4b729a) | [confirmed] |
| 0.1.16.1 | 2026-10-15 07:43, 149.4 MB | v0.1.16 = v0.1.16.1 (8b95ea6) | 2.14.0+xpu | default attention preset | [confirmed] |
| GH-release v0.1.16 wheel | GitHub asset 2026-10-16, 520 MB | same commit | 2.14.0+xpu | very likely a **full attention-config** build (libattn_xe_2 1.74 GB uncompressed). Evidence is circumstantial (size, #63977 "use full config", image growth); the build workflow was not read | [confirmed] as strong circumstantial evidence |
| 0.1.16.2 | 2026-10-26 08:58, 150.2 MB | e52a057 = v0.1.16 + #663 (Qwen2-VL head-80 config) | 2.14.0+xpu | **last torch-2.14 PyPI wheel** | [confirmed] |
| 0.1.16.3 | not on PyPI | 4e38707 (= v0.1.16.4) | 2.15.0+xpu | GH release marked Latest, 10-31 | [confirmed] |
| 0.1.16.4 | 2026-10-30 14:21, 157.0 MB | 4e38707 = v0.1.16.2 + #621 (torch 2.15) | **2.15.0+xpu** | no `Requires-Dist` on torch, so pip will not block a mismatch | [confirmed] |
| v0.1.17 | — | tracker #676: cut 2026-11-04, release 2026-11-06 | 2.15 | planned: #668, #672, #665, #647, #606, #621, #617, #651; #660/#662 only requested | [confirmed] ([#676](https://github.com/vllm-project/vllm-xpu-kernels/issues/676)) |
| main | — | 3063346 (2026-11-02, #641) | 2.15.0 | MoE interface rewrite and xe2/xe3 split (#583, #664, #669), in no tag | [confirmed] |

- AOT lists on main, unchanged since v0.1.15: `pvc,bmg,bmg-g21-a0,bmg-g31-a0` (Xe3: `cri,cri-a0`). There is **no lnl member**. The `VLLM_XPU_*AOT_DEVICES` env overrides still work. **[confirmed]**
- Source-build note **[corrected]**: the v0.1.15 tag's CMake targets torch **2.13.0**, and the 2.13→2.14 move happened in v0.1.16. For a torch-2.14 source build, use **v0.1.16.2**. "Pin to 0.1.15.4" is not a verified source tag. ([v0.1.15 CMakeLists](https://raw.githubusercontent.com/vllm-project/vllm-xpu-kernels/v0.1.15/CMakeLists.txt))

### 2.4 torch, driver, oneAPI, triton, Level Zero

| Item | Status | Tag |
|---|---|---|
| torch XPU | 2.14.0 (PyPI 2026-09-02), 2.14.1 (2026-09-30, no XPU notes, same oneAPI 2026.1.0), **2.15.0 (2026-10-28 15:16 UTC)**. vLLM v0.33.0 and main pin 2.14.0 | [confirmed] |
| torch 2.15 XPU runtime | oneAPI pip 2026.1.2 ("DLE 2026.1.3"), triton-xpu 3.9.0, pyzes 0.1.3 | [unverified] |
| torch #195987 (iGPU usable fraction 0.94 in `getMemoryInfo`) | in v2.15.0, but compiled only when `SYCL_COMPILER_VERSION >= 20260200`. The 2026.1-built wheels therefore probably do not include it (inferred, not checked against a wheel) | [confirmed] (header); activation [unverified] |
| compute-runtime | latest release is still **26.35.39758.10 (2026-09-17, IGC 2.41.5)**. After Oct 17 only tags appeared: 26.35.39758.18/.20/.21 and 26.40.40196.7/.8/.15, with no release notes. LNL-Linux commits on master only: d322804 (10-19, ULLS 1 ms), 02d2ef7 (10-16, KMD notify off; already off in tags), a9797c6 (10-19, dGPU headroom 0.95→0.98; iGPU stays 0.94) | releases [confirmed]; commit contents [unverified] |
| vLLM Dockerfile.xpu | v0.32.0 = v0.33.0 drivers: NEO 26.27.39122.11, IGC 2.38.2, L0 1.32.0, gmmlib 22.10.0, pip oneAPI override 2026.1.1. v0.33.0 bumps UCX v1.21→v1.23 and NIXL v1.3.2→v1.5.0. main adds NIXL v1.5.2 (#64579, 10-31). No NEO, Triton or vLLM cache variables | [corrected] (it is not "identical" to v0.32.0; only the drivers are) |
| oneAPI pip 2026.2.0 | on PyPI 2026-10-16 (DPC++ v7.2.0, 10-17); no torch wheel uses it. Do not drop 2026.2 UR packages into a 2026.1 venv: it gave undefined `urKernelSetArg*` in PyTorch CI (#201257) | [unverified] |
| triton-xpu | v3.9.0 released 2026-10-29 (removes block pointers, #7557); PyPI `triton-xpu` still 3.3.0b1; vLLM keeps the 3.8.0+xpu shim | [unverified] |
| L0 V2 adapter | still the default for IP ≥ 0x05004000 (BMG+, includes LNL) in DPC++ v7.2.0 and on sycl HEAD; `SYCL_UR_USE_LEVEL_ZERO_V2=0` still forces V1. No upstream report on whether V1 is still needed for eager on LNL, so a local A/B is required | [unverified] |
| L0 loader / IGC | loader v1.34.0 stable (09-17), v1.35.0 pre-release (10-26); IGC v2.42.6 (10-26) | [unverified] |

---

## 3. What changed on XPU since Oct 17 (merged)

"v0.33" = in the v0.33.0 tag. "main" = merged after the v0.33.0rc0 cut (10-26), so not in any tag. Membership was checked with `git merge-base --is-ancestor` unless a row says otherwise.

### 3.1 Attention, Gemma 4 and graphs
- [#62014](https://github.com/vllm-project/vllm/pull/62014), 2026-10-21, **v0.33** (not in v0.32.1rc0) **[confirmed]**:
  - Gemma4Config no longer forces TRITON_ATTN on XPU.
  - XPU FlashAttention accepts head_size ≤512.
  - With no backend chosen, multimodal prefix-LM models get the TRITON_FLASH_ATTN composite (Triton for image-masked batches, SYCL FA for causal batches), and text-only models get FLASH_ATTN.
  - An explicit FLASH_ATTN on an mm-prefix model is honoured, with a warning that image/video results will be wrong.
  - In v0.32.0 the force applied only when no backend was given, so the owner's explicit flag already worked there.
- [#62244](https://github.com/vllm-project/vllm/pull/62244), 2026-10-21, **v0.33** **[confirmed]**:
  - Enables spec-decode adaptive verification on XPU.
  - Side effect: the XPU FLASH_ATTN builder now declares `AttentionCGSupport.ALWAYS`, where v0.32.0 declared UNIFORM_BATCH. With graphs on, mixed prefill/decode batches can now be captured FULL. Eager lanes are unaffected.
  - Adaptive verification was measured only under FULL_DECODE_ONLY graphs on 8×B60 TP8: −2.6 to −5.0% at K3, +2.2 to +11.7% at K5/K7. It does not need kernels #647.
- The XPU graph default is unchanged: graphs are still ON at -O2. xpu.py forces `cudagraph_mode=NONE` only for torch <2.11 or `--enable-sleep-mode`. **[confirmed]** ([v0.33.0 xpu.py](https://raw.githubusercontent.com/vllm-project/vllm/v0.33.0/vllm/platforms/xpu.py))
- [#52670](https://github.com/vllm-project/vllm/pull/52670), 2026-10-20, v0.33: pluggable KV-cache dtypes (generic refactor; fp8_e4m3 still built in). [#56155](https://github.com/vllm-project/vllm/pull/56155), 2026-11-01, main: per-layer KV dtype in the attention runtime. **[confirmed]** (membership only)
- [#62375](https://github.com/vllm-project/vllm/pull/62375), 2026-10-19, v0.33: `vllm bench forward` graph capture and profiling on XPU (tooling). **[confirmed]**

### 3.2 GDN, Mamba and Qwen3.5/3.6
- [#56835](https://github.com/vllm-project/vllm/pull/56835), 2026-10-19, **v0.33**: XPU `--gdn-prefill-backend sycl|triton`. `triton` runs the whole GDN layer through Triton/FLA and so bypasses the SYCL op and its `is_bmg()` chunk-inverse gate. The default `auto` still picks SYCL. **[confirmed]** (dispatch not traced line by line)
- [#62683](https://github.com/vllm-project/vllm/pull/62683), 2026-10-23, v0.33: Intel CI runs `test_mamba_prefix_cache.py` and `test_hybrid.py -k apc` on BMG only. **[unverified]**
- [#54163](https://github.com/vllm-project/vllm/pull/54163), 2026-10-22, v0.33: DFlash/DSpark no longer drop the last prefix-cache block. [#62841](https://github.com/vllm-project/vllm/pull/62841), 2026-10-23, v0.33: hybrid prompt-tail cache marking fix. **[confirmed]**
- [#51012](https://github.com/vllm-project/vllm/pull/51012), 2026-10-29, main: an eager/breakable-graph Qwen3.5 rewrite that applies only on CUDA, so XPU keeps the old implementation. [#62111](https://github.com/vllm-project/vllm/pull/62111) and [#63761](https://github.com/vllm-project/vllm/pull/63761), 2026-10-30, main. **[confirmed]** (membership; XPU-unchanged claim [unverified])

### 3.3 MoE and quantization
- [#50396](https://github.com/vllm-project/vllm/pull/50396), 2026-10-19, **v0.33**: `INCARKWNA16MoEMethod`. With the default `VLLM_XPU_INC_WNA16_BACKEND=auto`, INC/AutoRound sym INT4 MoE now goes to the ARK kernel. [#61354](https://github.com/vllm-project/vllm/pull/61354), 2026-10-21, v0.33: INC ARK narrow-shard loading fix. **[confirmed]**
- [#51605](https://github.com/vllm-project/vllm/pull/51605), 2026-10-26, main: XPU batched Triton MoE tile and grid retune. [#61362](https://github.com/vllm-project/vllm/pull/61362), 2026-10-26, main: XPU per-token dynamic FP8 activations. [#59938](https://github.com/vllm-project/vllm/pull/59938), 2026-10-29, main: XPU mrope kernel. [#64065](https://github.com/vllm-project/vllm/pull/64065), 2026-10-30, main: XPUMLASparseImpl fix. **[confirmed]**
- `xpu_moe.py` and `compressed_tensors_moe_wna16.py` are byte-identical from v0.32.0 to main, so the CT sym INT4 gs32/gs128 path is unchanged. **[unverified]**

### 3.4 Speculative decoding
- **v0.33**:
  - [#61589](https://github.com/vllm-project/vllm/pull/61589) (10-26): lookahead block hashes for EAGLE/MTP prefix caching. It disables the trailing-block drop by default when no KV connector is used.
  - [#59973](https://github.com/vllm-project/vllm/pull/59973) (10-15): `draft_lm_head_quantization` fp8/nvfp4.
  - [#59468](https://github.com/vllm-project/vllm/pull/59468), [#62784](https://github.com/vllm-project/vllm/pull/62784) and [#54448](https://github.com/vllm-project/vllm/pull/54448): DFlash budget, draft skip and the XPress refiner.
  - [#63251](https://github.com/vllm-project/vllm/pull/63251) (10-23): drafters stay within their rotary cache.
  - **[confirmed]**
- **main only**:
  - [#53104](https://github.com/vllm-project/vllm/pull/53104) (10-27): fixes MRV2 drafter lm_head sharing when the target lm_head is quantized. Without it, startup aborts with `'ParallelLMHead' object has no attribute 'weight'`.
  - [#63436](https://github.com/vllm-project/vllm/pull/63436) (10-26): DFlash aux FC in the draft dtype.
  - [#63819](https://github.com/vllm-project/vllm/pull/63819) (10-28): DFlash JIT warmup gaps.
  - [#63205](https://github.com/vllm-project/vllm/pull/63205) (10-30): MRV2 speculator config classes.
  - [#62796](https://github.com/vllm-project/vllm/pull/62796) and [#65041](https://github.com/vllm-project/vllm/pull/65041): adaptive-verification graph padding.
  - [#60339](https://github.com/vllm-project/vllm/pull/60339) (11-03): suffix decoding for MRV2.
  - **[confirmed]**
- **Regression on main:** [#57770](https://github.com/vllm-project/vllm/pull/57770) (10-30) broke NaN handling in the torch greedy-argmax fallback that XPU uses. The fix [#64897](https://github.com/vllm-project/vllm/pull/64897) is open. Which spec methods are affected on XPU is not established. **[confirmed]**
- The Gemma 4 MTP `torch.cuda.CUDAGraph` hard-code is still in the **V1** proposer only (`vllm/v1/spec_decode/gemma4.py`). The V2 Gemma4Speculator has no torch.cuda calls. **[confirmed]**

### 3.5 Sleep mode, loader and core
- [#63138](https://github.com/vllm-project/vllm/pull/63138), 2026-10-22, **v0.33**: `--sleep-mode-backend` (pluggable). **[confirmed]** (present in v0.33.0 arg_utils, absent in v0.32.0)
- [#64630](https://github.com/vllm-project/vllm/pull/64630), 2026-10-30, main: an exact-size, reusable pinned CPU backup for level-1 sleep (also cumem.py). On XPU the backup is no longer freed on wake, and the change is untested on XPU hardware. **[confirmed]**
- [#63548](https://github.com/vllm-project/vllm/pull/63548), 2026-10-27: moves MRV2 model state out of the level-2 discard pool on CUDA/ROCm only; XPU is excluded ("no nested pools"). [#64057](https://github.com/vllm-project/vllm/pull/64057): Intel CI deselects the Mamba deep-sleep test. **[confirmed]**
- [#61183](https://github.com/vllm-project/vllm/pull/61183), 2026-10-13, **v0.33**: `unify_kv_cache_spec_page_size` moved to `vllm/v1/core/kv_cache_planning.py`, with no re-export. **[confirmed]**
- **main only**:
  - [#56492](https://github.com/vllm-project/vllm/pull/56492) (11-01): extensible KV cache. CUDA/HIP VMM only, but it rewrites `gpu_worker.py`, which XPUWorker subclasses.
  - [#57678](https://github.com/vllm-project/vllm/pull/57678) and [#64692](https://github.com/vllm-project/vllm/pull/64692) (11-02): post-load hook changes.
  - [#64357](https://github.com/vllm-project/vllm/pull/64357) and [#63247](https://github.com/vllm-project/vllm/pull/63247) (10-30), [#63174](https://github.com/vllm-project/vllm/pull/63174) (10-31).
  - [#64336](https://github.com/vllm-project/vllm/pull/64336) (10-31): direct reload, which lists XPU weight padding as a limitation (that limitation is [unverified]).
  - **[confirmed]** (membership)
- Memory accounting is unchanged: `mem_utils.py` and the xpu.py memory path have no diff from v0.32.0 to main (11-03), and XPUPlatform still has no `is_integrated_gpu` override. **[confirmed]**

### 3.6 Server and tool parsers (new to the owner; landed after the v0.32 branch cut)
- **v0.33**: [#59928](https://github.com/vllm-project/vllm/pull/59928) (10-14) qwen3_coder recovers after an unclosed `<function=`; [#62027](https://github.com/vllm-project/vllm/pull/62027) (10-17) qwen3 incremental argument streaming; [#61494](https://github.com/vllm-project/vllm/pull/61494) (10-14) gemma4 incremental string arguments. **[unverified]** (dates from file history)
- The `--tool-strict-level` default changed from `"auto"` in v0.32.0 to `None` (per-parser) in v0.33.0. That gives AUTO for gemma4 and qwen3_coder (no behaviour change) and FUNCTION for glm47. **[confirmed]**
- [#64675](https://github.com/vllm-project/vllm/pull/64675), 10-31, main: fixes quadratic qwen3_coder re-parsing (the v0.33.0 sdist has no `_ClosedParams`). **[confirmed]**
- [#64177](https://github.com/vllm-project/vllm/pull/64177), 10-28, main: drops `prompt_tokens_details` when it is None. **[unverified]**

### 3.7 vllm-xpu-kernels (merged since Oct 17)
- [#621](https://github.com/vllm-project/vllm-xpu-kernels/pull/621) (10-30): torch 2.15 + OMIX. [#647](https://github.com/vllm-project/vllm-xpu-kernels/pull/647) (10-30): variable verify lengths in the FA spec path, main only. [#640](https://github.com/vllm-project/vllm-xpu-kernels/pull/640) (10-22): graph-mode variants of the kernel unit tests. [#674](https://github.com/vllm-project/vllm-xpu-kernels/pull/674) (10-29): GH-pages wheel index. **[confirmed]**
- GDN on main only, all after the torch-2.15 bump, so v0.1.17-only: [#658](https://github.com/vllm-project/vllm-xpu-kernels/pull/658) (10-26, 64-bit conv-state offsets), [#683](https://github.com/vllm-project/vllm-xpu-kernels/pull/683) (11-01, DS layout), [#660](https://github.com/vllm-project/vllm-xpu-kernels/pull/660) (11-02, skip NULL_BLOCK_ID padding rows; fixes graph-replay NaN [vllm#63620](https://github.com/vllm-project/vllm/issues/63620)). **[confirmed]**
- [#457](https://github.com/vllm-project/vllm-xpu-kernels/pull/457) was **closed unmerged 2026-11-02**, which leaves #586 (in 0.1.15+) as the grouped-GEMM race fix. **[confirmed]**
- The GDN chunk-inverse is still gated on `vllm::xpu::is_bmg()` (`chunk_gated_delta_rule_kernels_xe2.hpp:1351`, main 3063346). `getMemoryInfo` is unchanged, and `tests/test_get_memory_info.py` still covers only bmg_g21 and bmg_g31. **[confirmed]**

### 3.8 OpenVINO and NPU
- There has been no OpenVINO, GenAI, optimum-intel or nncf release since Oct 1. PyPI shows openvino 2026.4.1 and genai 2026.4.1.0 from 10-01. **[confirmed]**
- OVMS v2026.4.1 (10-06) and NPU driver v1.38.0 (09-11). **[unverified]** (not re-fetched)
- openvino.genai [#4795](https://github.com/openvinotoolkit/openvino.genai/pull/4795) (11-02) deprecates StaticWhisperPipeline on NPU. **[unverified]**

---

## 4. Idea uptake

| Idea | Upstream status | Evidence | Change to the plan |
|---|---|---|---|
| **S10** UMA accounting | **nothing** | `mem_utils.py` and the xpu.py memory path are unchanged v0.32.0→main (11-03); no `is_integrated_gpu`; kernels `mem_info.cpp` is identical v0.1.15→main; driver iGPU formula (0.94 × sharedSysMem − 450 MB − (MemTotal − MemAvailable)) unchanged on master. **[confirmed]** The #499 fallback is already in 0.1.15.4. torch #195987 is compiled out of 2.15 wheels and does not reach the lane. **[confirmed]** (header) | **Unchanged**, still day 1. Add one reading, `torch.xpu.mem_get_info()`, next to `getMemoryInfo(0)` and MemAvailable. **#40295:** decide now (auto-close about 11-03). **#41370:** not the owner's, so replace "close it" with an optional comment. Upgrading to v0.33.0 does not affect S10. Re-run it after any move past v0.33.0, because #56492 rewrites `gpu_worker.py`. |
| **S12** FP8 KV | **S12a: nothing. S12b: partly** (backend selection only) | No XPU FP8-KV change in vLLM or kernels; `calculate_kv_scales` absent in v0.32/v0.33, so scales default to 1.0 **[confirmed]**. #45540 was closed by its reporter on 2026-06-21 as fixed in kernels 0.1.10 (TP=2 dGPU GDN case) **[confirmed]**. v0.33.0 FA accepts head ≤512 on XPU **[confirmed]**. That FP8-KV dispatch covers every compiled tuple, and the `fa_utils` KV-dtype gate, are **[unverified]**. The default prefill preset has no head-256 causal+local line, and how that request is served is unresolved **[confirmed]** (source fact). Kernels #638 (FP8-KV prefill speed-up) is open and its numbers are **[unverified]** | **S12a unchanged.** The KL gate becomes a confirmation, not an expected failure. Keep it, because scale = 1.0 is the real risk. **S12b: reorder.** Run it first on stock v0.33.0 (text-only, or explicit FLASH_ATTN) with no home kernel. Keep the "no reference-attention fallback" log check. Remove "FP8 KV in the home kernel" from the S47 triggers unless stock S12b fails. Do not plan on #638. |
| **S15** n-gram | **partly** | On v0.32.0, `ngram` and `ngram_gpu` both run the V2 Triton `NgramGPUSpeculator` (#40704). Unmatched slots are filled with the last token as rejectable drafts, so every step verifies k+1 tokens. Async scheduling is auto-off for `ngram` and on for `ngram_gpu`. **[confirmed]** The #600 GDN fix is in 0.1.15.4 **[confirmed]**, but #54740 has had no reporter retest. Draft #63373 would remove V1. A B70 cookbook on v0.31/V1 reported +27–41% with ngram k=5 and negative KV memory on a single card **[unverified]** | **Shrink and re-frame.** "CPU n-gram" is wrong for the default lane. Run three arms on v0.32.0: `ngram` (V2, sync), `ngram_gpu` (V2, async) and `ngram` + `VLLM_USE_V2_MODEL_RUNNER=0` (V1 numba). Do the V1 arm now, because it is time-limited. Read "Available KV cache memory" for each arm. In S42, judge by accepted per draft token and accepted per step, since draft rate reads about 100% under fixed-K drafting. **Drop the "never on GDN" cut** and replace it with a short smoke test on the Qwen3.6 lane that watches for the #54740 RuntimeError. |
| **S16** DFlash / EAGLE-3 | **partly** (generic fixes only) | #54163 (v0.33) DFlash prefix block; #63436 and #63819 main only **[confirmed]**. The XPU DFlash context-norm fixes (#56431, kernels #579, #58896) are in v0.32.0 + 0.1.15.4 **[confirmed]**. #57470 (fused DFlash2 conv on XPU) was closed unmerged 10-18 **[unverified]**. No DFlash or EAGLE-3 result on any Arc iGPU, and no eager data; the only Arc DFlash numbers are dGPU **[confirmed]** | **Unchanged** ("you would be first" still holds). Run phase 0 on whichever lane has fresh A/A baselines; prefer v0.33.0 for #54163. Keep adaptive verification **off**: it is graph-measured and a loss at K3. If you see dtype or first-request JIT stalls, the fixes are main-only (#63436, #63819). Stay off main while #64897 is open. |
| **S13** REAP Qwen3-Coder | **partly** | #50396 ARK sym INT4 MoE (v0.33) **[confirmed]**; llm-compressor 0.15.0 (PyPI 2026-10-28) and compressed-tensors 0.20.1 **[confirmed]**; non-uniform REAP #61916 open **[confirmed]**. XPU top-k takes a generic path for non-power-of-2 expert counts, and REAP-25B's expert count is not known (HF blocked) **[unverified]**. Kernels #644 (INT4 GEMV MoE decode) is open and will land after the torch-2.15 bump **[confirmed]** | **Unchanged.** Keep uniform pruning and the CT sym INT4 export; avoid AWQ-packed (#48552 still open, stale since 10-23). Add a log check for which MoE/top-k path runs. If the export goes through AutoRound/INC and the lane is on v0.33.0, set `VLLM_XPU_INC_WNA16_BACKEND=w4a16`, because ARK is BMG-validated and #53211 is open. |
| **S37** fingerprints + foundry | **S37a: nothing. S37b: partly** | `draft_lm_head_quantization` (v0.33) quantizes only a draft copy and adds memory **[confirmed]**. **#53104 (main only)**: with a quantized lm_head, MRV2 MTP/EAGLE startup aborts on v0.32/v0.33 (plausible; not tested on XPU) **[confirmed]** (code). Even fixed, the drafter keeps its own head copy, about 0.95 GiB bf16 for 248k×2048 *est.* FP8 lm_head #61815 is open **[confirmed]** (the gain numbers are [unverified]). #53912 MTP+APC corruption is still open **[confirmed]**. #62781 (fp16 SSM cache breaks long-context GDN on B70) is open **[unverified]** (body not read). | **S37a unchanged. S37b: shrink and reorder.** Keep `ignore=['lm_head','re:.*mtp.*']`. Before the INT4-lm_head A/B, either carry #53104 or expect a startup crash, and budget about +0.95 GiB *est.* for the drafter's head copy. Add an FP8-lm_head arm (#61815). Keep `--no-enable-prefix-caching` on GDN+MTP and the SSM cache at fp32 or default. On v0.33.0, add the SYCL-vs-Triton GDN backend (#56835) as a fingerprint axis. |
| **S06** load time + caches | **nothing** | No XPU ingest PR with numbers; Dockerfile.xpu sets no NEO/Triton/vLLM cache variables at v0.33.0 or main **[confirmed]**. NEO defaults per compute-runtime master docs: `NEO_CACHE_PERSISTENT=1`, dir `$XDG_CACHE_HOME` or `~/.cache/neo_compiler_cache`, `NEO_CACHE_MAX_SIZE=1GB` **[unverified]** (one researcher read it; the skeptic did not re-fetch) | **Unchanged.** Replace "verify the defaults" with these values and still check them on the box. Point `HOME`/`XDG_CACHE_HOME` (or `NEO_CACHE_DIR`) at the named volume, or the cache lands in the ephemeral layer. Raise `NEO_CACHE_MAX_SIZE`, which matters more if the 1.74 GB full-config attention .so is ever used. Main's loader rewrites (#57678, #64692, #64357) change the profile on the next upgrade. |
| **S47** kernel wheel | **partly** (attention tuples) / **nothing** (lnl AOT, `is_bmg`) | The GH v0.1.16 full-config wheel, which vLLM main pins, probably contains the Gemma 4 head-256/512 causal+local tuples **[confirmed]** (circumstantial). AOT has no lnl; `is_bmg()` gate unchanged **[confirmed]**. `is_xe2_arch()` includes PVC, which the gate excludes for accuracy **[confirmed]**. | **Shrink.** If S45 shows the only live csrc patch is an attention tuple or `chunk_prefill_full.conf`, first try the GH v0.1.16 wheel (torch 2.14) on v0.33.0, and measure host RSS, load time and first-kernel JIT, since the attention .so is about 5× larger *est.* **Fix the patch condition** to `is_bmg() \|\| arch == intel_gpu_lnl_m`. Before building that, A/B `--gdn-prefill-backend sycl` vs `triton` on v0.33.0 with S37 KL. Build from tag **v0.1.16.2**, never main or 0.1.16.3+. Keep `lnl-m` in `VLLM_XPU_XE2_AOT_DEVICES`. Target the PR at v0.1.18; v0.1.17 cuts 11-04. |
| **S02** cached_tokens probe + wire shim | **nothing** (probe and shim) | Backend selection changed (#62014), but nothing in the probe is upstream **[corrected]**: the researcher said "partly", the skeptic said not_started. SWA layers follow `--prefix-cache-retention-interval` (default 0, semantic checkpoints only) **[confirmed]** in cache.py. Server-side wire bugs: `parameters=None` and `tool_choice:required` needing a parser **[unverified]**. Client-side fixes qwen-code #12222 and #13919 are open **[unverified]**. Prefix-cache CI on XPU is still disabled (#62656 open) **[unverified]** | **Shrink.** Before writing the shim, reproduce both 400s with curl against v0.32/v0.33 with the gemma4 and qwen3_coder parsers loaded (about 5 min). If they don't reproduce, drop the shim. Run three turns, not two, so junction pinning shows. Log the retention interval. On v0.33.0 with multimodal on and FLASH_ATTN forced, record the "image inputs will be wrong" warning. |
| **S17** prefix warmer | **partly** (MTP block-drop fix only) | On v0.33.0 #61589 removes the EAGLE trailing-block drop by default (no KV connector); #64808's default case is fixed **[confirmed]**. SWA retention 0 means an identical resend hits, but a warmer followed by a different suffix diverges inside an unretained SWA prompt **[confirmed]** (default value; behaviour [unverified] on XPU) | **Reorder.** Change the verification from an identical resend to a **different-suffix** request. Either set `--prefix-cache-retention-interval` to a block multiple (e.g. 1024) or None, or warm twice with different suffixes. On v0.32.0 keep MTP off on the warmed lane. On v0.33.0 MTP is fine unless a KV connector or `disable_lookahead_block_hashes` is used. |
| **S51** replay bench | **nothing** | No agent replay bench upstream; #51155 is open **[unverified]**. With the flag off, v0.33.0 returns no or null `prompt_tokens_details` **[confirmed]** (code) | **Unchanged.** Treat a missing or null `prompt_tokens_details` as "flag off", not zero. Log the retention interval and image digest in the run header. |
| **S07** sleep/wake | **partly** | XPU `xpumem.py` is byte-identical v0.32.0→v0.33.0; the `freed_bytes >= 0` assert has a 5 s grace on ROCm only **[confirmed]**. MRV2 level-2 model state is lost on XPU (#63548 excludes XPU; #64057) **[confirmed]**. `--sleep-preserve-parameter-names` is in v0.32.0; `--sleep-mode-backend` is in v0.33.0 **[confirmed]**. #64630 (main) keeps a weight-sized pinned backup after a level-1 wake **[confirmed]**. No iGPU sleep test anywhere **[confirmed]** | **Still conditional on S08. Shrink if it runs.** Level 2 only. Run the parity check on default MRV2 **and** `VLLM_USE_V2_MODEL_RUNNER=0` (time-limited by #63373). Replace "worker extension that reloads the drafter" with `--sleep-preserve-parameter-names 'draft.*'`. If a UMA fix is needed, deliver it as a `--sleep-mode-backend` plugin, which needs v0.33.0. Stay off main for this (#64630). Watch #62277 (OOM with `--kv-cache-memory-bytes` + sleep, CUDA report) **[unverified]**. |
| **S21** NPU contention | **nothing** | No OVMS, OpenVINO or NPU-driver release since Oct 17 **[confirmed]** (PyPI side). OVMS on NPU is Stateful-only, and `enable_prefix_caching` is ignored there; models must be INT4 channel-wise sym; `--max_length` caps NPU memory **[unverified]**. Fedora/Nobara NPU driver packaging could not be checked **[unverified]** | **Unchanged** (parked, week 4). Use OVMS 2026.4.1 or wait for 2026.5; add `--max_length`. Confirm a Qwen3-1.7B int4-cw-ov model exists (HF blocked). For T14, target stateful `WhisperPipeline('NPU')`. |
| **S52** parity twin | **nothing** | Kernels #640 adds graph-mode unit tests, but no LNL parity tests **[confirmed]**. XPU batch-invariant mode rejects quantized models **[unverified]**. Open #58131 (forward `num_splits` to XPU FA; bs=1 vs bs=N logprob mismatches) **[unverified]** | **Unchanged.** Keep the twin `--enforce-eager`. If #58131 lands in a tree you move to, re-take the A/A noise floor, since FLASH_ATTN determinism changes. |
| **S45** patch-set dry run | **partly** | v0.33.0 pins and drivers equal v0.32.0 **[confirmed]**. The Python diff that matters is broader than attention selection: Gemma4Config + FA ≤512 + FA cudagraph ALWAYS, `--tool-strict-level` default, #61589, #63138, the parser changes, the #61183 function move and the ARK default **[confirmed]**. The official v0.33.0 image exists **[confirmed]** | **Reorder.** Dry-run against the **v0.33.0 tag and image**, not main. Take the site-packages diff straight from `vllm/vllm-openai-xpu:v0.33.0`. Classify the #40128 patch as "already upstream" without a dry run. Expect the Gemma4Config hunks to be obsolete. Re-check the eager opt-out flag in the dry run. |
| **S46** plugin overlay | **partly** | The `MemorySnapshot.measure` wrap target is stable through v0.33.0 **[confirmed]**; main's #56492 changes the caller path **[corrected]** (it is not "applies unchanged to main"). The #40128 wrap is unnecessary, and keyed on `kv_cache_utils` it would silently miss on v0.33.0 (#61183) **[confirmed]** | **Shrink.** Drop the #40128 wrap. Keep the UMA wrap only if S10 fails. Add a sleep-backend plugin slot only if S07 needs it. Re-test the overlay on the release after v0.33.0. |
| **#40128 cherry-pick** | **done** | PR closed unmerged by its author 2026-04-21. Equivalent padding landed via #39931 (2026-05-04) + #45181 (2026-06-21), first in v0.24.0. v0.32.0 already pads MambaSpec and non-MLA attention pages **[confirmed]** | **Skip.** Move `xpu_hybrid_moe_page_size_pr40128.patch` to `attic/`. **Do not post** `issues/pr-40128-comment-draft.md`. If KV capacity on hybrid models matters, watch open #63347 instead. |

---

## 5. Revised first-week sequence

The plan does not stand as written. Changes are in bold. Everything else is as in the 2026-10-29 plan.

**Today (before Day 1, 10 min): #40295 decision.** [2026-11-11: did not happen; #40295 is still open and stale; the decision moves to after the Day-1 S10 run, work plan v2.] Its stale window ends about 2026-11-03. Either post a short holding comment ("re-measuring on v0.32/v0.33 with driver-side usable-memory accounting; will update or close with readings") or let it auto-close and reopen later as "driver-independence for compute-runtime <26.18" if S10 fails. Do nothing on #41370 beyond an optional comment, since it is not the owner's PR.

**Day 1 (console, v0.32.0):**
- S02 probe + request capture, **three turns**.
- S10 (45 min), **adding a `torch.xpu.mem_get_info()` reading**.
- S06 launch/microbench/neo-cache check, **with `HOME`/`XDG_CACHE_HOME`/`NEO_CACHE_DIR` pointed at the volume and the cache defaults read on the box**.
- **Optional, 5 min: curl repro of the two wire-bug 400s with a parser loaded.**

**Day 2:**
- S50 prompts; Spark S37a captures.
- Console: S45 diff, **dry-run against the official `vllm/vllm-openai-xpu:v0.33.0` image (pins identical to v0.32.0)**. **Drop the #40128 patch from the list without testing.**

**Days 3–4 (v0.32.0):**
- S51/S42 A/A baselines on Gemma 4 and Qwen3-Coder.
- S12a FP8 KV on Qwen3-Coder; S12c headless delta; S05 formula check.
- **S05 computes the 0.94/450 MB formula and cross-checks it against live `getMemoryInfo`. Do not hard-code it** [corrected per skeptic].

**Day 5:**
- **S15 with three arms (`ngram` V2, `ngram_gpu` V2, `ngram` V1).** Do the V1 arm now; draft #63373 would remove it. [2026-11-11: #63373 merged into v0.34.0rc0; work plan v2 drops the V1 arm and the GDN smoke.]
- **Plus a 20-minute n-gram smoke test on the Qwen3.6 GDN lane, watching for the #54740 RuntimeError.**

**End of week 1: cutover decision.** If the S45 dry run is clean, rebuild the lanes on v0.33.0 with:
- eager pinned explicitly (`--enforce-eager` or `cudagraph_mode NONE`);
- `--attention-backend FLASH_ATTN` kept as an explicit pin on Gemma 4;
- `VLLM_XPU_INC_WNA16_BACKEND=w4a16` on any INC/AutoRound lane;
- kernels pinned to `==0.1.15.4`.

Re-take the A/A noise floor on v0.33.0 before S16. If the dry run is not clean, stay on v0.32.0 for S16 phase 0. [2026-11-11: superseded; work plan v2 moves the cutover to Day 0/Day 1 so all baselines are taken on v0.33.0 once.]

**Week 2 changes:**
- S12b runs first on **stock v0.33.0** (no home kernel).
- S17 verification uses a **different-suffix** request.
- S37b's INT4-lm_head arm either carries #53104 or is replaced by an FP8-lm_head arm.

---

## 6. Watchlist for the next check

**vLLM**
- [#63977](https://github.com/vllm-project/vllm/pull/63977): torch 2.15 + triton-xpu 3.9.0 + kernels 0.1.16.3 (GH full config) + OMIX 0.4.1 driver. Draft, last head 10-31. It is the next container-stack change.
- [#63373](https://github.com/vllm-project/vllm/pull/63373): removes Model Runner V1 and `VLLM_USE_V2_MODEL_RUNNER`. Draft, needs-rebase, 11-02. Kills the S15 V1 arm and the S07 MRV1 parity run. [2026-11-11: merged, in v0.34.0rc0.]
- [#64897](https://github.com/vllm-project/vllm/pull/64897): XPU greedy_argmax NaN fix for the main regression #57770. Open, 10-31. [2026-11-11: merged on main Nov 9 as 0f0b352, in no tag; v0.34.0rc0 contains #57770 without it, so no spec decode on rc0.]
- [#64687](https://github.com/vllm-project/vllm/pull/64687) + kernels [#670](https://github.com/vllm-project/vllm-xpu-kernels/pull/670): eager Qwen3.5 port + fused MoE decode. FP8-per-tensor only, accuracy drops reported. Open, 11-02.
- [#61630](https://github.com/vllm-project/vllm/pull/61630): XPU head-256 tuned Triton prefill. Approved 10-23, open; changes the Triton half of the Gemma 4 composite.
- [#58131](https://github.com/vllm-project/vllm/pull/58131): XPU FA `num_splits` forwarding. Approved and ready 11-02; affects determinism and A/A.
- [#61731](https://github.com/vllm-project/vllm/pull/61731) (asym GPTQ dense on XPU), [#52652](https://github.com/vllm-project/vllm/pull/52652) (TD int4 fused_moe as XPU default), [#61815](https://github.com/vllm-project/vllm/pull/61815) (FP8 lm_head), [#49539](https://github.com/vllm-project/vllm/pull/49539) (BF16 MTP experts in GPTQ, approved 11-02), [#61916](https://github.com/vllm-project/vllm/pull/61916) (non-uniform REAP), [#48552](https://github.com/vllm-project/vllm/pull/48552) (AWQ MoE repack, stale, auto-close about 11-22).
- [#63347](https://github.com/vllm-project/vllm/pull/63347): grow padded full-attention blocks; KV capacity on heterogeneous pages.
- [#62656](https://github.com/vllm-project/vllm/pull/62656): re-enable prefix-cache e2e tests on XPU.
- [#65107](https://github.com/vllm-project/vllm/pull/65107): mirror images to the `vllm/vllm-xpu` alias.
- [#65154](https://github.com/vllm-project/vllm/pull/65154) / [#64396](https://github.com/vllm-project/vllm/pull/64396): XPU batch invariance.
- [#64479](https://github.com/vllm-project/vllm/pull/64479): reuse GDN metadata on XPU.
- [#60539](https://github.com/vllm-project/vllm/pull/60539): llm-scaler eager decode fusions, draft.
- Open graph bugs: [#54785](https://github.com/vllm-project/vllm/issues/54785) (batch-invariance label 11-02), [#54698](https://github.com/vllm-project/vllm/issues/54698), [#63620](https://github.com/vllm-project/vllm/issues/63620) (GDN NULL_BLOCK NaN), [#53997](https://github.com/vllm-project/vllm/pull/53997) (structured outputs + graphs, unmerged).
- [#53912](https://github.com/vllm-project/vllm/issues/53912) (MTP+APC on GDN) with fixes [#48375](https://github.com/vllm-project/vllm/pull/48375) / [#57128](https://github.com/vllm-project/vllm/pull/57128).
- [#62781](https://github.com/vllm-project/vllm/issues/62781) (fp16 SSM GDN on XPU), [#63621](https://github.com/vllm-project/vllm/issues/63621) (XPU KV-offload double staging; kernels [#662](https://github.com/vllm-project/vllm-xpu-kernels/pull/662)), [#64808](https://github.com/vllm-project/vllm/issues/64808) / [#65263](https://github.com/vllm-project/vllm/pull/65263) (Gemma 4 MTP prefix hits).
- [#49760](https://github.com/vllm-project/vllm/pull/49760): UMA host reserve, 8 GiB default, gated on `is_integrated_gpu`. Would cost the Claw 8 GiB *est.* if it merged together with an XPU `is_integrated_gpu`.
- [#63666](https://github.com/vllm-project/vllm/issues/63666): RFC, MoE expert offload on single-GPU XPU incl. iGPU.
- [#40295](https://github.com/vllm-project/vllm/pull/40295): auto-close about 11-03.

**vllm-xpu-kernels**
- v0.1.17 tracker [#676](https://github.com/vllm-project/vllm-xpu-kernels/issues/676): cut 11-04, release 11-06, torch 2.15; whether #660 and #662 make it.
- [#644](https://github.com/vllm-project/vllm-xpu-kernels/pull/644) (INT4 GEMV MoE decode; will need a rebase onto the MoE rewrite), [#638](https://github.com/vllm-project/vllm-xpu-kernels/pull/638) (FP8-KV prefill on Xe2), [#605](https://github.com/vllm-project/vllm-xpu-kernels/pull/605) (Xe2 grouped-GEMM prefetch, updated 11-03), [#389](https://github.com/vllm-project/vllm-xpu-kernels/issues/389) / [#391](https://github.com/vllm-project/vllm-xpu-kernels/pull/391), [#593](https://github.com/vllm-project/vllm-xpu-kernels/issues/593), [#606](https://github.com/vllm-project/vllm-xpu-kernels/pull/606).
- Whether any torch-2.14 maintenance release (0.1.16.x) follows.

**Runtime**
- [compute-runtime releases](https://github.com/intel/compute-runtime/releases): first release after 26.35.39758.10, carrying the graph-replay sync fixes and LNL ULLS d322804.
- PyTorch [#202396](https://github.com/pytorch/pytorch/pull/202396): DLE 2026.2 / OMIX 0.5.0. That is the build that would activate #195987's iGPU compensation; re-run S10 then.
- [intel/llvm](https://github.com/intel/llvm/issues) V2-adapter issues that name LNL.
- OpenVINO/OVMS 2026.5 with genai [#4667](https://github.com/openvinotoolkit/openvino.genai/pull/4667) (CB + paged KV on NPU) and openvino #38606 (continuous prefill default).
- Fedora `intel-npu-driver` ≥ 1.38.

---

## 7. Corrections to the previous status reports

### Sep 2 report (`vllm-xpu-status-2026-09-02.md`)
- **"N-gram: broken on GDN models (#54740 assert)" and "`TORCH_CHECK(spec_token == …)` still present"** were true for 0.1.14.1 and are **outdated for the owner's 0.1.15.4**. The check is now `<=` (kernels #600, release-branch cherry-pick f4b729a, confirmed in the PyPI binary on 2026-11-03). #54740 is still open and untested on hardware. **[confirmed]**
- **"Model Runner V2 default (falls back to V1 for `ngram`)" and "`ngram_gpu` uses torch.cuda.* → unusable"** → since v0.32.0 (#40704) both `ngram` and `ngram_gpu` run the V2 Triton `NgramGPUSpeculator`. The CPU numba path is V1-only. **[confirmed]**
- **"#499 getMemoryInfo fallback, unreleased"** → it shipped in v0.1.15 and is in 0.1.15.4. **[confirmed]**
- **"Your PRs: #40295 and #41370"** → #41370 was written by arnokamphuis. Only #40295 is the owner's. **[confirmed]**
- **"GDN `is_bmg()` → `is_xe2_arch()`" (PR candidate)** → wrong replacement: `is_xe2_arch()` includes PVC, which the gate excludes for accuracy. Use `is_bmg() || arch == intel_gpu_lnl_m`. **[confirmed]**
- **"Gemma 4 MTP: `torch.cuda.CUDAGraph` hard-coded in `gemma4.py`"** → narrower than stated: only the V1 proposer, and only for ordered-embedding drafts. The V2 `Gemma4Speculator` used by default on XPU has no torch.cuda calls. **[confirmed]**
- **"Grouped-GEMM race fixes #457/#524 OPEN"** → #586 fixed it (in 0.1.15+), and #457 was closed unmerged on 2026-11-02. #524 was not re-checked. **[confirmed]**
- **#45540** is listed only as a source in the Sep 2 report. The work plan calls it a "stale corruption report". It was actually closed by its reporter as completed on 2026-06-21 ("fixed in latest vLLM + kernels 0.1.10"), and it was a TP=2 dGPU GDN case. **[confirmed]**

### Oct 9 report (`xpu-graph-status-2026-10-09.md`)
- **"FLASH_ATTN on XPU reports `UNIFORM_BATCH`"** → true for v0.31/v0.32. From **v0.33.0** it reports `ALWAYS` (#62244), so graphs-on can capture mixed batches FULL. **[confirmed]**
- **"`Gemma4Config` forces TRITON_ATTN on XPU (workaround stays)"** → still true on v0.32.0. On v0.33.0 the force is removed (#62014): text-only Gemma 4 defaults to FLASH_ATTN, and multimodal defaults to the Triton/FA composite. Keep the explicit flag as a pin. **[confirmed]**
- **"The kernels repo has zero graph-capture/replay tests"** → #640 (2026-10-22) added graph-mode variants of the attention, MoE, grouped-GEMM, GDN, MLA and MQA-logits tests. **[confirmed]**
- **Watchlist "#524/#457 (close expected)"** → #457 closed 2026-11-02. **[confirmed]**
- **Watchlist "compute-runtime next release carrying the Oct 1/7 graph replay sync fixes (26.40-ish)"** → there has been no release since 26.35.39758.10 (09-17). The 26.40.40196.x tags carry no graph commits after Sep 23. **[confirmed]** (no release); tag contents **[unverified]**
- **#53993 "closed Sep 24 as linked to #51600"** → two researchers (B5, B8) independently report it was closed on **Aug 27** via the reporter's fork PR (CySpiegel/vllm-intel#8). The upstream fix #53997 is still unmerged either way. **[unverified]** (conflicting dates; the conclusion is unchanged)
- **#54698 "retest on 2.14 pending"** → the reporter ran 90 min clean on torch 2.14 + vLLM 0.31.0 with MTP off on 10-10, but on a different model (Qwen3.8-27B), and called it possibly inconclusive. The issue is still open. **[unverified]** (not re-fetched by a skeptic)

### Work plan (`stack-work-plan-2026-10-29.md`), for completeness
- S15 "CPU n-gram", "never on GDN", and the "#41370: close it" action: corrected as above.
- The #40128 row: the PR is closed and the work is upstream, so skip it.
- S46's #40128 wrap: drop it.
- S47's `is_xe2_arch()`: fix the condition.
- S05 "hard-code with confidence": cross-check against a live reading instead.

---

## 8. What could not be fetched

- **api.github.com and `gh api`** for vllm-project/*, pytorch and intel/* returned 403 for every researcher and skeptic ("GitHub access to this repository is not enabled for this session"; add_repo was denied). Data came from `git ls-remote`, blobless or treeless clones, raw.githubusercontent.com, WebFetch of github.com pages, PyPI and Docker Hub. PR timelines and labels come from summarised HTML, so exact comment timestamps are not fully verified.
- **wheels.vllm.ai** (DNS/CONNECT 403): whether an XPU wheel exists for 0.33.0, and what the nightly XPU index contains, is unverified.
- **vllm-project.github.io/vllm-xpu-kernels/whl** (CONNECT 403): the exact wheel behind main's `vllm_xpu_kernels==0.1.16` was not inspected. "Full attention config" is inferred, and the kernels build workflow was not read.
- **download.pytorch.org** (CONNECT 403): the torch 2.15 XPU and `whl/test/xpu` indexes were not listed.
- **huggingface.co**: Gemma 4 and Qwen3-Coder chat templates (so whether a template rejects `parameters=None`), the z-lab DFlash draft config, REAP-25B `num_experts`, the Ornith-1.5 cards, and the OpenVINO NPU model collection.
- **repositories.intel.com, dgpu-docs.intel.com**: the compute-runtime and IGC versions inside OMIX 0.4.1/0.5.0.
- **packages.fedoraproject.org, src.fedoraproject.org, copr.fedorainfracloud.org, bugzilla.redhat.com**: Fedora/Nobara NPU driver packaging.
- **benchlm.ai, gigazine.net, llmrun.dev** (DNS blocked). docs.vllm.ai and docs.openvino.ai were not used.
- **GitHub code search** needs a login, so repo-wide searches (GPUReclaim, block pointers in all of vLLM, getMemoryInfo callers) were partial.
- **v0.33.0 release notes**: the release page had no body, and whether it carries the Latest badge is conflicting (see §2.1).
- Only **page 1** of the GitHub issue/PR searches was read (e.g. 25 of 139 open XPU-titled PRs, page 1 of the intel-gpu label). Smaller items may be missing.
- The **vllm #62851** body was truncated, so its platform is unknown. The **#62781** body was not read by a skeptic.
- The local vllm-xpu-kernels **partial clone** has a damaged tree object, so a git log grep for the #600/#586 cherry-picks failed there. The #600 question was settled instead from the PyPI binary (by the B3 skeptic and by me on 2026-11-03).
- **Not re-verified by any skeptic**: #54698, #64177, #64336 limitations, #51155, #61415, #63621 numbers, kernels #644/#638/#662 review states, COMPILER_CACHE.md defaults, intel/llvm V2-adapter code, the OVMS/NPU-driver release pages, openvino.genai #4795/#4667, the triton-xpu 3.9.0 notes, the torch 2.15 runtime pins, and the compute-runtime tag contents.
- **No hardware run** backs any statement here. All behaviour on the Arc 140V (FLASH_ATTN for Gemma 4, the n-gram fix on GDN, sleep mode, the full-config kernel wheel, JIT vs `64.bmg` load) is untested.
