# Sub-30B MoE models for the Claw 8: progress review and line-up recommendation (2026-11-16)

Tags: [confirmed] = read on a page by a researcher or skeptic; [corrected] = a researcher's statement changed after checking; [unverified] = search snippet or single source only. Speeds and memory are "measured on this console" only where stated; everything else is est.

## 1. Short answer

- **Groot-19B-A3B is real** [confirmed]: GrootTeam/Groot-19B-A3B-261023, about 2026-10-23, Apache 2.0, a 19.4B cut of Qwen3.6-35B-A3B with about 3B active parameters. It is not NVIDIA GR00T and not a new pre-train.
- **"Better than Qwen3.6-35B-A3B at IQ2" is unverified.** Nobody has measured it, and the closest real measurement leans against it for knowledge and plain code generation. For multi-step tool calling there is no evidence either way.
- **On this console the question is moot.** Qwen3.6-35B-A3B fits at 3-4 bits under the 21 GiB ceiling, so IQ2 is never needed for single-model use.
- **Do not change the models; change the runtime.** Keep Qwen3.6 / Ornith-1.5 at about 4 bits as the coder and assistant, now as GGUF on a current llama.cpp build.
- **Two llama.cpp bugs matter more than the model choice:** the Qwen3.6 tool-call leak (#30118) and the Intel Vulkan prompt-corruption race (#32370).
- **Try first:** unsloth Qwen3.6-35B-A3B UD-IQ4_XS (17.7 GB) as the baseline, and Groot Q5_K_S (12.68 GiB) as an A/B challenger whose only sure benefit is 4-6 GiB of freed memory.
- Nothing else released since mid-2026 displaces Qwen3.6 in this class. LFM2.5-24B-A2B is worth a trial for the bots.

## 2. The Groot-19B-A3B question in full

**What it is** [confirmed]. Two skeptics cloned GrootTeam's GitHub repos (first commits 2026-10-24 +0800) and three unrelated repos that loaded or catalogued the weights.

- Lineage: "built from Qwen3.6-35B-A3B", post-trained on fewer than 100M tokens, "not training from scratch".
- Architecture: 40 layers (30 Gated DeltaNet, 10 full attention), 128 routed experts (top-8) plus one shared, 0.4B MTP layer, vision tower, 256K context, thinking on by default.
- [corrected] The maker never says "pruned" and publishes no method. The 128-expert figure comes from third-party READMEs, and "half of 256 experts removed" is an inference that fits the arithmetic.
- The team is anonymous: no people, company or paper found.

**Where the claim comes from.** No source compares Groot with any IQ2 file of Qwen3.6 [confirmed by all six researchers and both skeptics]. It looks like an inference from two facts:

- The sizes match: Groot Q4_K_M is 11.6-12.5 GB, and Qwen3.6 IQ2_M is 11.5-13 GB (sources disagree by about 1 GB) [unverified, snippets].
- The maker's chart (maker-reported, full precision) has Groot ahead on 12 of 16 benchmarks, average 41.44 vs 38.08, with 23% fewer output tokens.

The nearest maker statement is a capacity estimate ("on a 16GB card Groot runs 128K while Qwen3.6 does not load"), not a quality test.

**What the evidence supports.**

- The maker's chart does not carry the claim [corrected]. Its Qwen3.6 baseline is the maker's own run and scores 36.27 on SWE-bench Verified, against Qwen's published 73.4. Four benchmarks supply two thirds of the gap, and Groot still loses SWE-bench Verified (31.66 vs 36.27) and HLE on that chart.
- The best analogue points the other way (independent, single author, page read). On the predecessor Qwen3.5-35B-A3B at equal file size, the unpruned model at Unsloth UD-IQ2_M (10.61 GiB) averaged 91.4 on GSM8K, HumanEval and MMLU. A 128-of-256 expert prune at Q4_K_M (10.59 GiB) averaged 82.7. That prune had no retraining, so Groot should do somewhat better.
- "Qwen3.6 keeps about 87% at IQ2" [corrected: not re-found]. The measured loss for a good dynamic IQ2_M was 1.5 points.
- The MLX 4-bit comparison (78.37 vs 80.03; MMLU 70.6 vs 83.7) is uploader-run and [unverified]. Another snippet gives the same Qwen baseline as 76.78, which would reverse the order.
- The report of weak Groot tool calling (32 of 60 opencode replays) is [unverified]: one snippet, not re-found, and the tester compares against their own model.
- Independent leaderboard runs of Groot were still "queued" on 2026-11-16 [confirmed]. Worth re-checking in a week or two.

**The fair comparison for this owner** is Groot at Q5/Q6 (about 14.5-17.5 GiB resident, est.) against Qwen3.6 at UD-IQ4_XS (about 18 GiB, est.), plus Qwen3.6 UD-IQ3_XXS (13.2 GB) as the true size-matched rival. Nothing independent shows Groot beating either. Groot will not be faster: it has the same layers and active parameters.

## 3. What changed in sub-30B MoE since mid-2026

All rows are [unverified] snippet-level unless marked. Benchmarks are maker-reported unless marked independent.

| Model | Maker, date | Total / active | License | Context | Formats | Tool calling | Independent evidence | Verdict |
|---|---|---|---|---|---|---|---|---|
| Groot-19B-A3B-261023 | GrootTeam, ~2026-10-23 [confirmed] | 19.4B / ~3B | Apache 2.0 [confirmed] | 256K | Many GGUFs; BF16, FP8, NVFP4; no INT4 for vLLM-XPU found | Qwen3.6 template; no multi-step test | None finished | A/B candidate, not a default |
| LFM2.5-24B-A2B | Liquid AI, 2026-10-13 | 24B / 2-2.45B | LFM Open v1.0 (revenue cap reported by one third party) | 131K | Official GGUF incl. QAD Q4_0 | Lost calls in llama.cpp #30490 [confirmed]; fix unclear | None | Trial for bots after a lost-call test |
| Nemotron 3.5 Lightning 30B-A3B | NVIDIA, 2026-08-11 | ~30B / ~3B | OpenMDW-1.1 | 1M | GGUF (unsloth, ggml-org) | Validated on vLLM, not llama.cpp | None | Skip: trails Qwen3.6 on NVIDIA's own table |
| Ornith-1.5-35B-A3B | Ornith, Aug 2026 | 35.1B / ~3B | MIT | 262K | GGUF, GPTQ INT4 | Works in the owner's vLLM setup | None | Keep; over 30B but already fits |
| Gemma 4 26B-A4B (same-name update) | Google, April 2026 + later fix | 25.2B / 3.8B | Apache 2.0 | 256K | GGUF, QAT INT4 | Fixed by update; open streaming bug #32362 [confirmed] | Artificial Analysis: slightly below Qwen3.6 | Keep on vLLM; re-download weights |
| LFM2.5-8B-A1B | Liquid AI, ~2026-05-28 | 8.3B / 1.5B | LFM Open v1.0 | 128K | Official GGUF (Q4_K_M 5.16 GB) | Tokenizer re-uploaded for llama.cpp | None | Optional second resident bot model |
| Laguna XS 2.1 | Poolside, mid-2026 | 33B / 3B | OpenMDW-1.1 | 256K | Q4_K_M ~20.3 GB | Not found | None | Too big at Q4; llama.cpp support unclear |
| Mellum2.1 | JetBrains, 2026 | 12B / 2.5B | Apache 2.0 | 131K | GGUF "coming soon" | BFCL v4 62.3 | None | Too weak for the coding job |
| Qwen3.6 REAP prunes (19B-28B) | Community, 2026 | 19-28B / ~3B | Inherited (not confirmed) | 262K | GGUF, MLX | Untested | One app dropped a 19B prune for hallucinating more [confirmed] | Skip |
| Nex-N2-mini | Nex AGI, by June 2026 | 35B / ~3B | Apache 2.0 | 256K | GGUF | qwen3_coder parser | "Topped the leaderboard" [corrected: refuted; the plain Qwen3.5 base reportedly scores higher] | Do not rank on it |

Not found as of today: a GPT-OSS-20B successor, any Qwen3.6 or Qwen3.8 MoE below 35B-A3B, Gemma 4.5, a Phi-5 MoE, Granite 4.5. For the DGX Sparks, the dense Nanbeige4.3-33B (2026-10-24/25, Apache 2.0, self-reported scores) is the October release to look at.

## 4. Low-bit big model versus 4-bit smaller model

**The rule the evidence supports:** at equal file size, the bigger model at fewer bits wins down to about 2.6-3 bits per weight, provided the file is a dynamic quant that keeps attention and router tensors at higher precision.

Independent measurement on Qwen3.5-35B-A3B (page read; intervals about ±3 to ±6 points):

| File | Size | Average of GSM8K, HumanEval, MMLU |
|---|---|---|
| Unpruned Q4_K_M | not found | 92.9 |
| Unpruned UD-IQ3_XXS | 12.18 GiB | 92.2 |
| Unpruned UD-IQ2_M | 10.61 GiB | 91.4 |
| 128/256 prune, Q4_K_M | 10.59 GiB | 82.7 |
| 128/256 prune, IQ3_XXS | not found | 81.6 |
| 128/256 prune, IQ2_M | not found | 76.4 |

**Where it breaks:**

- **Tool-call formatting at 2 bits.** Qwen3.6 UD-IQ2_XXS dropped the opening `<tool_call>` tag on 8 of 10 seeds and solved 10 of 16 tasks, against 14 of 16 at 4-bit on a different engine [confirmed]. A lenient parser brought the same file to 14 of 16, so the damage is a recoverable tag, not lost reasoning.
- **"96% of tool decisions flip at 2 bits"** [corrected]. The summary that was read gives a median 3% at Q4_K_M, 10% at IQ3_XXS and 23% at IQ2_M, on dense models.
- **Retrained prunes.** Groot's post-training may recover part of the prune loss. This is unmeasured.
- **Intel Vulkan.** In #32370 the 2-bit file broke at a smaller micro-batch than the 4-bit one [confirmed].
- **Cross-model KLD.** KL divergence cannot compare Groot with Qwen; each is measured against its own BF16.
- Nobody has tested Groot below Q4, where prune loss and quantization loss stack.

## 5. Fit and speed on this console

Resident = file + about 0.6 GiB KV cache at f16 (arithmetic, not measured) + about 1 GiB buffers.

[corrected] The 16-22 tok/s estimate for current llama.cpp is not supported. The Vulkan kernel fix (#29476) raised token generation only 8.4% end to end, so the honest range is 13-20 tok/s until measured. Decode here is limited by per-layer overhead, not memory bandwidth: Gemma 4 (3.8B active) runs faster than Qwen3.6 (3B active) in the owner's own vLLM numbers.

| Candidate, quant | File size | Est. resident at 32K | llama.cpp decode | vLLM decode | Caveats |
|---|---|---|---|---|---|
| Qwen3.6-35B-A3B UD-IQ4_XS (unsloth) | 17.7 GB | ~18 GiB | 13-20 tok/s est. (Qwen3.5 IQ4_XS: 13.3 measured, March build) | 19, or 27.7 with MTP, measured (GPTQ INT4, 18-20 GiB) | #30118. bartowski IQ4_XS is 19.70 GB (~20 GiB, tight); Q4_K_M 22.29 GB does not fit |
| Qwen3.6-35B-A3B UD-IQ3_XXS | 13.2 GB | ~14 GiB | 13-20 est. | n/a | Tool calling at 3 bits untested here |
| Groot-19B-A3B Q5_K_S (TokenTinkerer) | 12.68 GiB | ~14.5 GiB | 13-20 est.; equal or slightly slower than Qwen3.6 IQ4_XS | None: no INT4 checkpoint found | Thinking on by default; same template, so #30118 likely applies |
| Groot-19B-A3B Q6_K | 15.6-16.8 GB | 16.5-17.5 GiB | As above or slower, est. | None | Same |
| Ornith-1.5-35B-A3B IQ4_XS | not found | ~18 GiB (assumed as Qwen3.6) | As Qwen3.6, est. | 19, or 27.7 with MTP, measured | GGUF sizes not found |
| Gemma 4 26B-A4B Q4 / QAT GGUF | not found | 16-18 GiB | 16-24 est., low confidence | ~30 measured (~16 GiB) | Needs a build from 2026-11-15; #32362 open |
| LFM2.5-24B-A2B QAD Q4_0 | not found | ~14 GiB | 21-28 est. (LFM2-24B: 20.7 measured, March build) | not found | Reasoning model; #30490 |
| GPT-OSS-20B MXFP4 | 12.1 GB | 13-14 GiB | 9.5 measured (March build); current build not measured | ~23 measured (12.9 GiB) | #30781 open |

The only Groot run on Intel Xe2 hardware (Arc Pro B60, SYCL) hit a GPU reset during the first decode [confirmed]. The same reset hit another model, so it is not shown to be Groot's fault.

## 6. Recommended line-up versus the old setting

| Job | Old setting | llama.cpp-first recommendation | vLLM checkpoint to keep |
|---|---|---|---|
| Coding agent (pi, Qwen Code) | Qwen3.6 / Ornith-1.5 GPTQ INT4 on vLLM | Same model as GGUF: Qwen3.6 UD-IQ4_XS, or Ornith IQ4_XS. Groot Q5_K_S only if it passes the test plan | Qwen3.6 / Ornith GPTQ INT4 with MTP (27.7 tok/s measured) |
| Telegram bots | Not fixed | Share the coder's server. If a separate fast model is wanted, trial LFM2.5-24B-A2B after the lost-call test | Gemma 4 26B-A4B QAT INT4 (~30 tok/s measured) |
| General assistant | Gemma 4 or Qwen3.6 on vLLM | Qwen3.6 at about 4 bits, which keeps the knowledge that pruning loses | Gemma 4 QAT INT4; GPT-OSS-20B MXFP4 as the small fallback |

**Preconditions on llama.cpp**, ahead of any model choice:

- **#30118 (open)** [corrected: worse than reported]. On b11382 with Qwen3.6 UD-IQ4_XS, 9 of 24 tool calls leaked into message content, about 37%, not 9%. Add a fallback parser in the harness, patch the build, or pin an older one.
- **#32370 (open, Meteor Lake; not shown on Lunar Lake).** Use `-ub 512`, and test a long prompt with flash attention on and off and with `GGML_VK_DISABLE_ASYNC=1` (about 5% slower).
- Keep the KV cache at f16; quantizing it reportedly hurts Qwen tool calling [unverified].

**Drop:**

- Any IQ2 file of a 35B model on this box: not needed, and the most fragile on Intel Vulkan.
- Qwen3-Coder-30B-A3B: superseded, with a KV cache of about 3 GiB at 32K (est.).
- GPT-OSS-20B on llama.cpp (9.5 tok/s measured); keep it on vLLM only.
- Nemotron 3.5 Lightning, Laguna XS 2.1, Mellum2.1 and Nex-N2-mini for this console.

If two models must be resident at once, Qwen3.6 UD-IQ3_XXS (~14 GiB) is the measured middle step. Groot Q5_K_S is the alternative to test.

## 7. Two-evening test plan

**Evening 1: baseline and runtime safety.**

- Build llama.cpp (Vulkan) from 2026-11-15 or later.
- Download unsloth Qwen3.6-35B-A3B UD-IQ4_XS (17.7 GB) and Groot Q5_K_S from TokenTinkerer (12.68 GiB), or bartowski Q5_K_M (14.72 GB).
- Run each with:

```
llama-server -m <file>.gguf --jinja -c 32768 -ngl 99 -b 2048 -ub 512
```

- Corruption check: send an 8K-token prompt and ask for an exact line back, 10 times each, with flash attention on and off, then with `GGML_VK_DISABLE_ASYNC=1`. Pass: 10 of 10 correct in at least one configuration.
- Speed: record tok/s on a short prompt and at about 24K tokens deep. Pass: 17 tok/s or better shallow (within 10% of vLLM's 19) and 15 or better deep. Below that, vLLM stays the coding runtime.
- Tool calls: replay 50 real tool-call turns from pi or Qwen Code. Count parsed calls, calls leaked into content, and text answers with no call. Pass: 48 of 50 parsed and 0 leaked once the fallback parser or patched build is in place.
- For Groot, set a generous `max_tokens` or `enable_thinking=false` in the chat-template arguments.

**Evening 2: the Groot decision and extras.**

- Add Qwen3.6 UD-IQ3_XXS (13.2 GB) as the third arm, same build and settings.
- Quality: run `quant_kld.py` only within a model family (UD-IQ3_XXS and UD-IQ4_XS against the best Qwen3.6 reference available, for example one produced on a Spark). Do not use it to compare Groot with Qwen.
- Personal task set of 20: 10 coding-agent tasks, 5 bot or memory questions needing general knowledge, 5 assistant questions. Groot passes if it is within one task of Qwen3.6 UD-IQ4_XS on coding, no more than one behind on knowledge, and within 2 of 50 on tool calls.
- MTP: ggml-org Qwen3.6-35B-A3B-MTP-GGUF with `--spec-type draft-mtp` against the plain file. Pass: at least 15% faster. MTP was a net loss in the one Intel measurement found (Arc A770, SYCL).
- Optional: LFM2.5-24B-A2B QAD Q4_0, 50 bot tool turns. Pass: no reply with `finish_reason: stop` and empty `tool_calls`.

## 8. What could not be verified

Hugging Face, arxiv.org, Reddit, dev.to, ollama.com and quantdiff.com were unreachable. The following rest on search snippets only:

- Groot's model card text, evaluation settings, the `qwen35moe` architecture tag and every GGUF file size, including the TokenTinkerer KLD ladder.
- All Qwen3.6 file sizes (unsloth and bartowski differ by about 1 GB per step) and all repository names in the test plan.
- The MLX 4-bit comparison and the EXL3 card's 32-of-60 tool-call result.
- The REAP paper (arXiv 2510.13999), arXiv 2607.16721 and arXiv 2608.06564. The last was read only through a GitHub issue summary.
- All LFM2.5, Nemotron 3.5, Ornith, Laguna, Mellum and Nex-N2-mini benchmarks, dates and licenses.
- The `--spec-type draft-mtp` flag and the minimum build for Groot's MTP head.

Also not found:

- Any measurement of any candidate on a Lunar Lake Arc 140V with a current llama.cpp build. The "~25 tok/s on other Lunar Lake machines" figure remains unmeasured.
- Whether #32370 reproduces on Lunar Lake.
- Any Groot INT4 checkpoint for vLLM-XPU.
- The origin of the claim itself. A Reddit thread could exist unseen.
