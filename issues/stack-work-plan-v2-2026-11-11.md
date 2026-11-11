# Stack work plan v2 for the Claw 8 (2026-11-11)

Supersedes `issues/stack-work-plan-2026-10-29.md`. Inputs: that plan, `vllm-xpu-status-2026-11-01.md`, the two 2026-11-11 reports (`llamacpp-vs-vllm-memory-2026-11-11.md`, `vllm-linux-vs-llamacpp-windows-2026-11-11.md`), the 2026-11-11 upstream check, six reviews and two skeptic passes. A change is in this plan only if at least one skeptic accepted it and neither rejected it. Where the skeptics disagreed, the resolution is stated.

**Number labels.** *(here)* = measured on this console, with the stack named when it is not v0.32/v0.33. *(other: X)* = measured elsewhere, on hardware or OS X. *(est.)* = estimate, arithmetic or config-derived. All KV-per-token figures are config-derived (est.) until the "GPU KV cache size: N tokens" startup line confirms them. The 21 GiB "practical ceiling" is a device-side figure (LUNAR rule 3). The 2.5 GiB floor applies to host MemAvailable. The 2.29 GiB gap between MemAvailable and XPU free is the driver guard (0.06 × MemTotal + 450 MB). It limits device allocation and is **not** consumed RAM.

---

## 1. What changed and why

| # | Item | Old plan | New plan | Evidence | Impact |
|---|---|---|---|---|---|
| 1 | Stack for week 1 | Baselines on v0.32.0. Cut over at end of week 1, then re-take A/A before S16 | **Decision U on Day 0/Day 1.** S45 dry run on the laptop against the `vllm/vllm-openai-xpu:v0.33.0` image, plus a 15-min console startup smoke. If clean, every week-1 launch and S16 run on v0.33.0. If not, freeze weeks 1–2 on v0.32.0. The "re-take A/A before S16" step is deleted | v0.33.0 `requirements/xpu.txt` pins torch 2.14.0 / triton 3.8.0+xpu / kernels 0.1.15.4, the same as v0.32.0. No v0.33.1 tag (upstream check 11-11). #54163 (DFlash prefix block) is in v0.33.0 only | Saves 4–6 launches. S16 gets a matching noise floor and #54163 |
| 2 | S16 memory budget | 17 + 1.7 + 0.7 KV at 16K ≈ 19.4 GiB | 17 + 1.0–1.7 draft + 1.5 KV (16K bf16) + ~0.2 draft KV + 0.8–1.3 act ≈ **20.5–21.7 GiB est.**, at or over the ceiling. Phase 0 runs at **8K with pinned KV bytes**. 16K needs S12a (fp8) or a measured headless fit. 32K needs S13 (REAP) | Qwen3-Coder is 96 KiB/token bf16 (48 layers × 4 KV heads × 128; LUNAR "Current status"; llamacpp-vs-vllm "KV cache at 32K") | Stops a weekend from going into an OOM or a gate (d) failure built into the budget |
| 3 | Spec-decode tag rule | "Stay off main while #64897 is open" | **Never run S15, S16, MTP or S37b on v0.34.0rc0 or main.** The first torch-2.15 tag allowed for spec decode must contain 0f0b352 (#64897) | Clone: #57770 (the regression) is reachable from v0.34.0rc0. #64897 merged on main Nov 9 and is in no tag | Prevents NaN-corrupted spec results |
| 4 | torch-2.15 generation | Not in the plan | **Freeze on v0.33.0 through week 4.** A Cutover 2 gate opens when a non-rc tag contains 0f0b352. No prep is pre-scheduled | rc0/main: torch 2.15, triton 3.9.0, kernels 0.1.16.3/0.1.17, OMIX 0.4.1. V1 removed (#63373), gpu_worker rewrite (#56492), pinned sleep backup (#64630) | Keeps baselines valid for the whole four weeks |
| 5 | A/A and gate baselines | A/A undefined ("close enough"). Gates in absolute tok/s from 18 / 27.69 | A/A = 2 launches × 3 reps from the same pre-state. Tolerance = max(cross-launch Δ, 2 × within-launch SD). **Decision N.** S15/S16/S37b gates are ratios to v0.33 medians. 18 / 23 / 36–40 s are relabelled *(here, v0.19.0, 2026-04)*. 27.69 is relabelled *(here, v0.30, MXFP4, historical)* | `vllm-v19-lunar-lake-findings.md` header (vLLM 0.19.0, torch 2.10); `xpu-graph-status-2026-10-09.md:108-109` | Every gate becomes something that can fail, with the right baseline |
| 6 | Memory state and leak | Reboot when the leak is > ~1 GiB. S12c after the A/A. S10 after a launch | **Day-1 memory ladder** with no model loaded: desktop-on → multi-user.target → drop_caches → cat shard. Then a **leak gauge** after the first SIGTERM (**decision L**). S10's reverted launch starts from the same pre-state. All memory gates run headless | v19 Finding 3 (expanded): a graceful stop dropped XPU free from 28 to 7–8 GiB on v0.19. Never re-measured | Sets launches per day and makes A/Bs clean |
| 7 | OVMS on Linux | Not considered (OVMS only as an NPU server) | **Decision O.** A 1-h OVMS GPU gate on Day 5 after S15, before the S16 weekend. ≥1.4× the v0.33 vLLM median on the 8K replay turns, with tool calls parsed → postpone S16 and evaluate OVMS in week 2 | OVMS 38 tok/s median, 31 accuracy-safe, Qwen3-Coder-30B-A3B INT4 *(other: 258V, Windows, ~2.2K ctx)*. 15.7 tok/s at ~11.7K *(other: OpenVINO, 16 GB 258V)* | One hour decides whether a weekend of first-ever XPU DFlash is worth running |
| 8 | S13 REAP | Week-2 Spark quantize. Brainstorm called it "low value" | **Prerequisite for S16 at 32K.** Spark export starts in week 1 after S37a. Numeric pass rule | Unpruned + draft + 32K fp8 ≈ 20.5–21.7 GiB est.; REAP ≈ 17.5–18.7 est. | S16-32K comes a week earlier and is possible at all |
| 9 | Gemma 4 KV | ~1.5 GiB at 32K, 64K "at ~1.5 GiB" | **Unknown.** b (KiB/token) is read on Day 1 at pinned bytes. S12b targets 64K only if b ≤ ~32–40 KiB bf16 | The 40–48 KiB figure is GPT-OSS (llamacpp-vs-vllm correction 1); plan §7 item 4 already listed it as unmeasured | Removes an unsourced number from the S12b, brainstorm §3 and harness-study budgets |
| 10 | S15 n-gram | "CPU n-gram". V2 + V1 runs. Never on GDN. Chat bar ≥0.97× | Two V2 arms (`ngram` sync, `ngram_gpu` async), k≤2, pinned KV bytes. **V1 not measured** (removed in v0.34). GDN smoke **not run**. Chat ≥ 1 − max(0.03, decode A/A tolerance) | Status 11-01 §1/§4 S15 row (V2 Triton speculator, #600 in 0.1.15.4); #63373 merged into rc0 | About 1 h saved and no crash/reboot. The chat gate stops tripping on noise |
| 11 | S37b Qwen3.6 foundry | gs128 + INT4-lm_head A/B in week 2. Pass ≤19 GiB vs 27.69. Gain "+16K ctx" | Export **gs128 experts-only only**. The INT4-lm_head A/B moves to Cutover 2 (#53104 is torch-2.15 only; net ≈ +0.27 GiB est.). No FP8-lm_head arm until #61815 is in a tag. Pass vs a **measured week-3 Qwen3.6 A/A**. Gain stated in GiB (≈0.4–0.5 est.) | #53104 in rc0 only, #61815 unmerged; brainstorm line 231 (gs32+BF16 MTP ≈ 21.1, gs128 ≈ 19.6 est.); 20 KiB/token ⇒ 1 GiB ≈ 52K tokens | Avoids exporting a checkpoint that cannot be loaded in week 3 |
| 12 | Gemma 4 attention kernel | "Must keep" | Keep **unless the stock-v0.33 zero-patch test passes**: the Day-1 smoke shows no reference-attention fallback, then week-2 S12b KL/decode inside A/A on the stock lane | #62014 (v0.33) removes the Triton force; status 11-01 §3.1, §4 S12/S47 rows | May delete the only must-keep patch |
| 13 | Live patch set | est. 4–5. #40128 row and S46 wrap. Post the #40128 draft | est. **2–3** (UMA, Gemma 4 kernel, oneDNN if it is code). #40128 obsolete, draft marked do-not-post. **S46 skipped if S10 passes** | Patch is already in `vllm/patches/attic/`; `issues/pr-40128-comment-draft.md` is headed "OBSOLETE … do not post"; LUNAR patch table | Less work in Thread 6 |
| 14 | #40295 / #41370 / #49760 | Close #41370. Decide #40295 "today". Reframe on an `is_integrated_gpu()` hook | #41370 is not the owner's PR (optional comment only). **#40295 is a Day-1 output of S10.** No `is_integrated_gpu` override unless the same PR sets the #49760 reserve to 0 for lnl_m | Brief delta (#40295 still open, stale); status 11-01 line 44; #49760 unmerged (clone) | Avoids losing ~8 GiB est. to an upstream reserve |
| 15 | S47 kernel build | Unconditional trigger list; GDN `is_bmg()→is_xe2_arch()` | Only if S45 finds a live csrc patch or the stock-v0.33 Gemma 4 check fails. Build from v0.1.16.2 for the v0.33 lane, end of life at Cutover 2. Gate fix `is_bmg() || arch == intel_gpu_lnl_m`. Upstream PRs target kernels main | `is_xe2_arch()` includes PVC (status 11-01 §4 S47); v0.1.17 is torch-2.15 only (#621, #583/#664/#669) | Prevents a wrong upstream PR and a wheel with no future |
| 16 | S05 formula and accounting | `21 − resident − 2.5` (GiB, not tokens; floor double-counted). Brainstorm adds 2.3 "reserve" as usage | KV_budget = util × 28.57 − W − D − A; max_model_len = ⌊KV_budget / b⌋. Check MemAvailable ≥ 2.5 GiB separately, and also W+D+A+KV ≤ MemAvail_at_launch − 2.29. The 2.29 row is relabelled the driver guard | pr-xpu-integrated-gpu-body.md:49 (30.86 → 28.57); LUNAR rule 3 | Context limits become computed token counts |
| 17 | S07 sleep/wake | Gemma 4 sleep probe if S08 needs two models; MRV1 parity arm | Still only if S08 needs two models. Run a **15-min llama-server switch-time gate first** (≤15 s and decode ≥0.9× vLLM → skip S07). Otherwise: v0.33.0, level 2, MRV2 only, `--sleep-preserve-parameter-names 'draft.*'`, headless, awake coder KV 2.25 GiB bf16 / 1.5 GiB fp8 | #63548 (XPU MRV2 level-2 state loss), #64630 in rc0 (weight-sized pinned backup on UMA); llama-server README router flags | |
| 18 | S17 / brainstorm Exp. 4 | An identical second request proves APC | A third turn with a different suffix (~500 appended tokens). Log `--prefix-cache-retention-interval` and the backend per batch type | Status 11-01 §4 S17/S02 rows (SWA retention default 0) | Avoids a false "APC works" |
| 19 | S16 gates (e) and time | (e) "two restarts without reboot". No stop rule | (e) relative to the no-spec lane, within 0.5 GiB after drop_caches. **4-h time box** to the first valid bench number. No local patching of dtype, JIT or NaN failures (logged as "expected fixed with #63436/#63819/#64897") | v19 Finding 3; #63436/#63819 in rc0, not v0.33 | Bounds the weekend |
| 20 | S13 / S08 / S21 gates | No number / no tolerance / a range | S13: KL ≤ unpruned + 3 × quant_kld A/A p99, top-1 ≥ unpruned − 1 pt, resident ≤ 14.5 GiB. S08: ≥4/5 tasks, 0 unparsed streamed tool calls in ≥20, 22K TTFT within A/A of the other model or better. S21: drop < 10% | Plan lines S13/S08/S21; brainstorm Exp. 6 | Gates can fail |
| 21 | replay.py | vLLM-only | `--base-url` (allows /v3), client-side TTFT/TPOT from SSE, cached tokens from usage else null, per-turn tool-call-parsed flag, turn-type tag (tool-call vs text), API-server CPU time. S42 stays vLLM-only | OVMS serves /v3; #64675 (qwen3_coder quadratic re-parse) is in rc0, not v0.33 | OVMS gate reuses the tool. Gates the #64675 backport |
| 22 | FP8 KV target | Brainstorm aims it at Qwen3.6 | FP8 KV on Qwen3-Coder and Gemma 4 only. On Qwen3.6 it saves ≤0.6 GiB at 64K | 10 full-attn layers × 2 KV heads × 256 ⇒ ~20 KiB/token est. | |
| 23 | Bench hygiene | `rpm -q` + digest | Add torch / vllm_xpu_kernels / triton versions, session state (headless/desktop) and MemAvailable to every header. Never add the GH-pages kernels index to a torch-2.14 lane. A GH wheel only by exact asset URL after checking torch-2.14 metadata | 0.1.16.4 has no Requires-Dist on torch (status 11-01 §2.3); main/rc0 use the GH-pages index | Catches silent ABI mismatches |
| 24 | Service architecture text | llama-swap fronting a "≤6 GiB GGUF" fallback | Text only. The GGUF side is llama-server router mode (`--models-max 1 --sleep-idle-seconds 600 --cache-ram 2048`), a full-size GGUF, never co-resident with vLLM. Nothing scheduled | llama-server README; IQ4_XS + q8_0 32K ≈ 17.4–17.9 GiB device est. | |

**Rejected or not adopted, and why**
- **GDN n-gram #54740 smoke on Qwen3.6.** Skeptic 2 rejected it: it decides nothing (that lane uses MTP, and only one spec method runs at a time), and the expected crash leaks memory and costs a reboot.
- **Full three-engine bake-off S55** (llama.cpp coder arm, S55b Gemma-on-OVMS, outcome table, engine-parametric S15/S16, S16 gate (c) vs the "best non-vLLM arm", §7 items 16–18, per-engine provenance beyond the one OVMS run). Skeptic 2 rejected these: they would be weeks of work resting on one unverified number. The idea was reduced to the 1-h OVMS gate (decision O).
- **Qwen3.6 MTP on llama-server, and a current-llama.cpp re-bench as a fallback/gaming lane.** Rejected: no MTP GGUF is verified, there is no Lunar Lake figure, and the gaming lane is operations. llama.cpp is measured only as the 15-min switch-time gate before S07.
- **Qwen3.6 fused routed/shared MoE decode trial on kernels 0.1.17.** Both skeptics put it on the watchlist only: #670 hard-requires `float8_e4m3fn` weights (kernels 6a59a3d), was validated as FP16 TP2 on a dGPU, and its vLLM caller #64687 is unmerged. The owner's lane is INT4.
- **Conflicts resolved, and how:**
  - **V1 n-gram arm.** Skeptic 1: optional, last. Skeptic 2: drop. *Dropped* (a V1 result ships only on a frozen lane, and S15 is expected to be a discard).
  - **Cutover 2 entry.** Skeptic 1: rc or final with #64897. Skeptic 2: non-rc with #64897 and kernels ≥0.1.17. *A non-rc tag containing 0f0b352*, which satisfies both. Record the kernels pin. Treat a pin below 0.1.17 as a flag to re-check, not as a gate.
  - **Week-4 Cutover-2 prep and the rc0 informational dry run.** *Not pre-scheduled* (skeptic 2). They run when the qualifying tag exists.
  - **S16 k=2/k=3 arms.** Skeptic 1 wanted them. *Not added* (skeptic 2). Add the "expert traffic does not scale with k; unverified" caveat instead.
  - **S16 at 32K.** Skeptic 2: S13 AND S12a. Skeptic 1: S13 required, S12a preferred. *S13 required. fp8 KV if S12a passed (17.5–18.7 est.). bf16 (19.0–20.4 est.) only with a measured MemAvailable ≥ 2.5 GiB steady.* This keeps skeptic 2's "gate on measured MemAvailable".
  - **Qwen3.6 A/A.** Skeptic 1: week 1. Skeptic 2: week 3. *Week 3*, just before S37b.
  - **OVMS gate placement.** *Day 5 after S15* (skeptic 2).
  - **Leak protocol.** *SIGTERM gauge on planned stops only, no deliberate SIGKILL* (both).
  - **Host accounting.** *Gate on measured MemAvailable.* MemAvailable-left estimates are not carried in the plan (skeptic 2).
  - **Kernels guard.** *Recording only, no refusing launcher* (skeptic 2).
  - **GPT-OSS baseline.** *Not run* (not a planned lane).

---

## 2. Decisions to make first

These are cheap readings that decide which engine and which threads are worth the four weeks. They come in time order. Every one of them is measured on this console; none exists yet.

| ID | Measurement | Cost / when | Threshold | Switches |
|---|---|---|---|---|
| **U** | S45 `git apply --check --3way` of the live set against the v0.33.0 image (laptop). Then a console startup smoke: Gemma 4 text-only (eager, explicit `--attention-backend FLASH_ATTN`, **no home kernel**, kernels ==0.1.15.4) and Qwen3-Coder | Day 0 laptop + 5-min `podman export` of the current container; 15 min console at the start of Day 1 | Dry run clean **and** neither log has "falling back to PyTorch reference attention" for Qwen3-Coder. A Gemma 4 fallback alone does not fail U; it fails Z | **Pass:** every launch from Day 1 on is v0.33.0, and "re-take A/A before S16" is deleted. **Fail:** weeks 1–2 frozen on v0.32.0, and S16 runs there (without #54163) |
| **Z** | Same smoke: does stock v0.33 Gemma 4 fall back for any head-256/512 batch type? | 0 extra | No fallback line | **Pass:** week-2 S12b runs on the stock lane. A KL/decode pass there deletes the home kernel, removes the S08 zero-patch tiebreak, and makes the S52 target a week-4 decision. **Fail:** try the GH v0.1.16 full-config wheel (exact asset, torch-2.14 metadata checked). If that also fails, the kernel stays and S47 may trigger |
| **M** | Memory ladder, no model loaded, after a reboot: MemAvailable, `getMemoryInfo(0)`, `torch.xpu.mem_get_info()`, desktop-on, then `systemctl isolate multi-user.target`, then drop_caches, then `cat shard > /dev/null` | ~1 h with S10, Day 1 | free ≈ MemAvailable − constant (~2.29 GiB expected), unchanged by page cache | Gives the headless gain (replaces "TBD 1.5–3.9 GiB est."), the S05 inputs and the state all memory gates use (headless) |
| **S** | S10 verdict: launch 1 with the UMA patch, launch 2 reverted from the same pre-state; diff "Available KV cache memory" | in the Day-1 block | KV lines within 0.1 GiB **and** M passes | **Pass:** delete the UMA patch, skip S46, close #40295 with the readings, size KV by `--kv-cache-memory-bytes`. **Fail:** keep the patch (as an S46 overlay) and refresh #40295 on post-#56492 main without an `is_integrated_gpu` override |
| **L** | Leak gauge after the first planned SIGTERM: free / mem_get_info / MemAvailable / Shmem before and after `sync; echo 3 > drop_caches`. The same readings after every later stop (hygiene) | 10 min, Day 1 | free back within 0.5 GiB of the post-boot baseline | **Pass:** separate arms with drop_caches plus a preflight check. **Fail:** reboot before every A/B arm, cap at ~8 launches/day (est. 3–4 min overhead per arm) |
| **B** | b = pinned KV bytes / N from "GPU KV cache size: N tokens". Gemma 4 on Day 1, Qwen3-Coder bf16 Days 3–4, fp8 in S12a, Qwen3.6 in week 3 | 0 extra launches | Qwen3-Coder 96 KiB ± 5%. Gemma 4: record it. If N looks inconsistent (sliding layers), cross-check at a second `--max-model-len` | Gemma 4 b ≤ ~32–40 KiB bf16 (≤ ~64–80 fp8) → S12b targets 64K, otherwise it stays at 32K. b feeds S05 and every budget in §3 |
| **N** | A/A per lane: 2 launches × 3 reps, same pre-state, includes `bench serve 1024/1024 ×3` and the S51 replay | Days 3–4 (Gemma 4, Qwen3-Coder); week 3 (Qwen3.6) | — | Tolerance = max(cross-launch Δ, 2 × within-launch SD). If decode tolerance > 5% or TTFT > 10%, every speed gate uses ≥3 launches per arm and is judged on the median |
| **K** | S12a FP8 KV on Qwen3-Coder at the same pinned bytes | 1 h, Days 3–4 | N ≈ 2×, and KL/top-1 Δ (fp8 vs bf16) inside the A/A tolerance | **Pass:** S16 at 16K (fp8), S04 coder YAML at 32K fp8, fp8 for S16-32K on REAP. **Fail:** S16 at 8K, or at 16K headless only if M shows room. S13 becomes the only route to 32K |
| **O** | OVMS GPU (pinned digest, `--device /dev/dri --cache_size 3 --tool_parser qwen3coder`, `OpenVINO/Qwen3-Coder-30B-A3B-Instruct-int4-ov`) on the S51 8K turns, vLLM stopped | 1 h console, Day 5 after S15. Downloads off-console on D2 | Median decode on the 8K turns ≥ 1.4× the v0.33 vLLM Qwen3-Coder A/A median, with tool calls parsed | **Pass:** S16 waits. Week-2 OVMS evaluation (22K TTFT, MemAvailable at 32K, KL via the optimum-intel adapter), and S08 includes OVMS-Qwen3-Coder. **Fail:** drop OVMS, run S16 as planned |
| **R** | S13 REAP-25B load/KL | week-2 weekday | KL ≤ unpruned + 3 × A/A p99, top-1 ≥ unpruned − 1 pt, resident ≤ 14.5 GiB | **Pass:** S16 at 32K on the REAP target (week 3). **Fail:** S16 tops out at 16K |
| **T** | Qwen3.6+MTP 22K replay TTFT | in the week-3 A/A | < 20 s | **≥ 20 s:** Qwen3.6 is "bots/short context", dropped from S08 as a J1 candidate |
| **W** | llama-server (Vulkan, current commit) stop/start switch time and decode for the second model | 15 min, week 4, only if S08 says two models | ≤ 15 s and decode ≥ 0.9× the vLLM A/A median | **Pass:** skip S07. **Fail:** run S07 (level 2, MRV2, headless) |

By the end of Day 1 these must exist: the S10 verdict, the cold/warm load split, leak class L and 9 captured turns. Anything missing is finished on Day 2 before anything else.

---

## 3. Threads, rewritten

All console items run on the stack chosen by U (v0.33.0 expected): eager pinned, `--attention-backend FLASH_ATTN` pinned on Gemma 4, `VLLM_XPU_INC_WNA16_BACKEND=w4a16` on INC/AutoRound lanes, kernels ==0.1.15.4, headless for every memory gate, and `--kv-cache-memory-bytes` pinned on every A/B.

### Thread 1: Yardsticks

1. **S50 real prompt set** (1 h, laptop, D2). About 30 prompts from the S02 capture and persona chats, secrets stripped → `real_prompts.txt`. Pass: it loads in `quant_kld.py capture` and covers 1K/8K/22K. Fail retires nothing (fallback is the 12 built-in prompts).
2. **S37a BF16 references** (~2 h Spark, D2). `python scripts/quant_kld.py capture --url http://spark1:8000/v1 --model <bf16> --prompts real_prompts.txt --max-tokens 256 --top-k 20 --out ref-<model>.json` for Gemma 4, Qwen3-Coder and Qwen3.6. Header records the tokenizer sha, template sha, thinking flag and vLLM version. Two tiers: a 40-prompt smoke set and a 500-prompt set. Pass: three `ref-*.json` files. Prereq: S50.
3. **S51 replay** (laptop D3–4, console from D3). Capture 9 turns (3 each at ~1K/8K/22K) in the Day-1 S02 session. `gates/replay.py` posts in session order, single stream. It takes `--base-url` (allowing /v3), measures TTFT/TPOT client-side from SSE, reads cached tokens from `usage.prompt_tokens_details.cached_tokens` (else null), and logs per turn a tool-call-parsed flag, a turn-type tag and the API-server CPU time (`ps -o time` before and after). **Pass:** decision N computed (definition in §2). Fail: N's ≥3-launch rule applies. Prereq: S02 capture.
4. **S42 /metrics diff** (30 min). Deltas for `vllm:spec_decode_num_drafts_total`, `num_draft_tokens_total`, `num_accepted_tokens_total`, prefix-cache hits/queries and `kv_cache_usage_perc`. **Pass:** accepted per draft token and accepted per step are computable on the **first S15 arm** (draft rate reads ~100% under fixed-K drafting). vLLM-only.
5. **Bench hygiene.** Every result header records `rpm -q kernel intel-compute-runtime level-zero`, the image digest, `torch.__version__`, the vllm_xpu_kernels and triton versions, session state, and MemAvailable before load, after load and after stop. Arms are separated per decision L. No suspend during runs. The GH-pages kernels index is never added to a torch-2.14 lane.
6. **S52 twin** (week 4, conditional). Target decided in week 4: Gemma 4 if Z/S12b kept the home kernel, otherwise Qwen3.6+MTP or deferred. Preflight MemAvailable ≥ twin_device + 2.0 (second Python process) + 2.5. Measure the twin's MemAvailable delta, not only "Model loading took". Pass threshold: 3 × A/A p99. Never beside the 18–20 GiB GDN models.

### Thread 2: Memory ceiling

1. **S10 + ladder** (Day 1, decisions M/S/L). Inside the U-selected container:
   ```
   python -c "import torch, vllm_xpu_kernels._C; print(torch.ops._C_cache_ops.getMemoryInfo(0)); print(torch.xpu.mem_get_info())"
   grep -E 'MemTotal|MemAvailable|Shmem' /proc/meminfo
   ```
   Take readings desktop-on, after `systemctl isolate multi-user.target` (= S12c), after `sync; echo 3 > /proc/sys/vm/drop_caches`, and after `cat <shard>.safetensors > /dev/null`. The native lane is skipped if no native lane serves traffic (`rpm -q intel-compute-runtime` recorded only). Pass/fail as S in §2. After deletion, size KV by `--kv-cache-memory-bytes`, not util.
2. **S05 fit arithmetic** (30 min, D4). KV_budget = util × 28.57 − W − D − A (A = 0.8–1.3 GiB est., Finding 7). max_model_len = ⌊KV_budget / b⌋ with b from decision B. Check separately that MemAvailable is ≥ 2.5 GiB at steady state and that W+D+A+KV ≤ MemAvail_at_launch − 2.29. Harvest "Model loading took" and "GPU KV cache size" per (model, image) into a manifest. Pass: the shell formula matches `getMemoryInfo` within ~0.1 GiB. Fail: keep the torch-based check. Prereq: S10.
3. **S12a FP8 KV on Qwen3-Coder** (1 h, D3–4, decision K). Same `--kv-cache-memory-bytes 2415919104`, with `--kv-cache-dtype fp8_e4m3` added. Read N, check for no fallback, and run `quant_kld.py compare` for the bf16 and fp8 runs against `ref-qwen3coder.json`. The KL gate is needed because FP8 KV scales default to 1.0 (no `calculate_kv_scales`). Gain: 32K KV 3.0 → 1.5 GiB est. Prereq: N, S37a.
4. **S13 REAP-25B** (Spark from D3; console week-2 weekday; decision R). Quantize `cerebras/Qwen3-Coder-REAP-25B-A3B` as CT sym INT4 gs32 and keep a BF16 REAP reference on the Spark. On the Claw, record "Model loading took", run compare vs `ref-qwen3coder.json`, the replay and bench. Uniform pruning, 20–25%. Expect E=102 to hit the `FallbackTopK` path: verify BF16 end-to-end first. Fail retires pruning as the coder's GiB source; S16 tops out at 16K. Prereq: Qwen3-Coder quant_kld A/A.
5. **S12b Gemma 4 FP8 KV** (week-2 weekday). On the stock v0.33 lane if Z passed, else on the home-kernel lane: `--kv-cache-dtype fp8_e4m3 --attention-backend FLASH_ATTN --max-model-len <32768, or 65536 if B allows>`, pinned bytes, fallback check, compare vs `ref-gemma4.json`. Record the backend chosen per batch type. On the stock lane this run is also the zero-patch KL/decode check. Fail on the home-kernel lane → an S47 kernel task.
6. **S37b gs128 Qwen3.6** (Spark week 1 after S37a; console week 3). Port `quantize_tiel.py` to `DEV='cuda'` and export experts-only sym INT4 gs128 with `ignore=['lm_head','re:.*mtp.*']`. Pass: resident ≤ (measured current checkpoint − 0.4 GiB), MemAvailable ≥ 2.5 GiB at `--max-model-len 65536`, MTP k=2 decode inside the week-3 A/A tolerance, KL inside the threshold vs `ref-qwen36.json`. Fail retires gs128 for this architecture. Gain ≈ 0.4–0.5 GiB est. gs32 only if weights + draft + KV ≤ ~19.5 GiB (it is not, at ~21.1 est.).
7. **Qwen3.6 64K reading** (week 3, folded into one A/A launch). Only if the measured Qwen3.6 resident is ≤ ~19 GiB. `--max-model-len 65536 --kv-cache-memory-bytes 1610612736 --no-enable-prefix-caching`. Pass: N ≥ 65538, MemAvailable ≥ 2.5 steady, decode inside A/A. Scope: bots and long reads, because re-prefill without APC is ~20–27 s at 20K est.

**Memory budgets (device side unless noted; all est. unless labelled)**

| Configuration | Weights | Draft | KV (figure used) | Act. | Device total | Verdict |
|---|---|---|---|---|---|---|
| Qwen3-Coder gs32, 24K bf16 (A/A, S04 until K) | 17.0 (here, v0.19-era) | — | 2.25 (96 KiB/token est.) | 0.8–1.3 | 20.05–20.55 | fits the device; pin 2415919104; MemAvailable from M |
| Qwen3-Coder, 32K bf16 | 17.0 | — | 3.0 (96) | 0.8–1.3 | 20.8–21.3 | at the ceiling; not used |
| Qwen3-Coder, 32K fp8 (after K) | 17.0 | — | 1.5 (48 fp8) | 0.8–1.3 | 19.3–19.8 | target S04 lane; pin 1610612736 |
| S16 phase 0, 8K bf16 | 17.0 | 1.0–1.7 | 0.75 + ~0.1 draft KV (96; draft ~12 KiB guessed) | 0.8–1.3 | 19.65–20.85 | first read; pin 939524096 |
| S16, 16K bf16 | 17.0 | 1.0–1.7 | 1.5 + ~0.2 | 0.8–1.3 | 20.5–21.7 | does not fit the practical ceiling; only headless if M and the first 16K launch show MemAvailable ≥ 2.5 |
| S16, 16K fp8 (after K) | 17.0 | 1.0–1.7 | 0.75 + ~0.1 | 0.8–1.3 | 19.65–20.85 | allowed after K |
| S16, 32K fp8, unpruned | 17.0 | 1.0–1.7 | 1.5 + ~0.2 | 0.8–1.3 | 20.5–21.7 | no (headless does not lower a device total) |
| S16 on REAP, 32K fp8 (after R+K) | ~14 (est.) | 1.0–1.7 | 1.5 + ~0.2 | 0.8–1.3 | 17.5–18.7 | week 3; pin 1879048192 |
| S16 on REAP, 32K bf16 (after R, K failed) | ~14 | 1.0–1.7 | 3.0 + ~0.4 | 0.8–1.3 | 19.0–20.4 | only with measured MemAvailable ≥ 2.5 |
| Gemma 4 INT4, 32K | 16.0 (here) | — | **unknown** (b from B) | 0.8–1.3 | 16.8–17.3 + KV_G4 | decided by B |
| Qwen3.6+MTP, 64K | 18–20 resident (here, older stack) | in resident | 1.25 + 0.06 GDN + ~0.12 MTP-layer (~20 KiB est.) | 0.8–1.3 | 20.3–22.8 | only if measured resident ≤ ~19 |
| Extra 32K Qwen3.6 sequence | — | — | ~0.63 + ~0.06 GDN | — | +~0.7 | (was "~1.6") |
| S52 twin (host check) | — | — | — | — | twin_device | MemAvailable ≥ twin_device + 2.0 + 2.5 |

Host side: vLLM process ~2.0 GiB *(here)*, CPU services ~0.74 *(est., brainstorm §3)*, OS + desktop ~3.9 *(here, inferred)*, headless gain TBD *(1.5–3.9 est.)* until M. The 2.29 GiB driver guard is not consumed RAM. Fits are judged against the measured MemAvailable, not the estimates.

### Thread 3: Decode speed on the coder lane

1. **S15 n-gram, V2 GPU speculator** (2 h, D5). Two arms on v0.33.0, identical `--kv-cache-memory-bytes`, prefix caching ON:
   `--speculative-config '{"method":"ngram","num_speculative_tokens":2,"prompt_lookup_min":4,"prompt_lookup_max":8}'`, then the same with `"method":"ngram_gpu"`. For each arm, grep the startup log for the speculator class, async scheduling, "Available KV cache memory" and "GPU KV cache size", and take an S42 delta over 3 prompts. **Pass:** code-edit replay ≥ 1.2× **and** chat ≥ 1 − max(0.03, decode A/A tolerance), on the median per decision N. Fail retires n-gram as a feature and leaves the acceptance bar S16 must beat. Gain ~0 GiB est. V1: not measured (removed in v0.34). GDN lanes use MTP; n-gram is not evaluated there. Prereqs: S51, S42, N.
2. **OVMS gate** (1 h, D5 after S15, decision O). vLLM stopped. Record the OVMS image digest and MemAvailable before load, after load and after stop. Week-2 evaluation on a pass only: 22K TTFT, MemAvailable at 32K, KL via an optimum-intel `OVModelForCausalLM(device='GPU')` logits adapter (~2 h laptop; trusted only after an adapter A/A gives KL ≈ 0 / top-1 ≈ 100%).
3. **S16 DFlash phase 0** (weekend after week 1; skipped if O passed). On the v0.33.0 lane:
   ```
   --speculative-config '{"method":"dflash","model":"z-lab/Qwen3-Coder-30B-A3B-DFlash","num_speculative_tokens":4}'
   --max-model-len 8192 --kv-cache-memory-bytes 939524096   # prefix caching ON
   ```
   Then k=5. Record "Model loading took" (the real draft GiB) and "GPU KV cache size" on the first launch. 16K only after K (fp8) or a measured headless fit. 32K only on the REAP target (week 3). **4-h time box** to the first valid bench number. eagle_ops dtype asserts, first-request JIT stalls or NaN output → stop, log "expected fixed with #63436/#63819/#64897", no local patch. EAGLE-3 30-min smoke only if DFlash starts. **Pass (all):** (a) KL/top-1 vs no-spec inside tolerance (not bit-exact); (b) mean acceptance ≥ 2.0 on a real Qwen Code session; (c) `bench serve 1024/1024` ≥ 1.4× the v0.33 Qwen3-Coder A/A median; (d) MemAvailable ≥ 2.5 GiB steady (headless) plus a 30-min soak; (e) after two stop/start cycles, post-stop free (after drop_caches) within 0.5 GiB of the no-spec lane in the same cycle. **Fail (c)** retires phase 2 and the NPU drafter. The gain (≥1.4× est.) assumes expert traffic does not scale with k; that is unverified. Prereqs: S15, N, U; K for 16K; R for 32K.
4. **Watchlist (not scheduled):** suffix decoding (#60339). Runs after Cutover 2, only if S15 failed the edit gate: one S15-harness run with `suffix`, k≤4. Pass: ≥ 1.2× edit and chat inside A/A.

### Thread 4: Harness overhead and TTFT

1. **S02 probe** (Day 1, launch 1). Gemma 4 with prefix caching ON, `--enable-prompt-tokens-details --enable-force-include-usage`. **Three turns** each from pi, Qwen Code and OpenClaw, with request logging on for S51. Pass: turn 2 ≥ 90% cached for every harness. Fail: name the harness. Static diets as in the old plan (OpenClaw `tools.profile: messaging`; Qwen Code auto-memory off ≈ −6K tokens/turn est.).
2. **#64675 check** (0 extra, D3–4 replay). Pass condition for doing nothing: Qwen3-Coder tool-call-turn TPOT is < 10% above text-turn TPOT at similar output length. If the gap is ≥ 10%, backport #64675 as a feature-detected Python overlay (a no-op when `_ClosedParams` exists) and re-run.
3. **S17 prefix warmer** (week 2, only if S02 passed). Turn 3 is a different-suffix request (~500 appended tokens). Log `--prefix-cache-retention-interval` and the backend per batch type. Pass: cached_tokens ≥ 90% on that turn. Baseline reference: 20,337 ms cold / 398 ms warm at 57.7% hit *(here, GPT-OSS, older stack; LUNAR:284-290)*.
4. **S18** (week 4, conditional) and **S34** (optional) are unchanged.

### Thread 5: Load and switch time

1. **S06** (Day 1, launches 1, 3 and 4). Launch 1 (cold page cache) runs with `VLLM_LOGGING_LEVEL=DEBUG`, timestamped phases and `py-spy dump`. This becomes the new load baseline; 36–40 s is *(here, v0.19)*. Launch 3 is warm, after `cat` of the shards. Then mount `NEO_CACHE_DIR`, `TRITON_CACHE_DIR` and `VLLM_CACHE_ROOT` as volumes keyed by image digest (with `HOME`/`XDG_CACHE_HOME` pointed at them, `NEO_CACHE_PERSISTENT=1`, defaults read on the box), and run launch 4 as the warm JIT cycle. The copy microbench runs last, then reboot. Decision: raw copy ≥ 15 s → the lever is the UMA alloc/copy path. Fail retires T10 if warm ≈ cold.
2. **S08 bake-off** (week 3, 1 day). Pass: ≥ 4/5 tasks, 0 unparsed streamed tool calls across ≥ 20 calls, J1 22K TTFT within A/A of the other model or better. Includes OVMS-Qwen3-Coder only if O passed. Excludes Qwen3.6 if T fails. The zero-patch tiebreak applies only if Gemma 4 still carries the kernel.
3. **W, then S07** (week 4, only if S08 needs two models). As in §2 W and changelog row 17. S12c (headless) is a prerequisite.
4. **S04 YAML** (week 3, settled lanes). Qwen3-Coder: `--max-model-len 24576 --kv-cache-memory-bytes 2415919104` until K, then `--max-model-len 32768 --kv-cache-dtype fp8_e4m3 --kv-cache-memory-bytes 1610612736`. Never size KV from util. llama-swap and quadlets wait on S08 (and O).

### Thread 6: Patch and kernel loop

1. **S45** (Day 0 laptop, decision U). Diff pristine vllm 0.32.0 / kernels 0.1.15.4 wheels against the exported site-packages. Write `vllm/patches-lnl/series` with `Upstream-PR:` / `drop_when:` headers. `git apply --check --3way` against the **v0.33.0 tag/image (never an rc)**. Classify each failure as file-missing, hunk-fail or fuzz. Pass: a definitive list, est. 2–3 live patches. The 14 v0.19 patches go to `attic/`. Radar: newest **non-rc** tag plus first tag containing 0f0b352.
2. **S46** (week 2–3). Skipped if S10 passes. Otherwise a `vllm.general_plugins` overlay wrapping `MemorySnapshot.measure` with feature detection. Gain: rebases 3 → 1–2.
3. **S47** (week 4, conditional). Triggers only if S45 finds a live csrc patch or the stock-v0.33 Gemma 4 check (Z/S12b) fails. Build from v0.1.16.2 for torch 2.14 off-console (laptop/CI; the Sparks are aarch64). Keep `lnl-m` in `VLLM_XPU_XE2_AOT_DEVICES` and gate GDN with `is_bmg() || arch == intel_gpu_lnl_m`. Before building, A/B `--gdn-prefill-backend sycl` vs `triton` with KL (#56835). End of life at Cutover 2. Upstream PRs target kernels main.
4. **S53** lints (1 h, week 3): unchanged.

### Parked: NPU

**S21** (week 4, optional). Pass if Gemma 4 decode drops < 10% while the NPU generates. ≥ 10% → NPU jobs only while the LLM is idle.

### Cutover 2 (torch 2.15): an unscheduled gate

Entry: a non-rc tag containing 0f0b352 (#64897). Record its kernels pin. Before it is adopted, #49760 must be absent, or set to 0 for lnl_m.

Steps, in order:
1. S45 `git apply --check` against that tag (laptop). This includes sizing the #61183 move, the #56492 caller path and the Gemma 4 kernel ABI.
2. One console day: re-run the Day-1 block (S10 + `mem_get_info`, ladder, b readouts, leak gauge) and the A/A per lane.
3. Only as lanes need them: S16, S12a, S15 V2 arms, the Gemma 4 checks (backend log, 8K/22K TTFT, S12b KL, S17 different suffix; #61630 changes the Triton prefill), S07 level 2 (level 1 is not viable with #64630), the INT4-lm_head arm (with #53104, budget +~0.95 GiB drafter head, net ≈ +0.27 est.), Qwen3.6 MTP k=2 vs the v0.33 median, S06 load/RSS/JIT with a new digest-keyed cache volume (the full-config attention .so is ~1.74 GB), and S47 build 2 from v0.1.17 if a home kernel is still needed.

v0.33.0 stays production until these pass.

---

## 4. Revised sequence

| Week | Console (serial, one model) | Spark | Laptop |
|---|---|---|---|
| **0** | 5-min `podman export` of the current container | — | Pull v0.33.0 image; S45 diff + series + dry run (U) |
| **1** | **D1:** reboot → ladder (M, incl. S12c) → 15-min v0.33 smoke (U, Z) → launch 1 Gemma 4, UMA patch, cold: S06 DEBUG + S02 3-turn capture + b → SIGTERM → leak gauge (L) → launch 2 reverted, same pre-state: S10 verdict (S) → #40295 action → launch 3 warm → cache volumes → launch 4 warm JIT → microbench → reboot. **D2:** finish any missing D1 output; optional 5-min wire-bug curl repro. **D3–4 (headless):** Gemma 4 A/A (2 launches) → Qwen3-Coder A/A at 24576 / 2415919104 (b) → S12a fp8 (K) → S05 → #64675 gap check from the replay. **D5:** S15 `ngram`, `ngram_gpu` → OVMS gate (O) → reboot | D2: S37a refs. D3→: S13 REAP export + BF16 REAP ref, then S37b gs128 export | S50; replay.py; S42; OVMS image + int4-ov IR download (D2) |
| **2** | **Weekend:** S16 phase 0 at 8K (4-h box; skipped if O passed). **Weekdays:** S13 load/KL (R); S12b on stock or kernel lane (Z, B decide the context); S16 at 16K if K passed; S17 if S02 passed; S02 diets; OVMS evaluation if O passed | (exports finish) | S46 only if S10 failed; optimum-intel adapter only if O passed |
| **3** | Qwen3.6+MTP A/A (2 launches, one at 64K if resident ≤ ~19, replay incl., T) → S37b load + MTP k=2 → S16 32K on REAP if R passed → S08 (1 day) → S04 YAML | — | S53; PR texts (§5) |
| **4** | W → S07 only if S08 needs two models and W fails; S52 (target decided now); T10 if S06 showed cold ≫ warm; S18 if S17 left a cold-prefix cost; S21 optional | S16 phase-2 scoping only if phase 0 passed | S47 build only if triggered |
| **Cutover 2** | When the qualifying tag exists: one re-baseline day, then lane re-runs as listed in §3 | — | S45 against that tag first |

Hygiene between A/B arms: identical pre-state per decision L (drop_caches + preflight, or reboot), headless, pinned KV bytes, version header. Any crashed launch is followed by a reboot. Launches stay at ≤ ~8 per day if L failed. No stack change from Day 1 to the end of week 4.

**Dependencies:**
- S02 → S50 → S37a → every KL gate.
- S02 → S51 → N.
- M/S/L → S05, S46, #40295, every memory gate.
- N → every pass threshold.
- Qwen3-Coder quant_kld A/A → S13.
- U → everything on the console.
- S15 → O → S16.
- K → S16-16K and S04-32K. R (+K preferred) → S16-32K.
- Z → S12b lane, S47, S52 target, S08 tiebreak.
- T → S08 roster. S08 → W → S07.
- S45 → S46/S47.

---

## 5. Patches and PRs

| Item | Status (2026-11-11) | Action |
|---|---|---|
| UMA accounting (`MemorySnapshot.measure`) | Live. Upstream memory accounting unchanged through main | Delete if S passes. Otherwise an S46 overlay |
| Gemma 4 attention kernel + FLASH_ATTN | Live. Stock v0.33 may cover it (#62014) | Keep unless Z + S12b pass. The override stays an S04 flag either way |
| oneDNN "Backends mismatch" fix | Unlocated; may be packaging (LUNAR) | S45 locates it. Python → S46; csrc → S47 + upstream issue |
| #40128 hybrid-MoE cherry-pick | Obsolete (upstream since v0.24; in `attic/`) | None. `pr-40128-comment-draft.md` is do-not-post |
| #40295 (owner) | Open, stale. Stale-window action did not happen | Day-1 output of S10: close with readings, or rebase on post-#56492 main without an `is_integrated_gpu` override unless the PR sets the #49760 reserve to 0 for lnl_m. Mark `pr-xpu-integrated-gpu-body.md` "premise stale" |
| #41370 | Not the owner's | Optional comment with the S10 readings only |
| #49760 (8 GiB UMA host reserve on `is_integrated_gpu`) | Unmerged | Cutover-2 blocker check |
| #64897 / #57770 | Fix on main only; regression in rc0 | Defines the Cutover-2 tag |
| #53104 / #61815 | In rc0 / unmerged | INT4-lm_head at Cutover 2; FP8-lm_head after #61815 is in a tag |
| #64675 qwen3_coder re-parse | In rc0, not v0.33 | Overlay only if the D3–4 gap is ≥ 10% |
| Kernels GDN gate | `is_bmg()` excludes lnl_m | PR `is_bmg() || arch == intel_gpu_lnl_m` against kernels main (not `is_xe2_arch()`, which includes PVC) |
| `lnl_m` case in `tests/test_get_memory_info.py`; decode planner `gpu_subslice_count` 8 vs 20 | Candidates | Write after S10 |
| #670 / #64687 fused MoE decode; #644 | fp8-only / unmerged / not in 0.1.17 | Watch for an INT4/W4A16 variant plus a merged caller |
| #63373 V1 removal | Merged (rc0) | No action; noted on S15/S07 |

---

## 6. Open measurements (deduplicated, ordered by how many steps each unblocks)

1. **Memory ladder** (M/S): free vs MemAvailable, the ~2.29 GiB constant, `mem_get_info`, headless gain. Unblocks S10, S05, S12c, S46, #40295, S16 16K-headless, S07, S52, every MemAvailable gate.
2. **A/A noise floor per lane** (N), cross-launch, including the v0.33 Qwen3-Coder/Gemma 4 decode, TTFT and TPOT medians. Unblocks every gate (S12a, S13, S15, S16, S37b, O, S08, S52) and the INT4-lm_head re-estimate.
3. **b per model** (B): Gemma 4 (unknown); confirm Qwen3-Coder 96 / Qwen3.6 ~20. Unblocks S05, S12b context, S16/S04 budgets, the brainstorm §3 total, the harness-study 64K claim.
4. **Leak per clean stop on v0.33** (L). Unblocks the launches/day budget, S16 (e), S07.
5. **FP8 KV tokens/GiB and KL on Qwen3-Coder** (K). Unblocks S16-16K, S04-32K, S16-32K KV dtype.
6. **Stock-v0.33 Gemma 4 fallback, KL and decode** (Z + S12b). Unblocks kernel deletion, S47, S08 tiebreak, S52 target.
7. **Cold/warm load split on v0.33** (S06). Unblocks T10, S47 AOT, the S07/S08 switch-time gains.
8. **OVMS 8K-turn decode on Linux** (O). Unblocks S16 go/postpone, S08 roster.
9. **Per-harness cached_tokens, and a different-suffix hit on Gemma 4** (S02/S17). Unblocks S17, S18, diets.
10. **n-gram proposer and acceptance per arm** (S15). Unblocks the S15 verdict and the S16 bar.
11. **REAP-25B resident and KL** (R). Unblocks S16-32K.
12. **DFlash acceptance and per-draft overhead on XPU**; draft GiB from "Model loading took" (S16). Unblocks phase 2.
13. **Served Qwen3.6 checkpoint name, resident, MTP k=2 median, 22K TTFT** (T). Unblocks S37b, S08 roster, the 64K reading.
14. **Tool-call vs text TPOT gap** on v0.33. Unblocks #64675.
15. Sleep level-2 GiB returned and wake time (S07, conditional); twin load/resident (S52); NPU co-run penalty (S21); llama-server switch time (W).

---

## 7. Corrections to make in the older documents

Exact find/replace edits are attached separately. In summary:

| File | Location | Change |
|---|---|---|
| `issues/stack-work-plan-2026-10-29.md` | Title | Add a "superseded by v2" note |
| same | TL;DR rows 3–4 | Gemma 4 KV unknown; S15 is the V2 GPU speculator |
| same | Bench hygiene, S51 Pass, S42 Pass | Decision L; A/A definition; validate S42 on the first S15 arm |
| same | S10 first step | v0.33 container; skip native half |
| same | S05 Pass | Token formula + separate MemAvailable check |
| same | S12 Extends/Pass/Fail/Gain | Per-model KV table; FP8-scale reason replaces #45540; 8K/16K-headless; GiB gains |
| same | S13 Pass | Numeric thresholds |
| same | S37b first step/Pass/Gain | lm_head arm to Cutover 2; measured baseline; GiB |
| same | S15 title/first step/Pass/Gain/cut | Two V2 arms; A/A-based chat bar; no GDN |
| same | S16 first step/(c)/(e)/Gain/Memory | v0.33, 8K pinned; ratio gate; relative (e); k caveat; 20.5–21.7 GiB |
| same | S17, S08, S07, S04, S21 | Different suffix + LUNAR:284-290; numeric pass; S07 conditions; router text; <10% |
| same | S45, S46, S47 | v0.33 tag; 2–3 patches; no #40128 wrap; gate fix and triggers |
| same | §3 week rows, Dependencies | INT4-lm_head out; 32K on REAP; #40295 only |
| same | §4 table and #40295/#41370 block | #40128 obsolete; Gemma 4 conditional; #41370 not the owner's; #49760 hazard; gate fix |
| same | §7 items 4, 8 | Gemma 4 unknown; S16 budget |
| `LUNAR_LAKE_COMPATIBILITY.md` | Stack table, rule 4, models table, load line, patches table | "do not use main or v0.34.0rc0"; clean-stop leak on v0.19; v0.19/v0.30 labels; Gemma 4 kernel conditional |
| `issues/service-design-brainstorm-2026-10-23.md` | TL;DR line 5, §3 budget table + notes, S10 line 140, S12 line 213, S13 value line 289, line 301, S19 line 379, watchlist line 358, Exp. 2/4/6 | Gemma 4 KV unknown; device/host split; ≤6 GiB fallback → router full-size; FP8 KV not for Qwen3.6; REAP value flipped; ~0.7 GiB; new watchlist; Exp. 2 superseded; different suffix; router. Regenerate `service-design-brainstorm-2026-10-23.html` from the .md |
| `issues/agent-harness-study-2026-10-22.md` | line 33 | 40 KiB/token is not Gemma 4's figure |
| `issues/llamacpp-vs-vllm-memory-2026-11-11.md` | TL;DR, "Measured tok/s here", worked-budget 2.3 row and bottom line | Stale-baseline caveat; v0.19 labels; driver guard relabel |
| `issues/vllm-linux-vs-llamacpp-windows-2026-11-11.md` | §2 rows, §3 KV line and headless line, §6 vLLM command | "(here)" → "(est.)" on arithmetic; v0.19 labels; headless TBD; 24576 pinned KV |
| `issues/vllm-xpu-status-2026-11-01.md` | §5 "Today", Day 5, cutover paragraph, §6 #63373/#64897 | Superseded notes (11-11 state) |
| `issues/pr-xpu-integrated-gpu-body.md` | Title | "premise stale; #49760 hazard" note |
