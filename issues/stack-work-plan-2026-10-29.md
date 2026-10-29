# Stack work plan for the Claw 8: re-cut of the brainstorm (2026-10-29)

Inputs: the 70 ideas that survived the first pass, with the feasibility skeptic's cut and the value skeptic's verdict for each, re-scored by three judges (performance, continuity, cheapest decisive test). The plan only uses numbers that are already in your docs. Anything nobody has measured on this console is marked **est.** Section 7 lists those numbers.

---

## 1. TL;DR

**What changed.** The first ranking asked what makes the service run itself, so ops plumbing came out on top: S39 sleep policy (7.0), T11 bot-memory backups (6.5), S04 quadlet manifests, S38 watchdog and S41 sentinel bot (6.33 each), S40 canaries (6.0). Under the new lenses none of those reach the top 8. S39, T11, S38 and S41 got out-of-scope votes from all three judges. The new top 8 scored only 3.2–4.75 in the first pass. The value skeptic rated most of them "low" because they are already decided or written down in your own notes. This plan agrees they are not new ideas. They rank high now because they are the open measurements and deletions left in work you have already done.

**Top 8 by consensus, and the number each one moves:**

| # | Idea | Mean | Number it moves |
|---|---|---|---|
| 1 | **S16** Stock DFlash draft on the Qwen3-Coder lane (phase 0 only) | 7.83 | Qwen3-Coder decode 18 → 25–27 tok/s est.; +1–1.7 GiB resident for the BF16 draft |
| 2 | **S10** Verify driver UMA accounting, delete the UMA patch | 7.83 | patch count −1; KV +0–0.5 GiB est. |
| 3 | **S12** FP8 KV + headless profile (UVA dropped) | 7.50 | KV tokens per GiB ~2× est.; Gemma 4 32K → 64K at ~1.5 GiB KV est.; +1.5–2 GiB host headless est. |
| 4 | **S15** CPU n-gram speculation on Qwen3-Coder | 7.43 | edit-turn decode 18 → ≥21.6 tok/s (that is the 1.2× gate); 0 GiB; most likely result is a discard |
| 5 | **S37** BF16 fingerprints, then the Spark quant foundry | 7.23 | makes KL/top-1 measurable for every change; Qwen3.6+MTP resident 18–20 → ≤19 GiB est. |
| 6 | **S51** Agent-shaped replay bench | 7.13 | TTFT at 8K/22K and decode on real turns (only 1024/1024 is benched today) |
| 7 | **S06** Load-time profile + persistent JIT caches | 7.10 | load 36–40 s → 25–30 s est. (~19 s may be the copy floor) |
| 8 | **S02** Per-harness cached_tokens probe + static prompt diet | 6.70 | cached/prompt tokens per turn per harness (unmeasured); Qwen Code auto-memory off ≈ −6K tokens/turn est. |

**First week:**
1. **Day 1 (console):** S02 two-turn probe, logging requests, so you record the 9 real turns S51 needs. S10 (45 min). S06 instrumented launch and copy microbench (≈1.5–2 h).
2. **Day 2:** S50 cut, `real_prompts.txt` (1 h, laptop). Spark: S37 BF16 captures for Gemma 4, Qwen3-Coder and Qwen3.6. Console: S45 wheel diff (2 h).
3. **Days 3–4:** S51 `gates/replay.py` and the S42 /metrics diff (laptop). Then A/A baselines on Gemma 4 and Qwen3-Coder to get the noise floor. S12 FP8 KV on Qwen3-Coder (1 h). S12 headless delta while docked (15 min).
4. **Day 5:** S15 n-gram (2–3 h). This clears the way for S16 over the weekend.

---

## 2. Threads

Six threads. Thread 1 builds the yardsticks the other five are gated on. NPU offload is parked at the end of this section with a single gating test.

### Thread 1: Yardsticks (quant_kld, replay, metrics)

Every gate in this plan compares against these, so they come first.

**S50: Real-traffic prompt set for quant_kld (setup only)**
- *Extends:* `quant_kld.py --prompts FILE`, decided gate 5 (personal eval).
- *First step (1 h):* take ~30 prompts from the S02 capture and from persona chats, strip secrets, and save them as `real_prompts.txt`.
- *Pass:* the file loads in `quant_kld.py capture` and covers the 1K/8K/22K mix.
- *Fail retires:* nothing. Without it, gates fall back to the 12 built-in prompts.
- *Gain:* KL measured on your own prompt distribution.
- *Skeptic cut:* the request WAL, Spark judge, blind A/B and shadow sampling are deferred. Before any live `logprobs:true` sampling, run a 10-prompt check that it does not change greedy output under MTP.

**S37a: BF16 reference fingerprints on the Sparks**
- *Extends:* the `quant_kld.py` capture/compare/report workflow; the Sparks as reference farm.
- *First step (~2 h Spark, runs alongside console work):* serve BF16 Gemma 4 26B-A4B (the QAT unquantized checkpoint), Qwen3-Coder-30B-A3B and Qwen3.6-35B-A3B one at a time with vLLM CUDA. For each, run `python scripts/quant_kld.py capture --url http://spark1:8000/v1 --model <bf16> --prompts real_prompts.txt --max-tokens 256 --top-k 20 --out ref-<model>.json`.
- *Pass:* three committed `ref-*.json` files.
- *Fail retires:* nothing.
- *Gain:* none directly. It is the precondition for every quality gate below.
- *Skeptic cut:* add tokenizer sha, chat-template sha, the thinking flag and the vLLM version to the header. Ship two tiers, a 40-prompt smoke set and a 500-prompt nightly set.

**S51: Agent-shaped replay bench**
- *Extends:* the `vllm bench serve 1024/1024 x3` recipe; the xpu-graph-status §7 checklist; the decided 5-gate suite.
- *First step (half a day, laptop plus console):* record 9 real requests (3 each at ~1K, 8K and 22K) during the S02 session with `--enable-log-requests` or a 60-line tee proxy into `agent-turns.jsonl`. Write `gates/replay.py` (~150 lines). It POSTs messages and tools in session order, single stream, to `/v1/chat/completions`, and logs TTFT, TPOT, `usage.prompt_tokens`, `prompt_tokens_details.cached_tokens` and `finish_reason` per turn as JSONL.
- *Pass:* two A/A runs on the same lane agree closely enough to set a tolerance. Record that tolerance.
- *Fail retires:* if A/A noise is large, single-run A/Bs become untrustworthy, and S15/S16 need 3+ runs each.
- *Gain:* TTFT at 8K/22K and decode on real turns become numbers.
- *Skeptic cut:* use the replay script for agent turns, not `--dataset-name custom`. Keep `bench serve --dataset-name random` for 128/1024/2048 decode. The nightly tier is the resident profile only, with no profile swaps.

**S42: /metrics diff around each run (minimal form)**
- *Extends:* vLLM /metrics; the `vllm/monitor/` Grafana stack already in the fork.
- *First step (30 min):* a 30-line script that snapshots /metrics before and after a run and prints deltas for `vllm:spec_decode_num_drafts_total`, `num_draft_tokens_total`, `num_accepted_tokens_total`, prefix-cache hits/queries and `kv_cache_usage_perc`.
- *Pass:* acceptance length is computable for an MTP run.
- *Gain:* acceptance and prefix-hit rate on every run.
- *Skeptic cut:* store series generically and do not build the SQLite SLO layer. Bring up the Grafana compose later only if you want history.

**Bench hygiene.** These are the only ops items kept, because they unblock measurement:
- Append `rpm -q kernel intel-compute-runtime level-zero` plus the image digest to every result line (the 15-minute kernel of S44).
- Record MemAvailable and `free -h` shared against the post-boot baseline after every stop (the measurement half of S09).
- Reboot between A/B pairs when leaked shared memory exceeds ~1 GiB, as the S51 cut prescribes.
- Keep the box from suspending during a run (the AC kernel of S39).

**S52: 3-layer twin for kernel parity (week 4)**
- *Extends:* `quant_kld.py`; the Bug H lesson (the INT4 MoE kernel passed isolated tests and crashed only in the pipeline after attention); the home Gemma 4 attention kernel.
- *First step (weekend):* `make_twin.py` for Gemma 4 26B-A4B INT4, keeping embeddings, lm_head, final norm and one sliding plus one global layer, with `quantization_config` intact. Serve it with `--enforce-eager --max-num-seqs 1 --max-model-len 16384 --kv-cache-memory-bytes 268435456 --gpu-memory-utilization 0.15`, then run `quant_kld capture` twice (A/A).
- *Pass:* ready in ~30–45 s est., a 10-prompt × 64-token battery in 2–4 min est., A/A noise floor recorded. Set the gate threshold at ~3× the A/A p99.
- *Fail retires:* the "beside production" use. The twin stays an idle-window tool.
- *Gain:* kernel-parity turnaround drops from a 36–40 s load at 16–20 GiB to 2–3 GiB resident, est.
- *Skeptic cut:* never run it beside the 18–20 GiB GDN models. Run it beside Gemma-class models only after a preflight shows MemAvailable ≥ twin + 2 GiB. Capture one reference per stable image.

### Thread 2: Memory ceiling (21 GiB)

**S10: Verify driver UMA accounting and delete the UMA patch**
- *Extends:* the UMA `MemorySnapshot.measure` patch; Finding 5 in `vllm-v19-lunar-lake-findings.md`; open question `vllm-xpu-status-2026-09-02.md:188`.
- *First step (45 min, inside the frozen v0.32.0 container and again on the native lane):*
  ```
  python -c "import torch, vllm_xpu_kernels._C; print(torch.ops._C_cache_ops.getMemoryInfo(0))"
  grep -E 'MemTotal|MemAvailable' /proc/meminfo
  ```
  Take readings before and after `sync; echo 3 > /proc/sys/vm/drop_caches`, and after `cat <shard>.safetensors > /dev/null`. Then launch Gemma 4 once with the patch reverted and diff the "Available KV cache memory" line.
- *Pass:* total ≈ 28.57 GiB, and free ≈ MemAvailable minus a constant that page cache does not change. The skeptic expects ~2.29 GiB, which is the driver formula globalMemSize×0.94 − 450 MB − (MemTotal − MemAvailable). The KV line also matches the patched run.
- *Fail retires:* the deletion. The patch stays, and #40295 keeps its "underreporting" justification.
- *Gain:* patch count −1; possibly +0–0.5 GiB KV from a tighter margin est.
- *Skeptic cut:* drop the vmtouch/fadvise half (at most a one-off A/B, never in the unit). If the goal is "page cache goes before swap", set `vm.swappiness=10`. After deletion, lower `--gpu-memory-utilization` by ~0.08 or let `--kv-cache-memory-bytes` carry the budget. Keep the patch on the native lane until `rpm -q intel-compute-runtime` on Nobara shows ≥ 26.18. The container's user-mode driver (UMD) is the image's, not the host's, which is why both lanes are checked.

**S05: Preflight fit arithmetic (after S10)**
- *Extends:* the <18 GiB gate in `~/bin/vllm-qwen30b`, `check_vram()` in `run-model.sh`, the usable-memory formula.
- *First step (30 min):* compute the formula in shell from /proc/meminfo and compare it with `getMemoryInfo` across the three S10 states.
- *Pass:* they agree within ~0.1 GiB est. Then harvest "Model loading took X GiB" and "GPU KV cache size: N tokens" per (model, image) into a manifest, and compute max `--max-model-len` = 21 GiB − resident − 2.5 GiB floor.
- *Fail retires:* host-only preflight; keep the torch-based check.
- *Gain:* usable context per model becomes a computed number, re-baselined after FP8 KV and REAP.
- *Skeptic cut:* no torch in the preflight. Use a delta-based leak gauge, not an absolute one.

**S12: FP8 KV, then headless. UVA dropped.**
- *Extends:* `--kv-cache-dtype fp8_e4m3` (native on Xe2 since v0.20, #37731); the 40–48 KiB/token table; the home Gemma 4 FLASH_ATTN kernel.
- *First step, part a (1 h):* relaunch Qwen3-Coder (stock GQA attention) with `--kv-cache-dtype fp8_e4m3` at the same `--kv-cache-memory-bytes`. Read "GPU KV cache size: N tokens" before and after, check there is no reference-attention fallback, and run `quant_kld.py compare` for both the bf16-KV and fp8-KV runs against `ref-qwen3coder.json`.
- *First step, part b (1 h, week 2):* Gemma 4 with `--kv-cache-dtype fp8_e4m3 --attention-backend FLASH_ATTN --max-model-len 65536`, with the same log check and compare against `ref-gemma4.json`.
- *First step, part c (15 min, docked):* `systemctl stop display-manager`, record the MemAvailable delta.
- *Pass:* N ≈ 2× est. for the same bytes. The KL/top-1 delta between the fp8 and bf16 runs stays inside the A/A tolerance (one stale corruption report, #45540, is the reason to check rather than assume). Gemma 4 starts at 64K with no fallback.
- *Fail retires:* on Qwen3-Coder, FP8 KV as a memory lever, so S16 stays at 16K. On Gemma 4 only, a failure turns into a kernel task in Thread 6 (FP8 KV support in the home kernel) and 64K on Gemma 4 waits for it.
- *Gain:* 64K costs ~1.5 GiB instead of ~3 GiB est.; +1.5–2 GiB headless est.
- *Skeptic cut:* UVA (`--cpu-offload-gb`) is dropped as a production path, since your status doc §6 refutes any capacity gain. The headless profile is one-way for docked or night use. Prerequisite: the gateway and bots move to system units or get linger enabled. The INT4 lm_head belongs to S37b.

**S13: REAP-pruned Qwen3-Coder (adopt cerebras REAP-25B)**
- *Extends:* the compressed-tensors sym INT4 gs32 recipe (`tiel-coder-xpu/quantize_tiel.py`); quant_kld; Sparks as foundry.
- *First step (one Spark evening plus 1 h on the console):* quantize `cerebras/Qwen3-Coder-REAP-25B-A3B` with llm-compressor or AutoRound, sym INT4 gs32, exported as compressed-tensors. Load it on the Claw, record "Model loading took X GiB", then run `quant_kld compare` against `ref-qwen3coder.json`, the S51 replay and bench.
- *Pass (fix these thresholds before starting):* resident ≈ 14 GiB est.; mean KL and top-1 within a set delta of the unpruned INT4; personal eval ≥ baseline − 1.
- *Fail retires:* pruning as a GiB source on the coder. Fall back on FP8 KV.
- *Gain:* −3 GiB est. at unchanged decode, enough for the DFlash draft plus 32K KV est.
- *Skeptic cut:* Qwen3-Coder only (uniform pruning, since vLLM reads a single `num_experts`). Target 20–25%, not 35%. Never AutoRound a QAT model. Defer Gemma 4 (it is a 2–4 week REAP port). Keep a BF16 REAP reference on the Spark so pruning loss and quant loss can be told apart.

**S37b: Quant foundry for the MTP lane (week 2 on a Spark)**
- *Extends:* `quantize_tiel.py`, `vllm_xpu_draft_mtp_int4.py`, the `-:mtp` rule, and the AutoRound-vs-GPTQ finding (19.26 vs 23 GiB, v19 findings).
- *First step (half a day, spark1):* port `quantize_tiel.py` to `DEV='cuda'` and export Qwen3.6-35B-A3B experts-only sym INT4 at gs128 with `ignore=['lm_head','re:.*mtp.*']`. Load it on the Claw, record resident GiB and KV tokens, then bench MTP k=2 with `--no-enable-prefix-caching`. As a separate A/B export, try an INT4 lm_head (the source of the +5–15% decode est.).
- *Pass:* resident ≤ 19 GiB est.; MTP k=2 decode within the A/A tolerance of the current lane (27.69 tok/s on the MXFP4 checkpoint per xpu-graph-status, k=2); KL inside the threshold against `ref-qwen36.json`.
- *Fail retires:* gs128 for this architecture; stay at the current checkpoint.
- *Gain:* KV headroom on the fastest measured lane, +~16K context est.
- *Skeptic cut:* group size is chosen per model, and gs32 is used only when weights + draft + 32K KV ≤ ~19.5 GiB. Scope it to the 2–3 architectures you actually serve. No Gemma 4 requant. Calibrate on a public coding set until a request journal exists.

### Thread 3: Decode speed on the GQA coder lane

**S15: CPU n-gram speculation (run first, expect a discard)**
- *Extends:* the status doc's "Qwen3-Coder-30B-A3B ngram OK" row.
- *First step (2–3 h):* `--speculative-config '{"method":"ngram","num_speculative_tokens":2,"prompt_lookup_min":4,"prompt_lookup_max":8}'` with prefix caching ON. Run it under default V2 and under `VLLM_USE_V2_MODEL_RUNNER=0`, and read the startup log to learn which proposer actually runs. Bench the S51 code-edit replay and a chat set, and compute draft rate and accepted tokens per drafted step from the S42 counters.
- *Pass:* ≥ 1.2× on the code-edit replay **and** ≥ 0.97× on chat.
- *Fail retires:* n-gram as a service feature. You keep the acceptance baseline S16 must beat.
- *Gain:* 18 → 22–25 tok/s on edit turns est., 0 GiB.
- *Skeptic cut:* never on GDN models (#54740 assert). k ≤ 2. No E-core pinning. Do not claim P4/P5 as addressed.

**S16: Stock DFlash draft, phase 0 is the whole bet**
- *Extends:* the measured MTP result on this console (19.27 → 27.69 tok/s, k=2); the status doc's open item "test stock z-lab Qwen3-Coder DFlash / EAGLE3". DFlash on non-GDN targets has zero XPU reports, so you would be first.
- *First step (3–4 h for the first read, a weekend for the full gate):* on the frozen v0.32.0 lane, run Qwen3-Coder INT4 with `--speculative-config '{"method":"dflash","model":"z-lab/Qwen3-Coder-30B-A3B-DFlash","num_speculative_tokens":4}'` (then 5), prefix caching ON, `--max-model-len 16384`.
  - Startup log: no "falling back to PyTorch reference attention", Model Runner V2 and async scheduling on, no eagle_ops dtype assertion.
  - Same session: smoke-test the public Qwen3-Coder EAGLE-3 head.
- *Pass (all five):*
  - (a) KL/top-1 against the no-spec run within tolerance. Do not demand bit-exact greedy: batch-1 and batch-(k+1) MoE kernels differ on XPU.
  - (b) mean acceptance length ≥ 2.0 on a real Qwen Code session.
  - (c) `bench serve 1024/1024` ≥ 1.4× (≥ 25 tok/s).
  - (d) MemAvailable ≥ 2.5 GiB steady state plus a 30-min soak.
  - (e) two restarts without a reboot.
- *Fail retires:* if (c) fails, stop. Phase 2 (domain-tuned draft trained on the Sparks, ~1 month) and the NPU-drafter moonshot are both retired.
- *Gain:* 18 → 25–27 tok/s est. The skeptic's arithmetic from your MTP A/B infers ~25 ms of fixed eager overhead per draft forward, which puts τ=2 at ~1.5× and τ=3 at ~2× est.
- *Memory (est.):* 17 GiB weights + ~1.7 GiB draft + ~0.7 GiB KV at 16K ≈ 19.4 GiB. 32K needs S12a or S13 first.
- *Skeptic cut:* phase 2 only if phase 0 passes and training code exists for the family that both trains and runs.

The S08 bake-off (Thread 5) runs after this thread, because a coder at 25+ tok/s changes that comparison.

### Thread 4: Harness overhead and TTFT

**S02: cached_tokens probe, then a static diet**
- *Extends:* the per-turn prompt analysis in `agent-harness-study-2026-10-22.md`; prefix caching on the GQA lanes.
- *First step (1 h):* start Gemma 4 with prefix caching ON, `--enable-prompt-tokens-details` and `--enable-force-include-usage`. Run two consecutive turns each from pi, Qwen Code and OpenClaw, and log `usage.prompt_tokens_details.cached_tokens / prompt_tokens`. Turn on request logging in the same session so S51 gets its 9 turns.
- *Pass (meaning no rewriting needed):* turn 2 is ≥ 90% cached for every harness. P5 then reduces to cold starts, and S17 is the only follow-up.
- *Fail:* name the harness that busts the prefix. Retires nothing.
- *Gain:* prompt tokens re-prefilled per turn becomes a number; Qwen Code auto-memory off ≈ −6K tokens/turn est.
- *Skeptic cut:* diets are static and harness-side (OpenClaw `tools.profile: messaging` plus allowBundled skills; Qwen Code auto-memory off, minimal `tools.eager`). A stateless shim fixes only wire bugs: parameter-less tools, and `tool_choice:'required'` mapped to the named function when exactly one tool is offered. Drop `cache_salt`. A template mover only for a harness verified to bust the prefix per turn, pinned to a template fingerprint that fails open. None of this applies to the GDN+MTP lane, where prefix caching is off.

**S17: Prefix warmer (only if S02 shows stable prefixes)**
- *Extends:* the measured 16K TTFT of 20,337 ms cold vs 398 ms warm in `LUNAR_LAKE_COMPATIBILITY.md:216-223`. That measurement is from an earlier lane, so re-measure on Gemma 4.
- *First step (1–2 h):* prove Gemma 4 APC works with the FLASH_ATTN override (a second identical request shows cached_tokens > 0). Then set OpenClaw `agents.defaults.heartbeat.every=5m` and add a oneshot unit `qwen -p "reply OK"` with `After=vllm.service` (never `ExecStartPost` without `-`).
- *Pass:* cached_tokens ≥ 90% on the next real turn.
- *Fail retires:* warming. A < 90% hit means a stale or evicted prefix, so re-capture rather than re-touch.
- *Gain:* first-turn TTFT ~20 s → ~0.4 s at 16K est. for this lane.
- *Skeptic cut:* no static prompt dumps, no pinning claims. pi (~1.2K tokens) is not worth warming.

**S18 (conditional):** probe NVMe prefix KV only if S02/S17 show cold prefixes still matter.
- Afternoon on Qwen3-Coder with `--kv-transfer-config` using `"kv_connector":"ExampleConnector"`, `"kv_load_failure_policy":"recompute"` and a versioned path.
- Measure restore wall time and per-turn store overhead. The skeptic expects the store cost to be unacceptable.

**S34 (optional, 1 h):** expose vLLM on tailscale0 with `--api-key` and point the laptop's pi at it. Record the Claw's MemAvailable with and without a console-side pi (+0.5–1 GiB est.). It matters for the 18–20 GiB lane. Keep one pinned pi on the Claw for XPU work.

### Thread 5: Load and switch time

**S06: Where do the 36–40 s go? Plus cache volumes.**
- *Extends:* the load tables in v19 findings (40.1 s / 15.56 GiB, 36.6 s / 16.84 GiB, GPT-OSS 19.1 s warm safetensors at :427-433); the two-lane container decision.
- *First step (1.5–2 h):*
  - One Gemma 4 launch with `VLLM_LOGGING_LEVEL=DEBUG`, timestamped phases and a `py-spy dump`.
  - A standalone microbench that mmaps the warm 16–17 GiB checkpoint and copies it tensor by tensor into `torch.xpu`, with no vLLM involved.
  - The 15-minute S47 check folded in: after a cold container start, look at `~/.cache/neo_compiler_cache` growth, `-Wrecompiled-from-ir` in stderr, and first-request TTFT.
- *Decision:* if the raw copy alone takes ≥ 15 s, the lever is the UMA allocation/copy path (zero-copy host USM via the UVA weight path #51770, `safetensors load_file(device='xpu')`, or allocator pre-reservation). In that case the prepack and `sharded_state` loaders are dead.
- *Do regardless (1 h):* mount `NEO_CACHE_DIR`, `TRITON_CACHE_DIR` and `VLLM_CACHE_ROOT` as named podman volumes keyed by image digest, set `NEO_CACHE_PERSISTENT=1`, raise `NEO_CACHE_MAX_SIZE` (verify the defaults; they are from memory), and run one warm serve cycle per lane after each image change.
- *Fail retires:* T10 if warm ≈ cold; the S47 AOT headline if no JIT happens.
- *Gain:* 36–40 → 25–30 s est. About 19 s may be irreducible copy time.

**T10 (conditional):** only if S06 shows cold ≫ warm. Try a `fadvise(DONTNEED)`-per-shard hook. Ship any O_DIRECT loader only as a `register_model_loader` plugin, and have it assert via `fincore` that the page cache did not grow.

**S08: One-model bake-off (end of week 2 / week 3)**
- *Extends:* §5 of the harness study and its five-task bake-off.
- *First step (1 day):* the five coding tasks under pi with Gemma 4 and with Qwen3-Coder (using the best spec config from S15/S16), including streamed tool-call correctness (#41967), plus one run of Qwen3-Coder as the assistant.
- *Pass:* one model is acceptable for J1 and J2. Prefer the zero-patch model if quality is within tolerance, since the patched Gemma 4 path is the one that churns.
- *Fail retires:* the zero-switch plan. S07 becomes relevant.
- *Gain:* switches per day → 0, which removes the 36–40 s load from the daily path.

**S07: Sleep/wake probe (only if S08 says two models are needed)**
- *First step (2 h):* Gemma 4 with `--enable-sleep-mode` and `VLLM_SERVER_DEV_MODE=1`, then `POST /sleep?level=2`. Record MemAvailable and GPUReclaim. Then `/wake_up` plus `collective_rpc reload_weights`, timed, and a 10-prompt greedy/KL parity check against a fresh start.
- *Pass:* MemAvailable rises by about weights + KV, GPUReclaim does not grow, there is no `freed_bytes` assertion, and wake takes < 30 s.
- *Fail retires:* two-resident-engine plans, T16, and S04 lazy loading.
- *Gain:* switch 36–40 → 20–35 s est. The parked process is budgeted at 3 GiB.
- *Skeptic cut:* pair only Qwen3-Coder with Gemma 4, never Qwen3.6. Cap awake KV at 1.5–2 GiB. Run headless. MTP off on both, or a worker extension that reloads the drafter after wake.

**S04 (1 h, once a lane's flags settle):** write `models.d/<id>.yaml` and launch with `vllm serve --config`. This pins the measured flags (FLASH_ATTN for Gemma 4, `--no-enable-prefix-caching` for GDN+MTP, spec configs, KV bytes) so the gains are not lost. llama-swap and quadlets wait on S07/S08.

### Thread 6: Patch and kernel loop

**S45: Extract the live patch set and dry-run it on the newest tag**
- *Extends:* `vllm/patches/` (14 dead v0.14/v0.19 owner patches, 5 Intel ones, a truncated `glm4_moe_lite_mla_xpu.patch`); the upstream-or-delete decision.
- *First step (2 h):* unpack the pristine vllm 0.32.0 and vllm-xpu-kernels 0.1.15.4 wheels and `diff -ruN` them against the container's site-packages. Commit the result as `git format-patch` output in a new `vllm/patches-lnl/series` with `Upstream-PR:` and `drop_when:` headers. Run `git apply --check --3way` against the newest vLLM tag, and classify each failure as file-missing, hunk-fail or fuzz.
- *Pass:* a definitive list of live patches (est. 4–5), with each one marked as applies, already upstream, or needs rework.
- *Fail retires:* nothing. The result is information either way.
- *Gain:* the real patch count. It answers Section 4's open questions, in particular where the oneDNN fix lives.
- *Skeptic cut:* leave Intel's files untouched. Move the 14 v0.19 patches to `attic/`. A drift-radar timer on spark1 comes later.

**S46: Python-only patches as a `vllm.general_plugins` overlay (after S10 + S45)**
- *First step (2 h):* wrap `MemorySnapshot.measure` (call the original, then on XPU recompute free memory) only if S10 kept it. Wrap the #40128 `kv_cache_utils` function with feature detection (the `page_size_padded` LCM logic), not a hash. Confirm it loads in the frozen container.
- *Pass:* the plugin loads, and it no-ops when it detects the upstream feature.
- *Gain:* each per-upgrade rebase becomes a version-checked plugin that fails loudly. Patches to rebase go from 4 to 1–2.
- *Skeptic cut:* the Gemma 4 override stays a launch flag; do not monkeypatch Gemma4Config. A csrc oneDNN fix goes to S47. Budget 1–2 h per release, as today.

**S47 (conditional):** build the vllm-xpu-kernels wheel off-console only if S45 shows a csrc patch is live (the Gemma 4 attention tuple, `chunk_prefill_full.conf`, GDN `is_bmg()`→`is_xe2_arch()`, possibly oneDNN) or S12b needs FP8 KV in the home kernel. Keep `lnl-m` in `VLLM_XPU_XE2_AOT_DEVICES`. The Sparks are aarch64, so an x86 wheel build needs the laptop or CI. That is the console's 1.5–2 h per build moved off-box.

**S53 (1 h):** a pre-commit hook that fails on `source setvars.sh` and on `|| true` after `git apply` or `patch`, plus a COPY-target-exists check for `Dockerfile.lunar-lake`, which references missing patches.

### Parked: NPU offload

Only one gate is worth running before J3 work starts.

**S21 contention test (2–3 h, week 4 if time allows):** OVMS with `--target_device NPU` serving Qwen3-1.7B int4-cw, then `bench serve 1024/1024` on Gemma 4 with the NPU idle and again while it generates.
- *Pass:* decode drop < 5–15% est.
- *If decode drops ≥ 10%:* NPU jobs run only while the LLM is idle.

T14 (compile Whisper-turbo plus one LLM on the NPU, logging JSONL) follows only if S21 passes. S22, S23, S24, T12 and T13 stay deferred.

---

## 3. Four-week sequence

The console runs one model at a time, so console steps are serial. Spark and laptop work runs in parallel with them.

| Week | Console (serial) | Spark (parallel) | Laptop (parallel) |
|---|---|---|---|
| **1** | D1: S02 probe + request capture → S10 (45 min) → S06 launch, microbench, neo-cache check + cache volumes. D2: S45 diff (needs container site-packages). D3–4: S51/S42 A/A baselines on Gemma 4 and Qwen3-Coder → S12a FP8 KV on Qwen3-Coder → S12c headless delta (docked) → S05 formula check. D5: S15 n-gram. | D2: S37a captures (Gemma 4, Qwen3-Coder, Qwen3.6) on `real_prompts.txt` | S50 prompt set; `replay.py`; metrics diff; S45 series/attic layout |
| **2** | Weekend: S16 phase 0 (DFlash k=4/5 + EAGLE-3 smoke, full gate a–e). Weekday: S12b Gemma 4 FP8 KV at 64K; S17 if S02 passed; S02 static diets on the failing harness. | S13 REAP-25B quantize (evening); S37b Qwen3.6 gs128 export + INT4-lm_head A/B export | S46 overlay code |
| **3** | S13 load/KL/bench; S37b load + MTP k=2 bench; S16 at 32K if S12a or S13 freed room; S08 bake-off (1 day); S46 load test; S04 YAML for settled lanes | spark1 drift-radar timer for the series (optional) | S53 lints; PR texts (Section 4) |
| **4** | S07 probe only if S08 says two models; S52 twin (weekend, Gemma 4); T10 only if S06 showed cold ≫ warm; S18 only if S17 left a cold-prefix cost; S21 contention test (optional) | S16 phase 2 scoping only if phase 0 passed | S47 wheel build only if S45 found a live csrc patch or S12b needs one |

**Dependencies:**
- S02 capture → S50 prompts → S37a refs → every KL gate (S12, S13, S16, S37b, S52).
- S02 capture → S51 replay → the S15/S16 gates and the S17 verification.
- S42 counters → the S15/S16 acceptance readings.
- S10 → S05, S46 (UMA wrap present or not), and the #40295/#41370 action.
- S45 → S46 and S47. S12b failure → a kernel task in S47.
- S06 → T10, S47 (AOT), S18.
- S15 → S16 (the bar to beat); S12a/S13 → S16 at 32K; S16 → S08 → S07, S04 lazy loading, T16.
- A/A noise floor (week 1) → every pass threshold in this plan.

---

## 4. Patches and PRs

| Live patch | What decides it | Likely outcome |
|---|---|---|
| **UMA accounting** (`MemorySnapshot.measure`, psutil-based; PRs #40295/#41370) | **S10** in both lanes; `rpm -q intel-compute-runtime` ≥ 26.18 on the host | **Delete** from the container lane if `getMemoryInfo` tracks MemAvailable. Keep it on the native lane until the host runtime is verified, and if S46 runs, deliver it as a plugin wrap. |
| **Hybrid-MoE #40128 cherry-pick** (`kv_cache_utils.py`; a v0.19-era `xpu_hybrid_moe_page_size_pr40128.patch` exists in `vllm/patches/`) | **S45**: is the hunk still in v0.32 site-packages, and has an equivalent landed by the newest tag? | If upstream: **delete**. If not: **keep** as an S46 overlay with feature detection. Post `issues/pr-40128-comment-draft.md` (never posted) refreshed with v0.32 / Qwen3.6 data. |
| **Gemma 4 attention kernel + `--attention-backend FLASH_ATTN`** | S12b (FP8 KV support), S52 (parity twin), S45 (where the kernel lives) | **Must keep.** The override stays a launch flag in the S04 YAML. The kernel is carried by S47 if it is csrc. Upstreaming it is not scheduled in this plan; S52 is what would make a parity claim credible. |
| **oneDNN engine/stream context fix** ("Backends mismatch") | **S45** finds where it lives; your issues docs do not record it | **Keep** until located. Python → S46 overlay; csrc in vllm-xpu-kernels → S47 wheel and an upstream issue with a repro. |

**#40295 / #41370 status** (from `vllm-xpu-status-2026-09-02.md`, not re-checked since; refresh it with `gh api repos/vllm-project/vllm/pulls/40295`): both open, needs-rebase since Jun 30.
- **#41370:** its premise is stale, because since #47134 (v0.25) vLLM reads `getMemoryInfo` rather than `torch.xpu.mem_get_info`. If S10 passes, **close it** with the S10 readings as the reason.
- **#40295:** if S10 passes, either close it or reframe it as "driver-independence for compute-runtime < 26.18". It would be a ~37-line xpu.py-only change on the `is_integrated_gpu()` hook (#35356). Either way, the S10 readings make a small, credible upstream contribution: an `lnl_m` case for `tests/test_get_memory_info.py`. That case and the other two tweaks the status doc lists (decode planner reading `gpu_subslice_count` 8 vs a hard-coded 20; GDN `is_bmg()`→`is_xe2_arch()`, which needs an accuracy check) are the PR candidates this plan feeds.

---

## 5. Out of scope (operations hygiene and off-stack)

The first group was considered and excluded by out-of-scope votes from two or three judges.

- S39 sleep/AC policy: ops. Only the no-suspend-during-runs rule is kept (Thread 1 hygiene).
- S38 watchdog and Wi-Fi self-heal: ops.
- S41 sentinel ops bot: no metric.
- S33 Spark standby gateway: bot availability only.
- S31 Thunderbolt dock Ethernet: connectivity, not inference.
- S35 mosh/tmux sessions: ops.
- S43 crash bundles: post-mortem tooling.
- S44 snapshots/versionlock: ops. Only the `rpm -q` tag per bench result is kept.
- S27 kiosk face: costs 0.3–0.7 GiB.
- S30 power-state daemon: acts on numbers S29 has not produced.
- T11 bot memory git backup: ops.
- T17 off-grid channel: new product surface.
- T18 presence fusion: new subsystem.
- T08 models as OCI artifacts: load time unchanged.
- T09 btrfs layout: no measured gain.
- S03 sensitivity router, S28 fingerprint consent: privacy and security, not speed.
- S26 MCP body server, S36 Spark refines local answers: off-stack.
- T01–T07 (threat ledger, LUKS/TPM, key broker, per-bot users, network policy, stack.lock beyond digest pins, quarantine reader): security hygiene.

Single-vote exclusions and low-scoring ideas, parked rather than scheduled:
- S01 routing facade: ops. The S51 tee proxy covers measurement.
- S40 canary/rollback: ops. The greedy goldens live in S37/S52.
- S11 xe rebind: unverified root cause.
- S09: only its leak-table half is kept, as bench hygiene.
- S48 night gate runner: schedules measurements that do not exist yet. Revisit after week 4.
- S29 clock/PL1 sweep: real numbers, but joules are not on your list.
- S54 P-core pinning: your xpu-graph data says host dispatch is not the bottleneck. A 30-minute `taskset -c 0-3` A/B would close it.
- S14 LoRA roles: moot if S08 picks one model.
- S49 night batch jobs: measures nothing.
- S32 B60 eGPU: needs a purchase.
- T16 game-watch sleep: downstream of S07.
- S21/T14 NPU: see the parked gate in Section 2.
- S22, S23, S24, S25, T12, T13, T15: J3, deferred until the S21 gate.

---

## 6. Where the judges disagreed

- **S10 (perf 5.5 vs continuity 9 / cheapest 9).** Performance sees "patch −1" as a small number. The other two see a 45-minute test that deletes a patch, settles a PR and fixes the memory formula S05/S12 depend on. The plan runs it on day 1.
- **S12, which model first.** Performance wants Qwen3-Coder on stock attention, to isolate FP8 KV from the home kernel. Continuity wants Gemma 4 through the home kernel, because that tests your code. Cheapest wants the headless delta plus Gemma 4 at 64K. The plan runs Qwen3-Coder first (it is also where S16 needs the room), then Gemma 4 in week 2.
- **S16 vs S15 order and yardstick.** Continuity's thread puts DFlash first; the other two run n-gram first as a zero-GiB baseline. Performance gates on `bench 1024/1024`, cheapest on acceptance from the replay. The plan runs S15 first and requires both gates for S16.
- **S51 (perf 5 vs ~8+).** Performance treats it as an instrument that moves no number. The others treat it as the yardstick without which the speculation gates measure the wrong workload. It is built in week 1.
- **S07 (cheapest 7.9 vs continuity 4 / perf 5).** Cheapest sees one probe deciding three ideas. The others expect it to fail at the ceiling. The plan runs it only if S08 says two models are needed.
- **S08 (cheapest 7.4 vs perf 4).** Same split: it is a decision, not a number. It is scheduled after S16 because the coder's speed changes the outcome.
- **S02 / S06 (cheapest 8.6/8.8 vs continuity 5/6).** Cheapest values how many ideas each one settles. Continuity notes that both mostly re-measure known findings.
- **S45/S46 (continuity 8–8.5 vs perf 4–4.5).** Patch count is your number, but not a runtime one. S45 runs in week 1 (2 h, it answers Section 4); S46 waits.
- **S52 (continuity 7.5 vs cheapest 5), and which model to twin.** Cheapest objects that nothing is learned until a weekend build is done. The judges named different targets: Qwen3-Coder (perf), Gemma 4 (continuity), Qwen3.6+MTP (cheapest). The plan picks Gemma 4 because its attention kernel is the patch that cannot be deleted.
- **S04 (continuity 6.5 vs 3.5).** Codifying launch rules versus moving no number. It is kept as a 1-hour pin, without llama-swap or quadlets.
- **S21 (cheapest 6.6 vs 3–3.5).** The contention test decides the whole NPU thread, but it spends iGPU-adjacent bandwidth and GiB. It is parked as an optional week-4 gate.
- **Out-of-scope votes split:** S01, S11, S40, S49, S32, T13, T15, S25 and T16 each drew one vote.

---

## 7. Open measurements the plan relies on (none measured on this console)

1. `getMemoryInfo(0)` free vs MemAvailable in the container and on the native lane, the constant offset (~2.29 GiB expected), and the host compute-runtime version.
2. A/A noise floor for `quant_kld compare` and for the replay on each lane. Every threshold in the plan depends on it.
3. FP8 KV: tokens per GiB on Xe2 (≈2× est.), the KL delta against bf16 KV, and whether the home Gemma 4 kernel accepts FP8 KV at all.
4. KV KiB/token for Gemma 4 specifically. The 40–48 KiB figure is for the 30B-class MoEs, and Gemma 4's sliding layers differ.
5. MemAvailable gained by stopping the display manager (+1.5–2 GiB est.; your notes say ~4 GiB for the desktop session).
6. DFlash acceptance length on XPU for Qwen3-Coder (zero reports), and the real per-draft-forward eager overhead. The ~25 ms figure is inferred from the MTP A/B, not measured.
7. Which n-gram proposer actually executes on v0.32 XPU, and its draft rate on code edits.
8. Resident GiB for Qwen3-Coder + DFlash at 16K (≈19.4 GiB est.) and at 32K.
9. Per-harness `cached_tokens/prompt_tokens` on turn 2. Whether Gemma 4 APC works with the FLASH_ATTN override. Cold vs warm 16K TTFT on Gemma 4: the 20.3 s / 0.4 s pair is from an earlier lane.
10. Load-time split: raw mmap-to-XPU copy, repack and JIT. Whether JIT happens in the container. Cold vs warm load.
11. REAP-25B resident GiB (~14 est.) and its KL. Qwen3.6 gs128 resident (≤19 est.). INT4 lm_head decode gain (+5–15% est., from the judge's arithmetic of ~10 ms of a ~52 ms step).
12. Sleep mode: GiB returned to MemAvailable, GPUReclaim behaviour, wake time.
13. GiB leaked per exit type (clean stop vs kill), to decide when an A/B pair needs a reboot.
14. Twin load time and resident GiB (30–45 s, 2–3 GiB est.).
15. NPU co-run decode penalty on Gemma 4 (5–15% est.).
