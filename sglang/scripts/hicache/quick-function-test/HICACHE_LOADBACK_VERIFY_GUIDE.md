# HiCache L2 load_back (H→D restore) — verification guide

Target model: **Qwen3.6-35B-A3B** (hybrid GDN+attention, `hybrid_ssm=True`, uses a Mamba
state cache), Intel XPU/BMG, TP=2.

The server fix under test lives in
`python/sglang/srt/mem_cache/hybrid_cache/hybrid_cache_controller.py`
(XPU `record_stream` device guard + `load_stream.synchronize()` drain on `start_loading`).

---

## 0. What "load_back fired" actually means

`load_back` = the H→D path that restores a prefix's **KV *and* Mamba state** from the
pinned host pool (L2) back onto the device when a previously-evicted prefix is
re-requested. On this hybrid model the restore carries `extra_pools=2` (the two Mamba
state pools) — that is the exact path that used to `UR_RESULT_ERROR_DEVICE_LOST`.

**Three independent signals it fired (use all three — one alone lies):**

| Signal | Where | Fired look |
|---|---|---|
| `sglang:load_back_tokens_total` climbs | `curl :PORT/metrics` | delta > 0 on the re-send |
| `cached_tokens > 0` **at** `full token usage: 0.00` | response `usage` + scheduler `Prefill batch` log | prefix came from **host**, not device |
| `start_loading: ... extra_pools=2` + `XPU - load_stream sync done` | server log (`SGLANG_HICACHE_DEBUG=1`) | the mamba H→D transfer + the fix ran |

A device hit shows `cached_tokens>0` but with `full token usage > 0` and **no**
`load_back` delta — do **not** count that as load_back.

---

## 1. Non-obvious gotchas (these cost real debugging time)

1. **The host pool must have ROOM.** If `hicache_host_used_tokens ≈ hicache_host_total_tokens`,
   write_through backups are silently dropped → nothing is retained on host → every
   re-send recomputes → load_back never fires.
2. **`flush_cache` does NOT free the host pool.** It clears the device radix tree only;
   `host_used` stays put. To reset the host pool you must **restart the server**.
3. **write_through backup is async.** If a prefix is evicted *before* its D→H backup
   finishes, it is DELETED (no host copy) instead of demoted. Always leave a few
   seconds between priming a prefix and evicting it.
4. **Input length cap.** Max input = `max_total_tokens − reserve` (e.g. 2042 when
   `MAX_TOTAL_TOKENS=2048`). A prefix bigger than that is rejected HTTP 400 → the client
   prints `nan`/`ERR`, which is a *client* symptom, not a server crash.
5. **Gate B (logprob parity) alone does not prove a restore** — a recompute is
   deterministic and also matches golden. Gate B is only meaningful once Gate A
   (real `load_back` delta) has passed.
6. **Greedy tokens are not a gate** — use teacher-forced logprobs, never "identical
   output text".
7. **`echo=true` / `logprob_start_len=0` SUPPRESSES the prefix match — never probe
   with it.** The server's `_compute_max_prefix_len` (schedule_batch.py) caps the
   prefix-MATCH key at `logprob_start_len` whenever `return_logprob=True`: positions
   you ask logprobs for MUST be recomputed, so cache reuse is deliberately turned
   off for them. `echo=true` sets `logprob_start_len=0` → match key truncated to 0
   tokens → **load_back cannot fire**. This is correct server behavior; a probe built
   on `echo=true` is a TEST bug that makes a working restore path look dead. To keep
   the match nearly full AND still teacher-force a tail, use native `/generate` with
   `logprob_start_len = prompt_tokens − K` (K≈8): match covers the first
   `prompt_tokens − K` tokens (load_back fires and restores them), only the last K are
   recomputed, and their logprobs — which attend to the RESTORED KV — are the Gate B
   correctness signal.

---

## 2. Launch the server (debug tracing on)

Wrapper: `/tmp/launch_hicache_dbg.sh` (already set up). Key env:

```
export SETVARS_CALL=1                 # let conda mpivars skip its body under set -u
export SGLANG_HICACHE_DEBUG=1         # trace start_writing / start_loading / the XPU drain
export MODEL_PATH=/mnt/models/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0
export ZE_AFFINITY_MASK=0,1
export MAX_TOTAL_TOKENS=2048          # SMALL device pool -> easy eviction (quick test)
export HICACHE_RATIO=4                # host pool = 4x device -> evicted prefix survives on host
export HICACHE_WRITE_POLICY=write_through   # eager host backup -> retained for re-match
export PORT=30000
```

Launch detached (so it survives your shell) and DO NOT `pkill -f sglang.launch_server`
(that string matches your own shell → self-kill); use the bracket trick:

```bash
setsid bash /tmp/launch_hicache_dbg.sh > /tmp/hicache_dbg.log 2>&1 < /dev/null &
# wait for readiness:
until curl -s http://127.0.0.1:30000/v1/models | grep -q '"object"'; do sleep 2; done
# to stop it later:
pkill -f "[s]glang.launch_server"
```

**Always restart between load_back runs** so the host pool starts empty.

---

## 3. QUICK verify (small pool — the fast smoke test)

Server as in §2 (`MAX_TOTAL_TOKENS=2048`, `HICACHE_RATIO=4`), freshly restarted.

```bash
cd /home/karen/test-scripts
PORT=30000 DEBUG_LOG=/tmp/hicache_dbg.log SERVER_PAT='[s]glang.launch_server' \
  bash verify_loadback_proven.sh
```

What it does: primes prefix A (native `/generate`, `logprob_start_len=0` → captures
golden tail logprobs + `prompt_tokens=N`), **waits for the D→H backup**, sends B+C to
evict A off device, then re-sends A with `logprob_start_len = N − K` so the prefix
match stays nearly full and load_back fires; asserts all three gates. Exit 0 = PROVEN.
The probe uses `/generate` (NOT `echo=true`) — see gotcha #7 for why echo breaks it.

Tunables: `BACKUP_WAIT=6` (raise if Gate A flakes), `REPS=140` (~1800-tok prefix; keep
the prefix under the input cap), `K=8` (teacher-forced tail length), `PORT`.

If Gate A fails with `cached=0`: the host pool is full or the backup hadn't finished —
**restart the server** and/or raise `BACKUP_WAIT`.

### Manual one-liner cross-check (independent of the script)
```bash
BASE=http://127.0.0.1:30000
before=$(curl -s $BASE/metrics|grep -E '^sglang:load_back_tokens_total'|awk '{print $NF}')
# ... prime A, wait 6s, send B, send C, re-send A (distinct ~1800-tok prefixes) ...
after=$(curl -s $BASE/metrics|grep -E '^sglang:load_back_tokens_total'|awk '{print $NF}')
echo "load_back delta = $(python3 -c "print($after-$before)")"      # > 0 == fired
grep 'start_loading' /tmp/hicache_dbg.log | tail            # extra_pools=2 + XPU sync done
```

---

## 4. REAL-scenario verify (bigger `max_total_tokens`, production-like)

The quick test uses a tiny pool to force eviction fast. To prove HiCache in a realistic
setting, scale the pool up and drive genuine reuse-under-pressure.

### 4a. Sizing rules (independent of absolute pool size)
- **Each conversation prefix** should be a meaningful fraction of the device pool
  (e.g. 30–50%) and comfortably **under the input cap** (`max_total_tokens − reserve`).
- **Working set > device pool, < host pool.** Keep `N_conversations × prefix_tokens`
  **greater** than `max_total_tokens` (forces device eviction) but **less** than the host
  pool `= max_total_tokens × HICACHE_RATIO` (so evicted prefixes survive on host).
  Rule of thumb: `HICACHE_RATIO ≥ 3`, working set ≈ 1.5–2× device pool.
- **KV path vs Mamba path** are driven by *different* limits:
  - KV/attention load_back → **token-pool** pressure (`max_total_tokens`). Longer prefixes.
  - Mamba-state eviction → **per-sequence slot** pressure (`max_mamba_cache_size`, one slot
    per sequence). To exercise *mamba* eviction specifically you need **more distinct
    sequences than `max_mamba_cache_size`**, not longer prompts. Shrink it with
    `--max-mamba-cache-size 8` for a deterministic mamba-eviction test.

### 4b. Recommended real-scenario server
```
MAX_TOTAL_TOKENS=32768        # or whatever fits the card
HICACHE_RATIO=4               # host pool = 131072 tokens
HICACHE_WRITE_POLICY=write_through
# (optional, to force MAMBA eviction quickly:) --max-mamba-cache-size 8
```

### 4c. Multi-turn / multi-round scenario (closest to production chat)
Several distinct long "conversations" revisited round-robin; each revisit was evicted by
the others since its last turn, so it restores from host. Script:

```bash
cd /home/karen/test-scripts
PORT=30000 N_ROUNDS=5 N_CONV=6 PREFIX_FRAC=0.4 INTER_DELAY=2.0 \
  DEBUG_LOG=/tmp/hicache_dbg.log SERVER_PAT='[s]glang.launch_server' \
  bash verify_loadback_nround.sh
```
- `N_CONV × PREFIX_FRAC` must exceed 1.0 (working set > device pool). With
  `PREFIX_FRAC=0.4`, use `N_CONV ≥ 3` (here 6 → 2.4× the pool).
- `INTER_DELAY ≥ 2` so each async backup registers before the next revisit evicts it.
- Keep `PREFIX_FRAC` low enough that one prefix stays under the input cap (0.4 of 32768 =
  ~13k tok is fine; 0.9 of 2048 overshoots the 2042 cap — that was the earlier `nan` bug).

### 4d. Which scenarios actually exercise load_back
| Scenario | Fires load_back? | Notes |
|---|---|---|
| Single prompt, sent once | No | nothing to restore |
| Same prompt twice, no pressure | No | 2nd is a **device** hit |
| Prime → evict (distinct load) → re-send | **Yes** | the §3 pattern |
| Multi-turn chat, history > device pool | **Yes** | the §4c pattern; most production-like |
| > `max_mamba_cache_size` distinct seqs, re-send first | **Yes (mamba path)** | deterministic mamba eviction |
| Working set > host pool | No | evicted off host too → recompute |

---

## 5. Confirm HiCache is actually the thing that fired (attribution)

Run any scenario **twice** and compare:
1. **Feature-on** (`--enable-hierarchical-cache`, as configured).
2. **Feature-off control** (restart with hierarchical cache disabled).

On feature-off, `load_back_tokens_total` stays 0 and re-sends recompute (higher TTFT,
`full token usage` spikes on every re-send). On feature-on, re-sends show the
`load_back` delta + `cached_tokens>0 @ usage 0.00` + `start_loading extra_pools=2`.
The *difference* between the two runs is what proves the win is HiCache and not prefix
reuse on device. (Metrics prove attribution; the logprob parity proves correctness.)

### Server-side counters to watch (`curl :PORT/metrics`)
- `sglang:load_back_tokens_total`   — H→D restores (the headline)
- `sglang:evicted_tokens_total`     — D→H evictions (must climb, else no pressure)
- `sglang:hicache_host_used_tokens` / `..._total_tokens` — keep used < ~70% of total
- `sglang:kv_evictable_tokens`, `sglang:mamba_evictable_tokens` — current device residency

### Server-log lines (`SGLANG_HICACHE_DEBUG=1`)
- `start_writing: ... extra_pools=1` + `XPU - write_stream sync done` — D→H backup
- `start_loading: ... extra_pools=2` + `XPU - draining load_stream` + `load_stream sync done`
  — the H→D restore **and** the fix executing
- `Prefill batch ... #cached-token: N ... full token usage: 0.00` — the restored prefix

---

## 6. Stability gate (the crash the fix targets)
Throughout every run, the server process must stay alive:
```bash
pgrep -f "[s]glang.launch_server"        # must keep returning a pid
grep -iE "DEVICE_LOST|UR_RESULT_ERROR|level.zero|Traceback" /tmp/hicache_dbg.log
```
(Ignore matches inside the one-line `server_args=...` dump, e.g. `watchdog_timeout=300`.)
No `DEVICE_LOST` across many restores = the load-path record_stream/synchronize fix holds.
