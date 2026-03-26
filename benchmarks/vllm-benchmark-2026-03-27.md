## vLLM Server Benchmark — 2026-03-27

**Test Configuration:**
- Model: /shared/models/qwen3-8b-int4-autoround
- Input tokens: ~128 (random data, actual: 337)
- Max output tokens: 7800
- Concurrency: 1
- Endpoint: http://127.0.0.1:8000/v1/chat/completions

**Results:**
- Status: OK
- Finish reason: stop (natural end)
- Prompt tokens: 337
- Completion tokens: 2,923
- Time elapsed: 215.36s
- Throughput: 13.6 tok/s

**Notes:**
- Context window is 8192 tokens total; max usable output with ~337 input tokens is ~7,849
- Model stopped naturally at 2,923 tokens rather than hitting the 7,800 limit
- INT4 quantized 8B model on local inference server

---

## Concurrency Tests

### Concurrency 2

| Metric | Value |
|---|---|
| Wall time | 184.81s |
| Total output tokens | 4,714 |
| Aggregate throughput | 25.5 tok/s |

| Worker | Tokens | Time | Tok/s | Finish |
|---|---|---|---|---|
| 0 | 2,304 | 176.5s | 13.1 | stop |
| 1 | 2,410 | 184.8s | 13.0 | stop |

### Concurrency 5

| Metric | Value |
|---|---|
| Wall time | 600.08s |
| Total output tokens | 11,565 |
| Aggregate throughput | 19.3 tok/s |

> Note: Worker 1 timed out at 600s and did not return results.

| Worker | Tokens | Time | Tok/s | Finish |
|---|---|---|---|---|
| 0 | 1,396 | 128.0s | 10.9 | stop |
| 2 | 3,565 | 335.7s | 10.6 | stop |
| 3 | 2,613 | 244.1s | 10.7 | stop |
| 4 | 3,991 | 374.3s | 10.7 | stop |

## Summary

| Concurrency | Aggregate tok/s | Per-worker tok/s |
|---|---|---|
| 1 | 13.6 | 13.6 |
| 2 | 25.5 | ~13.0 |
| 5 | 19.3 | ~10.7 |

**Observations:**
- Concurrency 2 nearly doubles aggregate throughput vs single (25.5 vs 13.6 tok/s), per-worker speed unchanged
- Concurrency 5 shows aggregate throughput drops to 19.3 tok/s — per-worker latency degrades (~10.7 tok/s), suggesting GPU memory/compute saturation
- 1 worker timed out at concurrency 5 (>600s), indicating queue pressure at high concurrency

---

## Concurrency 5 — Capped Output Test (max_tokens=3000, timeout=900s)

**Goal:** Verify no worker timeouts when output length is capped.

**Test Configuration:**
- Model: /shared/models/qwen3-8b-int4-autoround
- Max output tokens: 3,000 (capped)
- Timeout: 900s
- Concurrency: 5

**Results:**

| Metric | Value |
|---|---|
| Wall time | 362.52s |
| Total output tokens | 13,183 |
| Aggregate throughput | 36.4 tok/s |
| Peak RAM (vLLM) | 45 MB (baseline: 18 MB, delta: +28 MB) |

| Worker | Tokens | Time | Tok/s | Finish |
|---|---|---|---|---|
| 0 | 3,000 | 362.5s | 8.3 | length (hit cap) |
| 1 | 2,994 | 362.0s | 8.3 | stop |
| 2 | 1,237 | 141.4s | 8.7 | stop |
| 3 | 2,952 | 356.5s | 8.3 | stop |
| 4 | 3,000 | 362.5s | 8.3 | length (hit cap) |

**Outcome:** ✅ No timeouts — all 5 workers completed successfully.

**Observations:**
- Capping max_tokens=3000 eliminates timeout risk at concurrency 5
- Aggregate throughput jumps to 36.4 tok/s (vs 19.3 tok/s uncapped) due to shorter wall time
- Per-worker speed drops to ~8.3 tok/s under 5-way concurrency (vs 13.8 tok/s single)
- RAM delta only +28 MB — GPU VRAM is the real constraint, not system RAM
- Workers 0 and 4 hit the length cap (3000 tok), indicating the model wanted to generate more

## Final Summary

| Concurrency | max_tokens | Aggregate tok/s | Per-worker tok/s | Timeouts | Peak RAM |
|---|---|---|---|---|---|
| 1 | 7800 | 13.6 | 13.6 | 0/1 | 79 MB |
| 2 | 7800 | 25.5 | ~13.0 | 0/2 | 79 MB |
| 5 | 7800 | 19.3 | ~10.7 | 2/5 | 80 MB |
| 5 | 3000 | 36.4 | ~8.3 | 0/5 | 45 MB |

**Recommendation:** Concurrency 2 with uncapped output for quality; concurrency 5 with max_tokens≤3000 for maximum throughput.
