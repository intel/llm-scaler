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
