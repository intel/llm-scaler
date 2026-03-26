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
