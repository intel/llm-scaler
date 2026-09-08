# SGLang Model Test Guide on Intel XPU

This guide covers accuracy, Function Calling, and serving performance tests for
the Qwen3.6 and Gemma4 configurations validated on Intel BMG.

## Test matrix

| Model | FP8 | GGUF |
|---|---|---|
| Qwen3.6-27B | Supported | Supported (`Q4_K_M`) |
| Qwen3.6-35B-A3B | Supported | Supported (`UD-Q4_K_M`) |
| Gemma4-31B | Supported | Not provided by the current scripts |
| Gemma4-26B-A4B | Supported | Not provided by the current scripts |

The complete matrix therefore contains six configurations: four FP8 services
and two Qwen GGUF services. Run one configuration at a time on the same GPUs
and port.

For each configuration, use this order:

1. Start the service and pass both health checks.
2. Run the 20-question GSM8K smoke test.
3. Run the BFCL Function Calling and multi-turn smoke tests.
4. Run the single-concurrency performance test.
5. If the smoke tests are stable, run GSM8K 200, BFCL 200, and the concurrent
   performance test.
6. Save the results, stop the service, and then start the next configuration.

## 1. Start the container

From the `sglang` directory on the host:

```bash
IMAGE_TAG=llm-scaler-sglang:dev-0907 \
MODEL_DIR=/path/to/models \
CONTAINER_NAME=sglang-dev \
bash scripts/run_container.sh
```

Enter the container:

```bash
docker exec -it sglang-dev bash
```

The commands below use `/llm-scaler/sglang`. In a source-mounted development
container, use `/workspace/llm-scaler/sglang` instead.

## 2. Start one model service

Use terminal A for the model service. Stop the current service before starting
the next configuration.

### Qwen3.6-27B FP8

```bash
cd /llm-scaler/sglang

MODEL_PATH=/models/Qwen3.6-27B \
ZE_AFFINITY_MASK=6,7 \
TP_SIZE=2 \
HOST=127.0.0.1 \
PORT=30000 \
bash scripts/start_qwen3_6_service.sh
```

### Qwen3.6-27B GGUF

```bash
cd /llm-scaler/sglang

MODEL_PATH=/models/Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_M.gguf \
GGUF_CFG_DIR=/models/Qwen3.6-27B \
ZE_AFFINITY_MASK=6,7 \
TP_SIZE=2 \
HOST=127.0.0.1 \
PORT=30000 \
bash scripts/start_qwen3_6_service.sh
```

### Qwen3.6-35B-A3B FP8

```bash
cd /llm-scaler/sglang

MODEL_PATH=/models/Qwen3.6-35B-A3B \
ZE_AFFINITY_MASK=6,7 \
TP_SIZE=2 \
HOST=127.0.0.1 \
PORT=30000 \
bash scripts/start_qwen3_6_service.sh
```

### Qwen3.6-35B-A3B GGUF

```bash
cd /llm-scaler/sglang

MODEL_PATH=/models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
GGUF_CFG_DIR=/models/Qwen3.6-35B-A3B \
ZE_AFFINITY_MASK=6,7 \
TP_SIZE=2 \
HOST=127.0.0.1 \
PORT=30000 \
bash scripts/start_qwen3_6_service.sh
```

### Gemma4-31B FP8

```bash
cd /llm-scaler/sglang

MODEL_PATH=/models/gemma-4-31B-it \
ZE_AFFINITY_MASK=6,7 \
HOST=127.0.0.1 \
PORT=30000 \
bash scripts/run_gemma4_31b.sh
```

### Gemma4-26B-A4B FP8

```bash
cd /llm-scaler/sglang

MODEL_PATH=/models/gemma-4-26B-A4B-it \
ZE_AFFINITY_MASK=6,7 \
TP_SIZE=2 \
SGLANG_FP8_DTYPE=e4m3 \
HOST=127.0.0.1 \
PORT=30000 \
bash scripts/run_gemma4_26b_moe.sh
```

## 3. Check service health

Use terminal B for all test clients:

```bash
curl --noproxy "*" --fail http://127.0.0.1:30000/health
curl --noproxy "*" --fail http://127.0.0.1:30000/v1/models
```

Do not start a test until both commands succeed.

## 4. Run GSM8K

Run GSM8K for every configuration in the matrix. Use the same settings when
comparing FP8 and GGUF.

Start with a 20-question smoke test and one client thread:

```bash
cd /llm-scaler/sglang

python3 scripts/run_gsm8k.py \
  --host 127.0.0.1 \
  --port 30000 \
  --num-examples 20 \
  --num-threads 1 \
  --api chat \
  --no-thinking \
  --temperature 0 \
  --out-prefix /workspace/results/MODEL_FORMAT_gsm8k_smoke
```

Then run the 200-question accuracy test:

```bash
python3 scripts/run_gsm8k.py \
  --host 127.0.0.1 \
  --port 30000 \
  --num-examples 200 \
  --num-threads 1 \
  --api chat \
  --no-thinking \
  --temperature 0 \
  --out-prefix /workspace/results/MODEL_FORMAT_gsm8k_200
```

Replace `MODEL_FORMAT` with a unique name such as `qwen27_fp8`,
`qwen27_gguf`, or `gemma31_fp8`. Keep `--num-threads 1` for the safest XPU
accuracy comparison. Increase it only for a separate concurrency test.

The harness writes:

```text
<out-prefix>_examples.jsonl
<out-prefix>_summary.txt
```

## 5. Run BFCL

BFCL is applicable to all six configurations because the Qwen FP8 and GGUF
services both enable the `qwen3_coder` tool parser, while the Gemma4 FP8
services enable the `gemma4` parser.

### 5.1 Install BFCL once

Follow `scripts/bfcl/README.md` Steps 1-5 to clone the pinned Gorilla fork and
install BFCL.

### 5.2 Prepare a workspace for each model

Use the HF model directory as `TOK_DIR`, including for GGUF. GGUF files do not
contain all tokenizer and model configuration files required by BFCL.

| Configuration | `TOK_DIR` | `MODEL_ID` | Example `WORKDIR` |
|---|---|---|---|
| Qwen3.6-27B FP8/GGUF | `/models/Qwen3.6-27B` | `Qwen/Qwen3.6-27B-FC` | `/workspace/bfcl_kit/workspace_qwen27_FORMAT` |
| Qwen3.6-35B-A3B FP8/GGUF | `/models/Qwen3.6-35B-A3B` | `Qwen/Qwen3.6-35B-A3B-FC` | `/workspace/bfcl_kit/workspace_qwen35_FORMAT` |
| Gemma4-31B FP8 | `/models/gemma-4-31B-it` | `google/gemma-4-31B-it-FC` | `/workspace/bfcl_kit/workspace_gemma31_fp8` |
| Gemma4-26B-A4B FP8 | `/models/gemma-4-26B-A4B-it` | `google/gemma-4-26B-A4B-it-FC` | `/workspace/bfcl_kit/workspace_gemma26_fp8` |

For example:

```bash
cd /llm-scaler/sglang/scripts/bfcl

export TOK_DIR=/models/Qwen3.6-27B
export MODEL_ID=Qwen/Qwen3.6-27B-FC
export WORKDIR=/workspace/bfcl_kit/workspace_qwen27_fp8
export PORT=30000

bash 03_prepare_workspace.sh
```

Use a different workspace for FP8 and GGUF so results are not reused across
formats.

### 5.3 Run a Function Calling smoke test

```bash
export SKIP_START=1
export BFCL_NUM_THREADS=1

bash 04_run.sh simple_python 0-15
```

### 5.4 Run a multi-turn smoke test

```bash
bash 04_run.sh multi_turn_base 6
```

### 5.5 Run all 200 multi-turn cases

```bash
bash 04_run.sh multi_turn_base
```

Use `BFCL_NUM_THREADS=1` for the safest XPU validation. Higher client
concurrency should be tested separately only after the model is stable.
Results are saved under the selected `WORKDIR` in `result/`, `score/`, and
`gen_*.log`.

## 6. Run serving performance tests

Use SGLang's online serving benchmark with a fixed random workload. Keep the
same input length, output length, prompt count, and concurrency when comparing
formats.

Set the tokenizer directory for the running model:

```bash
export TOK_DIR=/models/Qwen3.6-27B
export RESULT_PREFIX=/workspace/results/qwen27_fp8
mkdir -p /workspace/results
```

For Qwen GGUF, continue to use the matching HF directory as `TOK_DIR`.

### 6.1 Single-concurrency latency

This run is useful for comparing time to first token (TTFT), time per output
token (TPOT), and inter-token latency (ITL):

```bash
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 \
  --port 30000 \
  --dataset-name random \
  --tokenizer "$TOK_DIR" \
  --num-prompts 20 \
  --random-input-len 1024 \
  --random-output-len 256 \
  --random-range-ratio 1 \
  --request-rate inf \
  --max-concurrency 1 \
  --warmup-requests 2 \
  --output-file "${RESULT_PREFIX}_c1.jsonl"
```

### 6.2 Concurrent serving throughput

```bash
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 \
  --port 30000 \
  --dataset-name random \
  --tokenizer "$TOK_DIR" \
  --num-prompts 64 \
  --random-input-len 1024 \
  --random-output-len 256 \
  --random-range-ratio 1 \
  --request-rate inf \
  --max-concurrency 8 \
  --warmup-requests 2 \
  --output-file "${RESULT_PREFIX}_c8.jsonl"
```

Gemma4 launch scripts currently default to `MAX_RUNNING_REQUESTS=1`. Override
that setting when starting the service if the purpose is to measure concurrent
throughput:

```bash
MAX_RUNNING_REQUESTS=8 bash scripts/run_gemma4_31b.sh
```

Increase concurrency gradually (`1`, `2`, `4`, `8`) rather than starting at a
high value. Record at least:

- successful and failed requests;
- request throughput;
- input and output token throughput;
- mean and percentile TTFT;
- mean and percentile TPOT;
- mean and percentile ITL.

## 7. Record results

Use one row for every model-format combination:

| Model | Format | GSM8K | BFCL simple | BFCL multi-turn | Concurrency | Input tokens/s | Output tokens/s | Mean TTFT | P99 TTFT | Mean TPOT | Errors |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.6-27B | FP8 |  |  |  | 1 |  |  |  |  |  |  |
| Qwen3.6-27B | GGUF Q4_K_M |  |  |  | 1 |  |  |  |  |  |  |
| Qwen3.6-35B-A3B | FP8 |  |  |  | 1 |  |  |  |  |  |  |
| Qwen3.6-35B-A3B | GGUF UD-Q4_K_M |  |  |  | 1 |  |  |  |  |  |  |
| Gemma4-31B | FP8 |  |  |  | 1 |  |  |  |  |  |  |
| Gemma4-26B-A4B | FP8 |  |  |  | 1 |  |  |  |  |  |  |

Always record the image tag, source commit, model path, GPU affinity, tensor
parallel size, and launch command alongside the result table.

A configuration passes the workflow check when:

- both health endpoints respond successfully;
- GSM8K finishes without request errors or empty/runaway output anomalies;
- BFCL generation and evaluation both finish and produce a score file;
- the performance benchmark completes all requests without server errors;
- the server log contains no XPU out-of-resources, device-lost, scheduler, or
  worker exceptions.

Accuracy and performance are comparison results rather than universal pass
thresholds. Compare formats only when their prompts, sampling parameters,
input/output lengths, concurrency, image, source commit, and GPU allocation are
identical.
