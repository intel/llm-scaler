# BFCL Evaluation Kit on Intel XPU (SGLang)

This README provides a step-by-step guide to installing and running the
Berkeley Function Calling Leaderboard (BFCL) inside the GPU container. The
examples use `/llm-scaler/sglang`; replace it with
`/workspace/llm-scaler/sglang` in a source-mounted development container.

This workflow has been validated on Intel XPU with the following FP8 models:

- Qwen3.6-27B
- Qwen3.6-35B-A3B
- Gemma4-31B
- Gemma4-26B-A4B

## Prerequisites

Ensure the Intel oneAPI environment and SGLang are available in the container.

## Step 1. Define the kit root directory

```bash
export KIT_ROOT="/workspace/bfcl_kit"
mkdir -p "$KIT_ROOT"
cd "$KIT_ROOT"
```

## Step 2. Clone the BFCL fork

The fork adds native SGLang Function Calling handlers for Qwen3.6 and Gemma4.

```bash
git clone https://github.com/liu-shaojun/gorilla.git vendor
```

## Step 3. Check out the pinned commit

```bash
cd "$KIT_ROOT/vendor"
git checkout 9d49adb45bd765794fd6b0a0f8006e0b31b997d2
```

## Step 4. Configure environment variables

Choose one matching set of `TOK_DIR`, `MODEL_ID`, and `WORKDIR`:

| Model | `TOK_DIR` | `MODEL_ID` | `WORKDIR` |
|---|---|---|---|
| Qwen3.6-27B | `/models/Qwen3.6-27B` | `Qwen/Qwen3.6-27B-FC` | `/workspace/bfcl_kit/workspace_qwen27` |
| Qwen3.6-35B-A3B | `/models/Qwen3.6-35B-A3B` | `Qwen/Qwen3.6-35B-A3B-FC` | `/workspace/bfcl_kit/workspace_qwen35` |
| Gemma4-31B | `/models/gemma-4-31B-it` | `google/gemma-4-31B-it-FC` | `/workspace/bfcl_kit/workspace_gemma31` |
| Gemma4-26B-A4B | `/models/gemma-4-26B-A4B-it` | `google/gemma-4-26B-A4B-it-FC` | `/workspace/bfcl_kit/workspace_gemma26` |

For example, to evaluate Qwen3.6-35B-A3B:

```bash
export TOK_DIR="/models/Qwen3.6-35B-A3B"
export MODEL_ID="Qwen/Qwen3.6-35B-A3B-FC"
export WORKDIR="/workspace/bfcl_kit/workspace_qwen35"
export PORT="30000"
```

`TOK_DIR` must point to the tokenizer files for the selected model. Use a
different `WORKDIR` for each model to keep their results separate.

## Step 5. Install BFCL

Run the installation script from the BFCL scripts directory. It installs the
pinned vendored checkout and verifies that all four model IDs resolve to the
expected handlers.

```bash
cd /llm-scaler/sglang/scripts/bfcl

bash 02_install_bfcl_fork.sh
```

## Step 6. Prepare the workspace

Generate `RUN_CONFIG.sh` and stage the tokenizer configuration:

```bash
cd /llm-scaler/sglang/scripts/bfcl

bash 03_prepare_workspace.sh
```

Run this step again with matching `TOK_DIR` and `WORKDIR` values when switching
models.

## Step 7. Start the model service

Start one model at a time and keep the service running. Open another terminal
in the same container for Step 8.

### Qwen3.6-27B

```bash
cd /llm-scaler/sglang

MODEL_PATH=/models/Qwen3.6-27B \
ZE_AFFINITY_MASK=6,7 \
TP_SIZE=2 \
HOST=0.0.0.0 \
PORT=30000 \
bash scripts/start_qwen3_6_service.sh
```

### Qwen3.6-35B-A3B

```bash
cd /llm-scaler/sglang

MODEL_PATH=/models/Qwen3.6-35B-A3B \
ZE_AFFINITY_MASK=6,7 \
TP_SIZE=2 \
HOST=0.0.0.0 \
PORT=30000 \
bash scripts/start_qwen3_6_service.sh
```

### Gemma4-31B

```bash
cd /llm-scaler/sglang

MODEL_PATH=/models/gemma-4-31B-it \
ZE_AFFINITY_MASK=6,7 \
HOST=127.0.0.1 \
PORT=30000 \
bash scripts/run_gemma4_31b.sh
```

### Gemma4-26B-A4B

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

Wait for the service to become ready before starting BFCL:

```bash
curl http://127.0.0.1:30000/health
```

## Step 8. Run the BFCL evaluation

In the second terminal, select the same model configuration used in Step 4.
Set `SKIP_START=1` so `04_run.sh` connects to the service started in Step 7
instead of launching the bundled reference server. For Qwen3.6-35B-A3B:

```bash
cd /llm-scaler/sglang/scripts/bfcl

export MODEL_ID="Qwen/Qwen3.6-35B-A3B-FC"
export WORKDIR="/workspace/bfcl_kit/workspace_qwen35"
export SKIP_START=1
export BFCL_NUM_THREADS=1

bash 04_run.sh multi_turn_base
```

Usage options for `04_run.sh`:

```bash
# Run all 200 multi_turn_base cases.
bash 04_run.sh multi_turn_base

# Run one case as a smoke test.
bash 04_run.sh multi_turn_base 6

# Run cases base_0 through base_29.
bash 04_run.sh multi_turn_base 0-29

# Run an explicit list of cases.
IDS="6,10,42" bash 04_run.sh multi_turn_base
```

Keep the exported `SKIP_START`, `MODEL_ID`, and `WORKDIR` values when using
these selectors.

For the safest XPU validation, `BFCL_NUM_THREADS=1` is recommended because only
batch size 1 is guaranteed to be valid; Batch Invariant has not been achieved
for vLLM on XPU. `04_run.sh` currently defaults to 16 BFCL client workers when
the variable is not set, but higher concurrency should only be used after the
selected model and server configuration have been verified. The server may
also queue or limit requests according to its own `--max-running-requests`
setting, so the worker count does not guarantee the same device batch size.

Results are written under the selected `WORKDIR` in `result/`, `score/`, and
the timestamped `gen_*.log`.
