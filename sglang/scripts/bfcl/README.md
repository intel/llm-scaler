# BFCL Evaluation on Intel XPU

This directory contains the BFCL setup and evaluation scripts used with an
OpenAI-compatible SGLang endpoint. Run the commands inside the GPU container.
The examples below use the image's `/llm-scaler/sglang` source tree. For a
source-mounted development container, replace it with
`/workspace/llm-scaler/sglang`.

## 1. Install the pinned BFCL fork

The pinned fork adds native SGLang function-calling handlers for Qwen3.6 and
Gemma4:

```text
https://github.com/liu-shaojun/gorilla.git
9d49adb45bd765794fd6b0a0f8006e0b31b997d2
```

Install it into `/opt/venv`:

```bash
cd /llm-scaler/sglang/scripts/bfcl

python3 -m venv --system-site-packages /opt/venv

KIT_ROOT=/workspace/bfcl_kit \
VENV=/opt/venv \
USE_NETWORK=1 \
FORK_DIR=/workspace/bfcl_kit/vendor \
bash 02_install_bfcl_fork.sh
```

The installer checks that all four model IDs resolve to their expected
handlers. For an offline install, stage the pinned repository at
`/workspace/bfcl_kit/vendor` and omit `USE_NETWORK=1`.

## 2. Start one FP8 model

Run one model at a time on XPU 6 and 7. Keep the service running in the first
terminal.

### Qwen3.6-27B

```bash
cd /llm-scaler/sglang

MODEL_PATH=/models/Qwen3.6-27B \
ZE_AFFINITY_MASK=6,7 \
TP_SIZE=2 \
HOST=127.0.0.1 \
PORT=30000 \
bash scripts/start_qwen3_6_service.sh
```

### Qwen3.6-35B-A3B

```bash
cd /llm-scaler/sglang

MODEL_PATH=/models/Qwen3.6-35B-A3B \
ZE_AFFINITY_MASK=6,7 \
TP_SIZE=2 \
HOST=127.0.0.1 \
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

Wait until `http://127.0.0.1:30000/health` is ready before running BFCL.

## 3. Prepare a model workspace

In a second terminal, select the tokenizer directory and a unique workspace:

| Model | `TOK_DIR` | `MODEL_ID` | `WORKDIR` |
|---|---|---|---|
| Qwen3.6-27B | `/models/Qwen3.6-27B` | `Qwen/Qwen3.6-27B-FC` | `/workspace/bfcl_kit/workspace_qwen27` |
| Qwen3.6-35B-A3B | `/models/Qwen3.6-35B-A3B` | `Qwen/Qwen3.6-35B-A3B-FC` | `/workspace/bfcl_kit/workspace_qwen35` |
| Gemma4-31B | `/models/gemma-4-31B-it` | `google/gemma-4-31B-it-FC` | `/workspace/bfcl_kit/workspace_gemma31` |
| Gemma4-26B-A4B | `/models/gemma-4-26B-A4B-it` | `google/gemma-4-26B-A4B-it-FC` | `/workspace/bfcl_kit/workspace_gemma26` |

For example, prepare Qwen3.6-35B-A3B:

```bash
cd /llm-scaler/sglang/scripts/bfcl

TOK_DIR=/models/Qwen3.6-35B-A3B \
PORT=30000 \
VENV=/opt/venv \
WORKDIR=/workspace/bfcl_kit/workspace_qwen35 \
bash 03_prepare_workspace.sh
```

Run this step again with the matching values when switching models.

## 4. Run BFCL against the existing service

`SKIP_START=1` tells `04_run.sh` to use the service started in the first
terminal. It health-checks the endpoint but does not start or stop it.

Quick function-calling and concurrency check:

```bash
SKIP_START=1 \
MODEL_ID=Qwen/Qwen3.6-35B-A3B-FC \
VENV=/opt/venv \
WORKDIR=/workspace/bfcl_kit/workspace_qwen35 \
bash 04_run.sh simple_python 0-15
```

Multi-turn smoke test:

```bash
SKIP_START=1 \
MODEL_ID=Qwen/Qwen3.6-35B-A3B-FC \
VENV=/opt/venv \
WORKDIR=/workspace/bfcl_kit/workspace_qwen35 \
bash 04_run.sh multi_turn_base 6
```

Full multi-turn category:

```bash
SKIP_START=1 \
MODEL_ID=Qwen/Qwen3.6-35B-A3B-FC \
VENV=/opt/venv \
WORKDIR=/workspace/bfcl_kit/workspace_qwen35 \
bash 04_run.sh multi_turn_base
```

`04_run.sh` defaults to 16 BFCL client workers. Override it when needed:

```bash
BFCL_NUM_THREADS=1 SKIP_START=1 MODEL_ID=Qwen/Qwen3.6-35B-A3B-FC \
VENV=/opt/venv WORKDIR=/workspace/bfcl_kit/workspace_qwen35 \
bash 04_run.sh simple_python 0-15
```

The worker count controls concurrent BFCL clients. The server can queue or
limit requests according to its own `--max-running-requests` setting, so 16
workers does not guarantee a device batch size of 16.

Use the same `SKIP_START`, `MODEL_ID`, `VENV`, and `WORKDIR` values with other
selectors:

```bash
SKIP_START=1 MODEL_ID=Qwen/Qwen3.6-35B-A3B-FC \
VENV=/opt/venv WORKDIR=/workspace/bfcl_kit/workspace_qwen35 \
bash 04_run.sh multi_turn_base 0-29

IDS="6,10,42" SKIP_START=1 MODEL_ID=Qwen/Qwen3.6-35B-A3B-FC \
VENV=/opt/venv WORKDIR=/workspace/bfcl_kit/workspace_qwen35 \
bash 04_run.sh multi_turn_base
```

Results are written below the selected `WORKDIR` in `result/`, `score/`, and
the timestamped `gen_*.log`.

## Bundled reference server

Without `SKIP_START=1`, `04_run.sh` invokes `01_start_server.sh`. That script is
only the bundled Qwen3.6-35B-A3B GGUF reference configuration; it is not the
launcher for the four FP8 examples above.
