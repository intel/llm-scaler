#!/usr/bin/env bash
# Run all 200 BFCL multi_turn_base cases against four FP8 models, sequentially.
#
# Run inside the container:
#   cd /llm-scaler/sglang
#   bash scripts/bfcl/06_run_four_fp8_200.sh
#
# Defaults:
#   GPUs:           6,7
#   endpoint:       http://127.0.0.1:30000/v1
#   BFCL workers:   16
#   output root:    /workspace/bfcl_kit/runs/fp8_four_<timestamp>
#
# Optional overrides:
#   ZE_AFFINITY_MASK=6,7 PORT=30000 BFCL_NUM_THREADS=16 \
#   RUN_ROOT=/workspace/bfcl_kit/runs/my_run \
#     bash scripts/bfcl/06_run_four_fp8_200.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV="${VENV:-/opt/venv}"
PORT="${PORT:-30000}"
HOST="${HOST:-127.0.0.1}"
ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-6,7}"
BFCL_NUM_THREADS="${BFCL_NUM_THREADS:-16}"
READY_TIMEOUT="${READY_TIMEOUT:-1800}"
RUN_ROOT="${RUN_ROOT:-/workspace/bfcl_kit/runs/fp8_four_$(date +%Y%m%d_%H%M%S)}"
SUMMARY="${RUN_ROOT}/summary.tsv"
SERVER_PID=""

mkdir -p "${RUN_ROOT}"
printf "model\tstatus\tscore\tworkspace\tserver_log\n" >"${SUMMARY}"

stop_server() {
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill -INT "${SERVER_PID}" 2>/dev/null || true
        for _ in $(seq 1 30); do
            kill -0 "${SERVER_PID}" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "${SERVER_PID}" 2>/dev/null; then
            kill -TERM "${SERVER_PID}" 2>/dev/null || true
        fi
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
    SERVER_PID=""
}

trap stop_server EXIT
trap 'stop_server; exit 130' INT TERM

wait_for_server() {
    local elapsed=0
    while (( elapsed < READY_TIMEOUT )); do
        if curl --noproxy "*" -sf --max-time 5 \
            "http://${HOST}:${PORT}/health" >/dev/null; then
            return 0
        fi
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            return 1
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done
    return 1
}

run_model() {
    local label="$1"
    local model_id="$2"
    local tokenizer_dir="$3"
    local start_script="$4"
    shift 4

    local workspace="${RUN_ROOT}/${label}"
    local server_log="${workspace}/server.log"
    local run_log="${workspace}/run.log"
    local status score

    mkdir -p "${workspace}"
    echo "[$label] starting service"
    env \
        ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK}" \
        HOST="${HOST}" \
        PORT="${PORT}" \
        "$@" \
        bash "${start_script}" >"${server_log}" 2>&1 &
    SERVER_PID=$!

    if ! wait_for_server; then
        echo "[$label] service failed or timed out; see ${server_log}" >&2
        printf "%s\tserver_failed\t\t%s\t%s\n" \
            "${label}" "${workspace}" "${server_log}" >>"${SUMMARY}"
        stop_server
        sleep 5
        return
    fi

    echo "[$label] service ready; preparing BFCL workspace"
    TOK_DIR="${tokenizer_dir}" \
    PORT="${PORT}" \
    VENV="${VENV}" \
    WORKDIR="${workspace}" \
        bash "${SCRIPT_DIR}/03_prepare_workspace.sh" >"${workspace}/prepare.log" 2>&1
    status=$?

    if (( status == 0 )); then
        echo "[$label] running 200 multi_turn_base cases with ${BFCL_NUM_THREADS} workers"
        SKIP_START=1 \
        MODEL_ID="${model_id}" \
        BFCL_NUM_THREADS="${BFCL_NUM_THREADS}" \
        VENV="${VENV}" \
        WORKDIR="${workspace}" \
        no_proxy=localhost,127.0.0.1 \
        NO_PROXY=localhost,127.0.0.1 \
            bash "${SCRIPT_DIR}/04_run.sh" multi_turn_base 2>&1 | tee "${run_log}"
        status=${PIPESTATUS[0]}
    fi

    score="$(
        find "${workspace}/score" -type f \
            -name "*multi_turn_base*_score.json" -print -quit 2>/dev/null \
            | xargs -r head -n 1
    )"
    [[ -n "${score}" ]] || status=1

    if (( status == 0 )); then
        printf "%s\tcompleted\t%s\t%s\t%s\n" \
            "${label}" "${score}" "${workspace}" "${server_log}" >>"${SUMMARY}"
    else
        printf "%s\tbfcl_failed\t%s\t%s\t%s\n" \
            "${label}" "${score}" "${workspace}" "${server_log}" >>"${SUMMARY}"
    fi

    stop_server
    sleep 5
}

run_model \
    qwen27 \
    Qwen/Qwen3.6-27B-FC \
    /models/Qwen3.6-27B \
    "${SGLANG_ROOT}/scripts/start_qwen3_6_service.sh" \
    MODEL_PATH=/models/Qwen3.6-27B \
    TP_SIZE=2 \
    MEM_FRACTION_STATIC=0.8

run_model \
    qwen35 \
    Qwen/Qwen3.6-35B-A3B-FC \
    /models/Qwen3.6-35B-A3B \
    "${SGLANG_ROOT}/scripts/start_qwen3_6_service.sh" \
    MODEL_PATH=/models/Qwen3.6-35B-A3B \
    TP_SIZE=2 \
    MEM_FRACTION_STATIC=0.85

run_model \
    gemma31 \
    google/gemma-4-31B-it-FC \
    /models/gemma-4-31B-it \
    "${SGLANG_ROOT}/scripts/run_gemma4_31b.sh" \
    MODEL_PATH=/models/gemma-4-31B-it \
    MEM_FRACTION_STATIC=0.85 \
    MAX_RUNNING_REQUESTS=1

run_model \
    gemma26 \
    google/gemma-4-26B-A4B-it-FC \
    /models/gemma-4-26B-A4B-it \
    "${SGLANG_ROOT}/scripts/run_gemma4_26b_moe.sh" \
    MODEL_PATH=/models/gemma-4-26B-A4B-it \
    TP_SIZE=2 \
    SGLANG_FP8_DTYPE=e4m3 \
    MEM_FRACTION_STATIC=0.82 \
    MAX_RUNNING_REQUESTS=1

echo "All model attempts finished. Summary: ${SUMMARY}"
column -t -s $'\t' "${SUMMARY}" 2>/dev/null || cat "${SUMMARY}"
