#!/usr/bin/env bash
# GSM8K limit=20 no-regression gate (baseline 0.95). Run after any kernel/model
# change to confirm general generation isn't broken — orthogonal to the BFCL
# multi-turn correctness. Launches a fresh server, runs evalscope, scores, stops.
# Method is FIXED to match the baseline (R7): limit=20, max_tokens=1024, temp=0,
# enable_thinking=false, /v1/chat/completions. Do NOT change these to "save time".
#
# Runs INSIDE the container.
set -uo pipefail

VENV="${VENV:-/opt/venv}"
GGUF="${GGUF:-/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}"
PORT="${PORT:-9010}"
START="${START:-$(cd "$(dirname "$0")" && pwd)/01_start_server.sh}"
WORKDIR="${WORKDIR:-/workspace/bfcl_kit/workspace_xpu}"
TS="$(date +%Y%m%d_%H%M%S)"
SRV="$WORKDIR/gsm_srv_${TS}.log"
OUT="$WORKDIR/gsm_${TS}.log"

source "$VENV/bin/activate"
mkdir -p "$WORKDIR"

setsid bash -c "PORT=$PORT bash '$START' </dev/null >'$SRV' 2>&1" </dev/null >/dev/null 2>&1 &
for i in $(seq 1 180); do
  grep -q 'ready to roll' "$SRV" 2>/dev/null && break
  sleep 5
done
grep -q 'ready to roll' "$SRV" 2>/dev/null || { echo "SERVER TIMEOUT — see $SRV"; exit 1; }
echo "[gsm] server ready $(date)"

cd /tmp && evalscope eval --model "$GGUF" \
  --api-url "http://127.0.0.1:${PORT}/v1/chat/completions" --eval-type server \
  --datasets gsm8k --limit 20 \
  --generation-config '{"max_tokens":1024,"temperature":0,"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}' \
  --work-dir "$WORKDIR/gsm_work_${TS}" >>"$OUT" 2>&1
echo "[gsm] eval rc=$?"
python3 -c "import json,glob;fs=glob.glob('$WORKDIR/gsm_work_${TS}/*/reports/*/gsm8k.json');print('SCORE (gate ==0.95):',json.load(open(fs[0])).get('score') if fs else 'no report')"
pkill -INT -f "sglang.launch_server" 2>/dev/null || true; sleep 5
echo "[gsm] DONE  log=$OUT"
