#!/usr/bin/env bash
# Generate + evaluate BFCL multi_turn against sglang-on-XPU. Launches a FRESH
# server (cold radix — REQUIRED for a clean parity number; reusing a persistent
# server across runs lets radix state accumulate cross-run and contaminates the
# comparison), runs the eval, scores.
#
# Runs INSIDE the container. threads=1 ALWAYS (only bsz=1 is optimized on XPU;
# threads>1 gives meaningless numbers). No `| tail` anywhere (the generate child
# tail blocks on a dead pipe and hangs forever — R5).
#
# Usage (inside container):
#   bash 04_run.sh                       # full multi_turn_base (200)
#   bash 04_run.sh multi_turn_base 6     # single entry base_6 (smoke)
#   bash 04_run.sh multi_turn_base 0-29  # subset base_0..29
#   IDS="6,10,42" bash 04_run.sh         # explicit id list
set -uo pipefail

CAT="${1:-multi_turn_base}"
RANGE="${2:-}"                       # "", "6" (single), or "0-29" (range)
VENV="${VENV:-/opt/venv}"
WORKDIR="${WORKDIR:-/workspace/bfcl_kit/workspace_xpu}"
PORT="${PORT:-9010}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.6-35B-A3B-FC}"
START="${START:-$(cd "$(dirname "$0")" && pwd)/01_start_server.sh}"
TS="$(date +%Y%m%d_%H%M%S)"
SRV="${SRV:-$WORKDIR/srv_${TS}.log}"
GEN="${GEN:-$WORKDIR/gen_${TS}.log}"

source "$VENV/bin/activate"
[ -f "$WORKDIR/RUN_CONFIG.sh" ] || { echo "ERROR: run 03_prepare_workspace.sh first"; exit 1; }
source "$WORKDIR/RUN_CONFIG.sh"

# Build the run-ids filter file if a subset was requested.
RUNIDS_ARG=""
if [[ -n "${IDS:-}" ]]; then
  IDLIST=$(python3 -c "print(','.join(f'\"${CAT}_'+i.strip()+'\"' for i in '${IDS}'.split(',')))")
  echo "{\"$CAT\": [${IDLIST}]}" > "$WORKDIR/test_case_ids_to_generate.json"
  RUNIDS_ARG="--run-ids --allow-overwrite"
elif [[ -n "$RANGE" ]]; then
  if [[ "$RANGE" == *-* ]]; then LO=${RANGE%-*}; HI=${RANGE#*-}; else LO=$RANGE; HI=$RANGE; fi
  IDLIST=$(python3 -c "print(','.join(f'\"${CAT}_{i}\"' for i in range($LO,$HI+1)))")
  echo "{\"$CAT\": [${IDLIST}]}" > "$WORKDIR/test_case_ids_to_generate.json"
  RUNIDS_ARG="--run-ids --allow-overwrite"
else
  rm -f "$WORKDIR/test_case_ids_to_generate.json"   # full category
fi

# 1. Server. By default launch the REFERENCE server fresh (cold radix — required
#    for clean parity). Set SKIP_START=1 to use an already-running external
#    server (BYO server: any OpenAI-compatible endpoint at $REMOTE_OPENAI_BASE_URL);
#    then this script only health-checks it and does NOT start or stop it.
SKIP_START="${SKIP_START:-0}"
if [[ "$SKIP_START" == "1" ]]; then
  echo "[run] SKIP_START=1 — using external server at $REMOTE_OPENAI_BASE_URL"
  curl -sS --max-time 5 "$REMOTE_OPENAI_BASE_URL/models" >/dev/null 2>&1 \
    || { echo "[run] external server not reachable at $REMOTE_OPENAI_BASE_URL"; exit 1; }
  echo "[run] external server reachable"
else
  echo "[run] launching fresh reference server (cold radix) → $SRV"
  setsid bash -c "PORT=$PORT bash '$START' </dev/null >'$SRV' 2>&1" </dev/null >/dev/null 2>&1 &
  for i in $(seq 1 180); do
    grep -q 'ready to roll' "$SRV" 2>/dev/null && break
    # Abort ONLY on real fatals. Anchor to phrases that can't appear in the benign
    # server_args=... config dump (which contains 'custom_sigquit_handler=None' —
    # a bare 'sigquit'/'exception' match there gives a FALSE failure). Exclude the
    # config line explicitly as a belt-and-suspenders guard.
    if grep -E 'Scheduler hit an exception|^Traceback \(most recent|CompilationError:|SIGQUIT received|RuntimeError:' "$SRV" 2>/dev/null | grep -qv 'server_args='; then
      echo "[run] SERVER FAILED TO LOAD — see $SRV"; tail -25 "$SRV"; exit 1
    fi
    sleep 5
  done
  grep -q 'ready to roll' "$SRV" 2>/dev/null || { echo "[run] SERVER READY TIMEOUT — see $SRV"; exit 1; }
  echo "[run] server ready"
fi

# 2. Generate + evaluate. threads=1, temp=0, local tokenizer. NO | tail.
cd "$WORKDIR"
echo "[run] generate: cat=$CAT range='${RANGE:-full}' ids='${IDS:-}' $(date)"
bfcl generate --model "$MODEL_ID" --test-category "$CAT" --skip-server-setup \
  --num-threads 1 --temperature 0.0 --local-model-path "$TOK_DIR" $RUNIDS_ARG </dev/null >>"$GEN" 2>&1
echo "[run] GENDONE $(date)"
EVAL_PARTIAL=""; [ -n "$RUNIDS_ARG" ] && EVAL_PARTIAL="--partial-eval"
bfcl evaluate --model "$MODEL_ID" --test-category "$CAT" $EVAL_PARTIAL </dev/null >>"$GEN" 2>&1
echo "[run] EVALDONE $(date)"

# 3. Report score. Stop the reference server we started (R1: leave GPU clean) —
#    but NEVER touch an external BYO server (SKIP_START=1).
SCORE=$(cat "$WORKDIR"/score/*/multi_turn/*multi_turn_base*_score.json 2>/dev/null | head -1)
echo "[run] SCORE: $SCORE"
echo "$SCORE" >> "$GEN"
if [[ "$SKIP_START" != "1" ]]; then
  pkill -INT -f "sglang.launch_server" 2>/dev/null || true
  sleep 5
fi
echo "[run] DONE  gen=$GEN  srv=$SRV"
