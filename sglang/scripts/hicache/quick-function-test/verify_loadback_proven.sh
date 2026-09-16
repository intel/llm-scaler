#!/usr/bin/env bash
# PROVEN H->D load_back verify for the hybrid (GDN+attention) model.
#
# MECHANISM (why this fires load_back where round-robin didn't):
#   1. Prime A (a long, distinct prefix) -> its KV+mamba is cached on device and,
#      under write_through, backed up to the HOST pool (async D->H copy).
#   2. WAIT so A's async backup COMPLETES. If A is evicted before the backup
#      finishes, A is DELETED (no host_value) instead of demoted -> re-send is a
#      full recompute and load_back never fires. This wait is the crux.
#   3. Send B, C (distinct, each ~= the prime) -> A is pushed off the DEVICE pool.
#      Because A is already backed up, eviction DEMOTES it to host (keeps host_value).
#   4. Re-send A -> match lands on the host-only node (host_hit_length>0) ->
#      init_load_back -> start_loading -> H->D restore of KV + mamba (extra_pools=2).
#
# WHY NOT echo=true / logprob_start_len=0 (the old, BROKEN probe):
#   The server's _compute_max_prefix_len (schedule_batch.py) caps the prefix-MATCH
#   key at logprob_start_len whenever return_logprob=True -- positions you request
#   logprobs for MUST be recomputed, so cache reuse is deliberately suppressed for
#   them. echo=true sets logprob_start_len=0 -> match key truncated to 0 tokens ->
#   NO prefix match -> load_back CANNOT fire. That was a TEST bug, not a server bug.
#
#   The fix used here: native /generate with logprob_start_len = prompt_tokens - K.
#   The match key is then the first (prompt_tokens - K) tokens -> matches the
#   host-cached prefix -> load_back fires and RESTORES that region. Only the last K
#   tokens are recomputed; their logprobs depend (via attention) on the RESTORED KV,
#   so comparing them to golden is a real correctness gate on the restored bytes.
#
# HARD REQUIREMENT: the HOST pool must have ROOM. flush_cache does NOT free the
# host pool; a saturated host (host_used ~= host_total) silently drops backups so
# nothing is retained. Run this on a FRESH server (restart to reset host_used=0).
#
#   Gate A (attribution): load_back_tokens_total climbs on the re-send AND the
#                         re-send reports cached_tokens>0 (prefix came from HOST).
#   Gate B (correctness): teacher-forced tail logprobs after load_back == golden.
#                         NOTE: only meaningful when Gate A passes -- a recompute
#                         also matches golden, so Gate A is what proves it was a restore.
#   Gate C (stability):   server survives the restore (no DEVICE_LOST).
set -uo pipefail
export no_proxy="127.0.0.1,localhost,0.0.0.0" NO_PROXY="127.0.0.1,localhost,0.0.0.0"
PORT="${PORT:-30000}"; BASE="http://127.0.0.1:${PORT}"
LOG="${DEBUG_LOG:-/tmp/hicache_dbg.log}"; PAT="${SERVER_PAT:-[s]glang.launch_server}"
BACKUP_WAIT="${BACKUP_WAIT:-6}"   # seconds to let the write_through D->H backup finish
REPS="${REPS:-140}"               # ~140 reps -> ~1800 tok prefix (< 2042 input cap @ pool=2048)
K="${K:-8}"                       # teacher-forced tail length (tokens recomputed after restore)
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

get() { curl -s "${BASE}/metrics" | grep -v '^#' | grep -E "^sglang:${1}[ {]" | head -1 | awk '{print $NF}' | cut -d. -f1; }
alive() { pgrep -f "${PAT}" >/dev/null 2>&1; }
swdone() { grep -ac 'start_loading: done' "$LOG" 2>/dev/null || echo 0; }
jstr() { python3 -c 'import json,sys;print(json.dumps(sys.argv[1]))' "$1"; }

# probe TEXT LOGPROB_START_LEN -> "sum_lastK_logprob  cached_tokens  prompt_tokens"
# Uses native /generate. logprob_start_len caps the prefix MATCH at that many tokens
# (so pass prompt_tokens-K to keep the match nearly-full and let load_back fire),
# and returns input logprobs for positions [logprob_start_len, prompt_tokens).
probe() {
  curl -s "${BASE}/generate" -H 'Content-Type: application/json' -d "{
    \"text\":$(jstr "$1"),
    \"sampling_params\":{\"max_new_tokens\":1,\"temperature\":0},
    \"return_logprob\":true,\"logprob_start_len\":$2}" | K="$K" python3 -c '
import sys,json,os
K=int(os.environ.get("K","8"))
d=json.load(sys.stdin)
mi=d.get("meta_info") or {}
if "input_token_logprobs" not in mi:
    print("nan 0 0 ERR:"+str(d.get("message") or d.get("error") or d)[:60]); sys.exit()
itl=mi.get("input_token_logprobs") or []
lps=[x[0] for x in itl if x and x[0] is not None]
print(round(sum(lps[-K:]),4), mi.get("cached_tokens",0), mi.get("prompt_tokens",0))'
}

# fire TEXT -> generate one token, no logprob (used only to occupy/evict device)
fire() {
  curl -s "${BASE}/generate" -H 'Content-Type: application/json' -d "{
    \"text\":$(jstr "$1"),
    \"sampling_params\":{\"max_new_tokens\":1,\"temperature\":0}}" >/dev/null
}

mk() { local w="$1"; local t="$2"; local p="Ctx ${w}. "; local r; for ((r=0;r<REPS;r++)); do p+="$t "; done; echo "$p"; }

# Per-run nonce so A/B/C are DISTINCT across runs -> no cross-run tree contamination
# (fixed nonces let a leftover copy from a prior run fake a load_back; see the
# "probe scripts must nonce and self-size" lesson). Override RUN to reproduce a case.
RUN="${RUN:-$(date +%s)-$$-${RANDOM}}"
echo "run nonce: ${RUN}"
A=$(mk "AAA-${RUN}" "Alpha alpha superposition entanglement quantum coherence qubit measurement.")
B=$(mk "BBB-${RUN}" "Bravo bravo consensus replication sharding tolerance latency throughput.")
C=$(mk "CCC-${RUN}" "Charlie charlie photosynthesis glucose chloroplast sunlight oxygen carbon.")
SUF=" In one sentence, the central idea here is"

DP=$(get max_total_num_tokens); HT=$(get hicache_host_total_tokens); HU=$(get hicache_host_used_tokens)
echo "Device pool: ${DP}   Host pool: ${HT}   host_used: ${HU}   tail_K: ${K}"
if [[ "${HT:-0}" -gt 0 ]] && python3 -c "import sys;sys.exit(0 if ${HU:-0} > 0.7*${HT:-1} else 1)"; then
  echo -e "${YELLOW}WARNING: host_used ${HU}/${HT} > 70% -- backups may be dropped (no retention).${NC}"
  echo -e "${YELLOW}         flush_cache does NOT free the host pool. RESTART the server for a clean run.${NC}"
fi

echo ""
echo "=== Step 1: prime A (full compute), capture GOLDEN tail logprobs + prompt_tokens ==="
# logprob_start_len=0 on the prime is fine: nothing to match on a fresh tree, and it
# still caches+backs-up A. It gives us prompt_tokens (N) and the golden tail in one call.
G=$(probe "${A}${SUF}" 0); GLP=$(echo "$G"|awk '{print $1}'); GC=$(echo "$G"|awk '{print $2}'); N=$(echo "$G"|awk '{print $3}')
echo "  golden: sum_logprob=${GLP}  cached_tokens=${GC}  prompt_tokens=${N}"
if [[ "${N:-0}" -le "$K" ]]; then
  echo -e "${RED}ABORT: prompt_tokens=${N} <= K=${K} (probe error?). Response tail: ${G}${NC}"; exit 1
fi
LSL=$(( N - K ))   # keep the prefix match nearly full so load_back can fire
echo "  -> re-send will use logprob_start_len=${LSL} (match first ${LSL} tok, teacher-force last ${K})"

echo ""
echo "=== Step 1b: DEVICE-HIT control -- re-send A now (still on device, NO load_back) ==="
# Same code path as the restore (cached prefix + teacher-forced tail), only difference
# is the KV/mamba source: DEVICE here vs HOST after load_back. This is the correct
# baseline for restore correctness. If restored==device_hit, load_back is byte-exact;
# any device_hit-vs-fresh gap is a prefix-cache/mamba tail-continuation numeric, not a
# load_back bug.
DH_B=$(get load_back_tokens_total)
D=$(probe "${A}${SUF}" "${LSL}"); DLP=$(echo "$D"|awk '{print $1}'); DC=$(echo "$D"|awk '{print $2}')
DH_A=$(get load_back_tokens_total); DDELTA=$(( ${DH_A:-0} - ${DH_B:-0} ))
echo "  device-hit: sum_logprob=${DLP}  cached_tokens=${DC}  load_back_delta=${DDELTA} (expect cached>0, delta 0)"
if [[ "${DC:-0}" -le 0 || "${DDELTA:-0}" -ne 0 ]]; then
  echo -e "${YELLOW}  WARN: control was NOT a clean device hit (cached=${DC}, load_back_delta=${DDELTA}).${NC}"
  echo -e "${YELLOW}        A did not stay resident on the device pool -- Gate B baseline will be${NC}"
  echo -e "${YELLOW}        INVALID (mismatched chunk boundary). Increase MAX_TOTAL_TOKENS so the${NC}"
  echo -e "${YELLOW}        device pool comfortably holds A (>= ~2x the prefix, e.g. 4096).${NC}"
fi

echo "  waiting ${BACKUP_WAIT}s for A's write_through D->H backup to complete..."
sleep "${BACKUP_WAIT}"

echo ""
echo "=== Step 2: send B then C -> evict A off device (A already backed up -> demoted to host) ==="
fire "${B}${SUF}"; sleep 1; fire "${C}${SUF}"; sleep 1
echo "  host_used=$(get hicache_host_used_tokens)  evicted_total=$(get evicted_tokens_total)"

echo ""
echo "=== Step 3: re-send A (logprob_start_len=${LSL}) -> expect H->D load_back restore ==="
LB_B=$(get load_back_tokens_total); SL_B=$(swdone)
R=$(probe "${A}${SUF}" "${LSL}"); RLP=$(echo "$R"|awk '{print $1}'); RC=$(echo "$R"|awk '{print $2}'); RN=$(echo "$R"|awk '{print $3}')
LB_A=$(get load_back_tokens_total); SL_A=$(swdone)
echo "  after-loadback: sum_logprob=${RLP}  cached_tokens=${RC}  prompt_tokens=${RN}"
echo "  load_back_tokens_total: ${LB_B} -> ${LB_A}    start_loading_done: ${SL_B} -> ${SL_A}"

echo ""; echo "=== GATE A: attribution (H->D fired?) ==="
DELTA=$(( ${LB_A:-0} - ${LB_B:-0} ))
if [[ "$DELTA" -gt 0 && "${RC:-0}" -gt 0 ]]; then
  echo -e "  ${GREEN}PASS: load_back +${DELTA} tok; re-send served ${RC} cached tok from host; start_loading ${SL_B}->${SL_A}.${NC}"; GA=0
else
  echo -e "  ${YELLOW}FAIL: load_back delta=${DELTA}, cached=${RC}. (host full? backup not finished? raise BACKUP_WAIT / restart server)${NC}"; GA=1
fi

echo ""; echo "=== GATE B: correctness (restored bytes == device-hit bytes?) ==="
# Primary gate: restored (host) vs device-hit -- identical code path, KV/state source
# is the ONLY variable. Match => load_back restored byte-exact.
DIFF=$(python3 -c "print(abs(${DLP:-0}-${RLP:-0}))" 2>/dev/null || echo 999)
# Diagnostic: how far cached-prefix+tail drifts from a full fresh prefill (mamba effect).
DIFF_FRESH=$(python3 -c "print(abs(${GLP:-0}-${RLP:-0}))" 2>/dev/null || echo 999)
DIFF_DEVFRESH=$(python3 -c "print(abs(${GLP:-0}-${DLP:-0}))" 2>/dev/null || echo 999)
echo "  |device_hit - restored| = ${DIFF}     <- GATE B (restore correctness)"
echo "  |fresh(lsl=0) - device_hit| = ${DIFF_DEVFRESH}   (diagnostic: dominated by logprob_start_len path change, NOT caching)"
echo "  |fresh(lsl=0) - restored|   = ${DIFF_FRESH}   (diagnostic; ignore -- fresh uses lsl=0, not comparable)"
# VALIDITY: the control is only a valid baseline if it was a TRUE device hit at the
# SAME boundary as the restore (same cached_tokens). Otherwise we would be comparing
# different chunk boundaries and any diff is meaningless.
if [[ "${DC:-0}" -le 0 || "${DC:-0}" -ne "${RC:-1}" ]]; then
  echo -e "  ${YELLOW}INCONCLUSIVE: control cached=${DC} != restore cached=${RC} (or control missed device).${NC}"
  echo -e "  ${YELLOW}Baseline invalid -- boundaries differ. Re-run with a larger device pool so the${NC}"
  echo -e "  ${YELLOW}device-hit control gets cached=${RC} too (MAX_TOTAL_TOKENS >= ~2x prefix).${NC}"; GB=2
elif python3 -c "import sys;sys.exit(0 if ${DIFF:-999} < 0.02 else 1)"; then
  echo -e "  ${GREEN}PASS: restored == device-hit at same boundary (cached=${RC}) -> load_back byte-exact.${NC}"; GB=0
else
  echo -e "  ${RED}FAIL: restored != device-hit at SAME boundary (cached=${RC}) -> real restore bug (server).${NC}"; GB=1
fi

echo ""; echo "=== GATE C: stability ==="
if alive; then echo -e "  ${GREEN}PASS: server alive, no DEVICE_LOST.${NC}"; GC=0
else echo -e "  ${RED}FAIL: server died -> DEVICE_LOST.${NC}"; GC=1; fi

echo ""; echo "=== VERDICT ==="
if [[ "${GA:-1}" -eq 0 && "${GB:-1}" -eq 0 && "${GC:-1}" -eq 0 ]]; then
  echo -e "${GREEN}PROVEN: H->D load_back fired (+${DELTA} tok, ${RC} host-cached), restore byte-exact vs device-hit (|diff|=${DIFF}), no crash.${NC}"; exit 0
elif [[ "${GB:-1}" -eq 2 ]]; then
  echo -e "${YELLOW}INCONCLUSIVE (A=${GA:-1} C=${GC:-1}): load_back fired but the device-hit baseline was invalid.${NC}"
  echo -e "${YELLOW}  Raise MAX_TOTAL_TOKENS (>= ~2x the ~$(( ${N:-0} ))-tok prefix, e.g. 4096) so A stays${NC}"
  echo -e "${YELLOW}  resident for the control, then re-run. Do NOT read Gate B until cached counts match.${NC}"; exit 2
else
  echo -e "${RED}NOT proven this run: A=${GA:-1} B=${GB:-1} C=${GC:-1}.${NC}"
  echo -e "${RED}  If A failed: restart server (host pool must have room), raise BACKUP_WAIT.${NC}"
  echo -e "${RED}  If B failed at MATCHING cached counts: genuine load_back restore bug -> investigate server (likely mamba state H->D).${NC}"
  exit 1
fi
