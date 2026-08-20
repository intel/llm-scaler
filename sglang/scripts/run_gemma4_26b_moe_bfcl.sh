#!/usr/bin/env bash
# BFCL multi-turn profile: prefix reuse is required.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export GEMMA4_WORKLOAD_PROFILE=bfcl
export RADIX_CACHE=1
export MAX_RUNNING_REQUESTS=1
export SWA_FULL_TOKENS_RATIO=0.2

exec "${SCRIPT_DIR}/run_gemma4_26b_moe.sh"
