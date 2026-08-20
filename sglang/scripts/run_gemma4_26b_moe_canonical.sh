#!/usr/bin/env bash
# Canonical bsz=1 latency profile: radix must be disabled.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export GEMMA4_WORKLOAD_PROFILE=canonical_bench
export RADIX_CACHE=0
export MAX_RUNNING_REQUESTS=1

exec "${SCRIPT_DIR}/run_gemma4_26b_moe.sh"
