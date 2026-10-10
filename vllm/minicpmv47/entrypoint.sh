#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -eo pipefail
source /opt/intel/oneapi/setvars.sh --force >/dev/null
exec "$@"
