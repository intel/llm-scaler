#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(dirname "$0")"
PATCH_FILE="${SCRIPT_DIR}/../patches/oneapi-samples-enable-correctness-check.patch"

if [[ ! -f "${PATCH_FILE}" ]]; then
  echo "Error: patch file not found at ${PATCH_FILE}" >&2
  exit 1
fi

git clone https://github.com/oneapi-src/oneAPI-samples.git
cd oneAPI-samples
git apply "${PATCH_FILE}"
