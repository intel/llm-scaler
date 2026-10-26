#!/usr/bin/env bash
# Regenerate runtime-requirements.txt from runtime_requirements.in and the
# patched vLLM requirements. Re-run from any directory with Docker Buildx.
# Existing pins remain preferred; pass --upgrade to deliberately refresh them.
# Set BUILD_CA=/path/to/ca-bundle.crt when a custom CA is required.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
secret=()
[[ -z "${BUILD_CA:-}" ]] || secret=(--secret "id=build-ca,src=${BUILD_CA}")
docker buildx build -f docker/Dockerfile --target runtime-lock-export \
    --no-cache-filter runtime-lock \
    --build-arg "LOCK_COMPILE_ARGS=$*" \
    --build-arg http_proxy --build-arg https_proxy --build-arg no_proxy \
    "${secret[@]}" \
    --output type=local,dest=docker .
