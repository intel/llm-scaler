#!/usr/bin/env bash
# Build the release or dev BMG image from llm-scaler/sglang.
#
# Quick start from the llm-scaler repository root (choose one build):
#   cd sglang
#   bash scripts/build_image.sh dev
#   bash scripts/build_image.sh release
#
# Default: release. Image tags use the host's current date:
#   intel/llm-scaler-sglang:YYYYMMDD-dev
#   intel/llm-scaler-sglang:YYYYMMDD-release
#
# Custom image tag / proxy:
#   IMAGE_TAG=intel/llm-scaler-sglang:my-dev \
#     http_proxy=http://proxy https_proxy=http://proxy \
#     bash scripts/build_image.sh dev
#
# Dockerfile and build context resolve relative to this script, not the cwd.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_TYPE="${1:-release}"
if [[ $# -gt 1 || ! "$BUILD_TYPE" =~ ^(release|dev)$ ]]; then
    echo "Usage: $0 [release|dev]" >&2
    exit 2
fi
DOCKERFILE="${SGLANG_DIR}/docker/Dockerfile"
if [[ "$BUILD_TYPE" == dev ]]; then
    DOCKERFILE="${DOCKERFILE}.dev"
fi

IMAGE_TAG="${IMAGE_TAG:-intel/llm-scaler-sglang:$(date +%Y%m%d)-${BUILD_TYPE}}"
SGLANG_CACHEBUST="${SGLANG_CACHEBUST:-$(date +%s)}"

echo "Building ${IMAGE_TAG} from ${SGLANG_DIR}"
echo "  dockerfile=${DOCKERFILE}"
echo "  http_proxy=${http_proxy:-<unset>}"
echo "  https_proxy=${https_proxy:-<unset>}"

exec docker buildx build \
    --load \
    -f "${DOCKERFILE}" \
    -t "${IMAGE_TAG}" \
    --build-arg "SGLANG_CACHEBUST=${SGLANG_CACHEBUST}" \
    --build-arg "http_proxy=${http_proxy:-}" \
    --build-arg "https_proxy=${https_proxy:-}" \
    --build-arg "no_proxy=${no_proxy:-localhost,127.0.0.1,::1}" \
    "${SGLANG_DIR}"
