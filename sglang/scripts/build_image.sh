#!/usr/bin/env bash
# Build the llm-scaler-sgl BMG image from llm-scaler/sglang.
#
# Run from anywhere (it resolves the Dockerfile path next to itself):
#   ./scripts/build_image.sh
# or with a custom tag / proxy:
#   IMAGE_TAG=mytag http_proxy=http://proxy https_proxy=http://proxy \
#     ./scripts/build_image.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DOCKERFILE="${SGLANG_DIR}/docker/Dockerfile"

IMAGE_TAG="${IMAGE_TAG:-amr-registry.caas.intel.com/intelanalytics/llm-scaler-sglang:0.5.13-b1}"
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
    "${SGLANG_DIR}"
