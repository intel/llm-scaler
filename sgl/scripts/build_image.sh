#!/usr/bin/env bash
# Build the llm-scaler-sgl BMG image from llm-scaler/sgl.
#
# Run from anywhere (it resolves the Dockerfile path next to itself):
#   ./scripts/build_image.sh
# or with a custom tag / proxy:
#   IMAGE_TAG=mytag http_proxy=http://proxy https_proxy=http://proxy \
#     ./scripts/build_image.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SGL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE_TAG="${IMAGE_TAG:-llm-scaler-sgl:bmg}"

echo "Building ${IMAGE_TAG} from ${SGL_DIR}"
echo "  http_proxy=${http_proxy:-<unset>}"
echo "  https_proxy=${https_proxy:-<unset>}"

export DOCKER_BUILDKIT=1

exec docker buildx build --load -t "${IMAGE_TAG}" \
    --build-arg "http_proxy=${http_proxy:-}" \
    --build-arg "https_proxy=${https_proxy:-}" \
    "${SGL_DIR}"