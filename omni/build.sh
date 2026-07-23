#!/usr/bin/env bash

set -euo pipefail

HTTP_PROXY="${HTTP_PROXY:-${http_proxy:-}}"
HTTPS_PROXY="${HTTPS_PROXY:-${https_proxy:-${HTTP_PROXY}}}"
NO_PROXY="${NO_PROXY:-${no_proxy:-localhost,127.0.0.1,::1,intel.com,.intel.com}}"
export HTTP_PROXY HTTPS_PROXY NO_PROXY

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || true)"
VERSION_FILE="${SCRIPT_DIR}/omni_xpu_kernel/omni_xpu_kernel/_version.py"
TAG="$(sed -n 's/^__image_version__ = "\([^"]*\)"$/\1/p' "${VERSION_FILE}")"
if [ -z "${TAG}" ]; then
    echo "Unable to read Omni version from ${VERSION_FILE}" >&2
    exit 1
fi

DETECTED_SOURCE_REVISION=unknown
DETECTED_SOURCE_DIRTY=unknown
if [ -n "${REPOSITORY_ROOT}" ]; then
    DETECTED_SOURCE_REVISION="$(git -C "${REPOSITORY_ROOT}" rev-parse HEAD)"
    if [ -n "$(git -C "${REPOSITORY_ROOT}" status --porcelain --untracked-files=normal -- omni)" ]; then
        DETECTED_SOURCE_DIRTY=true
    else
        DETECTED_SOURCE_DIRTY=false
    fi
fi
SOURCE_REVISION="${OMNI_SOURCE_REVISION:-${DETECTED_SOURCE_REVISION}}"
SOURCE_DIRTY="${OMNI_SOURCE_DIRTY:-${DETECTED_SOURCE_DIRTY}}"

# XPU_TARGET is the canonical Docker build parameter. Keep OMNI_XPU_DEVICE as
# a backwards-compatible user-facing alias used by existing kernel scripts.
DEVICE_TARGET="${XPU_TARGET:-${OMNI_XPU_DEVICE:-bmg}}"
case "${DEVICE_TARGET}" in
    bmg|ptl-h) ;;
    *)
        echo "Unsupported XPU target '${DEVICE_TARGET}'; use bmg or ptl-h" >&2
        exit 1
        ;;
esac

BASE_IMAGE="${OMNI_BASE_IMAGE:-intel/omix:0.1.0-devel-ubuntu24.04}"
BUILD_MAX_JOBS="${MAX_JOBS:-8}"
IMAGE_REPOSITORY="${OMNI_IMAGE_REPOSITORY:-intel/llm-scaler-omni}"
IMAGE_FLAVOR="${OMNI_IMAGE_FLAVOR:-comfyui}"
KITCHEN_REPOSITORY="${COMFY_KITCHEN_REPOSITORY:-https://github.com/xiangyuT/comfy-kitchen-xpu.git}"
KITCHEN_COMMIT="${COMFY_KITCHEN_COMMIT:-223eea0c931bcf7a1bd0e83631e21b2a58961b28}"
KITCHEN_VERSION="${COMFY_KITCHEN_VERSION:-0.2.18}"
SYCL_TLA_REPOSITORY="${OMNI_SYCL_TLA_REPOSITORY:-https://github.com/intel/sycl-tla.git}"
SYCL_TLA_COMMIT="${OMNI_SYCL_TLA_COMMIT:-2fc09973bfdf15755090fcb0e3b6ad236408a992}"

case "${IMAGE_FLAVOR}" in
    comfyui)
        DOCKERFILE="${OMNI_DOCKERFILE:-./docker/Dockerfile}"
        DOCKER_TARGET="${OMNI_DOCKER_TARGET:-runtime-comfyui}"
        ;;
    full)
        DOCKERFILE="${OMNI_DOCKERFILE:-./docker/Dockerfile.full}"
        DOCKER_TARGET="${OMNI_DOCKER_TARGET:-runtime}"
        ;;
    *)
        echo "Unsupported image flavor '${IMAGE_FLAVOR}'; use comfyui or full" >&2
        exit 1
        ;;
esac

case "${DOCKERFILE}" in
    /*) DOCKERFILE_PATH="${DOCKERFILE}" ;;
    *) DOCKERFILE_PATH="${SCRIPT_DIR}/${DOCKERFILE#./}" ;;
esac

if [ ! -f "${DOCKERFILE_PATH}" ]; then
    echo "Dockerfile not found: ${DOCKERFILE_PATH}" >&2
    exit 1
fi

IMAGE_NAME="${IMAGE_REPOSITORY}:${TAG}-${IMAGE_FLAVOR}-${DEVICE_TARGET}"

cd "${SCRIPT_DIR}"

DOCKER_ARGS=(
    -f "${DOCKERFILE_PATH}"
    --target "${DOCKER_TARGET}"
    -t "${IMAGE_NAME}"
    --build-arg "BASE_IMAGE=${BASE_IMAGE}"
    --build-arg "IMAGE_TAG=${TAG}"
    --build-arg "XPU_TARGET=${DEVICE_TARGET}"
    --build-arg "MAX_JOBS=${BUILD_MAX_JOBS}"
    --build-arg "COMFY_KITCHEN_REPOSITORY=${KITCHEN_REPOSITORY}"
    --build-arg "COMFY_KITCHEN_COMMIT=${KITCHEN_COMMIT}"
    --build-arg "COMFY_KITCHEN_VERSION=${KITCHEN_VERSION}"
    --build-arg "https_proxy=${HTTPS_PROXY}"
    --build-arg "http_proxy=${HTTP_PROXY}"
    --build-arg "no_proxy=${NO_PROXY}"
)

if [ "${IMAGE_FLAVOR}" = "comfyui" ]; then
    DOCKER_ARGS+=(
        --build-arg "SYCL_TLA_REPOSITORY=${SYCL_TLA_REPOSITORY}"
        --build-arg "SYCL_TLA_COMMIT=${SYCL_TLA_COMMIT}"
        --build-arg "LLM_SCALER_SOURCE_REVISION=${SOURCE_REVISION}"
        --build-arg "LLM_SCALER_SOURCE_DIRTY=${SOURCE_DIRTY}"
    )
else
    DOCKER_ARGS+=(
        --build-arg "INSTALL_DISABLED_NODES=${INSTALL_DISABLED_NODES:-true}"
    )
fi

set -x
DOCKER_BUILDKIT=1 docker build "${DOCKER_ARGS[@]}" .
