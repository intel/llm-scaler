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
COMFYUI_REPOSITORY="${COMFYUI_REPOSITORY:-https://github.com/Comfy-Org/ComfyUI.git}"
COMFYUI_COMMIT="${COMFYUI_COMMIT:-b1693ecba9f5b65f8c80ab36b195ab963ec92413}"
COMFYUI_VERSION="${COMFYUI_VERSION:-0.30.0}"
COMFYUI_FRONTEND_VERSION="${COMFYUI_FRONTEND_VERSION:-1.47.12}"
COMFYUI_WORKFLOW_TEMPLATES_VERSION="${COMFYUI_WORKFLOW_TEMPLATES_VERSION:-0.11.28}"
COMFYUI_MANAGER_VERSION="${COMFYUI_MANAGER_VERSION:-4.2.2}"
KITCHEN_REPOSITORY="${COMFY_KITCHEN_REPOSITORY:-https://github.com/xiangyuT/comfy-kitchen-xpu.git}"
KITCHEN_COMMIT="${COMFY_KITCHEN_COMMIT:-f7250fa44cb6f593969ba869be803e7d03c80ec8}"
KITCHEN_VERSION="${COMFY_KITCHEN_VERSION:-0.2.26}"
AIMDO_REPOSITORY="${COMFY_AIMDO_REPOSITORY:-https://github.com/xiangyuT/comfy-aimdo.git}"
AIMDO_COMMIT="${COMFY_AIMDO_COMMIT:-6fda6e619e1647134d4ced4370e5fad488779d62}"
AIMDO_VERSION="${COMFY_AIMDO_VERSION:-0.4.13}"
AIMDO_SOURCE_DIR="${COMFY_AIMDO_SOURCE_DIR:-${REPOSITORY_ROOT}/../comfy-aimdo-xpu}"
GGUF_REPOSITORY="${COMFY_GGUF_REPOSITORY:-https://github.com/analytics-zoo/ComfyUI-GGUF-XPU.git}"
GGUF_COMMIT="${COMFY_GGUF_COMMIT:-39671fe73117ba97de7011e7e06e32599dcda06d}"
NUNCHAKU_REPOSITORY="${COMFY_NUNCHAKU_REPOSITORY:-https://github.com/xiangyuT/ComfyUI-nunchaku-XPU.git}"
NUNCHAKU_COMMIT="${COMFY_NUNCHAKU_COMMIT:-5cf4fa9886f45abff102d1dd91af5247b4950148}"
NUNCHAKU_VERSION="${COMFY_NUNCHAKU_VERSION:-1.2.1+xpu.3}"
SYCL_TLA_REPOSITORY="${OMNI_SYCL_TLA_REPOSITORY:-https://github.com/intel/sycl-tla.git}"
SYCL_TLA_COMMIT="${OMNI_SYCL_TLA_COMMIT:-2fc09973bfdf15755090fcb0e3b6ad236408a992}"

DOCKERFILE_PATH="${SCRIPT_DIR}/docker/Dockerfile"
DOCKER_TARGET=runtime-comfyui

if [ ! -f "${DOCKERFILE_PATH}" ]; then
    echo "Dockerfile not found: ${DOCKERFILE_PATH}" >&2
    exit 1
fi

if [ ! -d "${AIMDO_SOURCE_DIR}/.git" ]; then
    echo "Comfy AIMDO source checkout not found: ${AIMDO_SOURCE_DIR}" >&2
    exit 1
fi
AIMDO_SOURCE_REVISION="$(git -C "${AIMDO_SOURCE_DIR}" rev-parse HEAD)"
if [ "${AIMDO_SOURCE_REVISION}" != "${AIMDO_COMMIT}" ]; then
    echo "Comfy AIMDO source revision ${AIMDO_SOURCE_REVISION} does not match ${AIMDO_COMMIT}" >&2
    exit 1
fi
if [ -n "$(git -C "${AIMDO_SOURCE_DIR}" status --porcelain --untracked-files=normal)" ]; then
    echo "Comfy AIMDO source checkout must be clean: ${AIMDO_SOURCE_DIR}" >&2
    exit 1
fi

# Export an exact committed tree as a separate BuildKit context. This avoids
# sending ignored local build products or relying on an unpublished remote
# branch while preserving the full source commit as image provenance.
AIMDO_CONTEXT_DIR="$(mktemp -d)"
trap 'rm -rf -- "${AIMDO_CONTEXT_DIR}"' EXIT
git -C "${AIMDO_SOURCE_DIR}" archive --format=tar "${AIMDO_COMMIT}" \
    | tar -xf - -C "${AIMDO_CONTEXT_DIR}"
printf '%s\n' "${AIMDO_COMMIT}" \
    > "${AIMDO_CONTEXT_DIR}/.omni-source-revision"

IMAGE_NAME="${IMAGE_REPOSITORY}:${TAG}-comfyui-${DEVICE_TARGET}"

cd "${SCRIPT_DIR}"

DOCKER_ARGS=(
    -f "${DOCKERFILE_PATH}"
    --target "${DOCKER_TARGET}"
    -t "${IMAGE_NAME}"
    --build-arg "BASE_IMAGE=${BASE_IMAGE}"
    --build-arg "IMAGE_TAG=${TAG}"
    --build-arg "XPU_TARGET=${DEVICE_TARGET}"
    --build-arg "MAX_JOBS=${BUILD_MAX_JOBS}"
    --build-arg "COMFYUI_REPOSITORY=${COMFYUI_REPOSITORY}"
    --build-arg "COMFYUI_COMMIT=${COMFYUI_COMMIT}"
    --build-arg "COMFYUI_VERSION=${COMFYUI_VERSION}"
    --build-arg "COMFYUI_FRONTEND_VERSION=${COMFYUI_FRONTEND_VERSION}"
    --build-arg "COMFYUI_WORKFLOW_TEMPLATES_VERSION=${COMFYUI_WORKFLOW_TEMPLATES_VERSION}"
    --build-arg "COMFYUI_MANAGER_VERSION=${COMFYUI_MANAGER_VERSION}"
    --build-arg "COMFY_KITCHEN_REPOSITORY=${KITCHEN_REPOSITORY}"
    --build-arg "COMFY_KITCHEN_COMMIT=${KITCHEN_COMMIT}"
    --build-arg "COMFY_KITCHEN_VERSION=${KITCHEN_VERSION}"
    --build-arg "COMFY_AIMDO_REPOSITORY=${AIMDO_REPOSITORY}"
    --build-arg "COMFY_AIMDO_COMMIT=${AIMDO_COMMIT}"
    --build-arg "COMFY_AIMDO_VERSION=${AIMDO_VERSION}"
    --build-arg "COMFY_GGUF_REPOSITORY=${GGUF_REPOSITORY}"
    --build-arg "COMFY_GGUF_COMMIT=${GGUF_COMMIT}"
    --build-arg "COMFY_NUNCHAKU_REPOSITORY=${NUNCHAKU_REPOSITORY}"
    --build-arg "COMFY_NUNCHAKU_COMMIT=${NUNCHAKU_COMMIT}"
    --build-arg "COMFY_NUNCHAKU_VERSION=${NUNCHAKU_VERSION}"
    --build-arg "https_proxy=${HTTPS_PROXY}"
    --build-arg "http_proxy=${HTTP_PROXY}"
    --build-arg "no_proxy=${NO_PROXY}"
    --build-context "comfy_aimdo_source=${AIMDO_CONTEXT_DIR}"
)

DOCKER_ARGS+=(
    --build-arg "SYCL_TLA_REPOSITORY=${SYCL_TLA_REPOSITORY}"
    --build-arg "SYCL_TLA_COMMIT=${SYCL_TLA_COMMIT}"
    --build-arg "LLM_SCALER_SOURCE_REVISION=${SOURCE_REVISION}"
    --build-arg "LLM_SCALER_SOURCE_DIRTY=${SOURCE_DIRTY}"
)

set -x
DOCKER_BUILDKIT=1 docker build "${DOCKER_ARGS[@]}" .
