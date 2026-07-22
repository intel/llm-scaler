set -euo pipefail

HTTP_PROXY="${HTTP_PROXY:-${http_proxy:-}}"
HTTPS_PROXY="${HTTPS_PROXY:-${https_proxy:-${HTTP_PROXY}}}"
NO_PROXY="${NO_PROXY:-${no_proxy:-localhost,127.0.0.1,::1,intel.com,.intel.com}}"
export HTTP_PROXY HTTPS_PROXY NO_PROXY

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERSION_FILE="${SCRIPT_DIR}/omni_xpu_kernel/omni_xpu_kernel/_version.py"
TAG="$(sed -n 's/^__image_version__ = "\([^"]*\)"$/\1/p' "${VERSION_FILE}")"
if [ -z "${TAG}" ]; then
    echo "Unable to read Omni version from ${VERSION_FILE}" >&2
    exit 1
fi

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
INSTALL_OPTIONAL_NODES="${INSTALL_DISABLED_NODES:-true}"
IMAGE_REPOSITORY="${OMNI_IMAGE_REPOSITORY:-intel/llm-scaler-omni}"

cd "${SCRIPT_DIR}"
set -x

DOCKER_BUILDKIT=1 docker build -f ./docker/Dockerfile . \
    -t "${IMAGE_REPOSITORY}:${TAG}-${DEVICE_TARGET}" \
    --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
    --build-arg "IMAGE_TAG=${TAG}" \
    --build-arg "XPU_TARGET=${DEVICE_TARGET}" \
    --build-arg "MAX_JOBS=${BUILD_MAX_JOBS}" \
    --build-arg "INSTALL_DISABLED_NODES=${INSTALL_OPTIONAL_NODES}" \
    --build-arg "https_proxy=${HTTPS_PROXY}" \
    --build-arg "http_proxy=${HTTP_PROXY}" \
    --build-arg "no_proxy=${NO_PROXY}"
