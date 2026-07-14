set -euo pipefail

HTTP_PROXY="${HTTP_PROXY:-${http_proxy:-}}"
HTTPS_PROXY="${HTTPS_PROXY:-${https_proxy:-${HTTP_PROXY}}}"
export HTTP_PROXY HTTPS_PROXY

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERSION_FILE="${SCRIPT_DIR}/omni_xpu_kernel/omni_xpu_kernel/_version.py"
TAG="$(sed -n 's/^__version__ = "\([^"]*\)"$/\1/p' "${VERSION_FILE}")"
if [ -z "${TAG}" ]; then
    echo "Unable to read Omni version from ${VERSION_FILE}" >&2
    exit 1
fi

cd "${SCRIPT_DIR}"
set -x

DOCKER_BUILDKIT=1 docker build -f ./docker/Dockerfile . \
    -t "intel/llm-scaler-omni:${TAG}" \
    --build-arg "IMAGE_TAG=${TAG}" \
    --build-arg "https_proxy=${HTTPS_PROXY}" \
    --build-arg "http_proxy=${HTTP_PROXY}"
