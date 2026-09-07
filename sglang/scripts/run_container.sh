#!/usr/bin/env bash
# Start a container in the background; no model service is launched.
#
# Quick start from the llm-scaler repository root:
#   cd sglang
#
# Start with an existing image (replace MODEL_DIR with your host model root):
#   IMAGE_TAG=llm-scaler-sglang:dev-0907 \
#     MODEL_DIR=/path/to/models \
#     CONTAINER_NAME=sglang-dev \
#     bash scripts/run_container.sh
#
# Enter the running container manually:
#   docker exec -it sglang-dev bash
#
# Stop and delete it before creating a replacement (container changes are lost):
#   docker stop sglang-dev
#   docker rm sglang-dev
#
# IMAGE_TAG is required: pass the full image name and tag, dev or release.
# CONTAINER_NAME defaults to sglang-container; the example above overrides it.
# Optionally set SHM_SIZE (default 16g).
# Select GPUs when starting a service inside the container, not here.
#
# MODEL_DIR is the host model root mounted read-only at /models.
# Uses host networking; choose an unused host port when starting a service.
# Containers are retained on exit. Restart with: docker start sglang-dev
# Set CONTAINER_NAME to create another container without a name conflict.

set -euo pipefail

if [[ $# -gt 0 ]]; then
    echo "Usage: IMAGE_TAG=image:tag MODEL_DIR=/path/to/models $0" >&2
    exit 2
fi
if [[ -z "${IMAGE_TAG:-}" ]]; then
    echo "IMAGE_TAG must specify the image to run (for example, llm-scaler-sglang:dev-0907)." >&2
    exit 2
fi
if [[ -z "${MODEL_DIR:-}" || ! -d "$MODEL_DIR" ]]; then
    echo "MODEL_DIR must point to an existing host model directory." >&2
    exit 2
fi
MODEL_DIR="$(cd "$MODEL_DIR" && pwd)"

CONTAINER_NAME="${CONTAINER_NAME:-sglang-container}"
SHM_SIZE="${SHM_SIZE:-16g}"

args=(
    -dit
    --name "$CONTAINER_NAME"
    --device=/dev/dri
    --mount type=bind,src=/dev/dri/by-path,dst=/dev/dri/by-path,readonly
    --shm-size "$SHM_SIZE"
    --net=host
    --mount "type=bind,src=${MODEL_DIR},dst=/models,readonly"
    --env "no_proxy=${no_proxy:-localhost,127.0.0.1,::1}"
    --env "NO_PROXY=${no_proxy:-localhost,127.0.0.1,::1}"
)

echo "Starting ${CONTAINER_NAME} from ${IMAGE_TAG}"
echo "  models=${MODEL_DIR} network=host"
exec docker run "${args[@]}" --entrypoint /bin/bash "$IMAGE_TAG" -i
