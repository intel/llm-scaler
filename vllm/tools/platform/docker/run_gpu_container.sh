#!/bin/bash

# Usage check
if [ $# -ne 2 ]; then
  echo "Usage: $0 <docker-image> <host-directory-to-mount>"
  exit 1
fi

IMAGE_NAME="$1"
HOST_DIR="$2"

# Verify directory exists
if [ ! -d "$HOST_DIR" ]; then
  echo "Error: Directory '$HOST_DIR' does not exist."
  exit 2
fi

# Run the container
docker run \
  -it \
  --privileged \
  --device=/dev/dri \
  $(for dev in /dev/mei*; do echo --device $dev; done) \
  --group-add video \
  --cap-add=SYS_ADMIN \
  --mount type=bind,source=/dev/dri/by-path,target=/dev/dri/by-path \
  --mount type=bind,source=/sys,target=/sys \
  --mount type=bind,source=/dev/bus,target=/dev/bus \
  --mount type=bind,source=/dev/char,target=/dev/char \
  --mount type=bind,source="$(realpath "$HOST_DIR")",target=/mnt/workdir \
  "$IMAGE_NAME" \
  bash
