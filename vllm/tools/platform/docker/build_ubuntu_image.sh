#!/bin/bash
set -e

# Help message
usage() {
    echo "Usage: $0 [-n image_name:tag]"
    echo "Default image name: ubuntu:25.04-custom"
    exit 1
}

# Default image name
IMAGE_NAME="ubuntu:25.04-custom"

# Parse options
while getopts ":n:h" opt; do
  case ${opt} in
    n )
      IMAGE_NAME=$OPTARG
      ;;
    h )
      usage
      ;;
    \? )
      echo "Invalid option: -$OPTARG" >&2
      usage
      ;;
  esac
done

TAR_NAME="ubuntu-2504-rootfs.tar.gz"

echo "[+] Image name: $IMAGE_NAME"
echo "[+] Creating root filesystem archive..."

sudo tar --numeric-owner -czpf "$TAR_NAME" \
    --exclude=/proc \
    --exclude=/sys \
    --exclude=/dev \
    --exclude=/tmp/* \
    --exclude=/run/* \
    --exclude=/mnt \
    --exclude=/media \
    --exclude=/lost+found \
    --exclude=/var/tmp/* \
    --exclude=/home \
    --exclude=/root \
    --exclude=/etc/ssh \
    --exclude=/etc/hostname \
    --exclude=/etc/hosts \
    /

echo "[+] Archive created: $TAR_NAME"

echo "[+] Importing into Docker as image: $IMAGE_NAME"
cat "$TAR_NAME" | docker import - "$IMAGE_NAME"

echo "[✔] Done!"
echo "You can run the image using:"
echo "    docker run -it $IMAGE_NAME bash"
