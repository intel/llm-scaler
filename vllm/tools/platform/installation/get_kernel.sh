#!/bin/bash
# kernel-manager.sh
# Usage:
#   sudo ./kernel-manager.sh <kernel-version> <action>
# Example:
#   sudo ./kernel-manager.sh 6.14.0-1006-intel install
#   sudo ./kernel-manager.sh 6.14.0-1006-intel download

set -e

if [ $# -ne 2 ]; then
    echo "Usage: $0 <kernel-version> <action>"
    echo "  <kernel-version>: e.g. 6.14.0-1006-intel"
    echo "  <action>: install | download"
    exit 1
fi

KERNEL_VERSION="$1"
ACTION="$2"

# Kernel package list
PACKAGES=(
    "linux-image-${KERNEL_VERSION}"
    "linux-modules-${KERNEL_VERSION}"
    "linux-modules-extra-${KERNEL_VERSION}"
    "linux-headers-${KERNEL_VERSION}"
)

echo "Target kernel version: $KERNEL_VERSION"
echo "Action: $ACTION"
echo "Packages:"
for pkg in "${PACKAGES[@]}"; do
    echo "  $pkg"
done

echo
read -p "Do you want to continue? (y/N): " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Update package cache
sudo apt update

# Perform action
if [ "$ACTION" == "install" ]; then
    echo "Installing packages..."
    sudo apt install -y "${PACKAGES[@]}"
    echo "Installation complete. Please reboot to use the new kernel."
elif [ "$ACTION" == "download" ]; then
    echo "Downloading packages..."
    mkdir -p ./kernel-packages-"$KERNEL_VERSION"
    cd ./kernel-packages-"$KERNEL_VERSION"
    apt download "${PACKAGES[@]}"
    echo "Download complete. Packages saved in $(pwd)."
else
    echo "Invalid action: $ACTION"
    echo "Allowed values: install | download"
    exit 1
fi
