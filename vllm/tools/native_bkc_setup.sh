#!/bin/bash
 
###################################################################################
####### Please carefully read below [IMPORTANT NOTES!!!] before your start ########
###################################################################################
 
# Environment Setup Script for Intel GPU Multi-Arc solution Development
#
# [IMPORTANT NOTE!!!]:
# - This is based on Ubuntu 25.04 desktop version: https://releases.ubuntu.com/25.04/ubuntu-25.04-desktop-amd64.iso
# - Must run this script as root to ensure consistent environment.
# - Replace the http_proxy, https_proxy configuration to your own. If your network environment doesn't require proxy, just remove them. 
# - This script will also disable intel iommu through grub configuration intel_iommu=off for best P2P performance over PCIe.
# - This script covers kernel, GPU firmware, grub configuration and docker libraries, other necessary dirvers/tools/scripts will all be inside vllm/platform evaluation
#   docker image.
# - Do reboot to make all changes effect after installation.
 
########################################
##### Do Update the Proxy settings #####
########################################
export https_proxy=http://your-proxy.com:913
export http_proxy=http://your-proxy.com:913
export no_proxy=127.0.0.1
 
if [ "$(id -u)" -ne 0 ]; then
  echo "[ERROR] This script must be run as root. Exiting."
  exit 1
fi
 
# Enable strict mode
set -euo pipefail
trap 'echo "[ERROR] Script failed at line $LINENO."' ERR
 
# Output both to terminal and log file
exec > >(tee -i /var/log/multi_arc_setup_env.log) 2>&1

if [[ "$http_proxy" == "http://your-proxy.com:913" ]] || [[ "$https_proxy" == "http://your-proxy.com:913" ]]; then
  echo "[ERROR] Proxy environment variables http_proxy or https_proxy are still set to default placeholder values."
  echo "Please update them to your actual proxy before running this script."
  exit 1
fi

echo -e "\n[INFO] Starting environment setup..."
 
# Internet access check
echo "[INFO] Testing internet access to www.bing.com ..."
if ! wget -q --spider --timeout=10 https://www.bing.com; then
    echo "[WARNING] Internet access through proxy may be unavailable. Check your connection!"
    exit 1
fi

apt update

# Install kernel
# === Config ===
TARGET_VERSION="6.14.0-15-generic"
SUBMENU_TITLE="Advanced options for Ubuntu"
MENUENTRY_TITLE="Ubuntu, with Linux $TARGET_VERSION"
DEFAULT_FILE="/etc/default/grub"
GRUB_CFG="/boot/grub/grub.cfg"

# === Check running kernel ===
CURRENT_VERSION="$(uname -r)"
echo "Current running kernel: $CURRENT_VERSION"
echo "Target kernel version: $TARGET_VERSION"

if dpkg -s "linux-image-${TARGET_VERSION}" >/dev/null 2>&1; then
    echo "Kernel ${TARGET_VERSION} installed, skip..."
else
    echo "Kernel ${TARGET_VERSION} not installed, start installation..."
    for pkg in \
        "linux-image-$TARGET_VERSION" \
        "linux-headers-$TARGET_VERSION" \
        "linux-modules-$TARGET_VERSION" \
        "linux-modules-extra-$TARGET_VERSION";
    do
        echo "Installing $pkg ..."
        if ! apt install -y "$pkg"; then
            echo "❌ Failed to install $pkg. Check repository or kernel version."
            exit 1
        fi
    done
    echo "✅ Kernel $TARGET_VERSION installed successfully."
fi

# === Check GRUB top-level menu for target kernel ===
echo "🔍 Checking if current GRUB default kernel is '$TARGET_VERSION'..."

FOUND_IN_TOP=0

while IFS= read -r line || [[ -n "$line" ]]; do
    clean_line="$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"

    if [[ "$clean_line" =~ ^submenu ]]; then
        break
    fi

    if [[ "$clean_line" =~ ^menuentry[[:space:]]\'Ubuntu,[[:space:]]with[[:space:]]Linux[[:space:]]$TARGET_VERSION ]]; then
        FOUND_IN_TOP=1
        echo "✅ Found target kernel in top-level GRUB menu. No update needed."
        break
    fi
done < "$GRUB_CFG"

# === Set default GRUB entry if not in top menu ===
if [[ "$FOUND_IN_TOP" -ne 1 ]]; then
    echo "⚙️ Setting default GRUB entry to: $SUBMENU_TITLE > $MENUENTRY_TITLE"
    grub-set-default "$SUBMENU_TITLE>$MENUENTRY_TITLE"

    # === Ensure GRUB uses 'saved' as default ===
    if grep -q '^GRUB_DEFAULT=' "$DEFAULT_FILE"; then
        sed -i 's/^GRUB_DEFAULT=.*/GRUB_DEFAULT=saved/' "$DEFAULT_FILE"
    else
        echo 'GRUB_DEFAULT=saved' >> "$DEFAULT_FILE"
    fi

    echo "🔁 Updating GRUB configuration..."
    update-grub
fi

# Install GPU FW
WORK_DIR=/tmp/multi-arc

# Only create if it doesn't exist
if [[ ! -d "$WORK_DIR" ]]; then
  mkdir -p "$WORK_DIR"
fi

echo -e "\n[INFO] Downloading and installing GPU firmware..."

FIRMWARE_DIR="$WORK_DIR/firmware"
mkdir -p "$FIRMWARE_DIR"
cd "$FIRMWARE_DIR"
rm -rf ./*

wget https://gitlab.com/kernel-firmware/linux-firmware/-/raw/main/xe/bmg_guc_70.bin
wget https://gitlab.com/kernel-firmware/linux-firmware/-/raw/main/xe/bmg_huc.bin

zstd -1 bmg_guc_70.bin -o bmg_guc_70.bin.zst
zstd -1 bmg_huc.bin -o bmg_huc.bin.zst

if [ -d /lib/firmware/xe ]; then
  cp *.zst /lib/firmware/xe
else
  echo "[ERROR] /lib/firmware/xe does not exist. Ensure your system supports Xe firmware."
  exit 1
fi

update-initramfs -u
rm $FIRMWARE_DIR/* -rf
echo -e "✅ Update GPU firmware successfully"

echo -e "\n[INFO] Disabling intel_iommu..."
GRUB_FILE="/etc/default/grub"
if [ -f "$GRUB_FILE" ]; then
  cp "$GRUB_FILE" "${GRUB_FILE}.bak"
  sed -i 's/^GRUB_CMDLINE_LINUX_DEFAULT=.*/GRUB_CMDLINE_LINUX_DEFAULT="quiet splash intel_iommu=off"/' "$GRUB_FILE"
  update-grub
else
  echo "[ERROR] Could not find $GRUB_FILE"
  exit 1
fi

# Check if Docker is already installed
if command -v docker >/dev/null 2>&1; then
    echo "[INFO] Docker is already installed. Skipping installation."
else
    echo "[INFO] Docker is not installed. Starting installation..."

    # Add Docker GPG key if not present
    if [ ! -f /etc/apt/keyrings/docker.asc ]; then
        echo "[INFO] Adding Docker GPG key..."
        apt-get install -y ca-certificates curl
        install -m 0755 -d /etc/apt/keyrings
        curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
        chmod a+r /etc/apt/keyrings/docker.asc
    fi

    # Add Docker repository if not present
    if [ ! -f /etc/apt/sources.list.d/docker.list ]; then
        echo "[INFO] Adding Docker repository..."
        echo \
          "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
          $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
          tee /etc/apt/sources.list.d/docker.list > /dev/null
        apt-get update
    fi

    # Install Docker packages
    echo "[INFO] Installing Docker..."
    apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

    echo "[INFO] Docker installation completed successfully."
fi

echo -e "\n✅ [DONE] Environment setup complete. Please reboot your system to apply changes."
