#!/bin/bash
set -euo pipefail

trap 'echo -e "\033[1;31m[ERROR]\033[0m Command failed on line $LINENO: $BASH_COMMAND" >&2' ERR

# === Require root ===
if [[ "$EUID" -ne 0 ]]; then
  echo "This script must be run as root. Please run with sudo or as root user."
  exit 1
fi

if [ ! -d "./tools" ] || [ ! -d "./scripts" ]; then
    print_error "This script must be run from the root directory of the installation package."
    echo "Expected directories: ./tools and ./scripts"
    exit 1
fi

# === Config ===
TARGET_VERSION="6.14.0-1006-intel"
SUBMENU_TITLE="Advanced options for Ubuntu"
MENUENTRY_TITLE="Ubuntu, with Linux $TARGET_VERSION"
DEFAULT_FILE="/etc/default/grub"
GRUB_CFG="/boot/grub/grub.cfg"

# === Check running kernel ===
CURRENT_VERSION="$(uname -r)"
echo "Current running kernel: $CURRENT_VERSION"
echo "Target kernel version: $TARGET_VERSION"

if [[ "$CURRENT_VERSION" == "$TARGET_VERSION" ]]; then
    echo "✅ Already running the target kernel. No changes needed."
    exit 0
fi

# === Check if target kernel is installed ===
if dpkg -s "linux-image-${TARGET_VERSION}" >/dev/null 2>&1; then
    echo "Kernel ${TARGET_VERSION} installed, skip..."
else
    echo "⚠️ Target kernel is not installed. Installing..."
    dpkg -i kernel/*.deb
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

echo "💡 You may now reboot to use the new kernel."
