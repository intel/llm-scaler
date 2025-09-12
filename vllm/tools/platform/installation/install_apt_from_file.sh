#!/bin/bash
set -euo pipefail

# Check if package file is given
if [ $# -ne 1 ]; then
    echo "Usage: $0 <package_list.txt>"
    exit 1
fi

PACKAGE_FILE="$1"

if [ ! -f "$PACKAGE_FILE" ]; then
    echo "Error: File '$PACKAGE_FILE' not found."
    exit 2
fi

# Optionally refresh package index
echo "[INFO] Updating APT package index..."
apt update

while IFS= read -r line || [[ -n "$line" ]]; do
    # Skip empty lines or comments
    [[ -z "$line" || "$line" =~ ^# ]] && continue

    # line format: name=version
    echo "[INFO] Installing package: $line"
    apt install -y --allow-downgrades --allow-change-held-packages "$line"
done < "$PACKAGE_FILE"

echo "[INFO] All packages installed successfully."
