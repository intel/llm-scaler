#!/bin/bash

# Output header
echo "Category,Version"

# 1. Ubuntu version
UBUNTU_VERSION=$(grep '^VERSION=' /etc/os-release | cut -d '"' -f 2)
echo "Ubuntu,$UBUNTU_VERSION"

# 2. Linux kernel version
KERNEL_VERSION=$(uname -r)
echo "Linux Kernel,$KERNEL_VERSION"

# 3. Intel GPU firmware versions from dmesg

# Extract GuC firmware version
guc_ver=$(dmesg | grep -i 'Using GuC firmware' | head -n1 | grep -oP 'version \K[\d\.]+')
if [[ -n "$guc_ver" ]]; then
    echo "GPU Firmware (guc),$guc_ver"
else
    echo "GPU Firmware (guc),Not Found"
fi

# Extract HuC firmware version
huc_ver=$(dmesg | grep -i 'Using HuC firmware' | head -n1 | grep -oP 'version \K[\d\.]+')
if [[ -n "$huc_ver" ]]; then
    echo "GPU Firmware (huc),$huc_ver"
fi

# Extract DMC firmware version
dmc_ver=$(dmesg | grep -i 'Finished loading DMC firmware' | head -n1 | grep -oP '\(v\K[\d\.]+')
if [[ -n "$dmc_ver" ]]; then
    echo "GPU Firmware (dmc),$dmc_ver"
else
    echo "GPU Firmware (dmc),Not Found"
fi

# 4. OneAPI version (offline installed)
ONEAPI_LOG=$(ls /opt/intel/oneapi/logs/installer.install.intel.oneapi.lin.basekit.product,v=* 2>/dev/null | head -n1)
if [[ -n "$ONEAPI_LOG" ]]; then
    oneapi_ver=$(basename "$ONEAPI_LOG" | sed -n 's/.*basekit\.product,v=\(.*\)\..*/\1/p')
    echo "oneapi,oneapi-base-toolkit=$oneapi_ver"
else
    echo "oneapi,oneapi-base-toolkit=Not Installed"
fi

# 5. Parse passed-in package files
for file in "$@"; do
    [[ ! -f "$file" ]] && continue

    category=$(basename "$file" .txt)
    first=1

    while IFS= read -r pkg; do
        [[ -z "$pkg" || "$pkg" =~ ^# ]] && continue

        version=$(dpkg-query -W -f='${Version}\n' "$pkg" 2>/dev/null)
        version_output="$pkg=${version:-Not Installed}"

        if [[ $first -eq 1 ]]; then
            echo "$category,$version_output"
            first=0
        else
            echo ",$version_output"
        fi
    done < "$file"
done
