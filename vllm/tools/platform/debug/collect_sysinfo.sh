#!/bin/bash
set -euo pipefail

is_docker() {
  grep -qaE 'docker|kubepods|containerd' /proc/1/cgroup && return 0
  [[ "$(hostname)" =~ ^[0-9a-f]{12}$ ]] && return 0
  return 1
}

# Check for root privileges
if [[ "$EUID" -ne 0 ]]; then
  echo "[ERROR] This script must be run as root."
  exit 1
fi

if is_docker; then
  echo "[ERROR] Please run this script under native environment, not in docker"
  exit 1
fi

# Prepare output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="sysinfo_$TIMESTAMP"
mkdir -p "$OUTDIR"

echo "[INFO] Collecting system information into $OUTDIR..."

# 1. CPU governor
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor > "$OUTDIR/scaling_governor.txt" 2>/dev/null || echo "Not available" > "$OUTDIR/scaling_governor.txt"

# 2. CPU architecture
lscpu > "$OUTDIR/lscpu.txt"

# 3. PCI topology
lspci -tv > "$OUTDIR/lspci_tree.txt"
lspci -vvv > "$OUTDIR/lspci_verbose.txt"

# 4. Kernel messages
dmesg > "$OUTDIR/dmesg.txt"

# 5. DRI tree
tree /sys/kernel/debug/dri/ > "$OUTDIR/dri_tree.txt" 2>/dev/null || echo "Not available" > "$OUTDIR/dri_tree.txt"

# 6. Memory usage
free -h > "$OUTDIR/memory.txt"

# 7. Hardware info
dmidecode > "$OUTDIR/dmidecode.txt"

# 8. libze info
dpkg -l | grep libze > "$OUTDIR/libze_version.txt"

# Create tar archive first
TAR_FILE="sysinfo_$TIMESTAMP.tar"
XZ_FILE="$TAR_FILE.xz"

echo "[INFO] Creating archive $TAR_FILE..."
tar -cf "$TAR_FILE" "$OUTDIR"

echo "[INFO] Compressing with xz -9..."
xz -9 "$TAR_FILE"

echo "[INFO] Done. Output file: $XZ_FILE"

