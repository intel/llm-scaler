#!/bin/bash
# ==============================================================================
# Intel Multi-ARC Base Platform Offline Installer Script
# ------------------------------------------------------------------------------
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author:      James Tang
# Contact:     jun.tang@intel.com
# Created:     2025-07-27
# ==============================================================================

set -euo pipefail

trap 'echo -e "\033[1;31m[ERROR]\033[0m Command failed on line $LINENO: $BASH_COMMAND" >&2' ERR

# -------- Configuration --------
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGFILE="install_log_$TIMESTAMP.log"
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_NAME=$(basename "$0")

# -------- Color Output --------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info()  { echo -e "${GREEN}[INFO]${NC} $1" | tee -a "$LOGFILE"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "$LOGFILE"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOGFILE" >&2; }

# -------- Begin Logging --------
echo "=== Installer Log: $TIMESTAMP ===" > "$LOGFILE"
log_info "Running script: $SCRIPT_NAME"
log_info "Working directory: $WORK_DIR"
log_info "Log file: $LOGFILE"

# -------- Root Privileges --------
if [[ "$EUID" -ne 0 ]]; then
  log_error "This script must be run as root."
  exit 1
fi

# -------- Script Location Check --------
if [[ "$WORK_DIR" != "$(pwd)" ]]; then
  log_error "Please run this script from its own directory: $WORK_DIR"
  exit 1
fi

# -------- Docker Detection --------
is_docker() {
  if [ "${BUILD_ENV:-}" = "docker" ]; then return 0; fi
  grep -qaE 'docker|kubepods|containerd' /proc/1/cgroup && return 0
  [[ "$(hostname)" =~ ^[0-9a-f]{12}$ ]] && return 0
  return 1
}

if is_docker; then
  log_info "Detected Docker container environment."
else
  log_info "Detected native host environment."
fi

# -------- Unified .deb Installer --------
install_deb_packages() {
  local desc="$1"
  shift
  log_info "Installing $desc ..."
  dpkg -i "$@" 2>&1 | tee -a "$LOGFILE"
}

# -------- oneAPI Installer --------
install_oneapi() {
  local installer="$1"
  if [ -f "$installer" ]; then
    log_info "Installing Intel oneAPI Base Toolkit..."
    bash "$installer" -a -s --eula accept --install-dir=/opt/intel/oneapi \
      --components intel.oneapi.lin.dpcpp-ct:intel.oneapi.lin.dpcpp_dbg:intel.oneapi.lin.dpl:intel.oneapi.lin.tbb.devel:intel.oneapi.lin.ccl.devel:intel.oneapi.lin.dpcpp-cpp-compiler:intel.oneapi.lin.mkl.devel \
      2>&1 | tee -a "$LOGFILE"
    log_info "Intel oneAPI installed successfully."
  else
    log_error "oneAPI installer not found: $installer"
    exit 1
  fi
}

# -------- Install validated kernel (Host Only) --------
if ! is_docker; then
  log_info "Install kernel."
  ./scripts/installation/install_kernel.sh 2>&1 | tee -a "$LOGFILE"
fi

# -------- Disable IOMMU (Host Only) --------
if ! is_docker; then
  GRUB_FILE="/etc/default/grub"
  if [ -f "$GRUB_FILE" ]; then
    cp "$GRUB_FILE" "${GRUB_FILE}.bak"
    sed -i 's/^GRUB_CMDLINE_LINUX_DEFAULT=.*/GRUB_CMDLINE_LINUX_DEFAULT="quiet splash intel_iommu=off"/' "$GRUB_FILE"
    update-grub 2>&1 | tee -a "$LOGFILE"
    log_info "Disabled IOMMU in GRUB and updated configuration."
  else
    log_error "GRUB configuration not found at $GRUB_FILE."
    exit 1
  fi
fi

# -------- Disable Ubuntu Auto-Update (Host Only) --------
if ! is_docker; then
  log_info "Disabled Ubuntu Auto-Update to maintain consistent environment."
  ./scripts/config/disable_auto_update.sh 2>&1 | tee -a "$LOGFILE"
fi

# -------- Disable AppArmor (Host Only) --------
if ! is_docker; then
  log_info "Disabled Ubuntu AppArmor to avoid unnecessary message."
  ./scripts/config/disable_apparmor.sh 2>&1 | tee -a "$LOGFILE"
fi


# -------- Install Firmware (Host Only) --------
if ! is_docker; then
  FIRMWARE_DIR="$WORK_DIR/firmware"
  log_info "Installing GPU firmware from $FIRMWARE_DIR..."

  if [ -d "$FIRMWARE_DIR" ] && [ -d /lib/firmware/xe ]; then
    cp "$FIRMWARE_DIR"/*.zst /lib/firmware/xe/
    update-initramfs -u 2>&1 | tee -a "$LOGFILE"
    log_info "Firmware installed and initramfs updated."
  else
    log_error "Missing firmware source or target directory."
    exit 1
  fi
fi

cd "$WORK_DIR"

# -------- Install Base Libraries --------
install_deb_packages "base libraries" base/*.deb

if ! is_docker; then
  install_deb_packages "docker libraries" base/docker/*.deb
fi

# -------- Install Graphics Drivers --------
install_deb_packages "graphics base drivers" gfxdrv/base/*.deb
install_deb_packages "graphics GPGPU drivers" gfxdrv/gpgpu/*.deb

if ! is_docker; then
  install_deb_packages "graphics video drivers" gfxdrv/video/*.deb
  install_deb_packages "graphics display drivers" gfxdrv/graphics/*.deb
fi

# -------- Install Intel oneAPI Base Toolkit --------
# Only install oneapi in native environment since our docker image is based on
# onepai base image which already has oneapi installed
if ! is_docker; then
  ONEAPI_DIR="/opt/intel/oneapi/2025.1"
  ONEAPI_INSTALLER="$WORK_DIR/oneapi/intel-oneapi-base-toolkit-2025.1.3.7_offline.sh"

  if [ -d "$ONEAPI_DIR" ]; then
    log_info "Intel oneAPI already installed at $ONEAPI_DIR. Skipping."
  else
    install_oneapi "$ONEAPI_INSTALLER"
  fi
fi

# -------- Install Evaluation Tools --------
TOOLS_DIR=$WORK_DIR/tools
install_deb_packages "1ccl tool" "$TOOLS_DIR/1ccl/"*.deb || true
install_deb_packages "gemm tool" "$TOOLS_DIR/gemm/"*.deb  || true
install_deb_packages "xpu-smi tool" "$TOOLS_DIR/xpu-smi/"*.deb || true

cd "$WORK_DIR"

# -------- Final Message --------
log_info "Intel Multi-ARC base platform installation complete."

if is_docker; then
  log_info "Docker environment detected — reboot not required."
else
  log_info "Please reboot the system to apply changes."
fi

echo -e "\n${GREEN}Tools installed:${NC} gemm / 1ccl / xpu-smi in /usr/bin"
echo -e "${GREEN}level-zero-tests:${NC} ./tools/level-zero-tests"
echo -e "${GREEN}Support scripts:${NC} ./scripts"
echo -e "${GREEN}Installation log:${NC} ./$LOGFILE"
