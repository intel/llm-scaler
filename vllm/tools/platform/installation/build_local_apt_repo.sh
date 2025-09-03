#!/bin/bash
#
# build_local_apt_repo.sh
# Build a local APT repository from a directory of .deb files
#
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
# Author: James Tang <jun.tang@intel.com>
# Date: 2025-07-26

set -euo pipefail

# === Colors ===
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1" >&2; }

# === Argument Check ===
if [[ $# -ne 1 ]]; then
    echo -e "${YELLOW}Usage: $0 <source_deb_directory>${NC}"
    exit 1
fi

DEB_SOURCE_DIR="$1"
REPO_DIR="/opt/local-apt-repo"
APT_SOURCE_FILE="/etc/apt/sources.list.d/local-repo.list"

# === Validation ===
if [[ ! -d "$DEB_SOURCE_DIR" ]]; then
    log_error "Directory '$DEB_SOURCE_DIR' does not exist."
    exit 2
fi

if [[ ! "$(ls -1 "$DEB_SOURCE_DIR"/*.deb 2>/dev/null)" ]]; then
    log_error "No .deb files found in '$DEB_SOURCE_DIR'."
    exit 3
fi

log_info "Creating local APT repository from '$DEB_SOURCE_DIR'..."

# === Step 1: Prepare Repository ===
log_info "Copying .deb files to repository directory: $REPO_DIR"
mkdir -p "$REPO_DIR"
cp -v "$DEB_SOURCE_DIR"/*.deb "$REPO_DIR"/

# === Step 2: Generate Packages.gz ===
log_info "Generating Packages.gz index..."
cd "$REPO_DIR"
dpkg-scanpackages . /dev/null | gzip -9c > Packages.gz

# === Step 3: Configure APT Source ===
log_info "Configuring APT source file: $APT_SOURCE_FILE"
echo "deb [trusted=yes] file://$REPO_DIR ./" | tee "$APT_SOURCE_FILE" > /dev/null

# === Step 4: Update APT Index ===
log_info "Updating APT package index..."
apt update

log_info "✅ Local APT repository is ready and active!"
