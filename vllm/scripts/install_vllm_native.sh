#!/bin/bash
# ============================================================
# vLLM XPU Native Install Script (from llm-scaler)
# For Intel iGPU shared-memory systems: Lunar Lake, Meteor Lake, Arrow Lake
#
# This extracts the Dockerfile build steps for bare-metal installation.
# Tested on: Ubuntu 25.04
#
# Usage:
#   chmod +x install_vllm_native.sh
#   sudo bash install_vllm_native.sh
#
# After install:
#   source ~/vllm-venv/bin/activate
#   source /opt/intel/oneapi/setvars.sh
#   vllm serve /path/to/model --device xpu
# ============================================================

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# ============================================================
# RAM-based build parallelism
# 16GB (Claw A1M, Meteor Lake): 3 jobs — avoids OOM
# 32GB (Claw 8 AI+, Lunar Lake): 6 jobs
# 64GB+: 8 jobs
# ============================================================
TOTAL_RAM_GB=$(free -g | awk '/^Mem:/{print $2}')
if [ "$TOTAL_RAM_GB" -le 16 ]; then
    export MAX_JOBS=3
elif [ "$TOTAL_RAM_GB" -le 32 ]; then
    export MAX_JOBS=6
else
    export MAX_JOBS=8
fi

# ============================================================
# GPU detection — identify Intel iGPU platform
# ============================================================
detect_intel_gpu() {
    local gpu_id
    gpu_id=$(lspci -nn | grep -i 'vga\|3d\|display' | grep -oP '8086:\K[0-9a-fA-F]+' | head -1)
    case "${gpu_id,,}" in
        64a0)           echo "lunar_lake $gpu_id" ;;
        7d55|7dd5|7d40|7d45) echo "meteor_lake $gpu_id" ;;
        7d51|7dd1|7d41|7d67) echo "arrow_lake $gpu_id" ;;
        e211|e210)      echo "arc_pro_b60 $gpu_id" ;;
        56a0)           echo "arc_a770 $gpu_id" ;;
        *)              echo "unknown $gpu_id" ;;
    esac
}

read -r PLATFORM GPU_ID <<< "$(detect_intel_gpu)"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN} vLLM XPU Native Install (llm-scaler)${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "  Platform:      ${PLATFORM} (8086:${GPU_ID})"
echo "  RAM detected:  ${TOTAL_RAM_GB}GB"
echo "  Build jobs:    MAX_JOBS=${MAX_JOBS}"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}Please run with sudo: sudo bash install_vllm_native.sh${NC}"
    exit 1
fi

REAL_USER=${SUDO_USER:-$USER}
REAL_HOME=$(eval echo ~"$REAL_USER")
INSTALL_DIR="${INSTALL_DIR:-$REAL_HOME/llm}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ============================================================
# Precompiled wheel configuration
# Set VLLM_BUILD_FROM_SOURCE=1 to skip wheel download and force compilation.
# Wheels are hosted on GitHub Releases for the llm-scaler repo.
# ============================================================
WHEEL_REPO="MegaStood/llm-scaler"
WHEEL_TAG="vllm-xpu-wheels-v0.14.0"
WHEEL_BASE_URL="https://github.com/${WHEEL_REPO}/releases/download/${WHEEL_TAG}"
VLLM_WHEEL="vllm-0.14.0-cp312-cp312-linux_x86_64.whl"
XPU_KERNELS_WHEEL="vllm_xpu_kernels-0.0.1-cp312-cp312-linux_x86_64.whl"
BUILD_FROM_SOURCE="${VLLM_BUILD_FROM_SOURCE:-0}"

# Helper: try to download and install a precompiled wheel.
# Returns 0 on success, 1 on failure (triggers source build fallback).
try_install_wheel() {
    local wheel_name="$1"
    local wheel_url="${WHEEL_BASE_URL}/${wheel_name}"
    local wheel_path="${INSTALL_DIR}/${wheel_name}"

    if [ "$BUILD_FROM_SOURCE" = "1" ]; then
        echo "  VLLM_BUILD_FROM_SOURCE=1, skipping wheel download."
        return 1
    fi

    echo "  Trying precompiled wheel: $wheel_name"
    if wget -q --spider "$wheel_url" 2>/dev/null; then
        echo "  Downloading from GitHub Releases..."
        wget -q --show-progress -O "$wheel_path" "$wheel_url"
        if pip install "$wheel_path" 2>&1 | tail -3; then
            rm -f "$wheel_path"
            return 0
        else
            echo -e "${YELLOW}  Wheel install failed. Falling back to source build.${NC}"
            rm -f "$wheel_path"
            return 1
        fi
    else
        echo "  Precompiled wheel not available yet. Building from source."
        return 1
    fi
}

echo "  Install dir:   $INSTALL_DIR"
echo "  Repo root:     $REPO_ROOT"
echo ""

# ============================================================
# Swap optimization
# On 16GB systems the xpu-kernels build peaks at ~21GB (14GB RAM + 8GB swap).
# Without disk swap, the OOM killer will terminate the compilation.
# On 32GB systems, swap is optional overflow but recommended.
# ============================================================
ZRAM_ACTIVE=$(swapon --show=NAME,TYPE --noheadings 2>/dev/null | grep -c zram || true)
SWAPFILE="$REAL_HOME/swapfile"
FS_TYPE=$(df -T "$REAL_HOME" | awk 'NR==2{print $2}')

create_swapfile() {
    local size="$1"
    if [ -f "$SWAPFILE" ]; then
        echo "  Swapfile already exists at $SWAPFILE. Activating."
    else
        echo "  Creating ${size} disk swapfile..."
        if [ "$FS_TYPE" = "btrfs" ]; then
            btrfs filesystem mkswapfile --size "$size" "$SWAPFILE"
        else
            fallocate -l "$size" "$SWAPFILE"
            chmod 600 "$SWAPFILE"
            mkswap "$SWAPFILE"
        fi
    fi
}

if [ "$TOTAL_RAM_GB" -le 16 ]; then
    echo -e "${YELLOW}[swap] 16GB system — disk swap required for compilation${NC}"

    # zram wastes precious RAM on 16GB systems; replace with disk swap
    if [ "$ZRAM_ACTIVE" -gt 0 ]; then
        echo "  Disabling zram (frees RAM for the build)..."
        swapoff /dev/zram0 2>/dev/null || true
        zramctl --reset /dev/zram0 2>/dev/null || true
        systemctl mask systemd-zram-setup@zram0.service 2>/dev/null || true
    fi

    if ! swapon --show=NAME --noheadings 2>/dev/null | grep -q "$SWAPFILE"; then
        create_swapfile 16G
        swapon "$SWAPFILE"
        sysctl vm.swappiness=10
        echo "vm.swappiness=10" > /etc/sysctl.d/99-swappiness.conf

        if ! grep -q "$SWAPFILE" /etc/fstab 2>/dev/null; then
            echo "$SWAPFILE none swap defaults 0 0" >> /etc/fstab
        fi
        echo -e "${GREEN}  16GB disk swapfile active. zram permanently disabled.${NC}"
    else
        echo "  Disk swapfile already active."
    fi
    echo ""

elif [ "$TOTAL_RAM_GB" -le 32 ]; then
    echo -e "${YELLOW}[swap] 32GB system — optional disk swap for overflow${NC}"

    if ! swapon --show=NAME --noheadings 2>/dev/null | grep -q "$SWAPFILE"; then
        read -p "  Create 32GB disk swapfile as overflow safety net? (y/n): " SWAP_CHOICE
        if [ "$SWAP_CHOICE" = "y" ] || [ "$SWAP_CHOICE" = "Y" ]; then
            create_swapfile 32G
            swapon --priority 10 "$SWAPFILE"

            if ! grep -q "$SWAPFILE" /etc/fstab 2>/dev/null; then
                echo "$SWAPFILE none swap pri=10 0 0" >> /etc/fstab
            fi
            echo -e "${GREEN}  32GB disk swapfile active (priority 10, below zram).${NC}"
        fi
    else
        echo "  Disk swapfile already active."
    fi

    # Temporarily disable zram to free all 32GB RAM for the build
    if [ "$ZRAM_ACTIVE" -gt 0 ]; then
        echo "  Temporarily disabling zram for the build (returns on reboot)..."
        swapoff /dev/zram0 2>/dev/null || true
        zramctl --reset /dev/zram0 2>/dev/null || true
    fi
    echo ""
fi

# ============================================================
# Phase 1: System dependencies
# ============================================================
echo -e "${YELLOW}[1/8] Installing system dependencies...${NC}"

# Add Intel oneAPI repo
if [ ! -f /usr/share/keyrings/oneapi-archive-keyring.gpg ]; then
    wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
        | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
        | tee /etc/apt/sources.list.d/oneAPI.list
fi

# Add Intel graphics PPA (Ubuntu only)
if command -v add-apt-repository &>/dev/null; then
    add-apt-repository -y ppa:kobuk-team/intel-graphics 2>/dev/null || true
fi

apt-get update -y

# Install Python 3.12 separately so failure is visible (not hidden by --fix-missing)
if ! command -v python3.12 &>/dev/null; then
    echo "  Installing Python 3.12..."
    apt-get install -y python3.12 python3.12-dev python3.12-venv
else
    echo "  Python 3.12 already installed: $(python3.12 --version)"
fi

apt-get install -y --no-install-recommends --fix-missing \
    curl ffmpeg git libsndfile1 libsm6 libxext6 libaio-dev \
    libgl1 lsb-release numactl wget vim linux-libc-dev \
    intel-oneapi-dpcpp-ct

# Install Intel GPU runtime packages needed for XPU:
# - Level Zero: low-level GPU interface that torch XPU uses to talk to Intel GPUs.
#   Without it, torch.xpu.device_count() returns 0 and vLLM fails at startup.
# - ocloc + IGC: ahead-of-time GPU compiler toolchain for SYCL kernel compilation.
#   Without it, the build fails at object ~925/933 with "ocloc tool could not be found".
echo "  Installing Intel Level Zero + IGC (GPU runtime)..."
apt-get install -y --no-install-recommends \
    level-zero intel-level-zero-gpu \
    intel-ocloc intel-igc-core intel-igc-opencl 2>&1 | tail -3

echo -e "${GREEN}[1/8] System dependencies installed.${NC}"
echo ""

# ── xe vs i915 driver check ─────────────────────────────────
if lsmod | grep -q "^xe "; then
    echo -e "${GREEN}  xe driver loaded — good.${NC}"
elif lsmod | grep -q "^i915 "; then
    echo -e "${YELLOW}  i915 driver loaded instead of xe.${NC}"
    if [ "$PLATFORM" = "meteor_lake" ] || [ "$PLATFORM" = "arrow_lake" ]; then
        echo -e "${YELLOW}  For best SYCL/oneAPI support, switch to the xe driver:${NC}"
        echo ""
        echo -e "    Add to GRUB: ${GREEN}i915.force_probe=!${GPU_ID} xe.force_probe=${GPU_ID}${NC}"
        echo ""
        echo -e "${YELLOW}  Then: sudo update-grub && sudo reboot${NC}"
        echo -e "${YELLOW}  The install will continue, but XPU features may not work until you switch.${NC}"
    else
        echo -e "${YELLOW}  For best SYCL support, switch to xe driver.${NC}"
    fi
fi

# ── Memory-based recommendations ────────────────────────────
GPU_UTIL_RECOMMEND="0.8"
if [ "$TOTAL_RAM_GB" -le 16 ]; then
    echo -e "${YELLOW}  ${TOTAL_RAM_GB}GB RAM — only small models (≤8B) will fit.${NC}"
    echo -e "${YELLOW}  Use --gpu-memory-utilization 0.6 and --quantization int4.${NC}"
    GPU_UTIL_RECOMMEND="0.6"
elif [ "$TOTAL_RAM_GB" -le 32 ]; then
    echo -e "${GREEN}  ${TOTAL_RAM_GB}GB RAM — good for 8B-14B models with INT4/FP8.${NC}"
    GPU_UTIL_RECOMMEND="0.7"
else
    echo -e "${GREEN}  ${TOTAL_RAM_GB}GB RAM — plenty for most models up to 14B FP16.${NC}"
fi
echo ""

# ============================================================
# Phase 2: Create Python 3.12 virtual environment
# ============================================================
echo -e "${YELLOW}[2/8] Setting up Python environment...${NC}"

# Detect Python version — PyTorch XPU requires Python 3.10-3.12
# Ubuntu 25.04+ may ship Python 3.13+ which is too new for PyTorch XPU wheels
PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "unknown")
echo "  System Python: $PY_VERSION"

VENV_PYTHON="python3.12"
if command -v python3.12 &>/dev/null; then
    echo "  Found python3.12 for venv."
elif python3 -c "import sys; sys.exit(0 if sys.version_info[:2] <= (3,12) else 1)" 2>/dev/null; then
    VENV_PYTHON="python3"
    echo "  System Python $PY_VERSION is compatible with PyTorch XPU."
else
    echo -e "${YELLOW}  Python $PY_VERSION is too new for PyTorch XPU (needs <=3.12).${NC}"
    echo "  Installing Python 3.12..."
    apt-get install -y python3.12 python3.12-dev python3.12-venv 2>&1 | tail -3
    if ! command -v python3.12 &>/dev/null; then
        echo -e "${RED}  Cannot install Python 3.12. PyTorch XPU requires Python <=3.12.${NC}"
        exit 1
    fi
fi

VLLM_VENV="$REAL_HOME/vllm-venv"
if [ -d "$VLLM_VENV" ] && [ -f "$VLLM_VENV/bin/python" ]; then
    echo "  Python venv already exists at $VLLM_VENV"
else
    echo "  Creating Python virtual environment at $VLLM_VENV..."
    sudo -u "$REAL_USER" $VENV_PYTHON -m venv "$VLLM_VENV"
fi

# Activate venv — all pip/python commands below use Python 3.12
source "$VLLM_VENV/bin/activate"

# Redirect pip temp files to disk — /tmp may be a tmpfs (RAM-backed, ~7.6GB on 16GB).
# Large XPU wheels (>2GB) will overflow it, causing "No space left on device".
export TMPDIR="$REAL_HOME/.pip-tmp"
mkdir -p "$TMPDIR"
chown "$REAL_USER:$REAL_USER" "$TMPDIR"

# Clean up corrupted pip metadata from interrupted installs
find "$VLLM_VENV" -type d -name '~*' -exec rm -rf {} + 2>/dev/null || true

pip install --upgrade pip setuptools wheel 2>&1 | tail -1
echo "  Using Python: $(python --version) from $(which python)"

echo -e "${GREEN}[2/8] Python environment ready.${NC}"
echo ""

# ============================================================
# Phase 3: Configure build environment
# ============================================================
echo -e "${YELLOW}[3/8] Configuring build environment...${NC}"

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/usr/local/lib/"
export VLLM_TARGET_DEVICE=xpu
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Source oneAPI if available
# Temporarily relax strict mode because setvars.sh has unbound variables
# and non-zero exits internally that conflict with set -e
if [ -f /opt/intel/oneapi/setvars.sh ]; then
    set +e
    source /opt/intel/oneapi/setvars.sh --force > /tmp/oneapi_init.log 2>&1 || true
    set -e
    grep -E "^::|initialized" /tmp/oneapi_init.log || true
    rm -f /tmp/oneapi_init.log
else
    echo -e "${RED}  Warning: /opt/intel/oneapi/setvars.sh not found. Build may fail.${NC}"
fi

# Fix MKL library path — PyTorch's bundled MKL stubs use relative RPATHs that
# break inside venvs. Preload the real oneAPI MKL to avoid runtime errors.
if [ -n "${MKLROOT:-}" ] && [ -f "$MKLROOT/lib/libmkl_core.so.2" ]; then
    export LD_PRELOAD="${MKLROOT}/lib/libmkl_core.so.2:${MKLROOT}/lib/libmkl_intel_thread.so.2:${MKLROOT}/lib/libmkl_intel_lp64.so.2${LD_PRELOAD:+:$LD_PRELOAD}"
    echo "  MKL preloaded from $MKLROOT"
fi

# Add DPCPP include path for vLLM compilation
DPCPP_INCLUDE=$(find /opt/intel/oneapi/dpcpp-ct/ -maxdepth 2 -name "include" -type d 2>/dev/null | head -1)
if [ -n "$DPCPP_INCLUDE" ]; then
    export CPATH="${DPCPP_INCLUDE}:${CPATH:-}"
    echo "  DPCPP include: $DPCPP_INCLUDE"
fi

# Ensure llvm-foreach is on PATH — the SYCL AOT linker needs it.
# It ships inside the oneAPI compiler but is not on PATH by default.
LLVM_FOREACH=$(find /opt/intel/oneapi/compiler/ -name "llvm-foreach" -type f 2>/dev/null | head -1)
if [ -n "$LLVM_FOREACH" ] && [ ! -f /usr/local/bin/llvm-foreach ]; then
    ln -sf "$LLVM_FOREACH" /usr/local/bin/llvm-foreach
    echo "  Symlinked llvm-foreach -> $LLVM_FOREACH"
fi

echo -e "${GREEN}[3/8] Build environment configured.${NC}"
echo ""

# ============================================================
# Phase 4: Clone and patch vLLM
# ============================================================
echo -e "${YELLOW}[4/8] Cloning and patching vLLM v0.14.0...${NC}"

mkdir -p "$INSTALL_DIR"

VLLM_DIR="$INSTALL_DIR/vllm"
if [ -d "$VLLM_DIR" ]; then
    echo "  $VLLM_DIR already exists. Skipping clone."
    echo "  To re-clone: rm -rf $VLLM_DIR and re-run."
else
    sudo -u "$REAL_USER" git clone -b v0.14.0 https://github.com/vllm-project/vllm.git "$VLLM_DIR"
fi

# Apply Intel multi-arc patch (idempotent: checks before applying)
PATCH_FILE="$REPO_ROOT/vllm/patches/vllm_for_multi_arc.patch"
if [ -f "$PATCH_FILE" ]; then
    cd "$VLLM_DIR"
    if sudo -u "$REAL_USER" git apply --check "$PATCH_FILE" 2>/dev/null; then
        sudo -u "$REAL_USER" git apply "$PATCH_FILE"
        echo "  Applied vllm_for_multi_arc.patch"
    else
        echo "  Patch already applied or conflicts. Skipping."
    fi
else
    echo -e "${RED}  Patch not found at $PATCH_FILE${NC}"
    echo "  Make sure you're running this from the llm-scaler repo."
    exit 1
fi

echo -e "${GREEN}[4/8] vLLM cloned and patched.${NC}"
echo ""

# ============================================================
# Phase 5: Install vLLM XPU requirements + build
# ============================================================
echo -e "${YELLOW}[5/8] Building vLLM (MAX_JOBS=${MAX_JOBS}, this will take a while)...${NC}"

VLLM_CHECK=$(pip show vllm 2>/dev/null | grep -c "Name: vllm" || true)
if [ "$VLLM_CHECK" -gt 0 ]; then
    echo -e "${GREEN}  vLLM already installed in venv. Skipping build.${NC}"
    echo "  To rebuild: pip uninstall vllm -y && re-run this script."
else
    # XPU requirements are needed regardless of wheel or source build
    cd "$VLLM_DIR"
    echo "  Installing XPU requirements..."
    pip install -r requirements/xpu.txt 2>&1 | tail -3
    pip install arctic-inference==0.1.1 2>&1 | tail -1

    # Try precompiled wheel first, fall back to source build
    if try_install_wheel "$VLLM_WHEEL"; then
        echo -e "${GREEN}  vLLM installed from precompiled wheel (~2 min vs 30-60 min).${NC}"
    else
        echo "  Building vLLM from source (MAX_JOBS=$MAX_JOBS)..."
        echo "  On 16GB RAM this can take 30-60 minutes. Do not interrupt."
        pip install --no-build-isolation . 2>&1 | tail -5
    fi

    if pip show vllm &>/dev/null; then
        echo -e "${GREEN}  vLLM installed successfully.${NC}"
    else
        echo -e "${RED}  vLLM install failed. Check errors above.${NC}"
        echo "  Retry: cd $VLLM_DIR && MAX_JOBS=$MAX_JOBS pip install --no-build-isolation ."
        exit 1
    fi
fi

# Patch xpu_worker.py: disable CCL all_reduce warmup for single-GPU
# oneCCL's KVS init fails on devices without wired Ethernet (e.g. handhelds).
# The all_reduce warmup is unnecessary for single-GPU (TP=1).
XPU_WORKER=$(python -c "import vllm; import os; print(os.path.join(os.path.dirname(vllm.__file__), 'v1/worker/xpu_worker.py'))" 2>/dev/null || true)
if [ -n "$XPU_WORKER" ] && [ -f "$XPU_WORKER" ]; then
    if grep -q "torch.distributed.all_reduce" "$XPU_WORKER"; then
        echo "  Patching xpu_worker.py to disable CCL all_reduce warmup (single-GPU fix)..."
        sed -i '/torch\.distributed\.all_reduce(/,/)/s/^/#/' "$XPU_WORKER"
        echo "  xpu_worker.py patched."
    else
        echo "  xpu_worker.py already patched or no all_reduce found."
    fi
fi

echo -e "${GREEN}[5/8] vLLM build complete.${NC}"
echo ""

# ============================================================
# Phase 6: Install additional Python dependencies
# ============================================================
echo -e "${YELLOW}[6/8] Installing additional dependencies...${NC}"

pip install accelerate hf_transfer 'modelscope!=1.15.0' 2>&1 | tail -1
pip install librosa soundfile decord 2>&1 | tail -1

# CRITICAL: Install transformers from git with --no-deps.
# Without --no-deps, pip resolves transformers' dependency on "torch" and pulls in
# torch==2.11.0+cu130 (CUDA build) from PyPI, which overwrites our torch==2.10.0+xpu.
# This causes: torch.xpu._is_compiled() == False, SYCL_HOME == None, and
# vllm-xpu-kernels build fails with "AssertionError: SYCL_HOME is not set".
pip install --no-deps git+https://github.com/huggingface/transformers.git 2>&1 | tail -1

# Verify torch XPU is still intact after all pip installs
TORCH_XPU_OK=$(python -c "import torch; print('yes' if torch.xpu._is_compiled() else 'no')" 2>/dev/null || echo "no")
if [ "$TORCH_XPU_OK" != "yes" ]; then
    echo -e "${YELLOW}  torch+xpu was overwritten — reinstalling from XPU index...${NC}"
    pip install torch==2.10.0+xpu --extra-index-url=https://download.pytorch.org/whl/xpu 2>&1 | tail -1
fi

pip install ijson 2>&1 | tail -1
pip install bigdl-core==2.4.0b2 2>&1 | tail -1

echo -e "${GREEN}[6/8] Dependencies installed.${NC}"
echo ""

# ============================================================
# Phase 7: Build vllm-xpu-kernels + fix triton
# ============================================================
echo -e "${YELLOW}[7/8] Building vllm-xpu-kernels...${NC}"

XPU_KERNELS_CHECK=$(pip show vllm-xpu-kernels 2>/dev/null | grep -c "Name:" || true)
if [ "$XPU_KERNELS_CHECK" -gt 0 ]; then
    echo -e "${GREEN}  vllm-xpu-kernels already installed. Skipping.${NC}"
else
    # Try precompiled wheel first — saves 30-70 minutes of SYCL compilation
    if try_install_wheel "$XPU_KERNELS_WHEEL"; then
        echo -e "${GREEN}  vllm-xpu-kernels installed from precompiled wheel.${NC}"
    else
        # Fall back to source build
        XPU_KERNELS_DIR="$INSTALL_DIR/vllm-xpu-kernels"
        if [ -d "$XPU_KERNELS_DIR" ]; then
            echo "  $XPU_KERNELS_DIR already exists. Skipping clone."
        else
            sudo -u "$REAL_USER" git clone https://github.com/vllm-project/vllm-xpu-kernels.git "$XPU_KERNELS_DIR"
        fi

        cd "$XPU_KERNELS_DIR"
        sudo -u "$REAL_USER" git checkout 4c83144 2>/dev/null || true

        # Comment out conflicting pinned deps (we already installed them)
        sed -i 's|^--extra-index-url=https://download.pytorch.org/whl/xpu|# &|' requirements.txt
        sed -i 's|^torch==2.10.0+xpu|# &|' requirements.txt
        sed -i 's|^triton-xpu|# &|' requirements.txt
        sed -i 's|^transformers|# &|' requirements.txt

        # Fix ownership — sed ran as root, leaving root-owned files that pip (as user) can't write
        chown -R "$REAL_USER:$REAL_USER" "$XPU_KERNELS_DIR"

        # Clean stale build artifacts from previous failed attempts
        rm -rf "$XPU_KERNELS_DIR/build"

        pip install -r requirements.txt 2>&1 | tail -1

        echo ""
        echo -e "${YELLOW}  Compiling vllm-xpu-kernels (oneDNN + SYCL kernels)...${NC}"
        echo -e "${YELLOW}  This takes 30-70 minutes. Do not close this window.${NC}"
        echo ""
        pip install --no-build-isolation . 2>&1 | tail -5
    fi

    if pip show vllm-xpu-kernels &>/dev/null; then
        echo -e "${GREEN}  vllm-xpu-kernels installed successfully.${NC}"
    else
        echo -e "${RED}  vllm-xpu-kernels build failed. Check errors above.${NC}"
    fi
fi

# Fix triton — replace any CUDA triton with XPU version
TRITON_XPU_CHECK=$(pip show triton-xpu 2>/dev/null | grep -c "Name:" || true)
if [ "$TRITON_XPU_CHECK" -gt 0 ]; then
    echo -e "${GREEN}  triton-xpu already installed. Skipping.${NC}"
else
    echo "  Fixing triton installation..."
    pip uninstall triton triton-xpu -y 2>/dev/null || true
    pip install triton-xpu==3.6.0 --extra-index-url=https://download.pytorch.org/whl/test/xpu 2>&1 | tail -1
    echo -e "${GREEN}  triton-xpu installed.${NC}"
fi

echo -e "${GREEN}[7/8] vllm-xpu-kernels + triton complete.${NC}"
echo ""

# ============================================================
# Phase 8: Configure environment for production
# ============================================================
echo -e "${YELLOW}[8/8] Configuring production environment...${NC}"

# Find the vllm_int4 library path (inside venv)
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || echo "$VLLM_VENV/lib/python3.12/site-packages")
Q40_LIB="${SITE_PACKAGES}/vllm_int4_for_multi_arc.so"

# Add environment to user's bashrc
BASHRC="$REAL_HOME/.bashrc"
if ! grep -q "VLLM_TARGET_DEVICE" "$BASHRC" 2>/dev/null; then
    cat >> "$BASHRC" << VLLM_ENV

# vLLM XPU environment (added by install_vllm_native.sh)
# oneAPI must be sourced FIRST — it resets LD_LIBRARY_PATH to its own paths.
# Our exports below then append to it rather than being overwritten.
source /opt/intel/oneapi/setvars.sh --force 2>/dev/null || true
source ${VLLM_VENV}/bin/activate
export VLLM_TARGET_DEVICE=xpu
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_QUANTIZE_Q40_LIB="${Q40_LIB}"
export VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export LD_LIBRARY_PATH="\${LD_LIBRARY_PATH}:/usr/local/lib/"

# Convenience aliases
alias vllm-activate='source ${VLLM_VENV}/bin/activate && source /opt/intel/oneapi/setvars.sh --force 2>/dev/null && export VLLM_TARGET_DEVICE=xpu'
alias oneapi='source /opt/intel/oneapi/setvars.sh --force'
VLLM_ENV
    echo "  Added vLLM environment + aliases to $BASHRC"
else
    echo "  vLLM env vars already in ~/.bashrc, skipping."
fi

# Clean up pip temp directory
rm -rf "$TMPDIR"
unset TMPDIR

echo -e "${GREEN}[8/8] Production environment configured.${NC}"
echo ""

# ============================================================
# Summary
# ============================================================
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN} Installation Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "  Platform:     ${PLATFORM} (8086:${GPU_ID})"
echo "  RAM:          ${TOTAL_RAM_GB}GB"
echo "  MAX_JOBS:     ${MAX_JOBS}"
echo "  vLLM:         ${INSTALL_DIR}/vllm"
echo "  XPU kernels:  ${INSTALL_DIR}/vllm-xpu-kernels"
echo "  Venv:         $VLLM_VENV"
echo ""
echo "  Quick start:"
echo "    vllm-activate"
echo "    vllm serve <model> \\"
echo "        --tensor-parallel-size 1 \\"
echo "        --gpu-memory-utilization ${GPU_UTIL_RECOMMEND} \\"
echo "        --enforce-eager \\"
echo "        --quantization fp8 \\"
echo "        --host 127.0.0.1 --port 8000"
echo ""
echo "  Recommended models:"
if [ "$TOTAL_RAM_GB" -le 16 ]; then
    echo "    Qwen/Qwen3-8B --quantization int4     (fits in ${TOTAL_RAM_GB}GB)"
elif [ "$TOTAL_RAM_GB" -le 32 ]; then
    echo "    Qwen/Qwen3-8B --quantization fp8      (best balance)"
    echo "    Qwen/Qwen3-14B --quantization int4     (needs INT4)"
else
    echo "    Qwen/Qwen3-8B --quantization fp8      (best balance)"
    echo "    Qwen/Qwen3-14B --quantization int4     (needs INT4)"
    echo "    Qwen/Qwen3-8B                          (FP16, no quant loss)"
fi

if [ "$PLATFORM" = "meteor_lake" ] && lsmod | grep -q "^i915 "; then
    echo ""
    echo -e "  ${YELLOW}⚠ IMPORTANT: Switch to xe driver for XPU support:${NC}"
    echo -e "    Add to GRUB: ${GREEN}i915.force_probe=!${GPU_ID} xe.force_probe=${GPU_ID}${NC}"
    echo "    Then: sudo update-grub && sudo reboot"
fi
echo ""
