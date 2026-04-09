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

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN} vLLM XPU Native Install (llm-scaler)${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
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

echo "  Install dir:   $INSTALL_DIR"
echo "  Repo root:     $REPO_ROOT"
echo ""

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

# ============================================================
# Phase 2: Create Python 3.12 virtual environment
# ============================================================
echo -e "${YELLOW}[2/8] Setting up Python environment...${NC}"

VLLM_VENV="$REAL_HOME/vllm-venv"
if [ -d "$VLLM_VENV" ] && [ -f "$VLLM_VENV/bin/python" ]; then
    echo "  Python 3.12 venv already exists at $VLLM_VENV"
else
    echo "  Creating Python 3.12 virtual environment at $VLLM_VENV..."
    sudo -u "$REAL_USER" python3.12 -m venv "$VLLM_VENV"
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
if [ -f /opt/intel/oneapi/setvars.sh ]; then
    source /opt/intel/oneapi/setvars.sh --force 2>/dev/null || true
else
    echo -e "${RED}  Warning: /opt/intel/oneapi/setvars.sh not found. Build may fail.${NC}"
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
    cd "$VLLM_DIR"

    echo "  Installing XPU requirements..."
    pip install -r requirements/xpu.txt 2>&1 | tail -3
    pip install arctic-inference==0.1.1 2>&1 | tail -1

    echo "  Building vLLM (MAX_JOBS=$MAX_JOBS)..."
    echo "  On 16GB RAM this can take 30-60 minutes. Do not interrupt."
    pip install --no-build-isolation . 2>&1 | tail -5

    if pip show vllm &>/dev/null; then
        echo -e "${GREEN}  vLLM built and installed successfully.${NC}"
    else
        echo -e "${RED}  vLLM build failed. Check errors above.${NC}"
        echo "  Retry: cd $VLLM_DIR && MAX_JOBS=$MAX_JOBS pip install --no-build-isolation ."
        exit 1
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

    if pip show vllm-xpu-kernels &>/dev/null; then
        echo -e "${GREEN}  vllm-xpu-kernels built successfully.${NC}"
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
VLLM_ENV
    echo "  Added vLLM environment to $BASHRC"
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
echo "  RAM:        ${TOTAL_RAM_GB}GB"
echo "  MAX_JOBS:   ${MAX_JOBS}"
echo "  vLLM:       ${INSTALL_DIR}/vllm"
echo "  XPU kernels: ${INSTALL_DIR}/vllm-xpu-kernels"
echo "  Venv:       $VLLM_VENV"
echo ""
echo "  To use vLLM, open a new terminal (or source ~/.bashrc) then:"
echo ""
echo "    # Serve a model"
echo "    vllm serve /path/to/model --device xpu"
echo ""
echo "    # With memory tuning for iGPU shared memory"
echo "    vllm serve /path/to/model --device xpu \\"
echo "        --gpu-memory-utilization 0.6 \\"
echo "        --enforce-eager"
echo ""
if [ "$TOTAL_RAM_GB" -le 16 ]; then
    echo -e "${YELLOW}  Note: 16GB RAM limits you to ~4B-8B models.${NC}"
    echo -e "${YELLOW}  Recommended: Qwen3.5-4B, Phi-4-mini, or similar.${NC}"
elif [ "$TOTAL_RAM_GB" -le 32 ]; then
    echo -e "${YELLOW}  Note: 32GB shared memory supports up to ~20B MoE models.${NC}"
    echo -e "${YELLOW}  Recommended: gpt-oss-20b (MXFP4), Qwen3.5-4B.${NC}"
fi
echo ""
