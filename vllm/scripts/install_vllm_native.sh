#!/bin/bash
# ============================================================
# vLLM XPU Native Install Script (from llm-scaler)
# For Intel iGPU shared-memory systems: Lunar Lake, Meteor Lake, Arrow Lake
#
# This extracts the Dockerfile build steps for bare-metal installation.
# Tested on: Ubuntu 25.04 / Nobara 41
#
# Usage:
#   chmod +x install_vllm_native.sh
#   sudo bash install_vllm_native.sh
#
# After install:
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
apt-get install -y python3.12 python3.12-dev python3-pip
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 2>/dev/null || true
update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 2>/dev/null || true

apt-get install -y --no-install-recommends --fix-missing \
    curl ffmpeg git libsndfile1 libsm6 libxext6 libaio-dev \
    libgl1 lsb-release numactl wget vim linux-libc-dev \
    intel-oneapi-dpcpp-ct

# Suppress pip externally-managed error
python3 -m pip config set global.break-system-packages true 2>/dev/null || true

echo -e "${GREEN}[1/8] System dependencies installed.${NC}"
echo ""

# ============================================================
# Phase 2: Set environment variables
# ============================================================
echo -e "${YELLOW}[2/8] Configuring environment...${NC}"

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib/"
export VLLM_TARGET_DEVICE=xpu
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Source oneAPI if available
if [ -f /opt/intel/oneapi/setvars.sh ]; then
    source /opt/intel/oneapi/setvars.sh --force 2>/dev/null || true
fi

# Add DPCPP include path for vLLM compilation
DPCPP_INCLUDE=$(find /opt/intel/oneapi/dpcpp-ct/ -name "include" -type d 2>/dev/null | head -1)
if [ -n "$DPCPP_INCLUDE" ]; then
    export CPATH="${DPCPP_INCLUDE}:${CPATH}"
    echo "  DPCPP include: $DPCPP_INCLUDE"
fi

echo -e "${GREEN}[2/8] Environment configured.${NC}"
echo ""

# ============================================================
# Phase 3: Clone and patch vLLM
# ============================================================
echo -e "${YELLOW}[3/8] Cloning and patching vLLM v0.14.0...${NC}"

mkdir -p "$INSTALL_DIR"

VLLM_DIR="$INSTALL_DIR/vllm"
if [ -d "$VLLM_DIR" ]; then
    echo "  $VLLM_DIR already exists. Skipping clone."
    echo "  To re-clone: rm -rf $VLLM_DIR and re-run."
else
    sudo -u "$REAL_USER" git clone -b v0.14.0 https://github.com/vllm-project/vllm.git "$VLLM_DIR"
fi

# Apply Intel multi-arc patch
PATCH_FILE="$REPO_ROOT/vllm/patches/vllm_for_multi_arc.patch"
if [ -f "$PATCH_FILE" ]; then
    cd "$VLLM_DIR"
    if git apply --check "$PATCH_FILE" 2>/dev/null; then
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

echo -e "${GREEN}[3/8] vLLM cloned and patched.${NC}"
echo ""

# ============================================================
# Phase 4: Install vLLM XPU requirements + build
# ============================================================
echo -e "${YELLOW}[4/8] Building vLLM (MAX_JOBS=${MAX_JOBS}, this will take a while)...${NC}"

cd "$VLLM_DIR"
pip install -r requirements/xpu.txt
pip install arctic-inference==0.1.1
pip install --no-build-isolation .

echo -e "${GREEN}[4/8] vLLM built and installed.${NC}"
echo ""

# ============================================================
# Phase 5: Install additional Python dependencies
# ============================================================
echo -e "${YELLOW}[5/8] Installing additional dependencies...${NC}"

pip install accelerate hf_transfer 'modelscope!=1.15.0'
pip install librosa soundfile decord
pip install git+https://github.com/huggingface/transformers.git
pip install ijson
pip install bigdl-core==2.4.0b2

echo -e "${GREEN}[5/8] Dependencies installed.${NC}"
echo ""

# ============================================================
# Phase 6: Build vllm-xpu-kernels
# ============================================================
echo -e "${YELLOW}[6/8] Building vllm-xpu-kernels...${NC}"

XPU_KERNELS_DIR="$INSTALL_DIR/vllm-xpu-kernels"
if [ -d "$XPU_KERNELS_DIR" ]; then
    echo "  $XPU_KERNELS_DIR already exists. Skipping clone."
else
    sudo -u "$REAL_USER" git clone https://github.com/vllm-project/vllm-xpu-kernels.git "$XPU_KERNELS_DIR"
fi

cd "$XPU_KERNELS_DIR"
sudo -u "$REAL_USER" git checkout 4c83144

# Comment out conflicting pinned deps (we already installed them)
sed -i 's|^--extra-index-url=https://download.pytorch.org/whl/xpu|# &|' requirements.txt
sed -i 's|^torch==2.10.0+xpu|# &|' requirements.txt
sed -i 's|^triton-xpu|# &|' requirements.txt
sed -i 's|^transformers|# &|' requirements.txt

pip install -r requirements.txt
pip install --no-build-isolation .

echo -e "${GREEN}[6/8] vllm-xpu-kernels built.${NC}"
echo ""

# ============================================================
# Phase 7: Fix triton
# ============================================================
echo -e "${YELLOW}[7/8] Fixing triton-xpu...${NC}"

pip uninstall triton triton-xpu -y 2>/dev/null || true
pip install triton-xpu==3.6.0 --extra-index-url=https://download.pytorch.org/whl/test/xpu

echo -e "${GREEN}[7/8] triton-xpu fixed.${NC}"
echo ""

# ============================================================
# Phase 8: Configure environment for production
# ============================================================
echo -e "${YELLOW}[8/8] Configuring production environment...${NC}"

# Find the vllm_int4 library path
PYTHON_SITE=$(python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || echo "/usr/local/lib/python3.12/dist-packages")

# Add environment to user's bashrc
BASHRC="$REAL_HOME/.bashrc"
if ! grep -q "VLLM_TARGET_DEVICE" "$BASHRC" 2>/dev/null; then
    cat >> "$BASHRC" << EOF

# vLLM XPU environment (added by install_vllm_native.sh)
source /opt/intel/oneapi/setvars.sh --force 2>/dev/null
export LD_LIBRARY_PATH="\$LD_LIBRARY_PATH:/usr/local/lib/"
export VLLM_TARGET_DEVICE=xpu
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_QUANTIZE_Q40_LIB="${PYTHON_SITE}/vllm_int4_for_multi_arc.so"
export VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
EOF
    echo "  Added vLLM environment to $BASHRC"
fi

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
echo "  vLLM:       $VLLM_DIR"
echo "  XPU kernels: $XPU_KERNELS_DIR"
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
