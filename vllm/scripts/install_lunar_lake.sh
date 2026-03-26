#!/bin/bash
# ==============================================================================
# Lunar Lake Native Install — llm-scaler vLLM on Nobara/Fedora
# ------------------------------------------------------------------------------
# Installs the SYCL/oneAPI + vLLM stack on Lunar Lake systems running
# Nobara or Fedora (no Docker required).
#
# Target: Intel Core Ultra 7 258V + Arc 140V (Xe2) on MSI Claw 8 AI+
# OS:     Nobara 43 / Fedora 42+ (DNF-based)
#
# Usage:
#   chmod +x install_lunar_lake.sh
#   ./install_lunar_lake.sh
#
# After install:
#   source ~/.bashrc
#   cd ~/llm-scaler-vllm
#   ./lunar_lake_serve.sh <model> --quantization int4
# ==============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

INSTALL_DIR="$HOME/llm-scaler-vllm"
VENV_DIR="$INSTALL_DIR/venv"

# ── Preflight checks ──────────────────────────────────────────────────────────

log_info "Checking system..."

# Check for Intel GPU
if ! lspci -nn | grep -qi '8086:64a0'; then
    log_warn "Arc 140V (64a0) not detected. Checking for any Intel GPU..."
    if ! lspci | grep -qi 'intel.*vga\|intel.*display\|intel.*3d'; then
        log_error "No Intel GPU detected. This script requires Lunar Lake Xe2."
        exit 1
    fi
fi
log_info "Intel GPU detected."

# Check xe driver
if lsmod | grep -q "^xe "; then
    log_info "xe driver loaded."
else
    log_warn "xe driver not loaded. It may load as 'i915' instead."
    log_warn "For best SYCL support, switch to xe driver (see OpenClaw-on-MSI-Claw-8 guide)."
fi

# ── Phase 1: System dependencies ─────────────────────────────────────────────

log_info "Phase 1/5: Installing system dependencies..."

sudo dnf install -y \
    cmake gcc gcc-c++ git wget curl \
    python3 python3-pip python3-devel python3-virtualenv \
    numactl \
    mesa-vulkan-drivers \
    libdrm-devel \
    2>&1 | tail -5

log_info "System dependencies installed."

# ── Phase 2: Intel oneAPI Base Toolkit + Level-Zero ───────────────────────────

log_info "Phase 2/5: Installing Intel oneAPI Base Toolkit + Level-Zero..."

# Add Intel repos first (needed for both oneAPI and Level-Zero)
if [ ! -f /etc/yum.repos.d/oneAPI.repo ]; then
    cat << 'REPO' | sudo tee /etc/yum.repos.d/oneAPI.repo
[oneAPI]
name=Intel oneAPI repository
baseurl=https://yum.repos.intel.com/oneapi
enabled=1
gpgcheck=1
repo_gpgcheck=1
gpgkey=https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
REPO
fi

# Intel compute-runtime repo (provides Level-Zero + Intel GPU runtime)
if [ ! -f /etc/yum.repos.d/intel-graphics.repo ]; then
    sudo dnf install -y 'dnf-command(config-manager)' 2>/dev/null || true
    # Try Intel's RPM repo for compute runtime
    sudo tee /etc/yum.repos.d/intel-graphics.repo > /dev/null << 'REPO'
[intel-graphics]
name=Intel Graphics Drivers
baseurl=https://repositories.intel.com/gpu/rhel/9/lts/2350/unified/leapfrog/
enabled=1
gpgcheck=1
repo_gpgcheck=0
gpgkey=https://repositories.intel.com/gpu/intel-graphics.key
REPO
fi

# Install Level-Zero + Intel compute runtime (GPU userspace driver)
# intel-compute-runtime provides libze_intel_gpu.so which bridges Level-Zero to the xe kernel driver
log_info "Installing Level-Zero GPU runtime + Intel compute runtime..."
sudo dnf install -y --skip-unavailable \
    level-zero level-zero-devel \
    intel-level-zero-gpu intel-level-zero-gpu-devel \
    oneapi-level-zero level-zero-loader \
    intel-compute-runtime \
    intel-ocloc \
    2>&1 | tail -10 || true

# Check if Level-Zero loader is available
if ldconfig -p 2>/dev/null | grep -q libze_loader; then
    log_info "Level-Zero loader found."
else
    log_warn "Level-Zero loader not found in ldconfig."
fi

# Check if GPU driver (compute runtime) is available
if ldconfig -p 2>/dev/null | grep -q libze_intel_gpu; then
    log_info "Intel GPU compute runtime found."
elif find /usr/lib64 /usr/local/lib64 /opt/intel -name "libze_intel_gpu.so*" 2>/dev/null | head -1 | grep -q .; then
    log_info "Intel GPU compute runtime found (not in ldconfig, may need LD_LIBRARY_PATH)."
else
    log_warn "Intel GPU compute runtime (libze_intel_gpu.so) NOT found."
    log_warn "This is required for XPU to detect your GPU."
    log_warn "Try: sudo dnf install intel-compute-runtime"
    log_warn "Or check: https://github.com/intel/compute-runtime/releases"
fi

# Install oneAPI
if [ -f /opt/intel/oneapi/setvars.sh ]; then
    log_info "oneAPI already installed. Skipping."
else
    log_info "Installing oneAPI (this takes a while — ~15GB download)..."
    sudo dnf install -y intel-oneapi-base-toolkit 2>&1 | tail -10

    # Add to bashrc
    if ! grep -q "setvars.sh" ~/.bashrc; then
        echo 'source /opt/intel/oneapi/setvars.sh --force 2>/dev/null' >> ~/.bashrc
    fi
fi

# Source it now — temporarily relax strict mode because setvars.sh
# has unbound variables and non-zero exits internally that conflict
# with our set -euo pipefail
log_info "Sourcing oneAPI environment..."
set +euo pipefail
source /opt/intel/oneapi/setvars.sh --force > /tmp/oneapi_init.log 2>&1 || true
set -euo pipefail
grep -E "^::|initialized" /tmp/oneapi_init.log || true
rm -f /tmp/oneapi_init.log
log_info "oneAPI configured."

# Fix MKL library path — PyTorch's bundled MKL stubs use relative RPATHs that
# break inside venvs. Preload the real oneAPI MKL to avoid runtime errors.
if [ -n "${MKLROOT:-}" ] && [ -f "$MKLROOT/lib/libmkl_core.so.2" ]; then
    export LD_PRELOAD="${MKLROOT}/lib/libmkl_core.so.2:${MKLROOT}/lib/libmkl_intel_thread.so.2:${MKLROOT}/lib/libmkl_intel_lp64.so.2${LD_PRELOAD:+:$LD_PRELOAD}"
    log_info "MKL preloaded from $MKLROOT"
fi

# ── Phase 3: Python venv + PyTorch XPU ───────────────────────────────────────

log_info "Phase 3/5: Setting up Python environment..."

mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# Detect Python version — PyTorch XPU requires Python 3.10-3.12
# Nobara 43 ships Python 3.14 which is too new for PyTorch XPU wheels
PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
log_info "System Python: $PY_VERSION"

VENV_PYTHON="python3"
if python3 -c "import sys; sys.exit(0 if sys.version_info[:2] <= (3,12) else 1)" 2>/dev/null; then
    log_info "Python $PY_VERSION is compatible with PyTorch XPU."
else
    log_warn "Python $PY_VERSION is too new for PyTorch XPU (needs <=3.12)."
    # Try to find Python 3.12
    if command -v python3.12 &> /dev/null; then
        VENV_PYTHON="python3.12"
        log_info "Found python3.12, will use it for the venv."
    else
        log_info "Installing Python 3.12..."
        sudo dnf install -y python3.12 python3.12-devel 2>&1 | tail -5
        if command -v python3.12 &> /dev/null; then
            VENV_PYTHON="python3.12"
            log_info "Python 3.12 installed."
        else
            log_error "Cannot install Python 3.12. PyTorch XPU requires Python <=3.12."
            log_error "Install manually: sudo dnf install python3.12 python3.12-devel"
            exit 1
        fi
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    $VENV_PYTHON -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

VENV_PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
log_info "Venv Python: $VENV_PY_VERSION"

pip install --upgrade pip wheel setuptools 2>&1 | tail -3

log_info "Installing PyTorch XPU (this downloads ~2-3 GB, may take a while)..."
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
    --index-url https://download.pytorch.org/whl/xpu

# Verify PyTorch XPU
python3 -c "
import torch
assert torch.xpu.is_available(), 'XPU not available!'
print(f'PyTorch {torch.__version__} with XPU: OK')
print(f'Device: {torch.xpu.get_device_properties(0).name}')
" || {
    log_error "PyTorch XPU verification failed."
    log_error "Ensure Level-Zero is installed and xe driver is loaded."
    exit 1
}
log_info "PyTorch XPU working."

# ── Phase 4: Clone and build vLLM with multi-arc patches ────────────────────

log_info "Phase 4/5: Building vLLM with Intel XPU support..."

REPO_URL="${LLM_SCALER_REPO:-https://github.com/MegaStood/llm-scaler.git}"
BRANCH="${LLM_SCALER_BRANCH:-claude/check-lunar-lake-compatibility-CB5w6}"

if [ ! -d "$INSTALL_DIR/llm-scaler" ]; then
    git clone -b "$BRANCH" "$REPO_URL" "$INSTALL_DIR/llm-scaler"
fi

# Build patched vLLM
# Use pip show (not import) to avoid false positives from local vllm/ directories
if ! pip show vllm &>/dev/null; then
    cd "$INSTALL_DIR/llm-scaler/vllm"

    if [ ! -d "$INSTALL_DIR/vllm" ]; then
        git clone -b v0.14.0 https://github.com/vllm-project/vllm.git "$INSTALL_DIR/vllm"
    fi

    cd "$INSTALL_DIR/vllm"
    git apply "$INSTALL_DIR/llm-scaler/vllm/patches/vllm_for_multi_arc.patch" 2>/dev/null || \
        log_warn "Patch already applied or failed. Continuing..."

    log_info "Installing vLLM XPU requirements..."
    pip install -r requirements/xpu.txt
    pip install arctic-inference==0.1.1 || log_warn "arctic-inference install failed, continuing..."

    export VLLM_TARGET_DEVICE=xpu
    export CPATH=/opt/intel/oneapi/dpcpp-ct/latest/include/:${CPATH:-}
    log_info "Building vLLM (this may take 10-30 minutes)..."
    pip install --no-build-isolation .

    log_info "vLLM built successfully."
else
    log_info "vLLM already installed. Skipping build."
fi

# Install extras
pip install accelerate hf_transfer transformers ijson

# ── Phase 5: Install XPU kernels + configure ─────────────────────────────────

log_info "Phase 5/5: Installing XPU kernels and configuring environment..."

# Remove stale build if previous attempt failed
if [ -d "$INSTALL_DIR/vllm-xpu-kernels" ] && ! pip show vllm-xpu-kernels &>/dev/null; then
    rm -rf "$INSTALL_DIR/vllm-xpu-kernels"
fi
if [ ! -d "$INSTALL_DIR/vllm-xpu-kernels" ]; then
    cd "$INSTALL_DIR"
    git clone https://github.com/vllm-project/vllm-xpu-kernels.git
    cd vllm-xpu-kernels
    git checkout 4c83144
    # Remove conflicting version pins
    sed -i 's|^--extra-index-url=https://download.pytorch.org/whl/xpu|# &|' requirements.txt
    sed -i 's|^torch==2.10.0+xpu|# &|' requirements.txt
    sed -i 's|^triton-xpu|# &|' requirements.txt
    sed -i 's|^transformers|# &|' requirements.txt
    pip install -r requirements.txt
    # Limit parallel SYCL compilations to avoid OOM on 32GB shared-memory systems
    # Each icpx process can use ~4GB; -j=8 default causes OOM kills
    export MAX_JOBS=2
    pip install --no-build-isolation .
fi

# Install triton-xpu
pip install triton-xpu==3.6.0 --extra-index-url=https://download.pytorch.org/whl/test/xpu

# Copy launch script
cp "$INSTALL_DIR/llm-scaler/vllm/scripts/lunar_lake_serve.sh" "$INSTALL_DIR/"
chmod +x "$INSTALL_DIR/lunar_lake_serve.sh"

# Add activation helper to bashrc
if ! grep -q "llm-scaler-vllm" ~/.bashrc; then
    cat << 'BASHRC' >> ~/.bashrc

# llm-scaler-vllm (Lunar Lake)
alias vllm-activate='source ~/llm-scaler-vllm/venv/bin/activate && source /opt/intel/oneapi/setvars.sh --force 2>/dev/null'
alias vllm-serve='cd ~/llm-scaler-vllm && source venv/bin/activate && source /opt/intel/oneapi/setvars.sh --force 2>/dev/null && ./lunar_lake_serve.sh'
BASHRC
fi

# ── Done ──────────────────────────────────────────────────────────────────────

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_info "Installation complete!"
echo ""
echo "  Install dir:  $INSTALL_DIR"
echo "  Python venv:  $VENV_DIR"
echo "  Launch script: $INSTALL_DIR/lunar_lake_serve.sh"
echo ""
echo "  Quick start:"
echo "    vllm-activate"
echo "    cd ~/llm-scaler-vllm"
echo "    ./lunar_lake_serve.sh Qwen/Qwen3-8B --quantization int4"
echo ""
echo "  Or use the alias:"
echo "    vllm-serve Qwen/Qwen3-8B --quantization int4"
echo ""
echo "  Recommended models for 32GB Lunar Lake:"
echo "    Qwen/Qwen3-8B --quantization fp8         (best balance)"
echo "    Qwen/Qwen3-14B --quantization int4        (needs INT4)"
echo "    Qwen/Qwen3.5-35B-A3B --quantization int4 --max-model-len 8192"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
