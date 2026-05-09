#!/bin/bash
# ==============================================================================
# Meteor Lake / Arrow Lake Native Install — llm-scaler vLLM on Nobara/Fedora
# ------------------------------------------------------------------------------
# Installs the SYCL/oneAPI + vLLM stack on Intel Core Ultra systems with
# integrated Xe-LPG (Meteor Lake) or Xe-LPG+ (Arrow Lake) GPUs.
#
# Supported platforms:
#   Meteor Lake (Xe-LPG):   Core Ultra 155H, 135H, etc.  — PCI: 7d55, 7dd5, 7d40, 7d45
#   Arrow Lake-H (Xe-LPG+): Core Ultra 255H, 245H, etc.  — PCI: 7d51, 7dd1, 7d41, 7d67
#   Lunar Lake (Xe2):        Core Ultra 258V, 238V, etc.  — PCI: 64a0
#
# OS: Nobara 43+ / Fedora 42+ (DNF-based)
#
# Usage:
#   chmod +x install_meteor_arrow_lake.sh
#   ./install_meteor_arrow_lake.sh
#
# After install:
#   source ~/.bashrc
#   vllm-activate
#   vllm serve <model> --tensor-parallel-size 1 --enforce-eager --quantization fp8
# ==============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step()  { echo -e "${CYAN}[STEP]${NC} $1"; }

INSTALL_DIR="$HOME/llm-scaler-vllm"
VENV_DIR="$INSTALL_DIR/venv"

# ── GPU Detection ─────────────────────────────────────────────────────────────

detect_intel_gpu() {
    # Returns: platform_name device_id
    # Scans PCI bus for known Intel iGPU device IDs
    local gpu_id
    gpu_id=$(lspci -nn | grep -i 'vga\|3d\|display' | grep -oP '8086:\K[0-9a-fA-F]+' | head -1)

    case "${gpu_id,,}" in
        # Lunar Lake — Xe2
        64a0)
            echo "lunar_lake $gpu_id" ;;
        # Meteor Lake — Xe-LPG
        7d55)
            echo "meteor_lake $gpu_id" ;;  # Arc Graphics (dual-channel)
        7dd5)
            echo "meteor_lake $gpu_id" ;;  # Intel Graphics (single-channel)
        7d40|7d45)
            echo "meteor_lake $gpu_id" ;;  # Intel Graphics variants
        # Arrow Lake — Xe-LPG+
        7d51)
            echo "arrow_lake $gpu_id" ;;   # Arc 130T/140T
        7dd1)
            echo "arrow_lake $gpu_id" ;;   # Intel Graphics
        7d41)
            echo "arrow_lake $gpu_id" ;;   # Intel Graphics
        7d67)
            echo "arrow_lake $gpu_id" ;;   # Arrow Lake-S (desktop)
        # Discrete GPUs (not the target of this script but detected)
        e211|e210)
            echo "arc_pro_b60 $gpu_id" ;;
        56a0)
            echo "arc_a770 $gpu_id" ;;
        *)
            echo "unknown $gpu_id" ;;
    esac
}

# ── Preflight checks ─────────────────────────────────────────────────────────

log_info "Checking system..."

read -r PLATFORM GPU_ID <<< "$(detect_intel_gpu)"

if [ "$PLATFORM" = "unknown" ]; then
    if lspci | grep -qi 'intel.*vga\|intel.*display\|intel.*3d'; then
        log_warn "Unrecognized Intel GPU device ID: $GPU_ID"
        log_warn "This may be a newer Intel iGPU not yet in this script's database."
        log_warn "Proceeding anyway — the software stack is device-agnostic."
        PLATFORM="unknown_intel"
    else
        log_error "No Intel GPU detected. This script requires an Intel Xe-class iGPU."
        exit 1
    fi
elif [ "$PLATFORM" = "arc_pro_b60" ] || [ "$PLATFORM" = "arc_a770" ]; then
    log_warn "Discrete GPU detected ($PLATFORM, $GPU_ID)."
    log_warn "This script is designed for integrated GPUs (Meteor/Arrow/Lunar Lake)."
    log_warn "For discrete GPUs, use the standard llm-scaler Docker install."
    log_warn "Proceeding anyway..."
fi

log_info "Detected platform: $PLATFORM (device ID: 8086:$GPU_ID)"

case "$PLATFORM" in
    meteor_lake)
        log_info "Platform: Meteor Lake (Xe-LPG) — Core Ultra 1st Gen (155H, 135H, etc.)"
        ;;
    arrow_lake)
        log_info "Platform: Arrow Lake (Xe-LPG+) — Core Ultra 200H (255H, 245H, etc.)"
        ;;
    lunar_lake)
        log_info "Platform: Lunar Lake (Xe2) — Core Ultra 200V (258V, 238V, etc.)"
        ;;
esac

# ── Xe driver check ──────────────────────────────────────────────────────────

if lsmod | grep -q "^xe "; then
    log_info "xe driver loaded — good."
elif lsmod | grep -q "^i915 "; then
    log_warn "i915 driver loaded instead of xe."
    if [ "$PLATFORM" = "meteor_lake" ]; then
        log_warn "Meteor Lake uses i915 by default. For best SYCL/oneAPI support,"
        log_warn "switch to the xe driver by adding these kernel boot parameters:"
        echo ""
        echo -e "    ${CYAN}i915.force_probe=!${GPU_ID} xe.force_probe=${GPU_ID}${NC}"
        echo ""
        log_warn "Add to /etc/default/grub GRUB_CMDLINE_LINUX_DEFAULT, then:"
        log_warn "  sudo grub2-mkconfig -o /boot/grub2/grub.cfg && sudo reboot"
        echo ""
        log_warn "The install will continue, but XPU features may not work until"
        log_warn "you switch to the xe driver."
    else
        log_warn "For best SYCL support, switch to xe driver."
        log_warn "Add kernel params: i915.force_probe=!${GPU_ID} xe.force_probe=${GPU_ID}"
    fi
else
    log_warn "Neither xe nor i915 driver detected for GPU."
    log_warn "Ensure your kernel supports your GPU (kernel 6.8+ recommended)."
fi

# ── Memory check ─────────────────────────────────────────────────────────────

TOTAL_MEM_GB=$(free -g | awk '/^Mem:/{print $2}')
log_info "System memory: ${TOTAL_MEM_GB}GB"

if [ "$TOTAL_MEM_GB" -lt 16 ]; then
    log_error "Only ${TOTAL_MEM_GB}GB RAM detected. Minimum 16GB required for vLLM inference."
    log_error "Meteor Lake laptops with single-channel RAM may only expose 8-16GB."
    log_error "Ensure dual-channel memory is installed for best iGPU performance."
    exit 1
elif [ "$TOTAL_MEM_GB" -lt 24 ]; then
    log_warn "${TOTAL_MEM_GB}GB RAM detected. Only small models (≤8B) will fit."
    log_warn "Use --gpu-memory-utilization 0.6 and --quantization int4."
    GPU_UTIL_RECOMMEND="0.6"
elif [ "$TOTAL_MEM_GB" -lt 48 ]; then
    log_info "${TOTAL_MEM_GB}GB RAM — good for 8B-14B models with INT4/FP8 quantization."
    GPU_UTIL_RECOMMEND="0.7"
else
    log_info "${TOTAL_MEM_GB}GB RAM — plenty for most models up to 14B FP16."
    GPU_UTIL_RECOMMEND="0.8"
fi

# ── Phase 1: System dependencies ─────────────────────────────────────────────

log_step "Phase 1/5: Installing system dependencies..."

sudo dnf install -y \
    cmake gcc gcc-c++ git wget curl \
    python3 python3-pip python3-devel python3-virtualenv \
    numactl \
    mesa-vulkan-drivers \
    libdrm-devel \
    2>&1 | tail -5

log_info "System dependencies installed."

# ── Phase 2: Intel oneAPI Base Toolkit + Level-Zero ──────────────────────────

log_step "Phase 2/5: Installing Intel oneAPI Base Toolkit + Level-Zero..."

# Add Intel repos
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

# Intel compute-runtime repo
if [ ! -f /etc/yum.repos.d/intel-graphics.repo ]; then
    sudo dnf install -y 'dnf-command(config-manager)' 2>/dev/null || true
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

# Install Level-Zero + Intel compute runtime
log_info "Installing Level-Zero GPU runtime + Intel compute runtime..."
sudo dnf install -y --skip-unavailable \
    level-zero level-zero-devel \
    intel-level-zero-gpu intel-level-zero-gpu-devel \
    oneapi-level-zero oneapi-level-zero-devel level-zero-loader \
    intel-compute-runtime \
    intel-ocloc \
    2>&1 | tail -10 || true

# Verify Level-Zero
if ldconfig -p 2>/dev/null | grep -q libze_loader; then
    log_info "Level-Zero loader found."
else
    log_warn "Level-Zero loader not found in ldconfig."
fi

if ldconfig -p 2>/dev/null | grep -q libze_intel_gpu; then
    log_info "Intel GPU compute runtime found."
elif find /usr/lib64 /usr/local/lib64 /opt/intel -name "libze_intel_gpu.so*" 2>/dev/null | head -1 | grep -q .; then
    log_info "Intel GPU compute runtime found (not in ldconfig, may need LD_LIBRARY_PATH)."
else
    log_warn "Intel GPU compute runtime (libze_intel_gpu.so) NOT found."
    log_warn "This is required for XPU to detect your GPU."
    log_warn "Try: sudo dnf install intel-compute-runtime"
fi

# Install oneAPI
if [ -f /opt/intel/oneapi/setvars.sh ]; then
    log_info "oneAPI already installed. Skipping."
else
    log_info "Installing oneAPI (this takes a while — ~15GB download)..."
    sudo dnf install -y intel-oneapi-base-toolkit 2>&1 | tail -10

    if ! grep -q "setvars.sh" ~/.bashrc; then
        echo 'source /opt/intel/oneapi/setvars.sh --force 2>/dev/null' >> ~/.bashrc
    fi
fi

# Source oneAPI
log_info "Sourcing oneAPI environment..."
set +euo pipefail
source /opt/intel/oneapi/setvars.sh --force > /tmp/oneapi_init.log 2>&1 || true
set -euo pipefail
grep -E "^::|initialized" /tmp/oneapi_init.log || true
rm -f /tmp/oneapi_init.log
log_info "oneAPI configured."

# Fix MKL library path
if [ -n "${MKLROOT:-}" ] && [ -f "$MKLROOT/lib/libmkl_core.so.2" ]; then
    export LD_PRELOAD="${MKLROOT}/lib/libmkl_core.so.2:${MKLROOT}/lib/libmkl_intel_thread.so.2:${MKLROOT}/lib/libmkl_intel_lp64.so.2${LD_PRELOAD:+:$LD_PRELOAD}"
    log_info "MKL preloaded from $MKLROOT"
fi

# ── Phase 3: Python venv + PyTorch XPU ────────────────────────────────────────

log_step "Phase 3/5: Setting up Python environment..."

mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# Detect Python version — PyTorch XPU requires Python 3.10-3.12
PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
log_info "System Python: $PY_VERSION"

VENV_PYTHON="python3"
if python3 -c "import sys; sys.exit(0 if sys.version_info[:2] <= (3,12) else 1)" 2>/dev/null; then
    log_info "Python $PY_VERSION is compatible with PyTorch XPU."
else
    log_warn "Python $PY_VERSION is too new for PyTorch XPU (needs <=3.12)."
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
    if [ "$PLATFORM" = "meteor_lake" ] && lsmod | grep -q "^i915 "; then
        log_error "Meteor Lake detected with i915 driver — switch to xe driver first!"
        log_error "Add kernel params: i915.force_probe=!${GPU_ID} xe.force_probe=${GPU_ID}"
    else
        log_error "Ensure Level-Zero is installed and xe driver is loaded."
    fi
    exit 1
}
log_info "PyTorch XPU working."

# ── Phase 4: Clone and build vLLM with multi-arc patches ─────────────────────

log_step "Phase 4/5: Building vLLM with Intel XPU support..."

REPO_URL="${LLM_SCALER_REPO:-https://github.com/MegaStood/llm-scaler.git}"
BRANCH="${LLM_SCALER_BRANCH:-claude/check-lunar-lake-compatibility-CB5w6}"

if [ ! -d "$INSTALL_DIR/llm-scaler" ]; then
    git clone -b "$BRANCH" "$REPO_URL" "$INSTALL_DIR/llm-scaler"
fi

# Build patched vLLM
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

# Patch xpu_worker.py: disable CCL all_reduce warmup for single-GPU
XPU_WORKER=$(python3 -c "import vllm; import os; print(os.path.join(os.path.dirname(vllm.__file__), 'v1/worker/xpu_worker.py'))" 2>/dev/null || true)
if [ -n "$XPU_WORKER" ] && [ -f "$XPU_WORKER" ]; then
    if grep -q "torch.distributed.all_reduce" "$XPU_WORKER"; then
        log_info "Patching xpu_worker.py to disable CCL all_reduce warmup (single-GPU fix)..."
        sed -i '/torch\.distributed\.all_reduce(/,/)/s/^/#/' "$XPU_WORKER"
        log_info "xpu_worker.py patched."
    else
        log_info "xpu_worker.py already patched or no all_reduce found."
    fi
fi

# Install extras
pip install accelerate hf_transfer transformers ijson

# ── Phase 5: Install XPU kernels + configure ────────────────────────────────

log_step "Phase 5/5: Installing XPU kernels and configuring environment..."

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
    # Limit parallel SYCL compilations to avoid OOM
    # Adjust MAX_JOBS based on available memory
    if [ "$TOTAL_MEM_GB" -lt 24 ]; then
        export MAX_JOBS=${MAX_JOBS:-4}
        log_warn "Low memory (${TOTAL_MEM_GB}GB) — using MAX_JOBS=$MAX_JOBS to avoid OOM during kernel build."
    else
        export MAX_JOBS=${MAX_JOBS:-6}
    fi
    log_info "Building XPU kernels with MAX_JOBS=$MAX_JOBS (933 SYCL files, expect 1.5-2 hours)..."
    pip install --no-build-isolation .
fi

# Install triton-xpu — MUST uninstall plain triton first!
# vllm-xpu-kernels pulls in plain 'triton' as a transitive dependency.
# Plain triton's libtriton.so lacks the Intel backend, causing:
#   ImportError: cannot import name 'intel' from 'triton._C.libtriton'
# which degrades @triton.jit kernels to plain functions (TypeError: not subscriptable).
pip uninstall triton triton-xpu -y 2>/dev/null || true
pip install triton-xpu==3.6.0 --extra-index-url=https://download.pytorch.org/whl/test/xpu

# Copy launch script
cp "$INSTALL_DIR/llm-scaler/vllm/scripts/lunar_lake_serve.sh" "$INSTALL_DIR/"
chmod +x "$INSTALL_DIR/lunar_lake_serve.sh"

# Add activation helper to bashrc
if ! grep -q "llm-scaler-vllm" ~/.bashrc; then
    cat << 'BASHRC' >> ~/.bashrc

# llm-scaler-vllm (Intel Xe iGPU)
alias vllm-activate='source ~/llm-scaler-vllm/venv/bin/activate && source /opt/intel/oneapi/setvars.sh --force 2>/dev/null && export VLLM_TARGET_DEVICE=xpu'
alias vllm-serve='cd ~/llm-scaler-vllm && source venv/bin/activate && source /opt/intel/oneapi/setvars.sh --force 2>/dev/null && ./lunar_lake_serve.sh'
alias oneapi='source /opt/intel/oneapi/setvars.sh --force'
BASHRC
fi

# ── Done ─────────────────────────────────────────────────────────────────────

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_info "Installation complete!"
echo ""
echo "  Platform:     $PLATFORM (8086:$GPU_ID)"
echo "  Install dir:  $INSTALL_DIR"
echo "  Python venv:  $VENV_DIR"
echo "  System RAM:   ${TOTAL_MEM_GB}GB"
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
if [ "$TOTAL_MEM_GB" -lt 24 ]; then
    echo "    Qwen/Qwen3-8B --quantization int4     (fits in ${TOTAL_MEM_GB}GB)"
elif [ "$TOTAL_MEM_GB" -lt 48 ]; then
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
    echo -e "    Add to GRUB: ${CYAN}i915.force_probe=!${GPU_ID} xe.force_probe=${GPU_ID}${NC}"
    echo "    Then: sudo grub2-mkconfig -o /boot/grub2/grub.cfg && sudo reboot"
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
