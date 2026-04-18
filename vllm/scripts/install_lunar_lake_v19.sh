#!/bin/bash
# ==============================================================================
# Lunar Lake Native Install — vLLM v0.19.0 on Nobara/Fedora
# ------------------------------------------------------------------------------
# Installs the SYCL/oneAPI + vLLM v0.19.0 stack on Lunar Lake systems.
# v0.19.0 has native Gemma 4, MoE, and multimodal support — no patches needed.
#
# Target: Intel Core Ultra 7 258V + Arc 140V (Xe2) on MSI Claw 8 AI+
# OS:     Nobara 43 / Fedora 42+ (DNF-based)
#
# This installs alongside v0.14.0 (if present) in a separate directory:
#   v0.14.0: ~/llm-scaler-vllm/
#   v0.19.0: ~/llm-scaler-vllm-v19/
#
# Usage:
#   chmod +x install_lunar_lake_v19.sh
#   ./install_lunar_lake_v19.sh
#
# After install:
#   source ~/.bashrc
#   vllm-v19-activate
#   vllm serve google/gemma-4-12b-it --enforce-eager --dtype float16
# ==============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

VLLM_VERSION="v0.19.0"
INSTALL_DIR="$HOME/llm-scaler-vllm-v19"
VENV_DIR="$INSTALL_DIR/venv"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_info "vLLM $VLLM_VERSION Lunar Lake Installer"
echo "  Install dir: $INSTALL_DIR"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ── Preflight checks ──────────────────────────────────────────────────────────

log_info "Checking system..."

# Guard: don't run as root (venv would land in /root/)
if [ "$EUID" -eq 0 ]; then
    log_error "Don't run as root — use your normal user. The script calls sudo when needed."
    exit 1
fi

# Check GPU access permissions
if [ ! -r /dev/dri/renderD128 ]; then
    log_error "/dev/dri/renderD128 not accessible. Add user to render group:"
    log_error "  sudo usermod -aG render $USER && newgrp render"
    exit 1
fi

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
# NOTE: Uses RHEL 9 repo — Intel doesn't publish Fedora-native packages.
# Works on Nobara/Fedora via --skip-unavailable; verify Level-Zero loaded below.
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

# Install Level-Zero + Intel compute runtime (GPU userspace driver)
log_info "Installing Level-Zero GPU runtime + Intel compute runtime..."
sudo dnf install -y --skip-unavailable \
    level-zero level-zero-devel \
    intel-level-zero-gpu intel-level-zero-gpu-devel \
    oneapi-level-zero oneapi-level-zero-devel level-zero-loader \
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

# Source it now
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

# ── Phase 3: Python venv + PyTorch XPU ───────────────────────────────────────

log_info "Phase 3/5: Setting up Python environment..."

mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# Detect Python version — PyTorch XPU requires Python 3.10-3.12
PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
log_info "System Python: $PY_VERSION"

VENV_PYTHON="python3"
if python3 -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)" 2>/dev/null; then
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
# Pinning matches vLLM 0.19.0 requirements/xpu.txt: torch==2.10.0+xpu via
# --extra-index-url (so PyPI is still searched for common deps).
pip install --extra-index-url https://download.pytorch.org/whl/xpu \
    torch==2.10.0+xpu torchvision==0.25.0 torchaudio==2.10.0

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

# ── Phase 4: Build vLLM v0.19.0 (no patches needed) ─────────────────────────

log_info "Phase 4/5: Building vLLM $VLLM_VERSION with native XPU support..."
log_info "v0.19.0 has upstream Intel XPU support — no multi-arc patch needed."

if ! pip show vllm &>/dev/null; then
    if [ ! -d "$INSTALL_DIR/vllm" ]; then
        git clone -b "$VLLM_VERSION" https://github.com/vllm-project/vllm.git "$INSTALL_DIR/vllm"
    fi

    cd "$INSTALL_DIR/vllm"

    log_info "Installing vLLM XPU requirements..."
    pip install -r requirements/xpu.txt 2>&1 | tail -5

    # Upgrade vllm_xpu_kernels v0.1.4 -> v0.1.5
    # v0.1.4 (pinned in vLLM 0.19.0's requirements/xpu.txt) has a bug where
    # is_xe2_arch() doesn't include intel_gpu_lnl_m — so Lunar Lake iGPU is
    # rejected with "Only XE2 cutlass kernel is supported currently." error.
    # v0.1.5 adds lnl_m to the allowlist. Source:
    # https://github.com/vllm-project/vllm-xpu-kernels/blob/v0.1.5/csrc/utils.h
    log_info "Upgrading vllm_xpu_kernels to v0.1.5 (Lunar Lake XE2 fix)..."
    pip install --force-reinstall --no-deps \
        https://github.com/vllm-project/vllm-xpu-kernels/releases/download/v0.1.5/vllm_xpu_kernels-0.1.5-cp38-abi3-manylinux_2_28_x86_64.whl \
        2>&1 | tail -3

    export VLLM_TARGET_DEVICE=xpu
    export CPATH=/opt/intel/oneapi/dpcpp-ct/latest/include/:${CPATH:-}
    log_info "Building vLLM (this may take 10-30 minutes)..."
    pip install --no-build-isolation . 2>&1 | tail -5

    if pip show vllm &>/dev/null; then
        log_info "vLLM $VLLM_VERSION built successfully."
    else
        log_error "vLLM build failed. Check errors above."
        log_error "Retry: cd $INSTALL_DIR/vllm && VLLM_TARGET_DEVICE=xpu pip install --no-build-isolation ."
        exit 1
    fi
else
    log_info "vLLM already installed. Skipping build."
fi

# Patch xpu_worker.py: disable CCL all_reduce warmup for single-GPU
# oneCCL's KVS init fails on devices without wired Ethernet (e.g. handhelds).
# Check if this is still needed in v0.19 — may have been fixed upstream.
XPU_WORKER=$(python3 -c "import vllm; import os; print(os.path.join(os.path.dirname(vllm.__file__), 'v1/worker/xpu_worker.py'))" 2>/dev/null || true)
if [ -n "$XPU_WORKER" ] && [ -f "$XPU_WORKER" ]; then
    if grep -q "torch.distributed.all_reduce" "$XPU_WORKER"; then
        log_info "Patching xpu_worker.py to disable CCL all_reduce warmup (single-GPU fix)..."
        python3 -c "
import re, sys
f = sys.argv[1]
src = open(f).read()
# Comment out torch.distributed.all_reduce(...) calls (may span multiple lines)
patched = re.sub(
    r'(\n)([ \t]*)(torch\.distributed\.all_reduce\([^)]*\))',
    r'\1\2# \3  # disabled for single-GPU',
    src)
if patched != src:
    open(f, 'w').write(patched)
    print('Patched.')
else:
    print('No match found.')
" "$XPU_WORKER"
        log_info "xpu_worker.py patched."
    else
        log_info "xpu_worker.py: no all_reduce warmup found (fixed in v0.19?)."
    fi
fi

# Install extras
# Gemma 4 may need transformers>=5.x for full support, but 5.x is not yet
# on PyPI stable. Try to install it; fall back to latest 4.x if unavailable.
pip install --no-deps accelerate hf_transfer ijson 2>&1 | tail -3
if pip install 'transformers>=5.5.0' 2>&1 | tail -3; then
    log_info "transformers 5.x installed (Gemma 4 full support)."
else
    log_warn "transformers 5.x not available on PyPI — installing latest 4.x."
    log_warn "Gemma 4 models may need transformers>=5.5.0 when it's released."
    pip install -U transformers 2>&1 | tail -3
fi

# Verify torch XPU is still intact after all pip installs
TORCH_XPU_OK=$(python3 -c "import torch; print('yes' if torch.xpu.is_available() else 'no')" 2>/dev/null || echo "no")
if [ "$TORCH_XPU_OK" != "yes" ]; then
    log_warn "torch+xpu was overwritten by a dependency — reinstalling from XPU index..."
    pip install --extra-index-url https://download.pytorch.org/whl/xpu \
        torch==2.10.0+xpu torchvision==0.25.0 torchaudio==2.10.0
fi

# ── Phase 5: Configure triton-xpu + environment ──────────────────────────────
#
# vllm-xpu-kernels: Phase 4's `pip install -r requirements/xpu.txt` already
# installed the pre-built wheel `vllm_xpu_kernels==0.1.4` (pinned in vLLM
# 0.19.0's requirements/xpu.txt). The 1.5-2 hour source build below is now
# OPT-IN via VLLM_BUILD_XPU_KERNELS=1 — only needed if you want newer kernels
# (e.g. v0.1.5) or a development build.

log_info "Phase 5/5: Configuring triton-xpu and environment..."

if [ "${VLLM_BUILD_XPU_KERNELS:-0}" = "1" ]; then
    log_info "VLLM_BUILD_XPU_KERNELS=1 — building vllm-xpu-kernels from source..."
    if [ -d "$INSTALL_DIR/vllm-xpu-kernels" ] && ! pip show vllm-xpu-kernels &>/dev/null; then
        rm -rf "$INSTALL_DIR/vllm-xpu-kernels"
    fi
    if [ ! -d "$INSTALL_DIR/vllm-xpu-kernels" ]; then
        cd "$INSTALL_DIR"
        git clone https://github.com/vllm-project/vllm-xpu-kernels.git
        cd vllm-xpu-kernels
        git checkout v0.1.5 2>/dev/null || git checkout main
        # Remove conflicting version pins from requirements.txt
        sed -i 's|^--extra-index-url=https://download.pytorch.org/whl/xpu|# &|' requirements.txt 2>/dev/null || true
        sed -i 's|^torch==2.10.0+xpu|# &|' requirements.txt 2>/dev/null || true
        sed -i 's|^triton-xpu|# &|' requirements.txt 2>/dev/null || true
        sed -i 's|^transformers|# &|' requirements.txt 2>/dev/null || true
        pip install -r requirements.txt 2>&1 | tail -3
        rm -rf "$INSTALL_DIR/vllm-xpu-kernels/build"
        export MAX_JOBS=${MAX_JOBS:-6}
        log_info "Building XPU kernels with MAX_JOBS=$MAX_JOBS (expect 1.5-2 hours)..."
        pip install --no-build-isolation . 2>&1 | tail -5
    fi
else
    log_info "Using pre-built vllm_xpu_kernels from requirements/xpu.txt (v0.1.4)."
    log_info "Set VLLM_BUILD_XPU_KERNELS=1 to build from source instead."
fi

if pip show vllm-xpu-kernels &>/dev/null || pip show vllm_xpu_kernels &>/dev/null; then
    log_info "vllm-xpu-kernels is installed."
else
    log_warn "vllm-xpu-kernels not installed — re-run with VLLM_BUILD_XPU_KERNELS=1 if needed."
fi

# Install triton-xpu per vLLM 0.19.0 XPU docs:
# https://github.com/vllm-project/vllm/blob/v0.19.0/docs/getting_started/installation/gpu.xpu.inc.md
# "For torch 2.10 (the version in requirements/xpu.txt), the matching package
#  is triton-xpu==3.6.0" — from the stable wheel index (NOT test/).
pip install --force-reinstall triton-xpu==3.6.0 \
    --extra-index-url=https://download.pytorch.org/whl/xpu 2>&1 | tail -5 || {
    log_warn "triton-xpu==3.6.0 install failed — keeping existing triton."
    pip install triton 2>&1 | tail -3 || true
}

# Add activation helper to bashrc (v19 variant, doesn't conflict with v14 aliases)
if ! grep -q "llm-scaler-vllm-v19" ~/.bashrc; then
    cat << 'BASHRC' >> ~/.bashrc

# llm-scaler-vllm v0.19 (Lunar Lake)
alias vllm-v19-activate='source ~/llm-scaler-vllm-v19/venv/bin/activate && source /opt/intel/oneapi/setvars.sh --force 2>/dev/null && export VLLM_TARGET_DEVICE=xpu && if [ -n "${MKLROOT:-}" ] && [ -f "$MKLROOT/lib/libmkl_core.so.2" ]; then export LD_PRELOAD="${MKLROOT}/lib/libmkl_core.so.2:${MKLROOT}/lib/libmkl_intel_thread.so.2:${MKLROOT}/lib/libmkl_intel_lp64.so.2${LD_PRELOAD:+:$LD_PRELOAD}"; fi'
alias oneapi='source /opt/intel/oneapi/setvars.sh --force'
BASHRC
fi

# ── Done ──────────────────────────────────────────────────────────────────────

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_info "Installation complete!"
echo ""
echo "  vLLM version: $VLLM_VERSION (native XPU, no patches)"
echo "  Install dir:  $INSTALL_DIR"
echo "  Python venv:  $VENV_DIR"
echo ""
echo "  Quick start:"
echo "    vllm-v19-activate"
echo "    vllm serve /shared/models/gpt-oss-20b \\"
echo "        --enforce-eager \\"
echo "        --gpu-memory-utilization 0.6 \\"
echo "        --max-model-len 2048 \\"
echo "        --max-num-seqs 2 \\"
echo "        --kv-cache-memory-bytes 2147483648 \\"
echo "        --port 8080"
echo ""
echo "  Notes for Lunar Lake (28 GiB shared memory):"
echo "    - --kv-cache-memory-bytes N bypasses vLLM's memory profile (v0.19"
echo "      replacement for v0.14's VLLM_SKIP_PROFILE_RUN=1 + multiplier hack)."
echo "    - XPU mem_get_info excludes page cache. If preflight fails:"
echo "        sync && echo 3 | sudo tee /proc/sys/vm/drop_caches"
echo "    - XPU shared memory leaks across crashed launches — reboot if you"
echo "      hit 'Free memory (< 3 GiB)' on preflight."
echo ""
echo "  Gemma 4 models (native support in v0.19):"
echo "    google/gemma-4-12b-it                     (FP16, ~24 GB)"
echo "    intel/gemma-4-26b-a4b-it-int4-autoround   (INT4, ~15 GB)"
echo ""
echo "  Other models:"
echo "    Qwen/Qwen3-8B --quantization fp8          (best balance)"
echo "    Qwen/Qwen3-14B --quantization int4         (needs INT4)"
echo ""
echo "  Switching between versions:"
echo "    vllm-activate        # v0.14.0 (if installed)"
echo "    vllm-v19-activate    # v0.19.0 (this install)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
