#!/bin/bash
# ==============================================================================
# Lunar Lake Xe2 iGPU Platform Evaluation Script
# ------------------------------------------------------------------------------
# Evaluates Intel Arc 140V (Xe2) integrated GPU on Lunar Lake platforms.
# Adapted from the B60 discrete GPU evaluation for single iGPU use.
# ==============================================================================

set -eo pipefail

# === Error Handling ===
CURRENT_STEP=""
function print_info()    { echo -e "\033[1;34m[INFO]\033[0m $1"; }
function print_success() { echo -e "\033[1;32m[SUCCESS]\033[0m $1"; }
function print_warn()    { echo -e "\033[1;33m[WARN]\033[0m $1"; }
function print_error()   { echo -e "\033[1;31m[ERROR]\033[0m $1"; }

function error_handler() {
    local exit_code=$?
    local line_no=$1
    print_error "Script failed during: '$CURRENT_STEP' (line $line_no, exit code $exit_code)"
    echo "[FAILED COMMAND] $BASH_COMMAND"
    echo "[DEBUG] Check log file: $LOG"
    exit $exit_code
}
trap 'error_handler $LINENO' ERR

function step() {
    CURRENT_STEP="$1"
    print_info "$CURRENT_STEP"
}

# === Detect GPU type ===
detect_gpu() {
    # Lunar Lake Arc 140V: 8086:64a0
    # Arc Pro B60: 8086:e211 or 8086:e210
    # Arc A770: 8086:56a0
    local gpu_id
    gpu_id=$(lspci -nn | grep -i 'vga\|3d\|display' | grep -oP '8086:\K[0-9a-fA-F]+' | head -1)

    case "$gpu_id" in
        64a0) echo "lunar_lake" ;;
        e211|e210) echo "arc_pro_b60" ;;
        56a0) echo "arc_a770" ;;
        *) echo "unknown:$gpu_id" ;;
    esac
}

GPU_TYPE=$(detect_gpu)
step "Detected GPU type: $GPU_TYPE"

if [[ "$GPU_TYPE" != "lunar_lake" ]]; then
    print_warn "This script is optimized for Lunar Lake Xe2 (Arc 140V)."
    print_warn "Detected: $GPU_TYPE. Proceeding anyway..."
fi

# === Timestamped result directory ===
step "Creating result directory"
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
RESULT_DIR="results/lunar_lake_$TIMESTAMP"
mkdir -p "$RESULT_DIR"
LOG="$RESULT_DIR/benchmark_detail_log.txt"

# === Load Intel oneAPI environment ===
step "Sourcing Intel oneAPI environment"
SETVARS="/opt/intel/oneapi/setvars.sh"
if [ ! -f "$SETVARS" ]; then
    print_error "$SETVARS not found. Install Intel oneAPI first."
    print_info "On Fedora/Nobara: sudo dnf install intel-oneapi-basekit"
    print_info "On Ubuntu: follow https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html"
    exit 1
fi
source "$SETVARS" --force >> "$LOG" 2>&1

# === Setup Environment Variables ===
export NEOReadDebugKeys=1
export RenderCompressedBuffersEnabled=0

# === List SYCL Devices ===
step "Listing SYCL devices"
sycl-ls 2>&1 | tee -a "$LOG"

# Verify Xe2 iGPU is visible
if ! sycl-ls 2>&1 | grep -qi "intel.*gpu"; then
    print_error "No Intel GPU detected by SYCL runtime."
    print_info "Ensure xe driver is loaded: lsmod | grep xe"
    print_info "Check Level-Zero: apt install level-zero level-zero-dev"
    exit 1
fi
print_success "Intel GPU detected by SYCL runtime"

# === xpu-smi (if available) ===
if command -v xpu-smi &> /dev/null; then
    step "xpu-smi discovery"
    xpu-smi discovery 2>&1 | tee -a "$LOG"
    xpu-smi dump -m 0,1,2,3,4,5,18,19,20 -n 1 2>&1 | tee -a "$LOG" || true
else
    print_warn "xpu-smi not found. Skipping GPU management info."
    print_info "Install with: apt install xpu-smi (or use nvtop for monitoring)"
fi

# === Memory Info ===
step "Checking GPU memory (shared iGPU)"
echo "--- System Memory (shared with iGPU) ---" | tee -a "$LOG"
free -h 2>&1 | tee -a "$LOG"
echo "" | tee -a "$LOG"

# Check total available for GPU via sysfs
if [ -f /sys/class/drm/card0/lmem_total_bytes ]; then
    LMEM=$(cat /sys/class/drm/card0/lmem_total_bytes 2>/dev/null || echo "N/A")
    echo "GPU local memory (lmem): $LMEM bytes" | tee -a "$LOG"
fi

# === Skip P2P tests (single iGPU) ===
print_info "Skipping P2P bandwidth tests (single integrated GPU)"

# === Host <-> Device Bandwidth Test ===
if command -v ze_peak &> /dev/null; then
    step "Running H2D/D2H transfer_bw test"
    ze_peak -t transfer_bw 2>&1 | tee -a "$LOG" || print_warn "ze_peak transfer_bw failed"

    step "Running device global_bw test"
    ze_peak -t global_bw 2>&1 | tee -a "$LOG" || print_warn "ze_peak global_bw failed"
else
    print_warn "ze_peak not found. Skipping bandwidth tests."
fi

# === GEMM Test (if available) ===
if command -v matrix_mul_mkl &> /dev/null; then
    step "Running GEMM MKL test (int8) — smaller size for iGPU"
    # Use smaller matrix for iGPU (shared memory constraints)
    matrix_mul_mkl int8 -m 8192 -n 8192 -k 8192 -c 0 2>&1 | tee -a "$LOG" || print_warn "GEMM test failed"
else
    print_warn "matrix_mul_mkl not found. Skipping GEMM test."
fi

# === Skip CCL tests (single GPU, no multi-rank) ===
print_info "Skipping 1CCL multi-GPU collective tests (single iGPU)"

# === PyTorch XPU Verification ===
step "Verifying PyTorch XPU support"
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'XPU available: {torch.xpu.is_available()}')
if torch.xpu.is_available():
    print(f'XPU device count: {torch.xpu.device_count()}')
    props = torch.xpu.get_device_properties(0)
    print(f'Device name: {props.name}')
    print(f'Total memory: {props.total_memory / 1024**3:.1f} GB')
    # Quick compute test
    x = torch.randn(1024, 1024, device='xpu')
    y = torch.mm(x, x)
    print(f'XPU compute test: PASSED ({y.shape})')
else:
    print('WARNING: XPU not available. Check Level-Zero and xe driver.')
" 2>&1 | tee -a "$LOG" || print_warn "PyTorch XPU check failed"

# === Final Message ===
print_success "Lunar Lake evaluation completed."
print_info "Logs saved to: $LOG"
print_info ""
print_info "Next steps:"
print_info "  1. If SYCL and PyTorch XPU are working, you can run vLLM with:"
print_info "     vllm serve <model> --device xpu --tensor-parallel-size 1"
print_info "  2. For small models (7B-14B), use FP8 or INT4 quantization"
print_info "  3. Monitor memory with: nvtop or watch -n1 free -h"
