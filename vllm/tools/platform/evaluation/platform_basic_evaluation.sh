#!/bin/bash
set -eo pipefail

# === Error Handling ===
CURRENT_STEP=""
function print_info()    { echo -e "\033[1;34m[INFO]\033[0m $1"; }
function print_success() { echo -e "\033[1;32m[SUCCESS]\033[0m $1"; }
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

# === Validate current directory ===
step "Validating directory structure"
if [ ! -d "./tools" ] || [ ! -d "./scripts" ]; then
    print_error "This script must be run from the root directory of the installation package."
    echo "Expected directories: ./tools and ./scripts"
    exit 1
fi

# === Timestamped result directory ===
step "Creating result directory"
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
RESULT_DIR="results/$TIMESTAMP"
mkdir -p "$RESULT_DIR"
LOG="$RESULT_DIR/benchmark_detail_log.txt"

# === Load Intel oneAPI environment ===
step "Sourcing Intel oneAPI environment"
SETVARS="/opt/intel/oneapi/setvars.sh"
if [ ! -f "$SETVARS" ]; then
    print_error "$SETVARS not found. Please check the path."
    exit 1
fi
source "$SETVARS" --force>> "$LOG" 2>&1

# === Setup Necessary Environment Variable ===
# For accurate memory bandwidth benchmark
export NEOReadDebugKeys=1
export RenderCompressedBuffersEnabled=0

# === List SYCL Devices ===
step "Listing SYCL devices"
sycl-ls 2>&1 | tee -a "$LOG"

# === Run xpu-smi ===
step "xpu-smi test"
xpu-smi discovery 2>&1 | tee -a "$LOG"
xpu-smi dump -m 0,1,2,3,4,5,18,19,20 -n 1 2>&1 | tee -a "$LOG"

# === P2P Bandwidth Test ===
count=$(lspci | grep -E "e211|e210" | wc -l)
echo "Detected GPU count: $count"
if [ "$count" -ge 2 ]; then
    step "Running ze_peer default test"
    ./tools/level-zero-tests/ze_peer -s 0 -d 1 2>&1 | tee -a "$LOG"

    step "Running ze_peer bi-directional write test"
    ./tools/level-zero-tests/ze_peer -o write -t transfer_bw -s 0 -d 1 -b 2>&1 | tee -a "$LOG"

    step "Running ze_peer bi-directional read test"
    ./tools/level-zero-tests/ze_peer -o read -t transfer_bw -s 0 -d 1 -b 2>&1 | tee -a "$LOG"
    
#    if [ "$count" -ge 4 ]; then
#        step "Running ze_peer 2 pair uni-directional write test"
#        ./tools/level-zero-tests/ze_peer -o write -t transfer_bw --parallel_pair_targets 0:1,2:3 2>&1 | tee -a "$LOG"
#
#       step "Running ze_peer 2 pair bi-directional write test"
#        ./tools/level-zero-tests/ze_peer -o write -t transfer_bw --parallel_pair_targets 0:1,2:3 -b 2>&1 | tee -a "$LOG"
#    fi
else
    echo "GPU count < 2, no need to do P2P benchmark"
fi

# === Host ↔ Device Bandwidth Test ===
step "Running H2D/D2H transfer_bw test"
./tools/level-zero-tests/ze_peak -t transfer_bw 2>&1 | tee -a "$LOG"

# === Device ↔ Device Bandwidth Test ===
step "Copying .spv files"
cp ./tools/level-zero-tests/*.spv ./ 2>/dev/null || print_info "No .spv files to copy."

step "Running D2D global_bw test"
./tools/level-zero-tests/ze_peak -t global_bw 2>&1 | tee -a "$LOG"

step "Cleaning up SPIR-V files"
rm -f *.spv

# === GEMM Test using MKL ===
step "Running GEMM MKL test (int8)"
matrix_mul_mkl int8 -m 40960 -n 40960 -k 40960 -c 0 2>&1 | tee -a "$LOG"

# === 1CCL Benchmarking ===
function run_ccl_test() {
    local op=$1
    local outfile="$RESULT_DIR/${op}_outplace_128M.csv"
    step "Running 1CCL ${op^^} test"
    mpirun -np 2 /usr/bin/1ccl_benchmark \
        -a gpu -m usm -u device -e in_order \
        -l "$op" -i 50 -w 20 -f 512 -t 67108864 \
        -j off -p 0 -d float16 -q 0 -o "$outfile" 2>&1 | tee -a "$LOG"

    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        print_success "1CCL $op test completed. Output: $outfile"
    else
        print_error "1CCL $op test failed."
        exit 1
    fi
}

run_ccl_test allreduce
run_ccl_test allgather
run_ccl_test alltoall


# === Generate Report ====
python3 ./scripts/evaluation/gen_evaluation_report.py $RESULT_DIR/benchmark_detail_log.txt results/reference_perf.csv $RESULT_DIR/benchmark_report.csv

# === Final Message ===
print_success "All tests completed."
print_info "Logs saved to: $LOG"
print_info "Result report saved in: $RESULT_DIR/benchmark_report.csv"
print_info "Result detailed log saved in: $RESULT_DIR/benchmark_detailed_log.txt"
