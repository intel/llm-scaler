#!/bin/bash
# Build script for omni_xpu_kernel
# 
# Usage:
#   ./scripts/build.sh          # Build and install
#   ./scripts/build.sh --dev    # Install in development mode
#   ./scripts/build.sh --clean  # Clean and rebuild

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Parse arguments
DEV_MODE=false
CLEAN=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --dev|-e)
            DEV_MODE=true
            shift
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dev|-e] [--clean]"
            exit 1
            ;;
    esac
done

# Source Intel oneAPI if available and not already sourced
if [ -z "$ONEAPI_ROOT" ]; then
    SETVARS_PATHS=(
        "/opt/intel/oneapi/setvars.sh"
        "$HOME/intel/oneapi/setvars.sh"
    )
    for setvars in "${SETVARS_PATHS[@]}"; do
        if [ -f "$setvars" ]; then
            echo "Sourcing Intel oneAPI from $setvars"
            source "$setvars" 2>/dev/null || true
            break
        fi
    done
fi

# Check for icpx
if ! command -v icpx &> /dev/null; then
    echo "Error: Intel icpx compiler not found."
    echo "Please install Intel oneAPI and source setvars.sh"
    exit 1
fi

echo "Using compiler: $(which icpx)"
echo "Project directory: $PROJECT_DIR"

# Clean if requested
if [ "$CLEAN" = true ]; then
    echo "Cleaning build artifacts..."
    rm -rf build/ dist/ *.egg-info
    rm -f omni_xpu_kernel/*.so omni_xpu_kernel/_C*.so
fi

# Build and install
if [ "$DEV_MODE" = true ]; then
    echo "Installing in development mode..."
    pip install -e . -v
else
    echo "Building and installing..."
    pip install . -v
fi

echo ""
echo "Build complete!"
echo ""
echo "Test with:"
echo "  python -c \"import omni_xpu_kernel; print(omni_xpu_kernel.is_available())\""
