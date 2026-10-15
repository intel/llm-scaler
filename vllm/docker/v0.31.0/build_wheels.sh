#!/usr/bin/env bash
set -euo pipefail
export VIRTUAL_ENV=/opt/venv PATH="/opt/venv/bin:$PATH"
export MAX_JOBS="${MAX_JOBS:-16}"
export VLLM_XPU_ENABLE_XE2=ON VLLM_XPU_ENABLE_XE3P=OFF
export VLLM_XPU_ENABLE_XE_DEFAULT=ON QWEN38_KERNELS_ENABLED=ON
export FETCHCONTENT_BASE_DIR=/build/deps
mkdir -p /build/kernels /build/vllm /opt/wheels
cd /src/vllm-xpu-kernels
ln -s /opt/venv .venv
ln -s /build/kernels build
git config --add safe.directory /src/vllm-xpu-kernels
VLLM_VERSION_OVERRIDE=0.1.15.4+llmscaler.1 \
  uv run --active --no-project python -m build --wheel --no-isolation --skip-dependency-check --outdir /opt/wheels
cd /src/vllm
ln -s /opt/venv .venv
ln -s /build/vllm build
git config --add safe.directory /src/vllm
VLLM_TARGET_DEVICE=xpu VLLM_VERSION_OVERRIDE=0.31.0+xpu.llmscaler \
  uv run --active --no-project python -m build --wheel --no-isolation --skip-dependency-check --outdir /opt/wheels
cd /opt/wheels
sha256sum *.whl > SHA256SUMS
