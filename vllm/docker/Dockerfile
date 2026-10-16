# syntax=docker/dockerfile:1.7
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Context: vllm/, plus prepared vllm-source and kernels-source contexts.
ARG OMIX_DEVEL=intel/omix:0.4.0-devel-ubuntu26.04@sha256:59cef13f8e87275846cc08e01d32b49082694a5c01d842d2e845552cbc8c199f
ARG OMIX_RUNTIME=intel/omix:0.4.0-runtime-ubuntu26.04@sha256:a69b77be1ecc5784edc61023ef8d91c88899939dfa9f422af5a4474151dc071c
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.13.0@sha256:cdc6093146eb3ff6a40107b38f008b789e050e77ad87865e381d9917da55a168
ARG RUST_BASE=ubuntu:22.04@sha256:5ec03bb3441e8b0bf3b4f9cd4629a1ae763010dc3035bb8da3ae6cf026486401
ARG PYTHON_VERSION=3.12.15

FROM ${UV_IMAGE} AS uv-tool

FROM ${RUST_BASE} AS rust-build
ENV DEBIAN_FRONTEND=noninteractive \
    CARGO_BUILD_JOBS=4 \
    VLLM_RS_BUILD_VERSION=0.31.0+xpu.llmscaler
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
      build-essential ca-certificates curl git python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace/vllm
COPY --from=vllm-source requirements/build/rust.txt requirements/build/rust.txt
RUN python3 -m pip install --no-cache-dir -r requirements/build/rust.txt
COPY --from=vllm-source rust rust
COPY --from=vllm-source rust-toolchain.toml rust-toolchain.toml
COPY --from=vllm-source tools/build_rust.py tools/build_rust.py
COPY --from=vllm-source tools/build_rust.sh tools/build_rust.sh
RUN --mount=type=cache,target=/root/.cargo/registry,sharing=locked \
    --mount=type=cache,target=/root/.cargo/git,sharing=locked \
    bash tools/build_rust.sh

FROM ${OMIX_DEVEL} AS native-builder
ARG http_proxy
ARG https_proxy
ARG no_proxy
ARG PYTHON_VERSION
ENV VIRTUAL_ENV=/opt/venv \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_SYSTEM_CERTS=true \
    UV_HTTP_TIMEOUT=500 \
    UV_EXTRA_INDEX_URL="https://download.pytorch.org/whl/xpu https://wheels.vllm.ai/xpu" \
    UV_INDEX_STRATEGY=unsafe-best-match \
    UV_LINK_MODE=copy \
    SYCL_HOME=/opt/intel/oneapi/compiler/2026.1 \
    CMPLR_ROOT=/opt/intel/oneapi/compiler/2026.1 \
    CC=/opt/intel/oneapi/compiler/2026.1/bin/icx \
    CXX=/opt/intel/oneapi/compiler/2026.1/bin/icpx \
    PATH=/root/.local/bin:/opt/venv/bin:/opt/intel/oneapi/compiler/2026.1/bin:${PATH}
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
WORKDIR /llm/
COPY --from=uv-tool /uv /uvx /usr/local/bin/
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update -y \
    && apt-get install -y --no-install-recommends \
      build-essential ca-certificates curl git libaio-dev ninja-build cmake pkg-config numactl \
    && uv python install ${PYTHON_VERSION} \
    && uv venv --python ${PYTHON_VERSION} --seed ${VIRTUAL_ENV}
COPY --from=kernels-source . /src/vllm-xpu-kernels/
COPY --from=vllm-source . /src/vllm/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python ${VIRTUAL_ENV}/bin/python \
      -r /src/vllm/requirements/xpu.txt \
      build pyproject-hooks psutil grpcio-tools protobuf nanobind

FROM native-builder AS wheels
ARG MAX_JOBS=16
COPY --from=rust-build /workspace/vllm/vllm/vllm-rs /src/vllm/vllm/vllm-rs
COPY --from=rust-build /workspace/vllm/vllm/_rust_*.so /src/vllm/vllm/
COPY docker/v0.31.0/build_wheels.sh /opt/build_wheels.sh
RUN --mount=type=tmpfs,target=/build \
    MAX_JOBS=${MAX_JOBS} bash /opt/build_wheels.sh

FROM scratch AS wheel-export
COPY --from=wheels /opt/wheels/ /

FROM ${OMIX_RUNTIME} AS runtime
ARG http_proxy
ARG https_proxy
ARG no_proxy
ARG PYTHON_VERSION
ENV VIRTUAL_ENV=/opt/venv \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_SYSTEM_CERTS=true \
    UV_HTTP_TIMEOUT=500 \
    UV_EXTRA_INDEX_URL="https://download.pytorch.org/whl/xpu https://wheels.vllm.ai/xpu" \
    UV_INDEX_STRATEGY=unsafe-best-match \
    UV_LINK_MODE=copy \
    PATH=/root/.local/bin:/opt/venv/bin:${PATH} \
    VLLM_WORKER_MULTIPROC_METHOD=spawn
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
WORKDIR /llm/
COPY --from=uv-tool /uv /uvx /usr/local/bin/
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update -y \
    && apt-get install -y --no-install-recommends \
      ca-certificates curl ffmpeg libaio-dev libsndfile1 libsm6 libxext6 libgl1 numactl \
      ocl-icd-opencl-dev \
    && uv python install ${PYTHON_VERSION} \
    && uv venv --python ${PYTHON_VERSION} --seed ${VIRTUAL_ENV}
COPY --from=wheels /opt/wheels/ /opt/wheels/
RUN --mount=type=cache,target=/root/.cache/uv \
    (cd /opt/wheels && sha256sum -c SHA256SUMS) \
    && uv pip install --python ${VIRTUAL_ENV}/bin/python \
      /opt/wheels/vllm-*.whl /opt/wheels/vllm_xpu_kernels-*.whl \
    && uv pip install --python ${VIRTUAL_ENV}/bin/python --no-deps \
      'transformers==5.18.0' 'tokenizers==0.23.2' \
      'huggingface-hub==1.31.0' 'safetensors==0.8.0' \
      'intel-cmplr-lib-rt==2026.1.1' 'intel-cmplr-lib-ur==2026.1.1' \
      'intel-cmplr-lic-rt==2026.1.1' 'intel-sycl-rt==2026.1.1' \
      'oneccl-devel==2022.1.2' 'oneccl==2022.1.2' \
      'dpcpp-cpp-rt==2026.1.1' 'intel-opencl-rt==2026.1.1' \
      'intel-openmp==2026.1.1' 'intel-pti==1.1.0'
COPY docker/v0.31.0/llmscaler_transformers_compat.py docker/v0.31.0/llmscaler_transformers_compat.pth /opt/venv/lib/python3.12/site-packages/
ENV LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib
RUN python -c 'import torch; assert torch.__version__ == "2.14.0+xpu"; import transformers; assert transformers.__version__ == "5.18.0"'
ENTRYPOINT ["vllm", "serve"]
