# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# ======== Base Stage ========
FROM amr-registry.caas.intel.com/intelanalytics/llm-scaler-platform:26.24.9.1 AS vllm-base

ARG https_proxy
ARG http_proxy
ARG PYTHON_VERSION=3.12

WORKDIR /llm/

# Install only packages not already in base image
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update -y && \
    apt-get install -y --no-install-recommends --fix-missing \
    curl \
    ffmpeg \
    git \
    libsndfile1 \
    libsm6 \
    libxext6 \
    libgl1 \
    libaio-dev \
    lsb-release \
    numactl \
    wget \
    linux-libc-dev

# Setup uv and Python virtual environment
ENV PATH="/root/.local/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    uv venv --python ${PYTHON_VERSION} --seed ${VIRTUAL_ENV}
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

SHELL ["bash", "-c"]

WORKDIR /llm/vllm

# Configure package index for XPU
ENV UV_HTTP_TIMEOUT=500
ENV PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/xpu"
ENV UV_EXTRA_INDEX_URL="https://download.pytorch.org/whl/xpu"
ENV UV_INDEX_STRATEGY="unsafe-best-match"
ENV UV_LINK_MODE="copy"
ENV PIP_NO_COMPILE=1

# Set vLLM build environment
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/lib/"
ENV VLLM_TARGET_DEVICE=xpu
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn

# Clone + patch + build vLLM v0.21.0 (shallow clone, strip .so, prune source after install)
COPY ./patches/vllm_for_multi_arc.patch /tmp/
RUN --mount=type=cache,target=/root/.cache/uv \
    touch /tmp/.ts && \
    git clone --depth 1 -b v0.21.0 https://github.com/vllm-project/vllm.git /llm/vllm && \
    cd /llm/vllm && \
    git apply /tmp/vllm_for_multi_arc.patch || true && \
    uv pip install --upgrade pip && \
    uv pip install -r requirements/xpu.txt && \
    uv pip install grpcio-tools protobuf nanobind && \
    source /opt/intel/oneapi/ccl/2021.15/env/vars.sh --force && \
    export CMAKE_PREFIX_PATH="$(python3 -c 'import site; print(site.getsitepackages()[0])'):${CMAKE_PREFIX_PATH}" && \
    uv pip install --no-build-isolation . && \
    rm -rf /llm/vllm/.git /tmp/vllm_for_multi_arc.patch && \
    rm -rf /llm/vllm/docs /llm/vllm/build /llm/vllm/csrc /llm/vllm/examples \
           /llm/vllm/benchmarks /llm/vllm/tools /llm/vllm/cmake /llm/vllm/.github /llm/vllm/.buildkite && \
    find /opt/venv/lib/python3.12/site-packages -newer /tmp/.ts -name '*.so' -exec strip --strip-unneeded {} + 2>/dev/null; \
    find /opt/venv/lib/python3.12/site-packages -newer /tmp/.ts -name '*.so.*' -exec strip --strip-unneeded {} + 2>/dev/null; \
    find /opt/venv/lib/python3.12/site-packages -newer /tmp/.ts -type f \( -name '*.a' -o -name '*.o' \) -delete; \
    find /opt/venv/lib/python3.12/site-packages -newer /tmp/.ts -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null; \
    rm -rf /opt/venv/lib/python3.12/site-packages/torch/include \
           /opt/venv/lib/python3.12/site-packages/torch/share \
           /opt/venv/lib/python3.12/site-packages/torch/test \
           /opt/venv/lib/python3.12/site-packages/torch/benchmarks \
           /opt/venv/lib/python3.12/site-packages/torch/_C/__init__.pyi \
           /opt/venv/lib/python3.12/site-packages/torch/lib/libtorch_cuda* \
           /opt/venv/lib/python3.12/site-packages/torch/lib/libc10_cuda* \
           /opt/venv/lib/python3.12/site-packages/torch/lib/libcudnn* \
           /opt/venv/lib/python3.12/site-packages/torch/lib/libnvfuser* \
           /opt/venv/lib/python3.12/site-packages/torch/lib/libcublas* \
           /opt/venv/lib/python3.12/site-packages/torch/lib/libcufft* \
           /opt/venv/lib/python3.12/site-packages/torch/lib/libcurand* \
           /opt/venv/lib/python3.12/site-packages/torch/lib/libcusparse* \
           /opt/venv/lib/python3.12/site-packages/torch/lib/libcusolver* \
           /opt/venv/lib/python3.12/site-packages/torch/lib/libnccl* \
           /opt/venv/lib/python3.12/site-packages/caffe2 \
           /opt/venv/lib/python3.12/site-packages/nvidia \
           /opt/venv/lib/python3.12/site-packages/torch/utils/hipify; \
    rm -f /tmp/.ts; true

# Install unified custom ESIMD kernels
COPY ./custom-esimd-kernels-vllm /tmp/custom-esimd-kernels-vllm
RUN --mount=type=cache,target=/root/.cache/uv \
    touch /tmp/.ts && \
    cd /tmp/custom-esimd-kernels-vllm && \
    TORCH_XPU_ARCH_LIST=bmg-g21 MAX_JOBS=1 python3 setup.py bdist_wheel && \
    uv pip install dist/*.whl --no-deps && \
    rm -rf /tmp/custom-esimd-kernels-vllm && \
    find /opt/venv/lib/python3.12/site-packages -newer /tmp/.ts -name '*.so' -exec strip --strip-unneeded {} + 2>/dev/null; \
    rm -f /tmp/.ts; true

# Clone + install MinerU (shallow, drop .git/docs/tests after install)
RUN --mount=type=cache,target=/root/.cache/uv \
    touch /tmp/.ts && \
    git clone --depth 1 -b release-2.6.2 https://github.com/opendatalab/MinerU.git /llm/MinerU && \
    cd /llm/MinerU && \
    uv pip install -e .[core] && \
    sed -i 's/kwargs.get("max_concurrency", 100)/kwargs.get("max_concurrency", 200)/' /llm/MinerU/mineru/backend/vlm/vlm_analyze.py && \
    sed -i 's/kwargs.get("http_timeout", 600)/kwargs.get("http_timeout", 1200)/' /llm/MinerU/mineru/backend/vlm/vlm_analyze.py && \
    rm -rf /llm/MinerU/.git /llm/MinerU/docs /llm/MinerU/tests && \
    find /opt/venv/lib/python3.12/site-packages -newer /tmp/.ts -name '*.so' -exec strip --strip-unneeded {} + 2>/dev/null; \
    find /opt/venv/lib/python3.12/site-packages -newer /tmp/.ts -name '*.so.*' -exec strip --strip-unneeded {} + 2>/dev/null; \
    rm -f /tmp/.ts; true

# Install bigdl-core
RUN --mount=type=cache,target=/root/.cache/uv \
    touch /tmp/.ts && \
    uv pip install bigdl-core==2.4.0b2 && \
    find /opt/venv/lib/python3.12/site-packages -newer /tmp/.ts -name '*.so' -exec strip --strip-unneeded {} + 2>/dev/null; \
    find /opt/venv/lib/python3.12/site-packages -newer /tmp/.ts -name '*.so.*' -exec strip --strip-unneeded {} + 2>/dev/null; \
    rm -f /tmp/.ts; true

# Cleanup
RUN rm -rf /tmp/*

CMD ["bash", "-c", "source /opt/intel/oneapi/ccl/2021.15/env/vars.sh --force && exec bash"]

# ======== OpenAI Serving Stage ========
FROM vllm-base AS vllm-openai

ARG http_proxy
ARG https_proxy

# Install additional dependencies for serving
RUN --mount=type=cache,target=/root/.cache/uv \
    touch /tmp/.ts && \
    uv pip install accelerate hf_transfer 'modelscope!=1.15.0' && \
    uv pip install librosa soundfile decord ijson && \
    uv pip install transformers==5.8.0 && \
    find /opt/venv/lib/python3.12/site-packages -newer /tmp/.ts -name '*.so' -exec strip --strip-unneeded {} + 2>/dev/null; \
    rm -f /tmp/.ts; true

# Build vllm-xpu-kernels: shallow fetch pinned commit, install, drop source
COPY ./patches/vllm_xpu_kernels.patch /tmp/
RUN --mount=type=cache,target=/root/.cache/uv \
    touch /tmp/.ts && \
    cd /llm && \
    mkdir vllm-xpu-kernels && cd vllm-xpu-kernels && \
    git init -q && \
    git remote add origin https://github.com/vllm-project/vllm-xpu-kernels.git && \
    git fetch --depth 1 origin 3cab97a && \
    git checkout FETCH_HEAD && \
    git apply /tmp/vllm_xpu_kernels.patch && \
    sed -i 's|^--extra-index-url=https://download.pytorch.org/whl/xpu|# --extra-index-url=https://download.pytorch.org/whl/xpu|' requirements.txt && \
    sed -i '/^torch==/s/^/# /' requirements.txt && \
    sed -i 's|^triton-xpu|# triton-xpu|' requirements.txt && \
    sed -i 's|^transformers|# transformers|' requirements.txt && \
    uv pip install -r requirements.txt && \
    uv pip install --no-build-isolation . && \
    cd /llm && \
    rm -rf /llm/vllm-xpu-kernels /tmp/vllm_xpu_kernels.patch && \
    find /opt/venv/lib/python3.12/site-packages -newer /tmp/.ts -name '*.so' -exec strip --strip-unneeded {} + 2>/dev/null; \
    find /opt/venv/lib/python3.12/site-packages -newer /tmp/.ts -name '*.so.*' -exec strip --strip-unneeded {} + 2>/dev/null; \
    rm -f /tmp/.ts; true

# Install UCX and NIXL for distributed communication (new in v0.21.0)
ARG UCX_VERSION=e5d98879705239d254ede40b4a52891850cb5349
ARG NIXL_VERSION=0.7.0

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    autoconf \
    automake \
    libtool \
    pkg-config \
    rdma-core \
    libibverbs-dev \
    ibverbs-utils \
    librdmacm-dev \
    libibumad-dev \
    infiniband-diags \
    perftest

ENV PKG_CONFIG_PATH=/tmp/ucx_install/lib/pkgconfig:${PKG_CONFIG_PATH}
ENV LD_LIBRARY_PATH=/tmp/ucx_install/lib:${LD_LIBRARY_PATH}
RUN --mount=type=cache,target=/root/.cache/uv \
    git clone --depth 1 https://github.com/openucx/ucx /tmp/ucx_source && \
    cd /tmp/ucx_source && git fetch --depth 1 origin "${UCX_VERSION}" && git checkout FETCH_HEAD && \
    bash autogen.sh && \
    ./configure --prefix=/tmp/ucx_install --with-ze=yes --enable-mt && \
    make CFLAGS="-Wno-error=incompatible-pointer-types" -j$(nproc) && make install && \
    git clone --depth 1 -b "${NIXL_VERSION}" https://github.com/ai-dynamo/nixl /tmp/nixl_source && \
    cd /tmp/nixl_source && \
    uv pip install --upgrade meson pybind11 patchelf && \
    uv pip install -r requirements.txt && \
    uv pip install . && \
    rm -rf /tmp/ucx_source /tmp/nixl_source && \
    find /tmp/ucx_install/lib -name '*.a' -delete && \
    find /tmp/ucx_install/lib -name '*.la' -delete && \
    strip --strip-unneeded /tmp/ucx_install/lib/*.so* 2>/dev/null; true

# Fix triton — use 3.7.0 for vLLM 0.21.0, then strip and remove unnecessary files
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip uninstall triton triton-xpu && \
    uv pip install triton-xpu==3.7.0 && \
    strip --strip-debug /opt/venv/lib/python3.12/site-packages/triton/_C/libtriton.so 2>/dev/null; \
    strip --strip-debug /opt/venv/lib/python3.12/site-packages/triton/_C/libproton.so 2>/dev/null; \
    strip --strip-debug /opt/venv/lib/python3.12/site-packages/triton/plugins/libMLIRDialectPlugin.so 2>/dev/null; \
    rm -f /opt/venv/lib/python3.12/site-packages/triton/plugins/libMLIRDialectPlugin.so.22.0git \
          /opt/venv/lib/python3.12/site-packages/triton/plugins/libTritonPluginsTestLib.so; \
    rm -rf /opt/venv/lib/python3.12/site-packages/triton/backends/amd; \
    true

# Remove torch-bundled oneccl to avoid conflicts with system oneCCL
RUN uv pip uninstall oneccl oneccl-devel || true

# Install test utilities
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -e /llm/vllm/tests/vllm_test_utils

COPY ./examples/offline_inference.py /llm/

# Production environment
ENV VLLM_QUANTIZE_Q40_LIB="/opt/venv/lib/python3.12/site-packages/vllm_int4_for_multi_arc.so"
ENV VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=1
ENV VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# Final cleanup: purge build-only packages, stale files, caches
RUN apt-get purge -y --auto-remove \
        build-essential \
        autoconf \
        automake \
        libtool \
        pkg-config \
        libaio-dev \
        libibverbs-dev \
        librdmacm-dev \
        libibumad-dev \
        linux-libc-dev && \
    rm -rf /usr/lib/python3/dist-packages/PyJWT-*.dist-info \
           /root/.cache /var/tmp/* /tmp/* /root/.local \
           /opt/venv/lib/python3.12/site-packages/torch/bin \
           /opt/venv/lib/python3.12/site-packages/torch/lib/libtorchbind_test.so \
           /opt/venv/lib/python3.12/site-packages/torch/lib/libjitbackend_test.so \
           /opt/venv/lib/python3.12/site-packages/cmake \
           /opt/venv/lib/python3.12/site-packages/opencv_python.libs && \
    find /opt/venv/lib/python3.12/site-packages -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null; \
    echo "source /opt/intel/oneapi/setvars.sh --force" >> /root/.bashrc; true

ENTRYPOINT ["bash", "-c", "source /opt/intel/oneapi/ccl/2021.15/env/vars.sh --force && exec vllm serve \"$@\"", "--"]
