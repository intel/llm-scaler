# ======== Build Stage ========
FROM intel/omix:latest-devel-ubuntu24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ARG http_proxy
ARG https_proxy
ARG no_proxy=localhost,127.0.0.1,::1
ARG DPCPP_SYCL_TARGET=bmg
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
ENV http_proxy=${http_proxy} https_proxy=${https_proxy} no_proxy=${no_proxy}
ENV TORCH_XPU_ARCH_LIST=${DPCPP_SYCL_TARGET}
ENV OMNI_XPU_DEVICE=${DPCPP_SYCL_TARGET}
ENV MAX_JOBS=1

WORKDIR /llm-scaler/sgl
COPY . .

# --- System packages + pip base setup ---
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3-pip python3-dev git \
 && python3 -m pip config set global.break-system-packages true \
 && pip3 install ninja \
 && pip3 install torch==2.11.0+xpu --index-url https://download.pytorch.org/whl/xpu \
 && rm -rf /root/.cache /root/.cmake /tmp/* /var/tmp/* /var/lib/apt/lists/*

# --- Build sgl-kernel-xpu wheel ---
RUN git clone --depth 1 -b dev-bmg https://github.com/analytics-zoo/sgl-kernel-xpu.git /root/sk \
 && pip3 wheel -v /root/sk --config-settings=cmake.define.DPCPP_SYCL_TARGET=${DPCPP_SYCL_TARGET} --wheel-dir /root/wheels --no-deps \
 && rm -rf /root/sk \
 && rm -rf /root/.cache /root/.cmake /tmp/* /var/tmp/* /var/lib/apt/lists/*

# --- Build sglang wheel ---
RUN git clone --depth 1 -b dev-bmg https://github.com/analytics-zoo/sglang.git /root/sglang \
 && cd /root/sglang/python \
 && cp pyproject_xpu.toml pyproject.toml \
 && sed -i '/sgl-kernel/d;/torch==/d;/torchaudio==/d;/torchvision==/d;/torchao==/d' pyproject.toml \
 && pip3 wheel -v . --wheel-dir /root/wheels --no-deps \
 && rm -rf /root/sglang \
 && rm -rf /root/.cache /root/.cmake /tmp/* /var/tmp/* /var/lib/apt/lists/*

# --- Build custom ESIMD kernels ---
RUN . /opt/intel/oneapi/setvars.sh --force \
 && cd /llm-scaler/sgl/custom-esimd-kernels \
 && CXX=icpx pip3 wheel -v . --wheel-dir /root/wheels --no-deps --no-build-isolation \
 && cd /llm-scaler/sgl/custom-esimd-kernels-sglang \
 && CXX=icpx pip3 wheel -v . --wheel-dir /root/wheels --no-deps --no-build-isolation \
 && rm -rf /root/.cache /root/.cmake /tmp/* /var/tmp/* /var/lib/apt/lists/*

# ======== Runtime Stage ========
FROM intel/pytorch:xpu-2.11.0-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ARG http_proxy
ARG https_proxy
ARG no_proxy=localhost,127.0.0.1,::1,intel.com,.intel.com
ARG DPCPP_SYCL_TARGET=bmg
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
ENV http_proxy=${http_proxy} https_proxy=${https_proxy} no_proxy=${no_proxy}
ENV TORCH_XPU_ARCH_LIST=${DPCPP_SYCL_TARGET}
ENV OMNI_XPU_DEVICE=${DPCPP_SYCL_TARGET}

WORKDIR /llm-scaler/sgl
COPY . .

# Single RUN: install UMD driver (BMG) + all pre-built wheels via bind-mounted /root/wheels
# (no COPY layer for wheels -> ~400 MB saved), then strip + prune caches/bytecode.
RUN --mount=type=bind,from=builder,source=/root/wheels,target=/tmp/wheels \
    export no_proxy=localhost,127.0.0.1,::1 \
 # --- Intel BMG user-mode driver (pytorch base does not bundle it) ---
 && apt-get update \
 && apt-get install -y --no-install-recommends software-properties-common python3-pip \
 && add-apt-repository -y ppa:kobuk-team/intel-graphics \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
        libze-intel-gpu1 libze1 intel-metrics-discovery intel-opencl-icd clinfo intel-gsc \
        libze-dev \
 # --- Python deps: pre-built wheels + explicit runtime deps sglang needs ---
 && python3 -m pip config set global.break-system-packages true \
 && pip3 install /tmp/wheels/*.whl --extra-index-url https://download.pytorch.org/whl/xpu \
 && pip3 install --no-deps xgrammar==0.1.33 || true \
 && chmod +x /llm-scaler/sgl/scripts/*.sh \
 # --- Size cleanup: drop caches + bytecode ---
 && find /opt/venv /usr/local/lib/python3* -name "*.pyc" -delete 2>/dev/null || true \
 && find /opt/venv /usr/local/lib/python3* -name "__pycache__" -type d \
        -exec rm -rf {} + 2>/dev/null || true \
 && find /tmp -mindepth 1 -maxdepth 1 ! -name wheels -exec rm -rf {} + \
 && rm -rf /root/.cache /root/.cmake /var/tmp/* /var/lib/apt/lists/*

CMD ["bash"]

