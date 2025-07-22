# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# ======== Base Stage ========
FROM intel/deep-learning-essentials:2025.0.2-0-devel-ubuntu24.04 AS vllm-base

ARG https_proxy
ARG http_proxy

# Add Intel oneAPI repo and PPA for GPU support
RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null && \
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list && \
    add-apt-repository -y ppa:kobuk-team/intel-graphics-testing

# Install dependencies and Python 3.10
RUN apt-get update -y && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update -y && \
    apt-get install -y python3.10 python3.10-distutils python3.10-dev && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 && \
    apt-get install -y --no-install-recommends --fix-missing \
        curl \
        ffmpeg \
        git \
        libsndfile1 \
        libsm6 \
        libxext6 \
        libgl1 \
        lsb-release \
        numactl \
        wget \
        vim \
        linux-libc-dev && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    # Install Intel GPU runtime packages
    apt-get update -y && \
    apt-get install -y libze1 libze-dev libze-intel-gpu1 intel-opencl-icd libze-intel-gpu-raytracing && \
    apt-get install -y intel-oneapi-dpcpp-ct=2025.0.1-17 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /llm
COPY ./patches/vllm_for_multi_arc.patch /tmp/
COPY ./patches/0001-oneccl-align-global-V0.1.1.patch /tmp/
COPY ./patches/vllm_int4_for_multi_arc.so /opt/lib/

# Set environment variables early
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/lib/"
ENV VLLM_TARGET_DEVICE=xpu
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn

# Clone + patch vllm
RUN git clone -b v0.9.0 https://github.com/vllm-project/vllm.git && \
    cd vllm && \
    git apply /tmp/vllm_for_multi_arc.patch && \
    pip install --no-cache-dir -r requirements/xpu.txt && \
    export CPATH=/opt/intel/oneapi/dpcpp-ct/2025.0/include/:${CPATH} && \
    python3 setup.py install

# ======= Add oneCCL build =======
RUN apt-get update && apt-get install -y \
    cmake \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Build 1ccl
RUN git clone https://github.com/oneapi-src/oneCCL.git && \
    cd oneCCL && \
    git checkout def870543749186b6f38cdc865b44d52174c7492 && \
    git apply /tmp/0001-oneccl-align-global-V0.1.1.patch && \
    mkdir build && cd build && \
    export IGC_VISAOptions=-activeThreadsOnlyBarrier && \
    /usr/bin/cmake .. \
        -DCMAKE_INSTALL_PREFIX=_install \
        -DCMAKE_C_COMPILER=icx \
        -DCMAKE_CXX_COMPILER=icpx \
        -DCOMPUTE_BACKEND=dpcpp \
        -DCCL_ENABLE_ARCB=1 && \
    make -j && make install && \
    mv _install /opt/intel/oneapi/ccl/2021.15.3 && \
    cd /opt/intel/oneapi/ccl/ && \
    ln -snf 2021.15.3 latest

# Install pypi dependencies
RUN pip install --pre --upgrade bigdl-core

# Install required triton version
RUN pip uninstall pytorch-triton-xpu triton -y && \
    pip install https://ubit-artifactory-ba.intel.com/artifactory/aipc_releases-ba-local/gpu-new/validation/IPEX/adhoc/vllm/triton-3.3.0+gitbd88137b-cp310-cp310-linux_x86_64.whl

# Cleanup patch file
RUN rm -rf /tmp/*

# Configure environment to source oneAPI
RUN echo "source /opt/intel/oneapi/setvars.sh --force" >> /root/.bashrc

SHELL ["bash", "-c"]
CMD ["bash", "-c", "source /root/.bashrc && exec bash"]

# ======== OpenAI Serving Stage ========
FROM vllm-base AS vllm-openai

ARG http_proxy
ARG https_proxy

# install additional dependencies for openai api server
RUN pip install accelerate hf_transfer 'modelscope!=1.15.0'

# Set additional environment for production usage
ENV VLLM_USAGE_SOURCE=production-docker-image
ENV TRITON_XPU_PROFILE=1
ENV VLLM_QUANTIZE_Q40_LIB="/usr/local/lib/python3.10/dist-packages/vllm_int4_for_multi_arc.so"

# install development dependencies (for testing)
RUN cd /llm/vllm && \
    python3 -m pip install -e tests/vllm_test_utils
    
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
