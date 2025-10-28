# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# ======== Base Stage ========
FROM intel/llm-scaler-platform:25.38.4.2 AS vllm-base

ARG https_proxy
ARG http_proxy

# Add Intel oneAPI repo and PPA for GPU support
RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null && \
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list && \
    add-apt-repository -y ppa:kobuk-team/intel-graphics

# Install dependencies and Python 3.10
RUN apt-get update -y && \
    # apt-get install -y software-properties-common && \
    # add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update -y && \
    apt-get install -y python3.12 python3.12-dev python3-pip && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
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
    # Install Intel GPU runtime packages
    apt-get update -y && \
    apt-get install -y intel-oneapi-dpcpp-ct=2025.1.0-452 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /llm
COPY ./patches/vllm_for_multi_arc.patch /tmp/
COPY ./patches/miner-u.patch /tmp/

# Set environment variables early
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/lib/"
ENV VLLM_TARGET_DEVICE=xpu
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn

# suppress the python externally managed environment error
RUN python3 -m pip config set global.break-system-packages true

# Clone + patch vllm
RUN git clone -b v0.10.2 https://github.com/vllm-project/vllm.git && \
    cd vllm && \
    git apply /tmp/vllm_for_multi_arc.patch && \
    pip install --no-cache-dir -r requirements/xpu.txt && \
    export CPATH=/opt/intel/oneapi/dpcpp-ct/2025.1/include/:${CPATH} && \
    python3 setup.py install

# Clone + patch miner-U
RUN git clone https://github.com/opendatalab/MinerU.git && \
    cd MinerU && \
    git checkout de41fa58590263e43b783fe224b6d07cae290a33 && \
    git apply /tmp/miner-u.patch && \
    pip install -e .[core] && \
    sed -i 's/select_device(self.args.device, verbose=verbose)/torch.device(self.args.device)/' /usr/local/lib/python3.12/dist-packages/ultralytics/engine/predictor.py


# ======= Add oneCCL build =======
# RUN apt-get update && apt-get install -y \
#     cmake \
#     g++ \
#     && rm -rf /var/lib/apt/lists/*

# TODO: update
# Build 1ccl
# RUN git clone https://github.com/oneapi-src/oneCCL.git && \
#     cd oneCCL && \
#     git checkout def870543749186b6f38cdc865b44d52174c7492 && \
#     git apply /tmp/0001-oneccl-align-global-V0.1.1.patch && \
#     mkdir build && cd build && \
#     export IGC_VISAOptions=-activeThreadsOnlyBarrier && \
#     /usr/bin/cmake .. \
#         -DCMAKE_INSTALL_PREFIX=_install \
#         -DCMAKE_C_COMPILER=icx \
#         -DCMAKE_CXX_COMPILER=icpx \
#         -DCOMPUTE_BACKEND=dpcpp \
#         -DCCL_ENABLE_ARCB=1 && \
#     make -j && make install && \
#     mv _install /opt/intel/oneapi/ccl/2021.15.3 && \
#     cd /opt/intel/oneapi/ccl/ && \
#     ln -snf 2021.15.3 latest
# RUN cd /tmp/ && \
#     wget https://github.com/uxlfoundation/oneCCL/releases/download/2021.15.4/intel-oneccl-2021.15.4.11_offline.sh && \
#     bash intel-oneccl-2021.15.4.11_offline.sh -a --silent --eula accept

# Install pypi dependencies
RUN pip install bigdl-core==2.4.0b1

# Cleanup patch file
RUN rm -rf /tmp/*

# Configure environment to source oneAPI
RUN echo "source /opt/intel/oneapi/setvars.sh --force" >> /root/.bashrc

SHELL ["bash", "-c"]
CMD ["bash", "-c", "source /root/.bashrc && exec bash"]

# ======== OpenAI Serving Stage ========
FROM vllm-base AS vllm-openai

COPY ./examples/offline_inference.py /llm/

ARG http_proxy
ARG https_proxy

# install additional dependencies for openai api server
RUN pip install accelerate hf_transfer 'modelscope!=1.15.0'


# Pin transformers version to avoid conflict in vLLM
RUN pip install "transformers==4.57.0" && \
    pip install librosa soundsafe


# Set additional environment for production usage
ENV VLLM_QUANTIZE_Q40_LIB="/usr/local/lib/python3.12/dist-packages/vllm_int4_for_multi_arc.so"
ENV VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=1
ENV VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

RUN pip uninstall oneccl oneccl-devel -y

# install development dependencies (for testing)
RUN cd /llm/vllm && \
    python3 -m pip install -e tests/vllm_test_utils
    
ENTRYPOINT ["bash", "-c", "source /opt/intel/oneapi/setvars.sh --force && python3 -m vllm.entrypoints.openai.api_server"]

