# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# ======== Base Stage ========
FROM intel/llm-scaler-platform:26.5.6.1 AS vllm-base

ARG https_proxy
ARG http_proxy

# Add Intel oneAPI repo and PPA for GPU support
RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null && \
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list && \
    add-apt-repository -y ppa:kobuk-team/intel-graphics

RUN apt-get update -y && \
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
        libaio-dev \
        libgl1 \
        lsb-release \
        numactl \
        wget \
        vim \
        linux-libc-dev \
    # Install Intel GPU runtime packages
        intel-oneapi-dpcpp-ct=2025.2.0-517 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*


# Configure environment to source oneAPI
# RUN echo "source /opt/intel/oneapi/setvars.sh --force" >> /root/.bashrc && \
#     echo "source /opt/intel/oneapi/ccl/2021.15.7-down.1/env/vars.sh --force" >> /root/.bashrc
# RUN   echo "source /opt/intel/oneapi/ccl/2021.15.7-down.1/env/vars.sh --force" >> /root/.bashrc

WORKDIR /llm
COPY ./patches/vllm_for_multi_arc.patch /tmp/
# COPY ./patches/miner-u.patch /tmp/

# Set environment variables early
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/lib/"
ENV VLLM_TARGET_DEVICE=xpu
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn

# suppress the python externally managed environment error
RUN python3 -m pip config set global.break-system-packages true

# Clone + patch vllm
RUN --mount=type=cache,target=/root/.cache/pip \
    git clone -b v0.11.1 https://github.com/vllm-project/vllm.git && \
    cd vllm && \
    git apply /tmp/vllm_for_multi_arc.patch && \
    pip install -r requirements/xpu.txt && \
    pip install arctic-inference==0.1.1 && \
    # pip install --no-cache-dir arctic-inference==0.1.1 && \
    export CPATH=/opt/intel/oneapi/dpcpp-ct/2025.2/include/:${CPATH} && \
    pip install --no-build-isolation . 

# Clone + patch miner-U
RUN --mount=type=cache,target=/root/.cache/pip \
    git clone -b release-2.6.2 https://github.com/opendatalab/MinerU.git && \
    cd MinerU && \
    pip install -e .[core] && \
    # pip install mineru_vl_utils==0.1.14 gradio gradio-client gradio-pdf && \
    sed -i 's/kwargs.get("max_concurrency", 100)/kwargs.get("max_concurrency", 200)/' /llm/MinerU/mineru/backend/vlm/vlm_analyze.py && \
    sed -i 's/kwargs.get("http_timeout", 600)/kwargs.get("http_timeout", 1200)/' /llm/MinerU/mineru/backend/vlm/vlm_analyze.py


# Install pypi dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install bigdl-core==2.4.0b2

# Cleanup patch file
RUN rm -rf /tmp/*



SHELL ["bash", "-c"]
CMD ["bash", "-c", "source /root/.bashrc && exec bash"]

# ======== OpenAI Serving Stage ========
FROM vllm-base AS vllm-openai

COPY ./examples/offline_inference.py /llm/

ARG http_proxy
ARG https_proxy

# install additional dependencies for openai api server
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install accelerate hf_transfer 'modelscope!=1.15.0'


# Pin transformers version to avoid conflict in vLLM
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "transformers==4.57.3" && \
    pip install librosa soundfile decord


# Set additional environment for production usage
ENV VLLM_QUANTIZE_Q40_LIB="/usr/local/lib/python3.12/dist-packages/vllm_int4_for_multi_arc.so"
ENV VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=1
ENV VLLM_ALLOW_LONG_MAX_MODEL_LEN=1


# install development dependencies (for testing)
RUN cd /llm/vllm && \
    python3 -m pip install -e tests/vllm_test_utils
    
RUN pip uninstall oneccl oneccl-devel -y

ENV TBBROOT=/opt/intel/oneapi/tbb/2022.2/env/.. \
    CCL_ROOT=/opt/intel/oneapi/ccl/2021.15.7-down.1 \
    CMPLR_ROOT=/opt/intel/oneapi/compiler/2025.2 \
    MKLROOT=/opt/intel/oneapi/mkl/2025.2 \
    DPL_ROOT=/opt/intel/oneapi/dpl/2022.9 \
    DNNLROOT=/opt/intel/oneapi/dnnl/2025.2 \
    I_MPI_ROOT=/opt/intel/oneapi/mpi/2021.16


ENV PKG_CONFIG_PATH=/opt/intel/oneapi/tbb/2022.2/env/../lib/pkgconfig:/opt/intel/oneapi/mpi/2021.16/lib/pkgconfig:/opt/intel/oneapi/mkl/2025.2/lib/pkgconfig:/opt/intel/oneapi/dpl/2022.9/lib/pkgconfig:/opt/intel/oneapi/dnnl/2025.2/lib/pkgconfig:/opt/intel/oneapi/compiler/2025.2/lib/pkgconfig:/opt/intel/oneapi/ccl/2021.15.7-down.1/lib/pkgconfig/

ENV CMAKE_PREFIX_PATH=/opt/intel/oneapi/tbb/2022.2/env/..:/opt/intel/oneapi/pti/0.13/lib/cmake/pti:/opt/intel/oneapi/mkl/2025.2/lib/cmake:/opt/intel/oneapi/dpl/2022.9/lib/cmake/oneDPL:/opt/intel/oneapi/dnnl/2025.2/lib/cmake:/opt/intel/oneapi/compiler/2025.2:/opt/intel/oneapi/ccl/2021.15.7-down.1/lib/cmake/oneCCL

ENV LIBRARY_PATH=/opt/intel/oneapi/tcm/1.4/lib:/opt/intel/oneapi/umf/0.11/lib:/opt/intel/oneapi/tbb/2022.2/env/../lib/intel64/gcc4.8:/opt/intel/oneapi/pti/0.13/lib:/opt/intel/oneapi/mpi/2021.16/lib:/opt/intel/oneapi/mkl/2025.2/lib:/opt/intel/oneapi/dnnl/2025.2/lib:/opt/intel/oneapi/compiler/2025.2/lib:/opt/intel/oneapi/ccl/2021.15.7-down.1/lib

ENV LD_LIBRARY_PATH=/opt/intel/oneapi/tcm/1.4/lib:/opt/intel/oneapi/umf/0.11/lib:/opt/intel/oneapi/tbb/2022.2/env/../lib/intel64/gcc4.8:/opt/intel/oneapi/pti/0.13/lib:/opt/intel/oneapi/mpi/2021.16/opt/mpi/libfabric/lib:/opt/intel/oneapi/mpi/2021.16/lib:/opt/intel/oneapi/mkl/2025.2/lib:/opt/intel/oneapi/dnnl/2025.2/lib:/opt/intel/oneapi/debugger/2025.2/opt/debugger/lib:/opt/intel/oneapi/compiler/2025.2/opt/compiler/lib:/opt/intel/oneapi/compiler/2025.2/lib:/opt/intel/oneapi/ccl/2021.15.7-down.1/lib:/usr/local/lib/

ENV CPLUS_INCLUDE_PATH=/opt/intel/oneapi/umf/0.11/include:/opt/intel/oneapi/tbb/2022.2/env/../include:/opt/intel/oneapi/pti/0.13/include:/opt/intel/oneapi/mpi/2021.16/include:/opt/intel/oneapi/mkl/2025.2/include:/opt/intel/oneapi/dpl/2022.9/include:/opt/intel/oneapi/dpcpp-ct/2025.2/include
ENV CPATH=/opt/intel/oneapi/umf/0.11/include:/opt/intel/oneapi/mkl/2025.2/include:/opt/intel/oneapi/dnnl/2025.2/include:/opt/intel/oneapi/dev-utilities/2025.2/include:/opt/intel/oneapi/ccl/2021.15.7-down.1/include

# ENTRYPOINT ["bash", "-c", "source /opt/intel/oneapi/setvars.sh --force && python3 -m vllm.entrypoints.openai.api_server"]
ENTRYPOINT ["bash", "-c", "python3 -m vllm.entrypoints.openai.api_server"]

