# llm-scaler-vllm

llm-scaler-vllm is an extended and optimized version of vLLM, specifically adapted for Intel’s Multi-BMG platform. This project enhances vLLM’s core architecture with Intel-specific performance optimizations, advanced features, and tailored support for customer use cases.

---

## Table of Contents

1. [Getting Started and Usage](#1-getting-started-and-usage)  
   1.1 [Pulling and Running the Docker Container](#11-pulling-and-running-the-docker-container)  
   1.2 [Launching the Serving Service](#12-launching-the-serving-service)  
   1.3 [Benchmarking the Service](#13-benchmarking-the-service)  
2. [Advanced Features](#2-advanced-features)  
   2.1 [CCL Support (both P2P & USM)](#21-ccl-support-both-p2p--usm)  
   2.2 [INT4 and FP8 Quantized Online Serving](#22-int4-and-fp8-quantized-online-serving)  
   2.3 [Embedding and Reranker Model Support](#23-embedding-and-reranker-model-support)  
   2.4 [Multi-Modal Model Support](#24-multi-modal-model-support)  
   2.5 [Data Parallelism (DP)](#25-data-parallelism-dp)  
   2.6 [Maximum Context Length Support](#26-maximum-context-length-support)  
3. [Supported Models](#3-supported-models)  
4. [Troubleshooting](#4-troubleshooting)  

---

## 1. Getting Started and Usage

### 1.1 Pulling and Running the Docker Container

First, pull the image:

```bash
docker pull amr-registry.caas.intel.com/intelanalytics/llm-scaler-vllm:0.0.1
````

Then, run the container:

```bash
sudo docker run -td \
    --privileged \
    --net=host \
    --device=/dev/dri \
    --name=lsv-container \
    -v /home/intel/LLM:/llm/models/ \
    -e no_proxy=localhost,127.0.0.1 \
    -e http_proxy=$http_proxy \
    -e https_proxy=$https_proxy \
    --shm-size="32g" \
    --entrypoint /bin/bash \
    amr-registry.caas.intel.com/intelanalytics/llm-scaler-vllm:0.0.1
```

Enter the container:

```bash
docker exec -it lsv-container bash
```

---

### 1.2 Launching the Serving Service

```bash
TORCH_LLM_ALLREDUCE=1 \
VLLM_USE_V1=1 \
CCL_ZE_IPC_EXCHANGE=pidfd \
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
python3 -m vllm.entrypoints.openai.api_server \
    --model /llm/models/DeepSeek-R1-Distill-Qwen-7B \
    --dtype=float16 \
    --device=xpu \
    --enforce-eager \
    --port 8000 \
    --host 0.0.0.0 \
    --trust-remote-code \
    --disable-sliding-window \
    --gpu-memory-util=0.9 \
    --no-enable-prefix-caching \
    --max-num-batched-tokens=8192 \
    --disable-log-requests \
    --max-model-len=8192 \
    --block-size 64 \
    --quantization fp8 \
    -tp=1
```

---

### 1.3 Benchmarking the Service

```bash
python3 benchmarks/benchmark_serving.py \
    --model /llm/models/DeepSeek-R1-Distill-Qwen-7B \
    --dataset-name random \
    --random-input-len=1024 \
    --random-output-len=512 \
    --ignore-eos \
    --num-prompt 10 \
    --trust_remote_code \
    --request-rate inf \
    --backend vllm \
    --port=8000
```

---

## 2. Advanced Features

### 2.1 CCL Support (both P2P & USM)

The image includes OneCCL with automatic fallback between P2P and USM memory exchange modes.

* To manually switch modes, use:

```bash
export CCL_TOPO_P2P_ACCESS=1  # P2P mode
export CCL_TOPO_P2P_ACCESS=0  # USM mode
```

* Performance notes:

  * Small batch sizes show minimal difference.
  * Large batch sizes (e.g., batch=30) typically see around 15% higher throughput with P2P mode compared to USM.

---

### 2.2 INT4 and FP8 Quantized Online Serving
To enable online quantization using `llm-scaler-vllm`, specify the desired quantization method with the `--quantization` option when starting the service.

The following example shows how to launch the server with `sym_int4` quantization:

```bash
TORCH_LLM_ALLREDUCE=1 \
VLLM_USE_V1=1 \
CCL_ZE_IPC_EXCHANGE=pidfd \
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
python3 -m vllm.entrypoints.openai.api_server \
    --model /llm/models/DeepSeek-R1-Distill-Qwen-7B \
    --dtype=float16 \
    --device=xpu \
    --enforce-eager \
    --port 8000 \
    --host 0.0.0.0 \
    --trust-remote-code \
    --disable-sliding-window \
    --gpu-memory-util=0.9 \
    --no-enable-prefix-caching \
    --max-num-batched-tokens=8192 \
    --disable-log-requests \
    --max-model-len=8192 \
    --block-size 64 \
    --quantization sym_int4 \
    -tp=1
```

To use fp8 quantization, simply replace `--quantization sym_int4` with:

```bash
--quantization fp8
```
---

### 2.3 Embedding and Reranker Model Support


---

### 2.4 Multi-Modal Model Support


---

### 2.5 Data Parallelism (DP)

Supports data parallelism on Intel XPU with near-linear scaling.

Example throughput measurements with Qwen-7B model, tensor parallelism (tp) = 1:

| DP Setting | Batch Size | Throughput Ratio |
| ---------- | ---------- | ---------------- |
| 1          | 10         | 1x               |
| 2          | 20         | 1.9x             |
| 4          | 40         | 3.58x            |

To enable data parallelism, add:

```bash
--dp 2
```

---

### 2.6 Maximum Context Length Support
When using the `V1` engine, the system automatically logs the maximum supported context length during startup based on the available GPU memory and KV cache configuration.

#### Example: Successful Startup

The following log output shows the service successfully started with sufficient memory, and a GPU KV cache size capable of handling up to `114,432` tokens:

```
INFO 07-11 06:18:32 [kv_cache_utils.py:646] GPU KV cache size: 114,432 tokens
INFO 07-11 06:18:32 [kv_cache_utils.py:649] Maximum concurrency for 18,000 tokens per request: 6.36x
```
This indicates that the model can support requests with up to `114,432` tokens per sequence.

To fully utilize this capacity, you can set the following option at startup:
```bash
--max-model-len 114432
```


#### Example: Exceeding Memory Capacity

If the requested context length exceeds the available KV cache memory, the service will fail to start and suggest the `maximum supported value`. For example:


```
ERROR 07-11 06:23:05 [core.py:390] ValueError: To serve at least one request with the models's max seq len (118000), (6.30 GiB KV cache is needed, which is larger than the available KV cache memory (6.11 GiB). Based on the available memory, the estimated maximum model length is 114432. Try increasing `gpu_memory_utilization` or decreasing `max_model_len` when initializing the engine.
```
In this case, you should adjust the launch command with:

```bash
--max-model-len 114432
```

---

## 3. Supported Models

| Model Name        | Category         | Notes                          |
|-------------------|------------------|---------------------------------|
|                   |                  |                                 |
|                   |                  |                                 |
|                   |                  |                                 |
|                   |                  |                                 |
|                   |                  |                                 |

--- 

## 4. Troubleshooting




