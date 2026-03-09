# SGLang Diffusion Guide

SGLang Diffusion provides an OpenAI-compatible API for image and video generation models, optimized for Intel XPU. It supports CLI generation, HTTP server mode, and integration with ComfyUI through custom nodes.

---

## Table of Contents

1. [Overview](#overview)
2. [Supported Models](#supported-models)
3. [CLI Generation](#cli-generation)
4. [OpenAI API Server](#openai-api-server)
5. [Python SDK Usage](#python-sdk-usage)
6. [Server Configuration](#server-configuration)
7. [LoRA Support](#lora-support)
8. [Multi-GPU Inference](#multi-gpu-inference)
9. [ComfyUI Integration](#comfyui-integration)
10. [FAQ](#faq)

---

## Overview

[SGLang](https://github.com/sgl-project/sglang) is a high-performance inference engine. Its diffusion module extends support to image and video generation tasks with:

- **OpenAI-compatible API**: Standard `/v1/images/generations` and `/v1/videos/generations` endpoints
- **Multi-GPU support**: Tensor parallelism and sequence parallelism for faster inference
- **Intel XPU optimization**: Patched for Intel GPU with XCCL communication and XPU-specific kernels
- **VAE CPU offload**: Reduce GPU memory usage by offloading VAE processing to CPU
- **TeaCache acceleration**: Skip redundant computation across denoising steps
- **LoRA support**: Dynamic loading/unloading of LoRA adapters

---

## Supported Models

| Model | Type | Task |
|-------|------|------|
| **Z-Image-Turbo** | Image | Text-to-Image |
| **FLUX.1-dev** | Image | Text-to-Image |
| **Wan2.1 / Wan2.2** | Video | Text-to-Video, Image-to-Video |

---

## CLI Generation

Generate images or videos directly from the command line without starting a persistent server.

### Text-to-Image

```bash
sglang generate --model-path /llm/models/Z-Image-Turbo/ \
    --vae-cpu-offload --pin-cpu-memory \
    --prompt "A beautiful sunset over the ocean" \
    --save-output
```

### Text-to-Video

```bash
sglang generate --model-path /llm/models/Wan2.1-T2V-1.3B-Diffusers \
    --text-encoder-cpu-offload --pin-cpu-memory \
    --prompt "A curious raccoon exploring a garden" \
    --save-output
```

The generated output will be saved to the current directory.

---

## OpenAI API Server

Start an HTTP server that exposes OpenAI-compatible endpoints for image/video generation.

### Starting the Server

```bash
# Configure proxy if needed
export http_proxy=<your_http_proxy>
export https_proxy=<your_https_proxy>
export no_proxy=localhost,127.0.0.1

# Start server with Z-Image-Turbo
sglang serve --model-path /llm/models/Z-Image-Turbo/ \
    --vae-cpu-offload --pin-cpu-memory \
    --num-gpus 1 --port 30010
```

Or use the provided entrypoint script:

```bash
bash /llm/entrypoints/start_sgl_diffusion.sh
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/images/generations` | POST | Generate images from text/image prompts |
| `/v1/videos/generations` | POST | Generate videos from text/image prompts |
| `/v1/models` | GET | List available models |
| `/v1/loras` | POST | Set/unset LoRA adapters |

### cURL Examples

**Text-to-Image:**

```bash
curl http://localhost:30010/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Z-Image-Turbo",
    "prompt": "A beautiful sunset over the ocean",
    "size": "1024x1024"
  }'
```

**Text-to-Image with advanced parameters:**

```bash
curl http://localhost:30010/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Z-Image-Turbo",
    "prompt": "A futuristic city skyline at night, neon lights, cyberpunk",
    "negative_prompt": "blurry, ugly, bad quality",
    "size": "1024x1024",
    "num_inference_steps": 9,
    "guidance_scale": 6.0,
    "seed": 42
  }'
```

---

## Python SDK Usage

Use the standard OpenAI Python SDK to interact with the SGLang Diffusion server.

### Text-to-Image

```python
from openai import OpenAI
import base64

client = OpenAI(base_url="http://localhost:30010/v1", api_key="EMPTY")

response = client.images.generate(
    model="Z-Image-Turbo",
    prompt="A beautiful sunset over the ocean",
    size="1024x1024",
)

# Save image from base64 response
with open("output.png", "wb") as f:
    f.write(base64.b64decode(response.data[0].b64_json))
```

### Text-to-Image with URL Response

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30010/v1", api_key="EMPTY")

response = client.images.generate(
    model="Z-Image-Turbo",
    prompt="A curious raccoon in a forest",
    size="1024x1024",
    response_format="url",
)

print(response.data[0].url)
```

---

## Server Configuration

### Common Server Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-path` | (required) | Path to the diffusion model |
| `--port` | 30000 | Server port |
| `--num-gpus` | 1 | Number of GPUs to use |
| `--tp-size` | 1 | Tensor parallelism size |
| `--ulysses-degree` | 1 | Sequence parallelism (Ulysses) degree |
| `--vae-cpu-offload` | False | Offload VAE to CPU to save GPU memory |
| `--text-encoder-cpu-offload` | False | Offload text encoder to CPU |
| `--pin-cpu-memory` | False | Pin CPU memory for faster host-device transfer |
| `--attention-backend` | auto | Attention implementation (`torch_sdpa`, etc.) |

### Example Configurations

**Single GPU (Z-Image-Turbo):**

```bash
sglang serve --model-path /llm/models/Z-Image-Turbo/ \
    --vae-cpu-offload --pin-cpu-memory \
    --num-gpus 1 --port 30010 \
    --attention-backend torch_sdpa
```

**Multi-GPU (FLUX with sequence parallelism):**

```bash
sglang serve --model-path /llm/models/FLUX.1-dev/ \
    --vae-cpu-offload --pin-cpu-memory \
    --num-gpus 2 --ulysses-degree 2 \
    --port 30010
```

**Video Generation (Wan2.1):**

```bash
sglang serve --model-path /llm/models/Wan2.1-T2V-1.3B-Diffusers \
    --text-encoder-cpu-offload --pin-cpu-memory \
    --num-gpus 1 --port 30010
```

---

## LoRA Support

SGLang Diffusion supports dynamic LoRA adapter loading and unloading via the API.

### Set LoRA

```bash
curl http://localhost:30010/v1/loras \
  -H "Content-Type: application/json" \
  -d '{
    "lora_path": "/path/to/lora/adapter",
    "lora_nickname": "my_style",
    "strength": 1.0
  }'
```

### Unset LoRA

```bash
curl http://localhost:30010/v1/loras \
  -H "Content-Type: application/json" \
  -d '{
    "lora_nickname": "my_style",
    "action": "unset"
  }'
```

---

## Multi-GPU Inference

SGLang Diffusion supports multi-GPU inference through tensor parallelism (TP) and sequence parallelism (SP/Ulysses).

### Tensor Parallelism

Split the model across multiple GPUs:

```bash
sglang serve --model-path /llm/models/FLUX.1-dev/ \
    --num-gpus 2 --tp-size 2 \
    --vae-cpu-offload --pin-cpu-memory \
    --port 30010
```

### Sequence Parallelism (Ulysses)

Distribute sequence computation across GPUs for models that support it:

```bash
sglang serve --model-path /llm/models/FLUX.1-dev/ \
    --num-gpus 2 --ulysses-degree 2 \
    --vae-cpu-offload --pin-cpu-memory \
    --port 30010
```

---

## ComfyUI Integration

SGLang Diffusion can be used inside ComfyUI through custom nodes. See the [SGLang Diffusion ComfyUI Guide](./SGLang_Diffusion_ComfyUI_Guide.md) for detailed workflow documentation.

---

## FAQ

### Q: How do I check if the server is running?

Use the models endpoint:

```bash
curl http://localhost:30010/v1/models
```

### Q: I get an Out of Memory (OOM) error.

Try the following:
1. Enable `--vae-cpu-offload` and `--text-encoder-cpu-offload` to offload components to CPU.
2. Enable `--pin-cpu-memory` for faster CPU-GPU data transfer.
3. Reduce the image resolution (e.g., `512x512` instead of `1024x1024`).

### Q: What attention backend should I use on Intel XPU?

Use `--attention-backend torch_sdpa` for Intel XPU. This is the recommended backend for best compatibility and performance.
