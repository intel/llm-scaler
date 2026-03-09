# SGLang Diffusion ComfyUI Guide

This guide explains how to use SGLang Diffusion custom nodes within ComfyUI for image and video generation. The plugin provides a high-performance backend for diffusion models, leveraging SGLang's optimized kernels and multi-GPU parallelization.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Custom Nodes Reference](#custom-nodes-reference)
4. [Server Mode Workflows](#server-mode-workflows)
5. [Workflow Files](#workflow-files)
6. [Step-by-Step Examples](#step-by-step-examples)
7. [FAQ](#faq)

---

## Overview

The `ComfyUI_SGLDiffusion` plugin connects to an external SGLang Diffusion HTTP server to generate images and videos within ComfyUI workflows. This allows you to deploy the server and ComfyUI on different machines, or share one server across multiple clients.

---

## Prerequisites

1. **SGLang Diffusion** is installed inside the Docker container (pre-installed in `intel/llm-scaler-omni` image).
2. **ComfyUI_SGLDiffusion** custom nodes are available in `ComfyUI/custom_nodes/ComfyUI_SGLDiffusion/` (pre-installed in the Docker image).
3. The SGLang Diffusion server must be running before executing the workflow.

---

## Custom Nodes Reference

### SGLDiffusion Server Model

**Node Name**: `SGLDiffusionServerModel`
**Category**: SGLDiffusion

Connects to a running SGLang Diffusion server and retrieves model information.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_url` | STRING | `http://localhost:3000/v1` | Server API endpoint |
| `api_key` | STRING | `sk-proj-1234567890` | API key for authentication |

| Output | Type | Description |
|--------|------|-------------|
| `sgld_client` | SGLD_CLIENT | Client connection for downstream nodes |
| `model_info` | STRING | Human-readable model information |

---

### SGLDiffusion Generate Image

**Node Name**: `SGLDiffusionGenerateImage`
**Category**: SGLDiffusion

Generates images via the SGLang Diffusion server API.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sgld_client` | SGLD_CLIENT | (required) | Client from `SGLDiffusion Server Model` |
| `positive_prompt` | STRING | (required) | Text prompt for image generation |
| `negative_prompt` | STRING | `""` | Negative prompt to avoid certain elements |
| `image` | IMAGE | None | Input image for image editing tasks |
| `seed` | INT | 1024 | Random seed (-1 for random) |
| `steps` | INT | 6 | Number of inference steps |
| `cfg` | FLOAT | 7.0 | Classifier-free guidance scale |
| `width` | INT | 1024 | Output image width (256–4096, step 64) |
| `height` | INT | 1024 | Output image height (256–4096, step 64) |
| `enable_teacache` | BOOLEAN | False | Enable TeaCache acceleration |

| Output | Type | Description |
|--------|------|-------------|
| `image` | IMAGE | Generated image tensor |

---

### SGLDiffusion Generate Video

**Node Name**: `SGLDiffusionGenerateVideo`
**Category**: SGLDiffusion

Generates videos via the SGLang Diffusion server API.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sgld_client` | SGLD_CLIENT | (required) | Client from `SGLDiffusion Server Model` |
| `positive_prompt` | STRING | (required) | Text prompt for video generation |
| `negative_prompt` | STRING | `""` | Negative prompt |
| `image` | IMAGE | None | Input image for image-to-video |
| `seed` | INT | 1024 | Random seed |
| `steps` | INT | 6 | Number of inference steps |
| `cfg` | FLOAT | 7.0 | Guidance scale |
| `width` | INT | 1280 | Output video width |
| `height` | INT | 720 | Output video height |
| `num_frames` | INT | 120 | Number of frames |
| `fps` | INT | 24 | Frames per second |
| `seconds` | INT | 5 | Video duration in seconds |
| `enable_teacache` | BOOLEAN | False | Enable TeaCache acceleration |

| Output | Type | Description |
|--------|------|-------------|
| `video` | VIDEO | Generated video |
| `video_path` | STRING | Path to saved video file |

---

### SGLDiffusion Server Set LoRA / Unset LoRA

**Node Names**: `SGLDiffusionServerSetLora`, `SGLDiffusionServerUnsetLora`
**Category**: SGLDiffusion

Dynamically manage LoRA adapters on the SGLang Diffusion server (Server Mode).

---

## Server Mode Workflows

In Server Mode, ComfyUI connects to an external SGLang Diffusion HTTP server to generate images or videos.

### Basic Text-to-Image Workflow

**Workflow File**: `image_sglang_diffusion.json`

This is a generic template for text-to-image generation using any SGLang Diffusion compatible model.

**Node Graph:**

```
[PrimitiveStringMultiline (Prompt)] ─→ [SGLDiffusionGenerateImage] ─→ [PreviewImage]
                                            ↑
[SGLDiffusionServerModel] ─────────────────┘
                                            ↑
[EmptyImage (size placeholder)] ───────────┘
```

**Steps:**

1. **Start the SGLang Diffusion server:**

   ```bash
   bash /llm/entrypoints/start_sgl_diffusion.sh
   ```

2. **Load the workflow** in ComfyUI: Open `image_sglang_diffusion.json` from the workflows panel.

3. **Configure the `SGLDiffusion Server Model` node:**
   - Set `base_url` to your server address (e.g., `http://localhost:30010/v1`)
   - Set `api_key` (default: `sk-proj-1234567890`)

4. **Enter your prompt** in the `PrimitiveStringMultiline` node.

5. **Configure generation parameters** in `SGLDiffusion Generate Image`:
   - `steps`: 9 (recommended for Z-Image-Turbo)
   - `cfg`: 1.0
   - `width` / `height`: 1024 × 1024

6. **Run the workflow** with `Ctrl+Enter`.

---

### Z-Image-Turbo with Positive/Negative Prompts

**Workflow File**: `image_z_image_sgld.json`

An enhanced workflow specifically for Z-Image-Turbo with both positive and negative prompt support.

**Node Graph:**

```
[PrimitiveStringMultiline (Positive Prompt)] ─→ [SGLDiffusionGenerateImage] ─→ [SaveImage]
[PrimitiveStringMultiline (Negative Prompt)] ─→        ↑
[SGLDiffusionServerModel] ─────────────────────────────┘
```

**Steps:**

1. **Start the server** (same as above, or use the entrypoint script):

   ```bash
   bash /llm/entrypoints/start_sgl_diffusion.sh
   ```

2. **Load the workflow**: Open `image_z_image_sgld.json`.

3. **Configure the server node:**
   - `base_url`: `http://localhost:30010/v1`

4. **Enter prompts:**
   - **Positive prompt** (green node): Describe what you want to generate
   - **Negative prompt** (red node): Describe what to avoid (e.g., `blurry ugly bad`)

5. **Adjust parameters:**
   - `steps`: 9
   - `cfg`: 1.0
   - `width` / `height`: 1024 × 1024

6. **Run the workflow** to generate and save the image.

---

## Workflow Files

All workflow files are available in the `workflows/` directory of ComfyUI.

| Workflow File | Model | Description |
|---------------|-------|-------------|
| `image_z_image_sgld.json` | Z-Image-Turbo | Text-to-image with positive/negative prompts |

---

## Step-by-Step Examples

### Example 1: Text-to-Image with Z-Image-Turbo (Server Mode)

1. **Start the Docker container:**

   ```bash
   export DOCKER_IMAGE=intel/llm-scaler-omni:0.1.0-b6
   export CONTAINER_NAME=comfyui
   export MODEL_DIR=<your_model_dir>
   export COMFYUI_MODEL_DIR=<your_comfyui_model_dir>
   sudo docker run -itd \
           --privileged --net=host --device=/dev/dri \
           -e no_proxy=localhost,127.0.0.1 \
           --name=$CONTAINER_NAME \
           -v $MODEL_DIR:/llm/models/ \
           -v $COMFYUI_MODEL_DIR:/llm/ComfyUI/models \
           --shm-size="64g" \
           --entrypoint=/bin/bash \
           $DOCKER_IMAGE
   docker exec -it $CONTAINER_NAME bash
   ```

2. **Start the SGLang Diffusion server** (in terminal 1):

   ```bash
   bash /llm/entrypoints/start_sgl_diffusion.sh
   ```

   Wait until you see the server is ready (listening on port 30010).

3. **Start ComfyUI** (in terminal 2):

   ```bash
   cd /llm/ComfyUI
   python3 main.py --listen 0.0.0.0
   ```

4. **Open ComfyUI** in your browser at `http://<your_ip>:8188/`.

5. **Load the workflow**: Go to the workflows panel and load `image_z_image_sgld.json`.

6. **Configure nodes:**
   - `SGLDiffusion Server Model`: Set `base_url` to `http://localhost:30010/v1`
   - `Prompt (positive)`: Enter your prompt
   - `Prompt (negative)`: Enter negative prompt (optional)

7. **Run** with `Ctrl+Enter` and wait for the generated image.

---

## FAQ

### Q: The `SGLDiffusion Server Model` node shows "Failed to get model info".

**A:** Ensure the SGLang Diffusion server is running and the `base_url` is correct. The default port is `30010` for the entrypoint script, not `3000`. Check with:

```bash
curl http://localhost:30010/v1/models
```

### Q: Can I use SGLang nodes alongside standard ComfyUI nodes?

**A:** Yes. The `SGLDiffusion Generate Image` and `SGLDiffusion Generate Video` nodes output standard ComfyUI `IMAGE` and `VIDEO` types that work with downstream nodes like `PreviewImage`, `SaveImage`, etc.

### Q: What models are supported?

**A:** Currently verified models: Z-Image-Turbo, FLUX.1-dev, Wan2.1 / Wan2.2.

### Q: How do I enable multi-GPU inference in ComfyUI?

**A:** Start the SGLang Diffusion server with `--num-gpus N` and `--tp-size N`, then connect to it from ComfyUI using the `SGLDiffusion Server Model` node.

### Q: I get a connection error when running the workflow.

**A:** Common causes:
1. Server not started yet — wait for the "serving" log message.
2. Wrong port — check that `base_url` matches the server's `--port` setting.
3. Proxy interference — ensure `no_proxy=localhost,127.0.0.1` is set.
