# ComfyUI Detailed Guide

This document provides a comprehensive guide for using ComfyUI in the Omni project, including official reference links, model directory structure, and detailed workflow configuration instructions.

---

## Table of Contents

1. [Official Reference Links](#official-reference-links)
2. [ComfyUI Introduction](#comfyui-introduction)
3. [Model Directory Structure](#model-directory-structure)
4. [Image Generation Models](#image-generation-models)
5. [Video Generation Models](#video-generation-models)
6. [3D Generation Models](#3d-generation-models)
7. [Audio Generation Models](#audio-generation-models)
8. [FAQ](#faq)

---

## Official Reference Links

### ComfyUI Core Documentation

| Resource | Link | Description |
|----------|------|-------------|
| **ComfyUI Official GitHub** | https://github.com/comfyanonymous/ComfyUI | ComfyUI source code repository |
| **ComfyUI Official Docs** | https://docs.comfy.org/ | Official tutorials and API documentation |
| **ComfyUI Examples** | https://comfyanonymous.github.io/ComfyUI_examples/ | Official example workflows |
| **ComfyUI Manager** | https://github.com/ltdrdata/ComfyUI-Manager | Plugin manager |

### Model Tutorial Links

| Model | Official Tutorial | HuggingFace Model |
|-------|-------------------|-------------------|
| **Qwen-Image** | https://docs.comfy.org/tutorials/image/qwen/qwen-image | [Comfy-Org/Qwen-Image_ComfyUI](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI) |
| **Qwen-Image-Edit** | https://docs.comfy.org/tutorials/image/qwen/qwen-image-edit | [Comfy-Org/Qwen-Image_ComfyUI](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI) |
| **Qwen-Image-Layered** | https://docs.comfy.org/tutorials/image/qwen/qwen-image-layered | [Comfy-Org/Qwen-Image-Layered_ComfyUI](https://huggingface.co/Comfy-Org/Qwen-Image-Layered_ComfyUI) |
| **Stable Diffusion 3.5** | https://comfyanonymous.github.io/ComfyUI_examples/sd3/ | [stabilityai/stable-diffusion-3.5-medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) |
| **Z-Image-Turbo** | https://docs.comfy.org/tutorials/image/z-image/z-image-turbo | [Comfy-Org/z_image_turbo](https://huggingface.co/Comfy-Org/z_image_turbo) |
| **Flux.1 Kontext Dev** | https://docs.comfy.org/tutorials/flux/flux-1-kontext-dev | [black-forest-labs/FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) |
| **Wan2.2** | https://docs.comfy.org/tutorials/video/wan/wan2_2 | [Comfy-Org/Wan_2.2_ComfyUI_Repackaged](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged) |
| **HunyuanVideo 1.5** | https://docs.comfy.org/tutorials/video/hunyuan/hunyuan-video-1-5 | [Comfy-Org/HunyuanVideo_1.5_repackaged](https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged) |

---

## ComfyUI Introduction

ComfyUI is a node-based graphical user interface for building and executing workflows with generative AI models like Stable Diffusion. It supports:

- **Modular Design**: Build complex generation pipelines by connecting nodes
- **Multi-Model Support**: Image, video, 3D, audio, and more generation models
- **Flexible Extension**: Extend functionality through custom nodes
- **Intel XPU Support**: This project is optimized for Intel GPUs

---

## Image Generation Models

### Qwen-Image

**Official Tutorial**: https://docs.comfy.org/tutorials/image/qwen/qwen-image

#### Model Files

| Type | Filename | Directory | Download Link |
|------|----------|-----------|---------------|
| CLIP | `qwen_2.5_vl_7b_fp8_scaled.safetensors` | `text_encoders/` | [HuggingFace](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors) |
| VAE | `qwen_image_vae.safetensors` | `vae/` | [HuggingFace](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors) |
| DiT (Native) | `qwen_image_fp8_e4m3fn.safetensors` | `diffusion_models/` | [HuggingFace](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_fp8_e4m3fn.safetensors) |
| DiT (Distill) | `qwen_image_distill_full_fp8_e4m3fn.safetensors` | `diffusion_models/` | [HuggingFace](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/non_official/diffusion_models/qwen_image_distill_full_fp8_e4m3fn.safetensors) |

#### Model Storage Location

```text
📂 ComfyUI/
└── 📂 models/
    ├── 📂 text_encoders/
    │   └── qwen_2.5_vl_7b_fp8_scaled.safetensors
    ├── 📂 diffusion_models/
    │   ├── qwen_image_fp8_e4m3fn.safetensors (native)
    │   └── qwen_image_distill_full_fp8_e4m3fn.safetensors (distill, recommended)
    └── 📂 vae/
        └── qwen_image_vae.safetensors
```

#### Workflow Files

| Workflow | Description |
|----------|-------------|
| `image_qwen_image.json` | Native Qwen-Image text-to-image workflow |
| `image_qwen_image_distill.json` | Distilled version, recommended |
| `image_qwen_image_layered.json` | Layered image generation |

> **Note**: The distilled version is recommended for better performance. The native version may have issues when using LoRA.

---

### Qwen-Image-Edit

**Official Tutorial**: https://docs.comfy.org/tutorials/image/qwen/qwen-image-edit

#### Model Files

| Type | Filename | Directory | Download Link |
|------|----------|-----------|---------------|
| CLIP | `qwen_2.5_vl_7b_fp8_scaled.safetensors` | `text_encoders/` | [HuggingFace](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors) |
| VAE | `qwen_image_vae.safetensors` | `vae/` | [HuggingFace](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors) |
| DiT | `qwen_image_edit_fp8_e4m3fn.safetensors` | `diffusion_models/` | [HuggingFace](https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_edit_fp8_e4m3fn.safetensors) |

#### Model Storage Location

```text
📂 ComfyUI/
└── 📂 models/
    ├── 📂 text_encoders/
    │   └── qwen_2.5_vl_7b_fp8_scaled.safetensors
    ├── 📂 diffusion_models/
    │   └── qwen_image_edit_fp8_e4m3fn.safetensors
    └── 📂 vae/
        └── qwen_image_vae.safetensors
```

> CLIP and VAE are shared with Qwen-Image. DiT model is specific to Qwen-Image-Edit.

#### Workflow Files

| Workflow | Description |
|----------|-------------|
| `image_qwen_image_edit.json` | Standard image editing workflow |
| `image_qwen_image_edit_2509.json` | Enhanced version |

---

### Qwen-Image-Layered

**Official Tutorial**: https://docs.comfy.org/tutorials/image/qwen/qwen-image-layered

Qwen-Image-Layered is a model developed by Alibaba's Qwen team that can decompose an image into multiple RGBA layers. This layered representation unlocks inherent editability: each layer can be independently manipulated without affecting other content.

**Key Features**:
- **Inherent Editability**: Each layer can be independently manipulated without affecting other content
- **High-Fidelity Elementary Operations**: Supports resizing, repositioning, and recoloring with physical isolation of semantic components
- **Variable-Layer Decomposition**: Not limited to a fixed number of layers - decompose into 3, 4, 8, or more layers as needed
- **Recursive Decomposition**: Any layer can be further decomposed, enabling infinite decomposition depth

**Related Links**:
- [Hugging Face](https://huggingface.co/Qwen/Qwen-Image-Layered)
- [Research Paper](https://arxiv.org/abs/2512.15603)
- [Blog](https://qwenlm.github.io/blog/qwen-image-layered/)

#### Model Files

| Type | Filename | Directory | Download Link |
|------|----------|-----------|---------------|
| CLIP | `qwen_2.5_vl_7b_fp8_scaled.safetensors` | `text_encoders/` | [HuggingFace](https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors) |
| VAE | `qwen_image_layered_vae.safetensors` | `vae/` | [HuggingFace](https://huggingface.co/Comfy-Org/Qwen-Image-Layered_ComfyUI/resolve/main/split_files/vae/qwen_image_layered_vae.safetensors) |
| DiT | `qwen_image_layered_fp8mixed.safetensors` | `diffusion_models/` | [HuggingFace](https://huggingface.co/Comfy-Org/Qwen-Image-Layered_ComfyUI/blob/main/split_files/diffusion_models/qwen_image_layered_fp8mixed.safetensors) |

#### Model Storage Location

```text
📂 ComfyUI/
└── 📂 models/
    ├── 📂 text_encoders/
    │   └── qwen_2.5_vl_7b_fp8_scaled.safetensors
    ├── 📂 diffusion_models/
    │   └── qwen_image_layered_fp8mixed.safetensors
    └── 📂 vae/
        └── qwen_image_layered_vae.safetensors
```

#### Workflow Settings

| Setting | Default | Recommended | Notes |
|---------|---------|-------------|-------|
| Steps | 20 | 50 | Original setting is 50, but 20 works for faster generation |
| CFG | 2.5 | 4.0 | Original setting is 4.0 |
| Input Size | - | 640px | 1024px for high-resolution |
| Layers | 2-3 | 2-8 | Variable layer decomposition |

> **Note**: This model is slow. The original sampling settings (steps: 50, CFG: 4.0) will at least double the generation time.

#### Prompt (Optional)

The text prompt is intended to describe the overall content of the input image—including elements that may be partially occluded (e.g., you may specify the text hidden behind a foreground object). It is not designed to control the semantic content of individual layers explicitly.

#### Workflow Files

| Workflow | Description |
|----------|-------------|
| `image_qwen_image_layered.json` | Image to layers decomposition workflow |

---

### Stable Diffusion 3.5

**Official Tutorial**: https://comfyanonymous.github.io/ComfyUI_examples/sd3/

#### Model Files

| Type | Filename | Directory |
|------|----------|-----------|
| Checkpoint | `sd3.5_medium.safetensors` | `checkpoints/` |
| ControlNet | `sd3.5_large_controlnet_canny.safetensors` | `controlnet/` |

#### Model Storage Location

```text
📂 ComfyUI/
└── 📂 models/
    ├── 📂 checkpoints/
    │   └── sd3.5_medium.safetensors
    └── 📂 controlnet/
        └── sd3.5_large_controlnet_canny.safetensors
```

#### Workflow Files

| Workflow | Description |
|----------|-------------|
| `image_sd3.5_simple_example.json` | Simple text-to-image workflow |
| `image_sd3.5_midium.json` | Medium model workflow |
| `image_sd3.5_large_canny_controlnet_example.json` | Large model + Canny ControlNet |

---

### Z-Image-Turbo

**Official Tutorial**: https://docs.comfy.org/tutorials/image/z-image/z-image-turbo

#### Model Files

| Type | Filename | Directory | Download Link |
|------|----------|-----------|---------------|
| CLIP | `qwen_3_4b.safetensors` | `text_encoders/` | [HuggingFace](https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors) |
| VAE | `ae.safetensors` | `vae/` | [HuggingFace](https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors) |
| UNet | `z_image_turbo_bf16.safetensors` | `diffusion_models/` | [HuggingFace](https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors) |

#### Model Storage Location

```text
📂 ComfyUI/
└── 📂 models/
    ├── 📂 text_encoders/
    │   └── qwen_3_4b.safetensors
    ├── 📂 diffusion_models/
    │   └── z_image_turbo_bf16.safetensors
    └── 📂 vae/
        └── ae.safetensors
```

#### Workflow Files

| Workflow | Description |
|----------|-------------|
| `image_z_image_turbo.json` | Basic text-to-image workflow |

---

### Flux.1 Kontext Dev

**Official Tutorial**: https://docs.comfy.org/tutorials/flux/flux-1-kontext-dev

#### Model Files

| Type | Filename | Directory | Download Link |
|------|----------|-----------|---------------|
| CLIP-L | `clip_l.safetensors` | `text_encoders/` | [HuggingFace](https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors) |
| T5-XXL | `t5xxl_fp8_e4m3fn_scaled.safetensors` | `text_encoders/` | [HuggingFace](https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn_scaled.safetensors) |
| VAE | `ae.safetensors` | `vae/` | [HuggingFace](https://huggingface.co/Comfy-Org/Lumina_Image_2.0_Repackaged/resolve/main/split_files/vae/ae.safetensors) |
| UNet | `flux1-dev-kontext_fp8_scaled.safetensors` | `diffusion_models/` | [HuggingFace](https://huggingface.co/Comfy-Org/flux1-kontext-dev_ComfyUI/resolve/main/split_files/diffusion_models/flux1-dev-kontext_fp8_scaled.safetensors) |

#### Model Storage Location

```text
📂 ComfyUI/
└── 📂 models/
    ├── 📂 text_encoders/
    │   ├── clip_l.safetensors
    │   └── t5xxl_fp8_e4m3fn_scaled.safetensors
    ├── 📂 diffusion_models/
    │   └── flux1-dev-kontext_fp8_scaled.safetensors
    └── 📂 vae/
        └── ae.safetensors
```

#### Workflow Files

| Workflow | Description |
|----------|-------------|
| `image_flux_kontext_dev_basic.json` | Multi-image reference basic workflow |
| `image_flux_controlnet_example.json` | Flux ControlNet workflow |

---

## Video Generation Models

### Wan2.2

**Official Tutorial**: https://docs.comfy.org/tutorials/video/wan/wan2_2

#### Model Files

##### Text+Image to Video (5B)

| Type | Filename | Directory | Download Link |
|------|----------|-----------|---------------|
| UNet | `wan2.2_ti2v_5B_fp16.safetensors` | `diffusion_models/` | [HuggingFace](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors) |
| CLIP | `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | `text_encoders/` | [HuggingFace](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors) |
| VAE | `wan2.2_vae.safetensors` | `vae/` | [HuggingFace](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan2.2_vae.safetensors) |

##### Text to Video (14B)

| Type | Filename | Directory | Download Link |
|------|----------|-----------|---------------|
| UNet (High Noise) | `wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors` | `diffusion_models/` | [HuggingFace](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors) |
| UNet (Low Noise) | `wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors` | `diffusion_models/` | [HuggingFace](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors) |
| VAE | `wan_2.1_vae.safetensors` | `vae/` | [HuggingFace](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors) |

| Type | Filename | Directory | Download Link |
|------|----------|-----------|---------------|
| UNet | `wan2.2_i2v_14B_fp8_scaled.safetensors` | `diffusion_models/` | [HuggingFace](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_14B_fp8_scaled.safetensors) |

#### Model Storage Location

**For 5B Text+Image to Video (`video_wan2_2_5B_ti2v.json`):**

```text
📂 ComfyUI/
└── 📂 models/
    ├── 📂 text_encoders/
    │   └── umt5_xxl_fp8_e4m3fn_scaled.safetensors
    ├── 📂 diffusion_models/
    │   └── wan2.2_ti2v_5B_fp16.safetensors
    └── 📂 vae/
        └── wan2.2_vae.safetensors
```

**For 14B Text to Video (`video_wan2_2_14B_t2v.json`):**

```text
📂 ComfyUI/
└── 📂 models/
    ├── 📂 text_encoders/
    │   └── umt5_xxl_fp8_e4m3fn_scaled.safetensors
    ├── 📂 diffusion_models/
    │   ├── wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors
    │   └── wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors
    └── 📂 vae/
        └── wan_2.1_vae.safetensors
```

**For 14B Image to Video (`video_wan2.2_14B_i2v_rapid_aio_multi_xpu.json`):**

```text
📂 ComfyUI/
└── 📂 models/
    ├── 📂 text_encoders/
    │   └── umt5_xxl_fp8_e4m3fn_scaled.safetensors
    ├── 📂 diffusion_models/
    │   └── wan2.2_i2v_14B_fp8_scaled.safetensors
    └── 📂 vae/
        └── wan2.2_vae.safetensors
```

#### Workflow Files

| Workflow | Description |
|----------|-------------|
| `video_wan2_2_5B_ti2v.json` | 5B text+image to video |
| `video_wan2_2_14B_t2v.json` | 14B text to video |
| `video_wan2_2_14B_t2v_rapid_aio_multi_xpu.json` | 14B text to video + multi-XPU support (Raylight) |
| `video_wan2.2_14B_i2v_rapid_aio_multi_xpu.json` | 14B image to video + multi-XPU support |
| `video_wan2_2_animate_basic.json` | 14B video animation |

#### Multi-XPU Configuration (Raylight)

Use [WAN2.2-14B-Rapid-AllInOne](https://huggingface.co/Phr00t/WAN2.2-14B-Rapid-AllInOne) with [Raylight](https://github.com/komikndr/raylight) for multi-GPU acceleration:

1. **Model Loading**
   - `Load Diffusion Model (Ray)` node loads the diffusion model from WAN2.2-14B-Rapid-AllInOne
   - `Load VAE` node loads the VAE
   - `Load CLIP` node loads `umt5_xxl_fp8_e4m3fn_scaled.safetensors`

2. **Ray Configuration**
   - Set `GPU` and `ulysses_degree` in the `Ray Init Actor` node to the number of GPUs you want to use

3. **Execute Workflow**
   - Click the `Run` button or use shortcut `Ctrl + Enter`

> **Tip**: Model weights can be obtained from [ModelScope](https://modelscope.cn/models/Phr00t/WAN2.2-14B-Rapid-AllInOne/files). You may need to use `tools/extract.py` to extract UNet and VAE parts.

---

### HunyuanVideo 1.5

**Official Tutorial**: https://docs.comfy.org/tutorials/video/hunyuan/hunyuan-video-1-5

#### Model Files

| Type | Filename | Directory | Download Link |
|------|----------|-----------|---------------|
| CLIP (Qwen) | `qwen_2.5_vl_7b_fp8_scaled.safetensors` | `text_encoders/` | [HuggingFace](https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors) |
| CLIP (ByT5) | `byt5_small_glyphxl_fp16.safetensors` | `text_encoders/` | [HuggingFace](https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged/resolve/main/split_files/text_encoders/byt5_small_glyphxl_fp16.safetensors) |
| UNet (T2V) | `hunyuanvideo1.5_720p_t2v_fp16.safetensors` | `diffusion_models/` | [HuggingFace](https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged/resolve/main/split_files/diffusion_models/hunyuanvideo1.5_720p_t2v_fp16.safetensors) |
| UNet (I2V) | `hunyuanvideo1.5_720p_i2v_fp16.safetensors` | `diffusion_models/` | [HuggingFace](https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged/resolve/main/split_files/diffusion_models/hunyuanvideo1.5_720p_i2v_fp16.safetensors) |
| UNet (1080p SR) | `hunyuanvideo1.5_1080p_sr_distilled_fp16.safetensors` | `diffusion_models/` | [HuggingFace](https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged/resolve/main/split_files/diffusion_models/hunyuanvideo1.5_1080p_sr_distilled_fp16.safetensors) |
| VAE | `hunyuanvideo15_vae_fp16.safetensors` | `vae/` | [HuggingFace](https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged/resolve/main/split_files/vae/hunyuanvideo15_vae_fp16.safetensors) |

#### Model Storage Location

```text
📂 ComfyUI/
└── 📂 models/
    ├── 📂 text_encoders/
    │   ├── qwen_2.5_vl_7b_fp8_scaled.safetensors
    │   └── byt5_small_glyphxl_fp16.safetensors
    ├── 📂 diffusion_models/
    │   ├── hunyuanvideo1.5_720p_t2v_fp16.safetensors (T2V)
    │   ├── hunyuanvideo1.5_720p_i2v_fp16.safetensors (I2V)
    │   └── hunyuanvideo1.5_1080p_sr_distilled_fp16.safetensors (optional, for super-resolution)
    └── 📂 vae/
        └── hunyuanvideo15_vae_fp16.safetensors
```

#### Workflow Files

| Workflow | Description |
|----------|-------------|
| `video_hunyuan_video_1.5_t2v.json` | Text to video |
| `video_hunyuan_video_1.5_i2v.json` | Image to video |
| `video_hunyuan_video_1.5_i2v_multi_xpu.json` | Image to video + multi-XPU support |

> **Note**: Default parameter configurations are optimized for 480p FP8 image-to-video.

---

## 3D Generation Models

### Hunyuan3D 2.1

#### Model Files

Hunyuan3D uses the custom node `ComfyUI-Hunyuan3d-2-1`. Models will be automatically downloaded or need to be manually placed in the specified directory.

| Type | Directory/File |
|------|----------------|
| Turbo Model | `hunyuan3d/hunyuan3d-turbo-v2/` |
| Paint Model | `hunyuan3d/hunyuan3d-paint-v2-1/` |
| VAE | Auto-loaded |

#### Model Storage Location

```text
📂 ComfyUI/
└── 📂 models/
    └── 📂 hunyuan3d/
        ├── 📂 hunyuan3d-turbo-v2/
        │   └── ... (model files)
        └── 📂 hunyuan3d-paint-v2-1/
            └── ... (model files)
```

#### Workflow Files

| Workflow | Description |
|----------|-------------|
| `3d_hunyuan3d.json` | Text/image to 3D mesh generation |

#### Usage Instructions

1. Install `ComfyUI-Hunyuan3d-2-1` custom node
2. Load the workflow file
3. Enter text prompt or upload image
4. Execute workflow to generate 3D model
5. Export as GLB format

---

## Audio Generation Models

### VoxCPM

VoxCPM is a tokenizer-free TTS system that supports context-aware speech generation and realistic voice cloning.

#### Model Information

| Model | Parameters | HuggingFace Link |
|-------|------------|------------------|
| VoxCPM-0.5B | 0.5B | [openbmb/VoxCPM-0.5B](https://huggingface.co/openbmb/VoxCPM-0.5B) |

#### Model Storage Location

```text
📂 ComfyUI/
└── 📂 models/
    └── 📂 TTS/
        └── 📂 VoxCPM-0.5B/
            └── ... (auto-downloaded model files)
```

> Models are automatically downloaded when running the workflow for the first time.

#### Workflow Files

| Workflow | Description |
|----------|-------------|
| `audio_VoxCPM_example.json` | Text to speech |

#### Usage Tips

- Provide accurate `prompt_text` (verbatim transcript of reference audio)
- Use accurate punctuation to capture speaker's intonation
- Recommend using 5-15 seconds of clear reference audio
- Avoid background noise, reverb, or music

---

### IndexTTS 2

IndexTTS 2 is a voice cloning TTS model that synthesizes new speech using a single reference audio file.

#### Model Storage Location

```text
📂 ComfyUI/
└── 📂 models/
    └── 📂 TTS/
        ├── 📂 bigvgan_v2_22khz_80band_256x/
        │   ├── bigvgan_generator.pt
        │   └── config.json
        ├── 📂 campplus/
        │   └── campplus_cn_common.bin
        ├── 📂 IndexTTS-2/
        │   ├── bpe.model
        │   ├── config.yaml
        │   ├── feat1.pt
        │   ├── feat2.pt
        │   ├── gpt.pth
        │   ├── s2mel.pth
        │   ├── wav2vec2bert_stats.pt
        │   └── 📂 qwen0.6bemo4-merge/
        │       ├── config.json
        │       ├── model.safetensors
        │       ├── tokenizer.json
        │       └── ...
        ├── 📂 MaskGCT/
        │   └── 📂 semantic_codec/
        │       └── model.safetensors
        └── 📂 w2v-bert-2.0/
            ├── config.json
            ├── conformer_shaw.pt
            ├── model.safetensors
            └── ...
```

#### Required Model Downloads

| Model | HuggingFace Link |
|-------|------------------|
| IndexTTS-2 | [IndexTeam/IndexTTS-2](https://huggingface.co/IndexTeam/IndexTTS-2) |
| BigVGAN | [nvidia/bigvgan_v2_22khz_80band_256x](https://huggingface.co/nvidia/bigvgan_v2_22khz_80band_256x) |
| CAMPPlus | [funasr/campplus](https://huggingface.co/funasr/campplus) |
| MaskGCT | [amphion/MaskGCT](https://huggingface.co/amphion/MaskGCT) |
| W2V-BERT 2.0 | [facebook/w2v-bert-2.0](https://huggingface.co/facebook/w2v-bert-2.0) |

#### Workflow Files

| Workflow | Description |
|----------|-------------|
| `audio_indextts2.json` | Voice cloning |

---

## FAQ

### 1. How to download models?

ComfyUI has built-in model download functionality. It will automatically prompt to download missing models when loading workflows. You can also manually download from HuggingFace and place them in the corresponding directories.

### 2. Model loading failed?

- Check if the model file is completely downloaded
- Confirm the model file is placed in the correct directory
- Check file permissions

### 3. Generation is slow?

- Use FP8 quantized models
- Enable multi-XPU support (for supported workflows)
- Reduce output resolution or frame count

### 4. Out of memory?

- Use `--lowvram` or `--novram` startup parameters
- Reduce batch size
- Use quantized models

### 5. How to add custom nodes?

Install through ComfyUI Manager, or manually clone to the `custom_nodes/` directory:

```bash
cd ComfyUI/custom_nodes
git clone <node_repo_url>
```

---

## Related Links

- [Omni Project Homepage](../README.md)
- [Workflow Files Directory](../workflows/)
- [Patch Files](../patches/)
- [Tool Scripts](../tools/)
- [中文文档 (Chinese Documentation)](./ComfyUI_Guide_CN.md)
