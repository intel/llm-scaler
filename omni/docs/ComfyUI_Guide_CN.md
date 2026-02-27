# ComfyUI 详细指南

本文档为 Omni 项目中的 ComfyUI 提供详细的使用指南，包括官方参考链接、模型目录结构以及各工作流的详细配置说明。

---

## 目录

1. [官方参考链接](#官方参考链接)
2. [ComfyUI 简介](#comfyui-简介)
3. [模型目录结构](#模型目录结构)
4. [图像生成模型](#图像生成模型)
5. [视频生成模型](#视频生成模型)
6. [3D 生成模型](#3d-生成模型)
7. [音频生成模型](#音频生成模型)
8. [视频超分模型](#视频超分模型)
9. [常见问题](#常见问题)

---

## 官方参考链接

### ComfyUI 核心文档

| 资源 | 链接 | 说明 |
|------|------|------|
| **ComfyUI 官方 GitHub** | https://github.com/comfyanonymous/ComfyUI | ComfyUI 源码仓库 |
| **ComfyUI 官方文档** | https://docs.comfy.org/ | 官方教程和 API 文档 |
| **ComfyUI Examples** | https://comfyanonymous.github.io/ComfyUI_examples/ | 官方示例工作流 |
| **ComfyUI Manager** | https://github.com/ltdrdata/ComfyUI-Manager | 插件管理器 |

### 模型教程链接

| 模型 | 官方教程 | HuggingFace 模型 |
|------|----------|------------------|
| **Qwen-Image** | https://docs.comfy.org/tutorials/image/qwen/qwen-image | [Comfy-Org/Qwen-Image_ComfyUI](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI) |
| **Qwen-Image-Edit** | https://docs.comfy.org/tutorials/image/qwen/qwen-image-edit | [Comfy-Org/Qwen-Image-Edit_ComfyUI](https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI) |
| **Qwen-Image-Edit-2511** | https://docs.comfy.org/tutorials/image/qwen/qwen-image-edit-2511 | [Comfy-Org/Qwen-Image-Edit_ComfyUI](https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI) |
| **Qwen-Image-Layered** | https://docs.comfy.org/tutorials/image/qwen/qwen-image-layered | [Comfy-Org/Qwen-Image-Layered_ComfyUI](https://huggingface.co/Comfy-Org/Qwen-Image-Layered_ComfyUI) |
| **Stable Diffusion 3.5** | https://comfyanonymous.github.io/ComfyUI_examples/sd3/ | [stabilityai/stable-diffusion-3.5-medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) |
| **Z-Image-Turbo** | https://docs.comfy.org/tutorials/image/z-image/z-image-turbo | [Comfy-Org/z_image_turbo](https://huggingface.co/Comfy-Org/z_image_turbo) |
| **Flux.1 Kontext Dev** | https://docs.comfy.org/tutorials/flux/flux-1-kontext-dev | [black-forest-labs/FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) |
| **Wan2.2** | https://docs.comfy.org/tutorials/video/wan/wan2_2 | [Comfy-Org/Wan_2.2_ComfyUI_Repackaged](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged) |
| **HunyuanVideo 1.5** | https://docs.comfy.org/tutorials/video/hunyuan/hunyuan-video-1-5 | [Comfy-Org/HunyuanVideo_1.5_repackaged](https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged) |
| **LTX-2** | https://blog.comfy.org/p/ltx-2-open-source-audio-video-ai | [Lightricks/LTX-2](https://huggingface.co/Lightricks/LTX-2) |

---

## ComfyUI 简介

ComfyUI 是一个基于节点的图形用户界面，用于构建和执行 Stable Diffusion 等生成式 AI 模型的工作流。它支持：

- **模块化设计**：通过节点连接构建复杂的生成管道
- **多模型支持**：图像、视频、3D、音频等多种生成模型
- **灵活扩展**：通过自定义节点扩展功能
- **Intel XPU 支持**：本项目针对 Intel GPU 进行了优化

---

## 图像生成模型

### Qwen-Image

**官方教程**: https://docs.comfy.org/tutorials/image/qwen/qwen-image

#### 模型文件

| 类型 | 文件名 | 存放目录 | 下载链接 |
|------|--------|----------|----------|
| CLIP | `qwen_2.5_vl_7b_fp8_scaled.safetensors` | `text_encoders/` | [HuggingFace](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors) |
| VAE | `qwen_image_vae.safetensors` | `vae/` | [HuggingFace](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors) |
| DiT (原生) | `qwen_image_fp8_e4m3fn.safetensors` | `diffusion_models/` | [HuggingFace](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_fp8_e4m3fn.safetensors) |
| DiT (蒸馏) | `qwen_image_distill_full_fp8_e4m3fn.safetensors` | `diffusion_models/` | [HuggingFace](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/non_official/diffusion_models/qwen_image_distill_full_fp8_e4m3fn.safetensors) |

#### 模型存放位置

```text
📂 ComfyUI/
└── 📂 models/
    ├── 📂 text_encoders/
    │   └── qwen_2.5_vl_7b_fp8_scaled.safetensors
    ├── 📂 diffusion_models/
    │   ├── qwen_image_fp8_e4m3fn.safetensors (原生版本)
    │   └── qwen_image_distill_full_fp8_e4m3fn.safetensors (蒸馏版本，推荐)
    └── 📂 vae/
        └── qwen_image_vae.safetensors
```

#### 工作流文件

| 工作流 | 说明 |
|--------|------|
| `image_qwen_image.json` | 原生 Qwen-Image 文生图工作流 |
| `image_qwen_image_distill.json` | 蒸馏版本，推荐使用 |
| `image_qwen_image_layered.json` | 分层图像生成 |

> **注意**: 推荐使用蒸馏版本以获得更好的性能。原生版本使用 LoRA 时可能存在问题。

---

### Qwen-Image-Edit

**官方教程**: https://docs.comfy.org/tutorials/image/qwen/qwen-image-edit

#### 模型文件

| 类型 | 文件名 | 存放目录 | 下载链接 |
|------|--------|----------|----------|
| CLIP | `qwen_2.5_vl_7b_fp8_scaled.safetensors` | `text_encoders/` | [HuggingFace](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors) |
| VAE | `qwen_image_vae.safetensors` | `vae/` | [HuggingFace](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors) |
| DiT | `qwen_image_edit_fp8_e4m3fn.safetensors` | `diffusion_models/` | [HuggingFace](https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_edit_fp8_e4m3fn.safetensors) |

#### 模型存放位置

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

> CLIP 和 VAE 与 Qwen-Image 共用，DiT 模型为 Qwen-Image-Edit 专用。

#### 工作流文件

| 工作流 | 说明 |
|--------|------|
| `image_qwen_image_edit.json` | 标准图像编辑工作流 |
| `image_qwen_image_edit_2509.json` | 增强版本 |

---

### Qwen-Image-Edit-2511 (Edit Plus)

**官方教程**: https://docs.comfy.org/tutorials/image/qwen/qwen-image-edit-2511

Qwen-Image-Edit-2511 是 Qwen-Image-Edit 的增强版本，通过 `TextEncodeQwenImageEditPlus` 节点支持多图参考编辑。该模型支持高级编辑场景，如图像之间的材质转换。

**主要特性**:
- **多图参考**：支持最多 3 张参考图像，用于复杂编辑任务
- **材质转换**：将纹理、图案或材质从一张图像转移到另一张图像

#### 模型文件

| 类型 | 文件名 | 存放目录 | 下载链接 |
|------|--------|----------|----------|
| CLIP | `qwen_2.5_vl_7b_fp8_scaled.safetensors` | `text_encoders/` | [HuggingFace](https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors) |
| VAE | `qwen_image_vae.safetensors` | `vae/` | [HuggingFace](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors) |
| DiT | `qwen_image_edit_2511_bf16.safetensors` | `diffusion_models/` | [HuggingFace](https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_edit_2511_bf16.safetensors) |

#### 模型存放位置

```text
📂 ComfyUI/
└── 📂 models/
    ├── 📂 text_encoders/
    │   └── qwen_2.5_vl_7b_fp8_scaled.safetensors
    ├── 📂 diffusion_models/
    │   └── qwen_image_edit_2511_bf16.safetensors
    └── 📂 vae/
        └── qwen_image_vae.safetensors
```

> CLIP 和 VAE 与 Qwen-Image/Qwen-Image-Edit 共用。DiT 模型为 Qwen-Image-Edit-2511 专用。

#### 工作流设置

| 设置 | 标准 | Comfy 默认 |
|------|------|------------|
| Steps | 40 | 20 |
| CFG | 4.0 | 4.0 |

> **注意**: 默认工作流使用 20 步。使用 40 步可获得更好效果（生成时间更长）。

#### 工作流文件

| 工作流 | 说明 |
|--------|------|
| `image_qwen_image_edit_2511.json` | 多图参考编辑工作流 (Edit Plus) |

#### 使用提示

1. **图像 1**: 要编辑的源图像
2. **图像 2**: 用于材质/纹理转换的参考图像
3. **图像 3** (可选): 额外的参考图像
4. **提示词**: 描述编辑操作（例如："将图像 1 中家具的皮革材质更改为图像 2 中的毛皮材质"）

---

### Qwen-Image-Layered

**官方教程**: https://docs.comfy.org/tutorials/image/qwen/qwen-image-layered

Qwen-Image-Layered 是阿里巴巴 Qwen 团队开发的模型，能够将图像分解为多个 RGBA 图层。这种分层表示解锁了固有的可编辑性：每个图层可以独立操作而不影响其他内容。

**主要特性**:
- **固有可编辑性**：每个图层可以独立操作而不影响其他内容
- **高保真基础操作**：支持语义组件的物理隔离，实现调整大小、重新定位和重新着色
- **可变图层分解**：不限于固定数量的图层 - 可根据需要分解为 3、4、5 或更多图层
- **递归分解**：任何图层都可以进一步分解，实现无限分解深度

**相关链接**:
- [Hugging Face](https://huggingface.co/Qwen/Qwen-Image-Layered)
- [研究论文](https://arxiv.org/abs/2512.15603)
- [博客](https://qwenlm.github.io/blog/qwen-image-layered/)

#### 模型文件

| 类型 | 文件名 | 存放目录 | 下载链接 |
|------|--------|----------|----------|
| CLIP | `qwen_2.5_vl_7b_fp8_scaled.safetensors` | `text_encoders/` | [HuggingFace](https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors) |
| VAE | `qwen_image_layered_vae.safetensors` | `vae/` | [HuggingFace](https://huggingface.co/Comfy-Org/Qwen-Image-Layered_ComfyUI/resolve/main/split_files/vae/qwen_image_layered_vae.safetensors) |
| DiT | `qwen_image_layered_fp8mixed.safetensors` | `diffusion_models/` | [HuggingFace](https://huggingface.co/Comfy-Org/Qwen-Image-Layered_ComfyUI/blob/main/split_files/diffusion_models/qwen_image_layered_fp8mixed.safetensors) |

#### 模型存放位置

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

#### 工作流设置

| 设置 | 默认值 | 推荐值 | 说明 |
|------|--------|--------|------|
| Steps | 20 | 50 | 原始设置为 50，但 20 可加快生成速度 |
| CFG | 2.5 | 4.0 | 原始设置为 4.0 |
| 输入尺寸 | - | 640px | 高分辨率使用 1024px |
| 图层数 | 2-3 | 2-8 | 可变图层分解 |

> **注意**: 该模型运行较慢。使用原始采样设置（steps: 50, CFG: 4.0）将至少使生成时间翻倍。

#### 提示词（可选）

文本提示词用于描述输入图像的整体内容，包括可能被部分遮挡的元素（例如，您可以指定隐藏在前景对象后面的文字）。它不是用来明确控制各个图层的语义内容。

#### 工作流文件

| 工作流 | 说明 |
|--------|------|
| `image_qwen_image_layered.json` | 图像到图层分解工作流 |

---

### Stable Diffusion 3.5

**官方教程**: https://comfyanonymous.github.io/ComfyUI_examples/sd3/

#### 模型文件

| 类型 | 文件名 | 存放目录 |
|------|--------|----------|
| Checkpoint | `sd3.5_medium.safetensors` | `checkpoints/` |
| ControlNet | `sd3.5_large_controlnet_canny.safetensors` | `controlnet/` |

#### 模型存放位置

```text
📂 ComfyUI/
└── 📂 models/
    ├── 📂 checkpoints/
    │   └── sd3.5_medium.safetensors
    └── 📂 controlnet/
        └── sd3.5_large_controlnet_canny.safetensors
```

#### 工作流文件

| 工作流 | 说明 |
|--------|------|
| `image_sd3.5_simple_example.json` | 简单文生图工作流 |
| `image_sd3.5_midium.json` | Medium 模型工作流 |
| `image_sd3.5_large_canny_controlnet_example.json` | Large 模型 + Canny ControlNet |

---

### Z-Image-Turbo

**官方教程**: https://docs.comfy.org/tutorials/image/z-image/z-image-turbo

#### 模型文件

| 类型 | 文件名 | 存放目录 | 下载链接 |
|------|--------|----------|----------|
| CLIP | `qwen_3_4b.safetensors` | `text_encoders/` | [HuggingFace](https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors) |
| VAE | `ae.safetensors` | `vae/` | [HuggingFace](https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors) |
| UNet | `z_image_turbo_bf16.safetensors` | `diffusion_models/` | [HuggingFace](https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors) |

#### 模型存放位置

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

#### 工作流文件

| 工作流 | 说明 |
|--------|------|
| `image_z_image_turbo.json` | 基础文生图工作流 |

---

### Flux.1 Kontext Dev

**官方教程**: https://docs.comfy.org/tutorials/flux/flux-1-kontext-dev

#### 模型文件

| 类型 | 文件名 | 存放目录 | 下载链接 |
|------|--------|----------|----------|
| CLIP-L | `clip_l.safetensors` | `text_encoders/` | [HuggingFace](https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors) |
| T5-XXL | `t5xxl_fp8_e4m3fn_scaled.safetensors` | `text_encoders/` | [HuggingFace](https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn_scaled.safetensors) |
| VAE | `ae.safetensors` | `vae/` | [HuggingFace](https://huggingface.co/Comfy-Org/Lumina_Image_2.0_Repackaged/resolve/main/split_files/vae/ae.safetensors) |
| UNet | `flux1-dev-kontext_fp8_scaled.safetensors` | `diffusion_models/` | [HuggingFace](https://huggingface.co/Comfy-Org/flux1-kontext-dev_ComfyUI/resolve/main/split_files/diffusion_models/flux1-dev-kontext_fp8_scaled.safetensors) |

#### 模型存放位置

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

#### 工作流文件

| 工作流 | 说明 |
|--------|------|
| `image_flux_kontext_dev_basic.json` | 多图参考基础工作流 |
| `image_flux_controlnet_example.json` | Flux ControlNet 工作流 |

---

## 视频生成模型

### Wan2.2

**官方教程**: https://docs.comfy.org/tutorials/video/wan/wan2_2

#### 模型文件

##### Text+Image to Video (5B)

| 类型 | 文件名 | 存放目录 | 下载链接 |
|------|--------|----------|----------|
| UNet | `wan2.2_ti2v_5B_fp16.safetensors` | `diffusion_models/` | [HuggingFace](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors) |
| CLIP | `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | `text_encoders/` | [HuggingFace](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors) |
| VAE | `wan2.2_vae.safetensors` | `vae/` | [HuggingFace](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan2.2_vae.safetensors) |

##### Text to Video (14B)

| 类型 | 文件名 | 存放目录 | 下载链接 |
|------|--------|----------|----------|
| UNet (High Noise) | `wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors` | `diffusion_models/` | [HuggingFace](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors) |
| UNet (Low Noise) | `wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors` | `diffusion_models/` | [HuggingFace](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors) |
| VAE | `wan_2.1_vae.safetensors` | `vae/` | [HuggingFace](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors) |


##### Image to Video (14B)

| 类型 | 文件名 | 存放目录 | 下载链接 |
|------|--------|----------|----------|
| UNet | `wan2.2_i2v_14B_fp8_scaled.safetensors` | `diffusion_models/` | [HuggingFace](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_14B_fp8_scaled.safetensors) |

#### 模型存放位置

##### Text+Image to Video (5B)

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

##### Text to Video (14B)

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

##### Image to Video (14B)

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

#### 工作流文件

| 工作流 | 说明 |
|--------|------|
| `video_wan2_2_5B_ti2v.json` | 5B 文本+图像生成视频 |
| `video_wan2_2_14B_t2v.json` | 14B 文本生成视频 |
| `video_wan2_2_14B_t2v_rapid_aio_multi_xpu.json` | 14B 文生视频 + 多 XPU 支持 (Raylight) |
| `video_wan2.2_14B_i2v_rapid_aio_multi_xpu.json` | 14B 图生视频 + 多 XPU 支持 |
| `video_wan2_2_animate_basic.json` | 14B 视频动画 |

#### Multi-XPU 配置 (Raylight)

使用 [WAN2.2-14B-Rapid-AllInOne](https://huggingface.co/Phr00t/WAN2.2-14B-Rapid-AllInOne) 与 [Raylight](https://github.com/komikndr/raylight) 实现多 GPU 加速：

1. **模型加载**
   - `Load Diffusion Model (Ray)` 节点加载 WAN2.2-14B-Rapid-AllInOne 的扩散模型
   - `Load VAE` 节点加载 VAE
   - `Load CLIP` 节点加载 `umt5_xxl_fp8_e4m3fn_scaled.safetensors`

2. **Ray 配置**
   - 在 `Ray Init Actor` 节点中设置 `GPU` 和 `ulysses_degree` 为要使用的 GPU 数量

3. **执行工作流**
   - 点击 `Run` 按钮或使用快捷键 `Ctrl + Enter`

> **提示**: 模型权重可从 [ModelScope](https://modelscope.cn/models/Phr00t/WAN2.2-14B-Rapid-AllInOne/files) 获取。可能需要使用 `tools/extract.py` 提取 UNet 和 VAE 部分。

---

### HunyuanVideo 1.5

**官方教程**: https://docs.comfy.org/tutorials/video/hunyuan/hunyuan-video-1-5

#### 模型文件

| 类型 | 文件名 | 存放目录 | 下载链接 |
|------|--------|----------|----------|
| CLIP (Qwen) | `qwen_2.5_vl_7b_fp8_scaled.safetensors` | `text_encoders/` | [HuggingFace](https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors) |
| CLIP (ByT5) | `byt5_small_glyphxl_fp16.safetensors` | `text_encoders/` | [HuggingFace](https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged/resolve/main/split_files/text_encoders/byt5_small_glyphxl_fp16.safetensors) |
| UNet (T2V) | `hunyuanvideo1.5_720p_t2v_fp16.safetensors` | `diffusion_models/` | [HuggingFace](https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged/resolve/main/split_files/diffusion_models/hunyuanvideo1.5_720p_t2v_fp16.safetensors) |
| UNet (I2V) | `hunyuanvideo1.5_720p_i2v_fp16.safetensors` | `diffusion_models/` | [HuggingFace](https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged/resolve/main/split_files/diffusion_models/hunyuanvideo1.5_720p_i2v_fp16.safetensors) |
| UNet (1080p SR) | `hunyuanvideo1.5_1080p_sr_distilled_fp16.safetensors` | `diffusion_models/` | [HuggingFace](https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged/resolve/main/split_files/diffusion_models/hunyuanvideo1.5_1080p_sr_distilled_fp16.safetensors) |
| VAE | `hunyuanvideo15_vae_fp16.safetensors` | `vae/` | [HuggingFace](https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged/resolve/main/split_files/vae/hunyuanvideo15_vae_fp16.safetensors) |

#### 模型存放位置

```text
📂 ComfyUI/
└── 📂 models/
    ├── 📂 text_encoders/
    │   ├── qwen_2.5_vl_7b_fp8_scaled.safetensors
    │   └── byt5_small_glyphxl_fp16.safetensors
    ├── 📂 diffusion_models/
    │   ├── hunyuanvideo1.5_720p_t2v_fp16.safetensors (T2V)
    │   ├── hunyuanvideo1.5_720p_i2v_fp16.safetensors (I2V)
    │   └── hunyuanvideo1.5_1080p_sr_distilled_fp16.safetensors (可选，用于超分)
    └── 📂 vae/
        └── hunyuanvideo15_vae_fp16.safetensors
```

#### 工作流文件

| 工作流 | 说明 |
|--------|------|
| `video_hunyuan_video_1.5_t2v.json` | 文本生成视频 |
| `video_hunyuan_video_1.5_i2v.json` | 图像生成视频 |
| `video_hunyuan_video_1.5_i2v_multi_xpu.json` | 图生视频 + 多 XPU 支持 (Raylight) |

> **注意**: 默认参数配置针对 480p FP8 图生视频进行了优化。

#### Multi-XPU 配置 (Raylight)

对于 `video_hunyuan_video_1.5_i2v_multi_xpu.json`，使用 [Raylight](https://github.com/komikndr/raylight) 实现多 GPU 加速：

##### Multi-XPU 额外模型文件

| 类型 | 文件名 | 存放目录 | 下载链接 |
|------|--------|----------|----------|
| UNet (480p I2V Distilled) | `hunyuanvideo1.5_480p_i2v_cfg_distilled_fp8_scaled.safetensors` | `diffusion_models/` | [HuggingFace](https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged/resolve/main/split_files/diffusion_models/hunyuanvideo1.5_480p_i2v_cfg_distilled_fp8_scaled.safetensors) |
| CLIP Vision | `sigclip_vision_patch14_384.safetensors` | `clip_vision/` | [HuggingFace](https://huggingface.co/Comfy-Org/sigclip_vision_384/resolve/main/sigclip_vision_patch14_384.safetensors) |

##### 模型存放位置 (Multi-XPU)

```text
📂 ComfyUI/
└── 📂 models/
    ├── 📂 text_encoders/
    │   ├── qwen_2.5_vl_7b_fp8_scaled.safetensors
    │   └── byt5_small_glyphxl_fp16.safetensors
    ├── 📂 diffusion_models/
    │   └── hunyuanvideo1.5_480p_i2v_cfg_distilled_fp8_scaled.safetensors
    ├── 📂 clip_vision/
    │   └── sigclip_vision_patch14_384.safetensors
    └── 📂 vae/
        └── hunyuanvideo15_vae_fp16.safetensors
```

##### Ray 配置

1. **Ray 初始化器**
   - 在 `RayInitializer` 节点中配置 `GPU` 数量和并行设置（`ulysses_degree`、`ring_degree`）
   - 设置为您要使用的 GPU 数量（例如 2 或 4）

2. **模型加载**
   - `RayUNETLoader` 节点使用 Ray 分布式支持加载扩散模型
   - `DualCLIPLoader` 加载文本编码器（Qwen + ByT5）
   - `VAELoader` 加载 VAE
   - `CLIPVisionLoader` 加载 CLIP Vision 模型用于图像条件

3. **采样**
   - `XFuserSamplerCustom` 在多个 GPU 上执行分布式采样
   - `RayModelSamplingSD3` 配置模型采样参数

---

### LTX-2

**官方教程**: https://blog.comfy.org/p/ltx-2-open-source-audio-video-ai

#### 模型文件

| 类型 | 文件名 | 存放目录 | 下载链接 |
|------|--------|----------|----------|
| Checkpoint | `ltx-2-19b-dev-fp8.safetensors` | `checkpoints/` | [HuggingFace](https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-dev-fp8.safetensors) |
| Text Encoder | `gemma_3_12B_it_fp4_mixed.safetensors` | `text_encoders/` | [HuggingFace](https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors) |
| Upscaler | `ltx-2-spatial-upscaler-x2-1.0.safetensors` | `latent_upscale_models/` | [HuggingFace](https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-spatial-upscaler-x2-1.0.safetensors) |
| LoRA | `ltx-2-19b-distilled-lora-384.safetensors` | `loras/` | [HuggingFace](https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-distilled-lora-384.safetensors) |

#### 模型存放位置

```text
📂 ComfyUI/
└── 📂 models/
    ├── 📂 checkpoints/
    │   └── ltx-2-19b-dev-fp8.safetensors
    ├── 📂 text_encoders/
    │   └── gemma_3_12B_it_fp4_mixed.safetensors
    ├── 📂 latent_upscale_models/
    │   └── ltx-2-spatial-upscaler-x2-1.0.safetensors
    └── 📂 loras/
        └── ltx-2-19b-distilled-lora-384.safetensors
```

#### 工作流文件

| 工作流 | 说明 |
|--------|------|
| `video_ltx2_t2v.json` | 文本生视频（含音频/动作） |
| `video_ltx2_i2v.json` | 图像生视频（含音频/动作） |
| `video_ltx_2_19B_t2v_distilled.json` | 文本生视频蒸馏工作流 |
| `video_ltx_2_19B_i2v_distilled.json` | 图像生视频蒸馏工作流 |

---

## 3D 生成模型

### Hunyuan3D 2.1

#### 模型文件

Hunyuan3D 使用自定义节点 `ComfyUI-Hunyuan3d-2-1`，模型会自动下载或需手动放置于指定目录。

| 类型 | 目录/文件 |
|------|----------|
| Turbo 模型 | `hunyuan3d/hunyuan3d-turbo-v2/` |
| Paint 模型 | `hunyuan3d/hunyuan3d-paint-v2-1/` |
| VAE | 自动加载 |

#### 模型存放位置

```text
📂 ComfyUI/
└── 📂 models/
    └── 📂 hunyuan3d/
        ├── 📂 hunyuan3d-turbo-v2/
        │   └── (模型文件)
        └── 📂 hunyuan3d-paint-v2-1/
            └── (模型文件)
```

#### 工作流文件

| 工作流 | 说明 |
|--------|------|
| `3d_hunyuan3d.json` | 文本/图像生成 3D 网格 |

#### 使用说明

1. 安装 `ComfyUI-Hunyuan3d-2-1` 自定义节点
2. 加载工作流文件
3. 输入文本提示或上传图像
4. 执行工作流生成 3D 模型
5. 导出为 GLB 格式

---

## 音频生成模型

### VoxCPM1.5

VoxCPM1.5 是一种无分词器的 TTS 系统，支持上下文感知语音生成和真实声音克隆。

#### 模型信息

| 模型 | 参数量 | HuggingFace 链接 |
|------|--------|------------------|
| VoxCPM1.5 | 800M | [openbmb/VoxCPM1.5](https://huggingface.co/openbmb/VoxCPM1.5) |
| VoxCPM-0.5B | 0.5B | [openbmb/VoxCPM-0.5B](https://huggingface.co/openbmb/VoxCPM-0.5B) |

#### 模型存放位置

```text
📂 ComfyUI/
└── 📂 models/
    └── 📂 tts/
        └── 📂 VoxCPM/
            └── 📂 VoxCPM1.5/
```

#### 工作流文件

| 工作流 | 说明 |
|--------|------|
| `audio_VoxCPM_example.json` | 文本转语音 |

#### 使用提示

- 提供精确的 `prompt_text`（参考音频的逐字转录）
- 使用准确的标点符号捕捉说话者的语调
- 建议使用 5-15 秒的清晰参考音频
- 避免背景噪音、混响或音乐

---

### IndexTTS 2

IndexTTS 2 是一种声音克隆 TTS 模型，使用单个参考音频文件合成新语音。

#### 模型目录结构

```text
TTS/
├── bigvgan_v2_22khz_80band_256x/
│   ├── bigvgan_generator.pt
│   └── config.json
├── campplus/
│   └── campplus_cn_common.bin
├── IndexTTS-2/
│   ├── bpe.model
│   ├── config.yaml
│   ├── feat1.pt
│   ├── feat2.pt
│   ├── gpt.pth
│   ├── s2mel.pth
│   ├── wav2vec2bert_stats.pt
│   └── qwen0.6bemo4-merge/
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer.json
│       └── ...
├── MaskGCT/
│   └── semantic_codec/
│       └── model.safetensors
└── w2v-bert-2.0/
    ├── config.json
    ├── conformer_shaw.pt
    ├── model.safetensors
    └── ...
```

#### 所需模型下载

| 模型 | HuggingFace 链接 |
|------|------------------|
| IndexTTS-2 | [IndexTeam/IndexTTS-2](https://huggingface.co/IndexTeam/IndexTTS-2) |
| BigVGAN | [nvidia/bigvgan_v2_22khz_80band_256x](https://huggingface.co/nvidia/bigvgan_v2_22khz_80band_256x) |
| CAMPPlus | [funasr/campplus](https://huggingface.co/funasr/campplus) |
| MaskGCT | [amphion/MaskGCT](https://huggingface.co/amphion/MaskGCT) |
| W2V-BERT 2.0 | [facebook/w2v-bert-2.0](https://huggingface.co/facebook/w2v-bert-2.0) |

#### 模型存放位置

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
        │       └── (模型文件)
        ├── 📂 MaskGCT/
        │   └── 📂 semantic_codec/
        │       └── model.safetensors
        └── 📂 w2v-bert-2.0/
            ├── config.json
            ├── conformer_shaw.pt
            └── model.safetensors

```

#### 工作流文件

| 工作流 | 说明 |
|--------|------|
| `audio_indextts2.json` | 声音克隆 |

---

## 视频超分模型

### SeedVR2

SeedVR2 是一个基于扩散模型的视频超分辨率模型，可对视频进行修复和超清放大。

#### 模型存放位置

```text
📂 ComfyUI/
└── 📂 models/
    └── 📂 SEEDVR2/
        ├── seedvr2_ema_3b_fp8_e4m3fn.safetensors  （DiT）
        └── ema_vae_fp16.safetensors               （VAE）
```

> 模型**首次使用时自动下载**。也可手动从 [numz/SeedVR2_comfyUI](https://huggingface.co/numz/SeedVR2_comfyUI)。

---

## 常见问题

### 1. 如何下载模型？

ComfyUI 内置模型下载功能，当加载工作流时会自动提示下载缺失的模型。也可以手动从 HuggingFace 下载并放置到相应目录。

### 2. 模型加载失败？

- 检查模型文件是否完整下载
- 确认模型文件放置在正确的目录
- 检查文件权限

### 3. 生成速度慢？

- 使用 FP8 量化模型
- 启用多 XPU 支持（对于支持的工作流）
- 降低输出分辨率或帧数

### 4. 内存不足？

- 使用 `--lowvram` 或 `--novram` 启动参数
- 减少 batch size
- 使用量化模型

### 5. 如何添加自定义节点？

通过 ComfyUI Manager 安装，或手动克隆到 `custom_nodes/` 目录：

```bash
cd ComfyUI/custom_nodes
git clone <node_repo_url>
```

---

## 相关链接

- [Omni 项目主页](../README.md)
- [工作流文件目录](../workflows/)
- [补丁文件](../patches/)
- [工具脚本](../tools/)
- [English Documentation](./ComfyUI_Guide.md)

