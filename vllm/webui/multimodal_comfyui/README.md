# Qwen2.5-VL-3B-Instruct Deployment Guide (ComfyUI + Intel GPU + Linux)

This document provides comprehensive instructions for deploying the `Qwen2.5-VL-3.5B-Instruct` multimodal LLM on Linux systems with `Intel GPU` acceleration via `ComfyUI` workflow.

## 🛠️ Installation Procedure
### 1. Environment Setup
```bash
# Install system dependencies
sudo apt update && sudo apt install -y \
    git python3-pip python3-venv \
    ocl-icd-opencl-dev

# Configure Intel GPU drivers (if not present)
sudo apt install -y \
    intel-opencl-icd \
    intel-level-zero-gpu \
    level-zero
```

### 2. Conda Environment Configuration
```bash
conda create -n comfyqwen python=3.11 
conda activate comfyqwen
```

### 3. ComfyUI Installation
```bash
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ./ComfyUI

# Install Intel-optimized PyTorch
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/xpu

# For nightly builds with potential performance improvements:
# pip install --pre torch torchvision torchaudio \
#     --index-url https://download.pytorch.org/whl/nightly/xpu

pip install -r requirements.txt
```

### 4. Qwen2.5-VL Custom Node Deployment
```bash
# Download node definition files
git clone https://github.com/IuvenisSapiens/ComfyUI_Qwen2_5-VL-Instruct

Move the ComfyUI_Qwen2_5-VL-Instruct folder into /ComfyUI/custom_nodes/ directory

Place the downloaded Qwen2.5-VL-3B-Instruct model folder into /ComfyUI/models/prompt_generator/
# If prompt_generator subdirectory doesn't exist under models, please create it first
```
## 🚀 Launching ComfyUI
```bash
python main.py
```
Access the web interface at: `http://localhost:8188`

## Post-Installation Configuration
1. Replace the final component node with `Preview Any` in your workflow
2. Reference model path: `./models/prompt_generator/Qwen2.5-VL-3B-Instruct/`

![Workflow Example](pic/image.png)

## References
- [ComfyUI GitHub](https://github.com/comfyanonymous/ComfyUI)
- [Intel PyTorch XPU](https://intel.github.io/intel-extension-for-pytorch/)
- [Qwen2.5 Model Card](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)

