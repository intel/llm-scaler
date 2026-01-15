# ComfyUI Portable Setup for Intel XPU (Windows)

This guide provides step-by-step instructions for setting up a portable ComfyUI environment with Intel XPU support on Windows systems.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running ComfyUI](#running-comfyui)
- [Installed Custom Nodes](#installed-custom-nodes)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)

---

## Overview

This setup script creates a fully portable ComfyUI installation optimized for Intel XPU (Arc GPUs). The installation includes:

- **Python 3.12 Embedded** - Self-contained Python environment
- **PyTorch with XPU Support** - Intel GPU acceleration
- **ComfyUI** - AI image generation workflow interface
- **Essential Custom Nodes** - Pre-installed plugins for extended functionality

### Key Features

- **Portable** - No system-wide installation required
- **Intel XPU Optimized** - Hardware-accelerated on Intel GPUs
- **Pre-configured** - Ready to use out of the box
- **Custom Nodes Included** - Popular extensions pre-installed

---

## Prerequisites

### 1. Install Git for Windows

1. Download Git from [git-scm.com](https://git-scm.com/download/win)
2. Run the installer with default options
3. Verify installation by opening Command Prompt and running:
   ```cmd
   git --version
   ```

### 2. (Optional) Configure Proxy

If you are behind a corporate proxy, edit `setup_portable_env.bat` before running:

```batch
REM Uncomment and modify these lines:
set "HTTP_PROXY=http://your-proxy-server:port"
set "HTTPS_PROXY=http://your-proxy-server:port"
set "NO_PROXY=localhost,127.0.0.1"
```

---

## Installation

### Step 1: Download Setup Files

Clone the LLM Scaler repository to your desired location:

```cmd
git clone https://github.com/intel/llm-scaler.git
cd llm-scaler\omni\comfyui_windows_setup
```

The required files are located in:
```
llm-scaler\omni\
├── comfyui_windows_setup\
│   └── setup_portable_env.bat
└── patches\
    ├── comfyui_for_multi_arc.patch
    └── comfyui_gguf_xpu.patch
```

### Step 2: Run Setup Script

1. **Right-click** on `setup_portable_env.bat`
2. Select **"Run as administrator"** (recommended)
3. Wait for the installation to complete (15-30 minutes depending on internet speed)

The script will automatically:
- Download and configure Python 3.12 Embedded
- Install PyTorch with Intel XPU support
- Clone ComfyUI from official repository
- Apply Intel XPU optimization patches
- Install essential custom nodes
- Create launcher scripts

### Step 3: Verify Installation

After installation, navigate to the `comfyui_windows_setup` folder and verify that everything is set up correctly:

```cmd
python_embeded\python.exe -c "import torch; print(f'XPU available: {torch.xpu.is_available()}')"
```

Expected output:
```
XPU available: True
```

---

## Running ComfyUI

### Standard Mode

Double-click `run_comfyui.bat` to start ComfyUI with default settings.

```cmd
run_comfyui.bat
```

> **Note**: The first launch will take longer time for initialization and dependency checking. Please be patient.

### Low VRAM Mode

For GPUs with limited VRAM (6-8 GB), use the low VRAM launcher:

```cmd
run_comfyui_lowvram.bat
```

### CPU Mode

To run on CPU only (no GPU acceleration):

```cmd
run_comfyui_cpu.bat
```

### Custom Arguments

You can pass additional arguments to any launcher:

```cmd
run_comfyui.bat --listen 0.0.0.0 --port 8188
```

### Accessing the Web Interface

Once ComfyUI starts, open your web browser and navigate to:

```
http://127.0.0.1:8188
```

---

## Installed Custom Nodes

The setup script automatically installs the following custom nodes:

| Node | Description | Repository |
|------|-------------|------------|
| **ComfyUI-Manager** | Plugin manager for easy node installation | [ltdrdata/ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager) |
| **ComfyUI-GGUF** | GGUF model format support | [city96/ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) |


### Installing Additional Nodes

Use ComfyUI-Manager (pre-installed) to install additional custom nodes:

1. Open ComfyUI in your browser
2. Click **"Manager"** button in the menu
3. Select **"Install Custom Nodes"**
4. Search and install desired nodes

---

## Directory Structure

After installation, your `comfyui_windows_setup` directory will look like:

```
comfyui_windows_setup/
├── python_embeded/          # Python environment
│   ├── python.exe
│   ├── Lib/
│   │   └── site-packages/   # Installed packages
│   └── ...
├── ComfyUI/                  # ComfyUI application
│   ├── main.py
│   ├── custom_nodes/         # Custom nodes
│   │   ├── comfyui-manager/
│   │   └── ComfyUI-GGUF/
│   ├── models/               # Model files (download separately)
│   │   ├── checkpoints/
│   │   ├── loras/
│   │   ├── vae/
│   │   └── ...
│   ├── input/                # Input images
│   └── output/               # Generated images
├── run_comfyui.bat           # Standard launcher
├── run_comfyui_lowvram.bat   # Low VRAM launcher
└── run_comfyui_cpu.bat       # CPU-only launcher
```

---

## Troubleshooting

### Common Issues

#### 1. "XPU not available" Error

**Symptom**: PyTorch reports `XPU available: False`

**Solutions**:
- Try reinstalling PyTorch XPU:
  ```cmd
  python_embeded\python.exe -m pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/xpu --force-reinstall
  ```

#### 2. "Git not found" Error

**Symptom**: Setup script fails with "Git is not installed"

**Solution**:
- Install Git from [git-scm.com](https://git-scm.com/download/win)
- Restart Command Prompt after installation
- Verify with `git --version`

#### 3. Out of Memory (OOM) Error

**Symptom**: ComfyUI crashes during generation with memory errors

**Solutions**:
- Use `run_comfyui_lowvram.bat` launcher
- Reduce image resolution
- Close other GPU-intensive applications
- Enable model offloading in workflow settings

#### 4. Patch Application Failed

**Symptom**: Warning about patch already applied or conflicts

**Solutions**:
- This is usually safe to ignore if ComfyUI runs correctly
- For fresh installation, delete `ComfyUI` folder and re-run setup

#### 5. Network/Proxy Issues

**Symptom**: Downloads fail or timeout

**Solutions**:
- Configure proxy settings in the batch file
- Check firewall settings
- Try running with VPN disabled (or enabled, depending on your network)

### Getting Help

If you encounter issues not covered above:

1. Check ComfyUI logs in the terminal window for detailed error messages
2. Search existing issues:
   - [ComfyUI GitHub Issues](https://github.com/comfyanonymous/ComfyUI/issues) - For general ComfyUI problems
   - [LLM Scaler Issues](https://github.com/intel/llm-scaler/issues) - For Intel XPU specific issues
3. Submit a new issue on [LLM Scaler repository](https://github.com/intel/llm-scaler/issues) with:
   - Your system information (GPU model, driver version)
   - Complete error logs from the terminal
   - Steps to reproduce the issue

---

## FAQ

### Q: Can I move the installation to another location?

**A**: Yes! The installation is fully portable. Simply move the entire folder to a new location. The relative paths will continue to work.

### Q: How do I update ComfyUI?

**A**: 
```cmd
cd ComfyUI
git stash
git fetch origin
git pull
git stash pop
```

> **Note**: After updating, there may be patch conflicts if ComfyUI has significant changes. If the patch fails to apply, you may need to download the latest setup script and patch files, then perform a fresh installation.

### Q: Where should I put my model files?

**A**: Place model files in the appropriate subfolders under `ComfyUI/models/`:
- Checkpoints (SDXL, SD1.5, etc.) → `models/checkpoints/`
- LoRA models → `models/loras/`
- VAE files → `models/vae/`
- ControlNet models → `models/controlnet/`

### Q: Can I use this with NVIDIA GPU?

**A**: This setup is optimized for Intel XPU. For NVIDIA GPUs, use the standard ComfyUI installation with CUDA support.

### Q: How do I uninstall?

**A**: Simply delete the entire installation folder. No registry entries or system files are modified.

### Q: Is this compatible with ComfyUI workflows from the internet?

**A**: Yes! Standard ComfyUI workflows are fully compatible. Some workflows may require additional custom nodes, which can be installed via ComfyUI-Manager.

---

## Version Information

| Component | Version |
|-----------|---------|
| Python | 3.12.10 |
| PyTorch | 2.9.0+xpu |
| ComfyUI | Commit 532e285 |
| Setup Script | v1.0 |

---

## License

This setup script is provided for use with Intel XPU hardware. ComfyUI and its custom nodes are subject to their respective licenses.
