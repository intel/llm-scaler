# omni_xpu_kernel (llm-scaler) Build and Install Notes

This document summarizes how to build the updated omni_xpu_kernel from llm-scaler on Windows using a conda environment, then install it into the ComfyUI embedded Python environment.

## 1. Prerequisites
- Intel oneAPI Base Toolkit installed
- VS2022 C++ components installed
- conda environment: omni_env
- PyTorch XPU 2.10.x, 2.11.x, or 2.12.x installed in the build environment
- ComfyUI embedded Python located at: %EMBED_PYTHON_DIR%
- Variables (adjust as needed):
  - %WORKSPACE%: workspace root
  - %EMBED_PYTHON_DIR%: embedded Python directory (example: %WORKSPACE%\omni\comfyui_windows_setup\python_embeded)
  - %LLM_SCALER_DIR%: llm-scaler repo directory (example: %WORKSPACE%\llm-scaler)
  - %OUTPUT_DIR%: build output directory (example: %WORKSPACE%\llm_scaler_dist)

### Variable setup example (cmd)
You can put these variables at the top of your command prompt or script for consistent configuration:

```
set "WORKSPACE=C:\workspace"
set "LLM_SCALER_DIR=%WORKSPACE%\llm-scaler"
set "EMBED_PYTHON_DIR=%WORKSPACE%\omni\comfyui_windows_setup\python_embeded"
set "OUTPUT_DIR=%WORKSPACE%\llm_scaler_dist"
set "OMNI_XPU_DEVICE=bmg"
```

## 2. Build in the configured conda environment

Initialize Visual Studio and oneAPI using the paths from the local
installation, then activate the conda environment that contains the selected
PyTorch XPU version. The repository does not provide a machine-specific conda
wrapper because the Visual Studio edition, oneAPI location, environment name,
workspace, and output path are host configuration.

Example commands:

```
@echo off
call "C:\path\to\VisualStudio\Common7\Tools\VsDevCmd.bat"
call "C:\path\to\oneAPI\setvars.bat"
call C:\ProgramData\miniforge3\Scripts\activate.bat omni_env
set "OMNI_XPU_REQUIRE_CUTE=0"
set "PATH=%CONDA_PREFIX%\Library\bin;%CONDA_PREFIX%\Lib\site-packages\torch\lib;%PATH%"
cd /d %LLM_SCALER_DIR%\omni\omni_xpu_kernel
python -m pip wheel . --wheel-dir %OUTPUT_DIR% --no-build-isolation --no-deps
```

Key points:
- Initialize oneAPI: setvars.bat
- Activate conda: omni_env
- Explicitly select a core-only build because CUTE FMHA is Linux-only
- Set `OMNI_XPU_DEVICE` to `bmg` or `ptl-h` for the destination GPU
- Set torch DLL search path
- Produce wheel to %OUTPUT_DIR%

Output:
- Torch 2.10.x/BMG: %OUTPUT_DIR%\omni_xpu_kernel-0.1.0b9.dev0+torch210.bmg-cp312-cp312-win_amd64.whl
- Torch 2.11.x/BMG: %OUTPUT_DIR%\omni_xpu_kernel-0.1.0b9.dev0+torch211.bmg-cp312-cp312-win_amd64.whl
- Torch 2.12.x/BMG: %OUTPUT_DIR%\omni_xpu_kernel-0.1.0b9.dev0+torch212.bmg-cp312-cp312-win_amd64.whl

PTL-H builds use the corresponding `.ptlh` local-version component. Build one
artifact for every Torch/GPU target pair rather than renaming a BMG wheel.

The build detects the installed Torch version automatically. Build one wheel
per Torch environment and GPU target, and install it only into an embedded
environment with the same Torch public version and GPU architecture.

## 3. Install the newly built wheel (embedded Python)
Use --no-deps to avoid pulling dependencies again:

 - Wheel path: select the matching Torch and `.bmg`/`.ptlh` artifact
 - Install command:
  - pip install --force-reinstall --no-deps <wheel>

Command example (with variables):

```
set "TORCH_TAG=torch211"
set "XPU_TAG=bmg"
"%EMBED_PYTHON_DIR%\python.exe" -m pip install --force-reinstall --no-deps "%OUTPUT_DIR%\omni_xpu_kernel-0.1.0b9.dev0+%TORCH_TAG%.%XPU_TAG%-cp312-cp312-win_amd64.whl"
```

## 4. Verify (optional)
- Import omni_xpu_kernel
- Check __version__

Command example (with variables):

```
"%EMBED_PYTHON_DIR%\python.exe" -c "import omni_xpu_kernel as ok, importlib.metadata as im; print('version:', im.version('omni-xpu-kernel')); print('target:', ok.__xpu_target__)"
```

---

To update or rebuild, repeat steps 2 → 3.

---

## FAQ

### 1) XPU symbol link errors (c10::xpu::XPUStream)
Add missing link libraries in llm-scaler code:

 - Source file:
  - [setup.py](setup.py)

 - Add on Windows:
  - torch_xpu.lib
  - c10_xpu.lib

 - Add on Linux:
  - -ltorch_xpu
  - -lc10_xpu

### 2) Restore embedded PyTorch to XPU build
Reinstall the XPU build of PyTorch in embedded Python (to avoid fallback to standard builds):

Choose Torch 2.10.x, 2.11.x, or 2.12.x and use the same exact public version
in both the build and embedded environments. Torch 2.9 is not supported by
this kernel release.

Command example (with variables):

```
set "TORCH_VERSION=2.11.0"
"%EMBED_PYTHON_DIR%\python.exe" -m pip install "torch==%TORCH_VERSION%+xpu" torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
```
