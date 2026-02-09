# omni_xpu_kernel (llm-scaler) Build and Install Notes

This document summarizes how to build the updated omni_xpu_kernel from llm-scaler on Windows using a conda environment, then install it into the ComfyUI embedded Python environment.

## 1. Prerequisites
- Intel oneAPI Base Toolkit installed
- VS2022 C++ components installed
- conda environment: omni_env
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
```

## 2. Build (in conda environment)
Build script location:
- [scripts/build_llm_scaler_conda.cmd](scripts/build_llm_scaler_conda.cmd)

Script contents (full example, with variables):

```
@echo off
set "VS2022INSTALLDIR=C:\Program Files\Microsoft Visual Studio\18\Community"
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
call C:\ProgramData\miniforge3\Scripts\activate.bat omni_env
set "WORKSPACE=C:\workspace"
set "LLM_SCALER_DIR=%WORKSPACE%\llm-scaler"
set "EMBED_PYTHON_DIR=%WORKSPACE%\omni\comfyui_windows_setup\python_embeded"
set "OUTPUT_DIR=%WORKSPACE%\llm_scaler_dist"
set "PATH=%CONDA_PREFIX%\Library\bin;%CONDA_PREFIX%\Lib\site-packages\torch\lib;%PATH%"
cd /d %LLM_SCALER_DIR%\omni\omni_xpu_kernel
python -m pip wheel . -w %OUTPUT_DIR% --no-build-isolation --no-deps > %WORKSPACE%\build_log.txt 2>&1
```

How to run the .cmd file (common options):
- Double-click it in File Explorer.
- From Command Prompt:
  - `cmd /c scripts\build_llm_scaler_conda.cmd`
- From PowerShell:
  - `cmd /c "scripts\build_llm_scaler_conda.cmd"`

Key points:
- Initialize oneAPI: setvars.bat
- Activate conda: omni_env
- Set torch DLL search path
- Produce wheel to %OUTPUT_DIR%

Output:
- %OUTPUT_DIR%\omni_xpu_kernel-0.1.0-cp312-cp312-win_amd64.whl

## 3. Install the newly built wheel (embedded Python)
Use --no-deps to avoid pulling dependencies again:

 - Wheel path:
  - %OUTPUT_DIR%\omni_xpu_kernel-0.1.0-cp312-cp312-win_amd64.whl
 - Install command:
  - pip install --force-reinstall --no-deps <wheel>

Command example (with variables):

```
"%EMBED_PYTHON_DIR%\python.exe" -m pip install --force-reinstall --no-deps "%OUTPUT_DIR%\omni_xpu_kernel-0.1.0-cp312-cp312-win_amd64.whl"
```

## 4. Verify (optional)
- Import omni_xpu_kernel
- Check __version__

Command example (with variables):

```
"%EMBED_PYTHON_DIR%\python.exe" -c "import omni_xpu_kernel as ok, importlib.metadata as im; print('omni_xpu_kernel:', ok); print('version:', im.version('omni-xpu-kernel'))"
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

 - Reference script:
  - [setup_portable_env.bat](../../../omni/comfyui_windows_setup/setup_portable_env.bat)
 - Key versions:
  - torch==2.9.0+xpu
  - torchvision==0.24.0+xpu
  - torchaudio==2.9.0+xpu
  - --index-url https://download.pytorch.org/whl/xpu

Command example (with variables):

```
"%EMBED_PYTHON_DIR%\python.exe" -m pip install torch==2.9.0+xpu torchvision==0.24.0+xpu torchaudio==2.9.0+xpu --index-url https://download.pytorch.org/whl/xpu
```
