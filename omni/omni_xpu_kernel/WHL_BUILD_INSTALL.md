# omni_xpu_kernel Build and Install Notes

This document records the validated Windows flow for building the `omni_xpu_kernel` wheel from `llm-scaler`, installing it into the portable ComfyUI embedded Python environment, and verifying it with an Intel XPU ComfyUI run.

## What Was Required To Make The Windows Flow Work

The validated Windows build/install flow depended on these fixes and guardrails:

- Build with the conda environment Python, not a user-site Python.
- Force `PYTHONNOUSERSITE=1` and clear `PYTHONPATH` before importing PyTorch.
- Compile the native extension with `/DNOMINMAX` and `/DWIN32_LEAN_AND_MEAN` to avoid Windows header macro conflicts.
- Link against the XPU PyTorch libraries from the conda environment.
- Register DLL directories at import time so the built `.pyd` files can find PyTorch and oneAPI runtime DLLs on Windows.
- Install the final wheel into the ComfyUI embedded Python with `--no-deps` so the XPU PyTorch stack is not replaced.

Without those pieces, the common failures were: wrong PyTorch selected during build, native compile errors, and a wheel that built successfully but failed to import on Windows.

## Validated Local Paths

The current validated workspace uses:

```text
Workspace:        C:\workspace
Repository:       C:\workspace\llm-scaler
Kernel source:    C:\workspace\llm-scaler\omni\omni_xpu_kernel
Build script:     C:\workspace\llm-scaler\omni\omni_xpu_kernel\scripts\build_llm_scaler_conda.cmd
Wheel output:     C:\workspace\llm_scaler_dist
Build log:        C:\workspace\build_log.txt
ComfyUI:          C:\workspace\llm-scaler\omni\comfyui_windows_setup\ComfyUI
Embedded Python:  C:\workspace\llm-scaler\omni\comfyui_windows_setup\python_embeded\python.exe
```

The validated wheel was:

```text
C:\workspace\llm_scaler_dist\omni_xpu_kernel-0.1.0-cp312-cp312-win_amd64.whl
```

It contains the native Python 3.12 Windows extensions, including `_C.cp312-win_amd64.pyd` and `lgrf_sdp.cp312-win_amd64.pyd`.

## Prerequisites

Install or prepare:

- Intel oneAPI Base Toolkit
- Visual Studio 2022 C++ build tools
- Miniforge or conda
- A conda environment named `omni_env`
- XPU PyTorch in the conda build environment
- The portable ComfyUI embedded Python environment if you want to install and test the wheel immediately

The current build script expects:

```text
C:\Program Files (x86)\Intel\oneAPI\setvars.bat
C:\ProgramData\miniforge3\Scripts\activate.bat
C:\Program Files\Microsoft Visual Studio\18\Community
```

Adjust `scripts\build_llm_scaler_conda.cmd` if your oneAPI, conda, Visual Studio, workspace, or output paths differ.

## Build The Wheel

From PowerShell:

```powershell
Set-Location C:\workspace
cmd /c "C:\workspace\llm-scaler\omni\omni_xpu_kernel\scripts\build_llm_scaler_conda.cmd"
```

From Command Prompt:

```cmd
cd /d C:\workspace
cmd /c "C:\workspace\llm-scaler\omni\omni_xpu_kernel\scripts\build_llm_scaler_conda.cmd"
```

The script does the important environment setup before invoking `pip wheel`:

```cmd
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
call C:\ProgramData\miniforge3\Scripts\activate.bat omni_env
set "PYTHONNOUSERSITE=1"
set "PYTHONPATH="
set "CONDA_PYTHON=%CONDA_PREFIX%\python.exe"
set "TORCH_LIB=%CONDA_PREFIX%\Lib\site-packages\torch\lib"
set "OMNI_XPU_DEVICE=bmg"
set "PATH=%CONDA_PREFIX%;%CONDA_PREFIX%\Scripts;%CONDA_PREFIX%\Library\bin;%TORCH_LIB%;%ONEDNN_LIB%;%PATH%"
"%CONDA_PYTHON%" -m pip wheel . -w "C:\workspace\llm_scaler_dist" --no-build-isolation --no-deps
```

Before building, it prints the Python executable, PyTorch version/path, and whether the XPU header exists:

```text
c10\xpu\impl\xpu_cmake_macros.h
```

This check is useful because one common failure mode is accidentally importing a user-site PyTorch instead of the conda environment PyTorch. The script prevents that with `PYTHONNOUSERSITE=1`, an empty `PYTHONPATH`, and an explicit `%CONDA_PREFIX%\python.exe`.

Expected output:

```text
Wheel build succeeded. Output directory: C:\workspace\llm_scaler_dist
omni_xpu_kernel-0.1.0-cp312-cp312-win_amd64.whl
```

The validated wheel contents included at least:

```text
omni_xpu_kernel/_C.cp312-win_amd64.pyd
omni_xpu_kernel/lgrf_sdp.cp312-win_amd64.pyd
```

If the build fails, inspect:

```text
C:\workspace\build_log.txt
```

The script prints the last 120 log lines automatically on failure.

## Install Into Portable ComfyUI Python

Install the wheel into the embedded Python environment with `--no-deps` so the ComfyUI PyTorch XPU stack is not replaced:

```cmd
"C:\workspace\llm-scaler\omni\comfyui_windows_setup\python_embeded\python.exe" -m pip install --force-reinstall --no-deps "C:\workspace\llm_scaler_dist\omni_xpu_kernel-0.1.0-cp312-cp312-win_amd64.whl"
```

The ComfyUI setup script also performs this step automatically. It uses this default wheel path:

```text
C:\workspace\llm_scaler_dist\omni_xpu_kernel-0.1.0-cp312-cp312-win_amd64.whl
```

To override it before running `setup_portable_env.bat`:

```cmd
set "OMNI_XPU_KERNEL_WHEEL=D:\path\to\omni_xpu_kernel-0.1.0-cp312-cp312-win_amd64.whl"
setup_portable_env.bat
```

This is the same integration path used by the validated Windows ComfyUI setup: build the wheel first, then let the portable setup consume that local wheel.

## Verify The Installed Wheel

Use the embedded Python, not the system Python:

```cmd
"C:\workspace\llm-scaler\omni\comfyui_windows_setup\python_embeded\python.exe" -c "import omni_xpu_kernel as ok; print(ok.__version__); print('available:', ok.is_available())"
```

Expected output:

```text
0.1.0
available: True
```

Also verify PyTorch XPU from the same embedded Python:

```cmd
"C:\workspace\llm-scaler\omni\comfyui_windows_setup\python_embeded\python.exe" -c "import torch; print(torch.__version__); print(torch.xpu.is_available())"
```

Validated output included:

```text
2.9.0+xpu
True
```

If `ok.is_available()` is `False`, do not continue to ComfyUI startup yet. Fix the embedded runtime first, because ComfyUI will only inherit the same broken environment.

## Why Windows DLL Search Setup Matters

On Windows Python 3.8 and newer, modifying `PATH` is not always enough for `.pyd` dependencies. The package initialization registers runtime DLL directories before importing the native extension. This is required for dependencies from locations such as:

```text
python_embeded\Lib\site-packages\torch\lib
python_embeded\Library\bin
Intel oneAPI runtime directories
```

Without this, importing `omni_xpu_kernel` can fail even when the dependent DLLs are visible in `PATH`.

## ComfyUI Validation

After installing the wheel and setting up ComfyUI, start the portable server from:

```text
C:\workspace\llm-scaler\omni\comfyui_windows_setup
```

Example manual launch on port `8190`:

```cmd
set "PYTHONNOUSERSITE=1"
set "PYTHONPATH="
set "PYTHONHOME="
set "PATH=%CD%\python_embeded;%CD%\python_embeded\Scripts;%CD%\python_embeded\Library\bin;%PATH%"
cd ComfyUI
..\python_embeded\python.exe main.py --listen 127.0.0.1 --port 8190
```

The validated startup log showed:

```text
pytorch version: 2.9.0+xpu
Device: xpu:0 Intel(R) Arc(TM) 140V GPU (16GB)
[omni_xpu_kernel] Loaded successfully
[omni_xpu_kernel] FP8 GEMM (oneDNN W8A16) loaded
[omni_xpu_kernel] Loaded rotary
```

A Z-Image-Turbo E2E workflow was submitted through the ComfyUI API using:

```text
UNETLoader: z_image_turbo_bf16.safetensors
CLIPLoader: qwen_3_4b.safetensors, type=lumina2
VAELoader: ae.safetensors
ModelSamplingAuraFlow: shift=3.0
KSampler: steps=4, cfg=1.0, sampler=res_multistep, scheduler=simple
EmptySD3LatentImage: 512x512, batch_size=1
```

The output was saved as:

```text
C:\workspace\llm-scaler\omni\comfyui_windows_setup\ComfyUI\output\z_image_turbo_e2e_00001_.png
```

It was verified as a valid `512x512` PNG. The first run completed in about 57 seconds after loading the model weights.

## Troubleshooting

### User-site PyTorch Is Imported During Build

Symptoms include missing XPU headers or linking against the wrong PyTorch install.

Use the build script settings:

```cmd
set "PYTHONNOUSERSITE=1"
set "PYTHONPATH="
"%CONDA_PREFIX%\python.exe" -c "import torch; print(torch.__version__, torch.__file__)"
```

The printed PyTorch path must be under `%CONDA_PREFIX%`.

### C++ `min` / `max` Macro Compile Errors

Windows headers can define `min` and `max` macros that conflict with C++ code. The Windows compile command uses:

```text
/DNOMINMAX
/DWIN32_LEAN_AND_MEAN
```

### Missing XPU Link Symbols

If link errors mention symbols such as `c10::xpu::XPUStream`, ensure the Windows link libraries include:

```text
torch_xpu.lib
c10_xpu.lib
```

The Linux equivalents are:

```text
-ltorch_xpu
-lc10_xpu
```

### oneDNN Runtime Not Found

The build script tries to auto-detect:

```text
%ProgramFiles(x86)%\Intel\oneAPI\dnnl\latest\include
%ProgramFiles(x86)%\Intel\oneAPI\dnnl\latest\lib
```

If your oneAPI layout differs, set `ONEDNN_INCLUDE` and `ONEDNN_LIB` before building.

### Embedded Python Reinstalls A Non-XPU PyTorch

Reinstall the XPU build in the embedded Python:

```cmd
"C:\workspace\llm-scaler\omni\comfyui_windows_setup\python_embeded\python.exe" -m pip install --force-reinstall torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/xpu
```

Then reinstall the local wheel with `--no-deps`.

### ComfyUI Port Already In Use

If startup fails with `WinError 10048` for `127.0.0.1:8188`, use a different port:

```cmd
..\python_embeded\python.exe main.py --listen 127.0.0.1 --port 8190
```

Check listeners with PowerShell:

```powershell
Get-NetTCPConnection -LocalPort 8188 -ErrorAction SilentlyContinue
```

## Rebuild Checklist

1. Confirm `omni_env` uses the desired XPU PyTorch.
2. Run `scripts\build_llm_scaler_conda.cmd`.
3. Confirm the wheel appears under `C:\workspace\llm_scaler_dist`.
4. Install it into `comfyui_windows_setup\python_embeded` with `--no-deps`.
5. Verify `omni_xpu_kernel.is_available()` from embedded Python.
6. Start ComfyUI and confirm XPU plus `omni_xpu_kernel` startup log messages.
7. Run a small workflow such as the Z-Image-Turbo `512x512`, 4-step API prompt.
