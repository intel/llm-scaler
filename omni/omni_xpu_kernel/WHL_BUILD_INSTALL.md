# omni_xpu_kernel Windows wheel build guide

This document describes the Windows workflow that is currently known to build and run `omni_xpu_kernel` successfully.

It is based on the configuration validated during local bring-up of:

- wheel build on Windows
- import of `omni_xpu_kernel._C`
- `norm.rms_norm(...)`
- `svdq.unpack_int4(...)`
- `sdp.sdp(...)`

## Validated version matrix

Use this combination unless you are intentionally re-validating a different stack:

| Component | Version |
| --- | --- |
| OS | Windows x64 |
| Python | 3.12 |
| PyTorch | `2.10.0+xpu` |
| torchvision | `0.25.0+xpu` |
| torchaudio | `2.10.0+xpu` |
| oneAPI compiler | `2025.3` |
| Intel XPU runtime packages | `2025.3.x` |
| MSVC toolchain | Visual Studio 2022 Build Tools |

The most important rule on Windows is: **the compiler/runtime used to build the wheel must match the SYCL runtime shipped with the active PyTorch XPU environment**.

The working combination above uses `sycl8.dll` from the `2025.3` stack. Building with a newer oneAPI toolchain and loading into an older PyTorch XPU environment can produce import failures or misleading runtime errors.

## Prerequisites

Install the following before building:

1. Intel oneAPI C++ Compiler `2025.3`
2. Visual Studio 2022 Build Tools with MSVC C++ components
3. Miniconda or Miniforge
4. Intel GPU driver capable of running PyTorch XPU

Recommended oneAPI component on Windows:

- Intel oneAPI C++ Essentials / DPC++ Compiler `2025.3`

## Create the build environment

The examples below use a conda environment named `omni_env`.

```cmd
conda create -n omni_env python=3.12 -y
conda activate omni_env
python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url https://download.pytorch.org/whl/xpu ^
  torch==2.10.0+xpu torchvision==0.25.0+xpu torchaudio==2.10.0+xpu
python -m pip install ^
  dpcpp-cpp-rt==2025.3.1 ^
  intel-cmplr-lib-rt==2025.3.1 ^
  intel-cmplr-lib-ur==2025.3.1 ^
  intel-cmplr-lic-rt==2025.3.1 ^
  intel-sycl-rt==2025.3.1 ^
  intel-opencl-rt==2025.3.1 ^
  intel-openmp==2025.3.1 ^
  intel-pti==0.15.0 ^
  onemkl-sycl-blas==2025.3.0 ^
  onemkl-sycl-dft==2025.3.0 ^
  onemkl-sycl-lapack==2025.3.0 ^
  onemkl-sycl-rng==2025.3.0 ^
  onemkl-sycl-sparse==2025.3.0 ^
  mkl==2025.3.0 ^
  tbb==2022.3.0 ^
  tcmlib==1.4.1 ^
  umf==1.0.2 ^
  pytorch-triton-xpu==3.5.0
```

Quick sanity check:

```cmd
python -c "import torch; print(torch.__version__); print(torch.xpu.is_available())"
```

## Build the wheel

The commands below assume:

- repo root: `C:\workspace\llm-scaler`
- package dir: `C:\workspace\llm-scaler\omni\omni_xpu_kernel`
- wheel output dir: `C:\workspace\llm_scaler_dist`

Open a Developer Command Prompt for VS 2022, or otherwise ensure the MSVC build tools are on `PATH`, then run:

```cmd
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
call C:\ProgramData\miniforge3\Scripts\activate.bat omni_env

set "WORKSPACE=C:\workspace"
set "LLM_SCALER_DIR=%WORKSPACE%\llm-scaler"
set "OUTPUT_DIR=%WORKSPACE%\llm_scaler_dist"

set "OMNI_XPU_ONEAPI_VERSION=2025.3"
set "OMNI_XPU_DEVICE=bmg"
set "OMNI_XPU_ENABLE_ONEDNN=0"
set "PATH=%CONDA_PREFIX%\Library\bin;%CONDA_PREFIX%\DLLs;%CONDA_PREFIX%\Lib\site-packages\torch\lib;%PATH%"

cd /d "%LLM_SCALER_DIR%\omni\omni_xpu_kernel"
python -m pip wheel . -w "%OUTPUT_DIR%" --no-build-isolation --no-deps
```

Expected wheel:

```text
C:\workspace\llm_scaler_dist\omni_xpu_kernel-0.1.0-cp312-cp312-win_amd64.whl
```

### Important build variables

| Variable | Meaning |
| --- | --- |
| `OMNI_XPU_ONEAPI_VERSION` | Prefer a specific installed oneAPI compiler version on Windows |
| `OMNI_XPU_DEVICE` | AOT target for the ESIMD SDP sidecar, default `bmg` |
| `OMNI_XPU_ENABLE_ONEDNN` | Enable oneDNN-backed kernels explicitly; current Windows guidance is `0` |

Notes:

- `OMNI_XPU_ENABLE_ONEDNN=0` is the recommended Windows default today.
- `OMNI_XPU_DEVICE=bmg` is the validated target for Arc B-series testing.
- If you want to validate oneDNN-backed kernels on Windows, treat that as a separate compatibility pass.

## Install the wheel into the build environment

```cmd
conda activate omni_env
python -m pip install --force-reinstall --no-deps ^
  C:\workspace\llm_scaler_dist\omni_xpu_kernel-0.1.0-cp312-cp312-win_amd64.whl
```

## Verify in the build environment

Use the same environment that built the wheel:

```cmd
python -c "import omni_xpu_kernel as ok; print(ok.__version__); print(ok.is_available())"
python -c "import torch, omni_xpu_kernel as ok; x=torch.randn(2,2560,device='xpu',dtype=torch.float16); w=torch.randn(2560,device='xpu',dtype=torch.float16); y=ok.norm.rms_norm(w,x,1e-6); torch.xpu.synchronize(); print(y.shape, y.device, y.dtype)"
```

Optional SDP smoke test:

```cmd
python -c "import torch; from omni_xpu_kernel import sdp; q=torch.randn(1,64,30,128,device='xpu',dtype=torch.bfloat16); k=torch.randn(1,64,30,128,device='xpu',dtype=torch.bfloat16); v=torch.randn(1,64,30,128,device='xpu',dtype=torch.bfloat16); y=sdp.sdp(q,k,v); torch.xpu.synchronize(); print(y.shape, y.device, y.dtype)"
```

## Install into another Python environment

For example, to install into an embedded ComfyUI Python:

```cmd
"%EMBED_PYTHON_DIR%\python.exe" -m pip install --force-reinstall --no-deps ^
  C:\workspace\llm_scaler_dist\omni_xpu_kernel-0.1.0-cp312-cp312-win_amd64.whl
```

The target environment should also use the same validated XPU stack:

```cmd
"%EMBED_PYTHON_DIR%\python.exe" -m pip install --force-reinstall --index-url https://download.pytorch.org/whl/xpu ^
  torch==2.10.0+xpu torchvision==0.25.0+xpu torchaudio==2.10.0+xpu
```

## Windows runtime notes

The wheel now relies on Windows DLL search setup performed in `omni_xpu_kernel.__init__`.

At import time it adds these locations when present:

1. active Python `Library\bin`
2. active Python `DLLs`
3. `torch\lib`
4. oneAPI compiler locale directory such as `compiler\2025.3\bin\1033`
5. any extra directories from `OMNI_XPU_DLL_DIRS`

This is important on Windows because the extension can depend on Intel compiler runtime resources such as `irc_msg.dll`.

## Troubleshooting

### `_C` import fails with a missing DLL or "找不到指定的模块"

Usually this means one of:

- oneAPI compiler/runtime version does not match the PyTorch XPU runtime
- the target environment is missing required Intel runtime DLLs
- the embedded environment is not using the XPU PyTorch build

Check:

```cmd
python -c "import torch; print(torch.__version__); print(torch.__file__); print(torch.xpu.is_available())"
python -c "import omni_xpu_kernel as ok; print(ok.is_available())"
```

### Wheel built, but kernels fail at runtime with "device or resource busy"

Do not assume this is only a device contention issue. On Windows this can also be a symptom of a mismatched SYCL runtime.

Re-check:

- oneAPI compiler version
- PyTorch XPU version
- XPU runtime package versions

The validated combination in this document is the known-good baseline.

### SDP fails to load the sidecar

The Windows loader now scans `lgrf_uni\` for `lgrf_sdp*.pyd`.

If this fails, confirm the wheel contains a packaged sidecar under:

```text
omni_xpu_kernel\lgrf_uni\
```

### oneDNN kernels on Windows

Windows builds currently default to `OMNI_XPU_ENABLE_ONEDNN=0`.

That keeps the validated Windows path focused on:

- `_C` import
- normalization kernels
- SVDQ ESIMD helpers
- SDP sidecar loading

If you enable oneDNN-backed kernels on Windows, re-validate that configuration explicitly.
