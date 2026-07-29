# omni_xpu_kernel Windows WHL 构建与 Portable 安装

本文记录 `omni_xpu_kernel` 在 Windows x64 上的独立构建、wheel
检查、ComfyUI Portable 安装和验收流程。

当前已实际验证的组合是：

```text
Python 3.13.12
PyTorch 2.12.0+xpu
Intel oneAPI DPC++/C++ Compiler 2025.3.3
oneDNN 2025.3.0 Python package / oneDNN 3.9.1 native API
Intel Arc Pro B70 / intel_gpu_bmg_g31
OMNI_XPU_DEVICE=bmg
Windows wheel tag: cp313-cp313-win_amd64
llm-scaler source: 0c6cd57c6034929c3f7ee6efc48d7aa0b72b7bed
```

本文不把 ComfyUI Portable 当作编译环境。编译环境位于项目目录内，
Portable 只用于最终安装和运行测试，避免修改其他项目的 Python 环境。

> [!IMPORTANT]
> Torch、Python ABI 和 GPU AOT 目标都属于 wheel 身份的一部分。不同
> Python ABI、Torch minor 或 GPU 架构必须分别构建，不能通过重命名 wheel
> 互换。Torch 2.13 尚未包含在本文的已验证范围内。

## 1. 已验证版本矩阵

### 1.1 系统工具链

| 组件 | 已验证版本 | 说明与获取地址 |
|---|---:|---|
| Windows | Windows 11 Pro x64，`10.0.26200` | Windows 10/11 x64；这里记录的是本次验证主机，而不是硬性最低版本 |
| Intel Arc Pro 驱动 | `32.0.101.8515` | 本机验证版本，不代表硬性最低版本；从 [Intel Arc Pro Windows 驱动页](https://www.intel.com/content/www/us/en/download/741626/intel-arc-pro-graphics-windows.html) 获取当前驱动 |
| Visual Studio Build Tools 2022 | `17.14.36` | 安装 `Desktop development with C++`；参见 [Microsoft C++ Build Tools 安装文档](https://learn.microsoft.com/en-us/cpp/overview/acquire-msvc) |
| MSVC v143 x64/x86 | `14.42.34433`，`cl 19.42.34444` | 本次构建显式使用 `-vcvars_ver=14.42`；工作负载组件见 [Microsoft Build Tools component IDs](https://learn.microsoft.com/en-us/visualstudio/install/workload-component-id-vs-build-tools) |
| Intel oneAPI DPC++/C++ Compiler | `2025.3.3`，build `20260319` | [编译器下载页](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler-download.html)；[2025 release notes](https://www.intel.com/content/www/us/en/developer/articles/release-notes/oneapi-dpcpp/2025.html) |
| Intel oneAPI oneDNN development install | `2025.3` | 必须包含 oneDNN `3.9.1` 的头文件和 `dnnl.lib`；可随 [Intel oneAPI Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/oneapi-toolkit.html) 安装 |
| Windows SDK NuGet package | `10.0.26100.3916` | NuGet 包内头文件版本为 `10.0.26100.0`；仅在系统没有 Windows SDK/UCRT 时需要项目内 fallback |

本次使用的三个 Windows SDK NuGet 包：

- [Microsoft.Windows.SDK.CPP 10.0.26100.3916](https://www.nuget.org/packages/Microsoft.Windows.SDK.CPP/10.0.26100.3916)
- [Microsoft.Windows.SDK.CPP.x64 10.0.26100.3916](https://www.nuget.org/packages/Microsoft.Windows.SDK.CPP.x64/10.0.26100.3916)
- [Microsoft.Windows.SDK.BuildTools 10.0.26100.3916](https://www.nuget.org/packages/Microsoft.Windows.SDK.BuildTools/10.0.26100.3916)

如果 Visual Studio 已经安装 Windows 10/11 SDK 和 Universal CRT，通常不需要
NuGet fallback。

### 1.2 独立 Python 构建环境

| 包 | 精确版本 | 用途与获取地址 |
|---|---:|---|
| CPython | `3.13.12` | 必须与目标 Portable 的 `cp313` ABI 一致；[Python 3.13.12](https://www.python.org/downloads/release/python-31312/) |
| uv | `0.11.21` | 仅用于在项目目录管理 Python 和 venv；[uv 0.11.21](https://pypi.org/project/uv/0.11.21/)、[安装文档](https://docs.astral.sh/uv/getting-started/installation/) |
| pip | `26.1.2` | Python 包安装器；[PyPI](https://pypi.org/project/pip/26.1.2/) |
| setuptools | `78.1.0` | wheel 构建后端；[PyPI](https://pypi.org/project/setuptools/78.1.0/) |
| wheel | `0.47.0` | wheel 打包；[PyPI](https://pypi.org/project/wheel/0.47.0/) |
| torch | `2.12.0+xpu` | 编译所针对的原生 ABI；[官方 XPU wheel index](https://download.pytorch.org/whl/xpu/torch/)、[PyTorch Intel GPU 指南](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html) |
| onednn | `2025.3.0` | oneDNN Windows 运行包；[PyPI](https://pypi.org/project/onednn/2025.3.0/) |
| onednn-devel | `2025.3.0` | 记录并锁定开发包版本；[PyPI](https://pypi.org/project/onednn-devel/2025.3.0/) |
| numpy | `2.5.1` | 测试环境数值依赖；[PyPI](https://pypi.org/project/numpy/2.5.1/) |
| pytest | `9.1.1` | 可选，运行源码测试；[PyPI](https://pypi.org/project/pytest/9.1.1/) |

Windows 上的 `onednn-devel` wheel 当前不能替代 oneAPI oneDNN development
install：构建仍然需要系统 oneAPI 目录中的 `oneapi/dnnl/dnnl.hpp` 和
`dnnl.lib`。`setup.py` 会校验原生头文件版本为 oneDNN `3.9.1`。

Torch/oneDNN 在本次解析出的关键原生传递依赖如下。通常不应逐项手工安装，
而应让 `torch==2.12.0+xpu` 和 `onednn==2025.3.0` 解析它们：

| 包 | 已验证版本 | 获取地址 |
|---|---:|---|
| dpcpp-cpp-rt | `2025.3.2` | [PyPI](https://pypi.org/project/dpcpp-cpp-rt/2025.3.2/) |
| intel-sycl-rt | `2025.3.2` | [PyPI](https://pypi.org/project/intel-sycl-rt/2025.3.2/) |
| intel-opencl-rt | `2025.3.2` | [PyPI](https://pypi.org/project/intel-opencl-rt/2025.3.2/) |
| intel-openmp | `2025.3.2` | [PyPI](https://pypi.org/project/intel-openmp/2025.3.2/) |
| intel-cmplr-lib-rt | `2025.3.2` | [PyPI](https://pypi.org/project/intel-cmplr-lib-rt/2025.3.2/) |
| intel-cmplr-lib-ur | `2025.3.2` | [PyPI](https://pypi.org/project/intel-cmplr-lib-ur/2025.3.2/) |
| intel-cmplr-lic-rt | `2025.3.2` | [PyPI](https://pypi.org/project/intel-cmplr-lic-rt/2025.3.2/) |
| intel-pti | `0.16.0` | [PyPI](https://pypi.org/project/intel-pti/0.16.0/) |
| mkl | `2025.3.1` | [PyPI](https://pypi.org/project/mkl/2025.3.1/) |
| tbb | `2022.3.1` | [PyPI](https://pypi.org/project/tbb/2022.3.1/) |
| tcmlib | `1.4.1` | [PyPI](https://pypi.org/project/tcmlib/1.4.1/) |
| umf | `1.0.3` | [PyPI](https://pypi.org/project/umf/1.0.3/) |
| triton-xpu | `3.7.1` | [PyTorch XPU index](https://download.pytorch.org/whl/xpu/triton-xpu/) |

### 1.3 已验证的 ComfyUI Portable 运行环境

| 包 | 精确版本 |
|---|---:|
| Python | `3.13.12` |
| torch | `2.12.0+xpu` |
| torchvision | `0.27.0+xpu` |
| torchaudio | `2.11.0+xpu` |
| onednn | `2025.3.0` |
| omni-xpu-kernel | `0.1.0b9.dev0+torch212.bmg` |
| comfy-kitchen | `0.2.18`，Intel XPU fork commit [`c7ae07e5...`](https://github.com/xiangyuT/comfy-kitchen-xpu/commit/c7ae07e5317d4a073562e278a41d98f05a4fe109) |
| ComfyUI | `0.28.0` |

`torchvision 0.27.0+xpu` 可从
[PyTorch XPU torchvision index](https://download.pytorch.org/whl/xpu/torchvision/)
获取。当前 XPU index 中 torchaudio 的可用最新版是 `2.11.0+xpu`；本次与
Torch 2.12 的导入测试通过，但 `omni_xpu_kernel` 本身不依赖 torchvision
或 torchaudio。

完整 Omni runtime 会从 ComfyUI requirements 中省略官方 `comfy-kitchen`
依赖，并由 Intel XPU 部署流程单独安装 fork。原因、安装和更新流程见
[`docs/WINDOWS_PORTABLE.md`](../docs/WINDOWS_PORTABLE.md)。

## 2. Windows wheel 的组成与限制

Windows 构建产出两个原生扩展：

```text
omni_xpu_kernel/_C.cp313-win_amd64.pyd
omni_xpu_kernel/lgrf_uni/lgrf_sdp.cp313-win_amd64.pyd
```

- `_C` 包含 norm、FP8、GGUF、SVDQ、INT8、rotary 和 oneDNN 等核心算子。
- `lgrf_sdp` 是独立的 ESIMD SDP sidecar。
- CUTE FMHA 当前是 Linux-only，Windows 必须设置
  `OMNI_XPU_REQUIRE_CUTE=0`。
- Windows ComfyUI 默认保留 PyTorch SDPA；ESIMD sidecar 仅在显式设置
  `OMNI_ATTN_BACKEND=esimd` 时由 Custom Node 使用。
- `OMNI_XPU_DEVICE=bmg` 会把核心扩展和 sidecar 都 AOT 编译为 BMG
  `spir64_gen` 镜像。
- PTL-H 必须单独使用 `OMNI_XPU_DEVICE=ptl-h` 构建，不能安装 BMG wheel。

项目当前识别 Torch XPU 2.10、2.11 和 2.12。识别某个 minor 不代表所有
组合都已经验收；本文只对 Torch `2.12.0+xpu`、Python 3.13、BMG 作出验证
声明。

## 3. 准备项目内独立构建环境

以下命令在 PowerShell 中执行。先把尖括号占位符替换为
`omni_xpu_kernel` 源码目录：

```powershell
$kernelRoot = (Resolve-Path "<omni_xpu_kernel-source-directory>").Path
$buildRoot = Join-Path $kernelRoot ".venv-win-py313-torch212"
$env:UV_PYTHON_INSTALL_DIR = Join-Path $buildRoot "python"
$env:UV_CACHE_DIR = Join-Path $buildRoot "cache"

Set-Location $kernelRoot

uv python install 3.13.12
uv venv --seed --python 3.13.12 (Join-Path $buildRoot "venv")

$buildPython = Join-Path $buildRoot "venv\Scripts\python.exe"

& $buildPython -m pip install `
    "pip==26.1.2" `
    "setuptools==78.1.0" `
    "wheel==0.47.0"

& $buildPython -m pip install `
    "torch==2.12.0+xpu" `
    --index-url "https://download.pytorch.org/whl/xpu"

& $buildPython -m pip install `
    "onednn==2025.3.0" `
    "onednn-devel==2025.3.0" `
    "numpy==2.5.1" `
    "pytest==9.1.1"
```

检查环境，不要依赖其他 Python：

```powershell
& $buildPython -c @"
import torch
print("torch:", torch.__version__)
print("torch XPU runtime:", torch.version.xpu)
print("XPU available:", torch.xpu.is_available())
print("devices:", [torch.xpu.get_device_name(i) for i in range(torch.xpu.device_count())])
"@

& $buildPython -m pip check
```

预期 Torch 版本为 `2.12.0+xpu`，`torch.version.xpu` 为 `20250302`。

## 4. 没有系统 Windows SDK 时的项目内 fallback

如果编译探针或正式构建报错：

```text
fatal error: 'assert.h' file not found
```

说明 MSVC/Intel 编译器环境没有取得 Windows SDK/UCRT 头文件。首选方案是
通过 Visual Studio Installer 安装 Windows 10/11 SDK 和 Universal CRT。
如果不希望修改系统安装，可以把已验证的 NuGet SDK 放进 `$buildRoot`。

```powershell
$sdkRoot = Join-Path $buildRoot "windows-sdk-nuget"
New-Item -ItemType Directory -Force -Path $sdkRoot | Out-Null

$sdkPackages = @(
    "Microsoft.Windows.SDK.CPP",
    "Microsoft.Windows.SDK.CPP.x64",
    "Microsoft.Windows.SDK.BuildTools"
)
$sdkPackageVersion = "10.0.26100.3916"

foreach ($package in $sdkPackages) {
    $fileName = "$($package.ToLowerInvariant()).$sdkPackageVersion.nupkg"
    $packageFile = Join-Path $sdkRoot $fileName
    $extractDir = Join-Path $sdkRoot $package.ToLowerInvariant()
    $downloadUrl = "https://www.nuget.org/api/v2/package/$package/$sdkPackageVersion"

    Invoke-WebRequest -Uri $downloadUrl -OutFile $packageFile
    New-Item -ItemType Directory -Force -Path $extractDir | Out-Null
    tar.exe -xf $packageFile -C $extractDir
}
```

本次解压后的关键目录是：

```text
windows-sdk-nuget\
  microsoft.windows.sdk.cpp\c\Include\10.0.26100.0\
    ucrt\
    shared\
    um\
    winrt\
    cppwinrt\
  microsoft.windows.sdk.cpp.x64\c\
    ucrt\x64\
    um\x64\
  microsoft.windows.sdk.buildtools\bin\10.0.26100.0\x64\
```

NuGet 包版本 `10.0.26100.3916` 与包内 SDK 文件版本
`10.0.26100.0` 不同，这是正常的。

## 5. 初始化编译环境并构建 wheel

建议打开普通 `cmd.exe`，显式初始化 MSVC 和 oneAPI，然后调用项目 venv
中的 Python。下面的命令适用于本次已验证机器：

```bat
@echo off

set "KERNEL_ROOT=<omni_xpu_kernel-source-directory>"
set "BUILD_ROOT=%KERNEL_ROOT%\.venv-win-py313-torch212"
set "BUILD_PYTHON=%BUILD_ROOT%\venv\Scripts\python.exe"

call "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=amd64 -host_arch=amd64 -vcvars_ver=14.42
if errorlevel 1 exit /b 1

call "%ProgramFiles(x86)%\Intel\oneAPI\setvars.bat" --force
if errorlevel 1 exit /b 1

set "DNNLROOT=%ProgramFiles(x86)%\Intel\oneAPI\dnnl\2025.3"
set "OMNI_XPU_DEVICE=bmg"
set "OMNI_XPU_REQUIRE_CUTE=0"

set "PATH=%BUILD_ROOT%\venv\Library\bin;%BUILD_ROOT%\venv\Lib\site-packages\torch\lib;%DNNLROOT%\bin;%PATH%"
```

如果使用第 4 节的项目内 Windows SDK，继续在同一个 `cmd.exe` 设置：

```bat
set "SDK_NUGET=%BUILD_ROOT%\windows-sdk-nuget"
set "SDK_CPP=%SDK_NUGET%\microsoft.windows.sdk.cpp\c"
set "SDK_X64=%SDK_NUGET%\microsoft.windows.sdk.cpp.x64\c"
set "SDK_TOOLS=%SDK_NUGET%\microsoft.windows.sdk.buildtools"
set "SDK_FILE_VERSION=10.0.26100.0"

set "INCLUDE=%SDK_CPP%\Include\%SDK_FILE_VERSION%\ucrt;%SDK_CPP%\Include\%SDK_FILE_VERSION%\shared;%SDK_CPP%\Include\%SDK_FILE_VERSION%\um;%SDK_CPP%\Include\%SDK_FILE_VERSION%\winrt;%SDK_CPP%\Include\%SDK_FILE_VERSION%\cppwinrt;%INCLUDE%"
set "LIB=%SDK_X64%\ucrt\x64;%SDK_X64%\um\x64;%LIB%"
set "PATH=%SDK_TOOLS%\bin\%SDK_FILE_VERSION%\x64;%PATH%"
```

先检查当前 shell 确实使用预期工具链：

```bat
where cl
where icx
cl /Bv
icx --version
"%BUILD_PYTHON%" -c "import torch; print(torch.__version__, torch.version.xpu)"
sycl-ls --verbose
```

在 B70 上，`sycl-ls --verbose` 应包含：

```text
Architecture: intel_gpu_bmg_g31
```

正式构建：

```bat
cd /d "%KERNEL_ROOT%"
if not exist "%BUILD_ROOT%\wheelhouse\patched" mkdir "%BUILD_ROOT%\wheelhouse\patched"

"%BUILD_PYTHON%" -m pip wheel . ^
  --wheel-dir "%BUILD_ROOT%\wheelhouse\patched" ^
  --no-build-isolation ^
  --no-deps
```

`--no-build-isolation` 是必需的：构建必须读取当前 venv 中已安装的
Torch XPU 头文件、库和版本。`--no-deps` 避免打包过程改变环境。

已验证输出：

```text
.venv-win-py313-torch212\wheelhouse\patched\
  omni_xpu_kernel-0.1.0b9.dev0+torch212.bmg-cp313-cp313-win_amd64.whl
```

已验证 artifact：

```text
size:   1,398,388 bytes
SHA256: 1F52D6F2BE86C9592432B367569B7859003C8ECE9A052FB3019D7A234D8C41FB
```

项目的 `scripts\build.bat` 可用于原地安装或开发安装，但需要发布或复制
artifact 时，应使用上面的 `pip wheel` 命令。

## 6. 检查 wheel 内容和元数据

```powershell
$wheelPath = Join-Path $buildRoot `
    "wheelhouse\patched\omni_xpu_kernel-0.1.0b9.dev0+torch212.bmg-cp313-cp313-win_amd64.whl"

& $buildPython -m zipfile -l $wheelPath
Get-FileHash -Algorithm SHA256 -LiteralPath $wheelPath
```

应至少看到：

```text
omni_xpu_kernel/_C.cp313-win_amd64.pyd
omni_xpu_kernel/lgrf_uni/lgrf_sdp.cp313-win_amd64.pyd
omni_xpu_kernel-0.1.0b9.dev0+torch212.bmg.dist-info/METADATA
```

metadata 应包含精确的 `torch==2.12.0` 要求，以及 Windows x64 上的
`onednn==2025.3.0` 条件依赖。

## 7. 安装到 ComfyUI Portable

下面仅以本次测试 Portable 为例。安装前先关闭所有正在使用该 Portable
Python 的 ComfyUI/Python 进程，并记录原环境：

```powershell
$portableRoot = (Resolve-Path "<ComfyUI_windows_portable-root>").Path
$embeddedPython = Join-Path $portableRoot "python_embeded\python.exe"

& $embeddedPython -m pip list
& $embeddedPython -c "import torch; print(torch.__version__)"
```

如果 Portable 原来不是 Torch 2.12，先安装匹配组合：

```powershell
& $embeddedPython -m pip install --force-reinstall `
    --index-url "https://download.pytorch.org/whl/xpu" `
    "torch==2.12.0+xpu" `
    "torchvision==0.27.0+xpu"
```

本次 Portable 保留了可用的 `torchaudio==2.11.0+xpu`。它不是
`omni_xpu_kernel` 的依赖；如果目标 Portable 不使用音频节点，可以不额外
安装 torchaudio。

安装 oneDNN runtime 和 wheel：

```powershell
& $embeddedPython -m pip install "onednn==2025.3.0"
& $embeddedPython -m pip install --force-reinstall --no-deps $wheelPath
& $embeddedPython -m pip check
```

安装本地 wheel 时必须保留 `--no-deps`，避免 pip 从其他 index 替换已确认的
Torch XPU build。预期 `pip check` 输出：

```text
No broken requirements found.
```

如果 Torch 降级后 pip 报告 `Ignoring invalid distribution ~orch`，检查
`python_embeded\Lib\site-packages` 是否留下旧版本的
`~orch-*.dist-info` 临时目录。仅删除这个已经确认的临时目录，不要删除
`torch` 或有效的 `torch-2.12.0+xpu.dist-info`。

## 8. 验收

所有安装后测试都应在源码目录之外运行，否则源码 checkout 可能遮蔽
Portable 中真正安装的 wheel。

### 8.1 包身份和 XPU

```powershell
Set-Location $portableRoot

& $embeddedPython -c @"
from pathlib import Path
import importlib.metadata as metadata
import torch
import omni_xpu_kernel as omni

print("torch:", torch.__version__)
print("torch XPU runtime:", torch.version.xpu)
print("devices:", [torch.xpu.get_device_name(i) for i in range(torch.xpu.device_count())])
print("package:", metadata.version("omni-xpu-kernel"))
print("module:", Path(omni.__file__).resolve())
print("metadata target:", omni.__xpu_target__)
print("core AOT target:", omni.core_aot_target())
print("available:", omni.is_available())
print("capabilities:", omni.native_capabilities())

assert torch.__version__ == "2.12.0+xpu"
assert torch.xpu.is_available()
assert omni.__xpu_target__ == "bmg"
assert omni.core_aot_target() == "bmg"
assert omni.is_available()
"@
```

### 8.2 最小原生 kernel correctness smoke

```powershell
@'
import torch
from omni_xpu_kernel import norm, sdp

for dtype in (torch.float16, torch.bfloat16, torch.float32):
    x = torch.randn(8, 2048, device="xpu", dtype=dtype)
    weight = torch.randn(2048, device="xpu", dtype=dtype)
    actual = norm.rms_norm(weight, x, eps=1e-6)
    x32 = x.float()
    expected = (
        x32
        / torch.sqrt(torch.mean(x32 * x32, dim=-1, keepdim=True) + 1e-6)
        * weight.float()
    ).to(dtype)
    tolerance = 1e-4 if dtype == torch.float32 else 1e-2
    torch.testing.assert_close(
        actual, expected, rtol=tolerance, atol=tolerance
    )

for dtype in (torch.float16, torch.bfloat16):
    q = torch.randn(1, 64, 8, 128, device="xpu", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    actual = sdp.sdp(q, k, v)
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.permute(0, 2, 1, 3).contiguous(),
        k.permute(0, 2, 1, 3).contiguous(),
        v.permute(0, 2, 1, 3).contiguous(),
    ).permute(0, 2, 1, 3).contiguous()
    tolerance = 5e-2 if dtype == torch.bfloat16 else 1e-2
    torch.testing.assert_close(
        actual, expected, rtol=tolerance, atol=tolerance
    )

torch.xpu.synchronize()
print("native kernel smoke: PASS")
'@ | & $embeddedPython -
```

本次额外验证通过：

- Windows 支持面源码测试：501 passed、36 skipped；
- packaging 测试：25 passed、4 skipped；
- RMSNorm：FP16、BF16、FP32；
- SVDQ UINT4 quantize/unpack/dequantize；
- GGUF Q4_1 capability 和 correctness；
- BMG GroupNorm；
- LTX direct split RoPE；
- INT8 shared/prequantized pair、rowwise 和 cache correctness；
- standalone SDP：`head_dim=128` 的 FP16、BF16；
- oneDNN FP8 GEMM correctness 和 primitive cache；
- 两张 Intel Arc Pro B70 的 XPU 张量、RMSNorm 和 D128 SDP correctness。

当前 BMG core-only Windows wheel 的 standalone SDP 实机验收范围是
`head_dim=128`。源码测试套件还包含 `head_dim=64` 用例；该 wheel 对这些
配置会明确返回 `kernel not available for this configuration`，不能把上面的
smoke 结果解释为已经覆盖 `head_dim=64`。

### 8.3 ComfyUI 启动 smoke

```powershell
& $embeddedPython (Join-Path $portableRoot "ComfyUI\main.py") `
    --windows-standalone-build `
    --disable-auto-launch `
    --quick-test-for-ci `
    --log-stdout `
    --verbose INFO
```

本次测试中该命令返回码为 `0`，ComfyUI `0.28.0` 正确识别：

```text
pytorch version: 2.12.0+xpu
Device: xpu:0 Intel(R) Arc(TM) Pro B70 Graphics
Device: xpu:1 Intel(R) Arc(TM) Pro B70 Graphics
```

这只证明 Portable 基础启动、设备发现和内核直接调用正常。具体模型工作流
仍需要对应的 Omni/ComfyUI 集成代码和模型做端到端验收。

## 9. 常见错误

### `assert.h` 或 Windows/UCRT 头文件缺失

安装 Visual Studio 的 Windows SDK/UCRT，或使用第 4 节的项目内 NuGet SDK，
并确认 `INCLUDE` 和 `LIB` 是在同一个已初始化的构建 shell 中设置的。

### Torch 头文件出现 `min`/`max` 宏冲突

当前 Windows build 已定义 `NOMINMAX` 和 `WIN32_LEAN_AND_MEAN`。旧 checkout
缺少这些定义时，会在 Torch 模板头文件中出现看似无关的语法错误。

### 找不到 oneDNN headers 或 `dnnl.lib`

确认：

```bat
set "DNNLROOT=%ProgramFiles(x86)%\Intel\oneAPI\dnnl\2025.3"
dir "%DNNLROOT%\include\oneapi\dnnl\dnnl.hpp"
dir "%DNNLROOT%\lib\dnnl.lib"
```

不要混用不同版本的 oneDNN header、import library 和 runtime DLL。

### `c10::xpu::XPUStream` 等链接错误

Windows 核心扩展必须链接 `torch_xpu.lib` 和 `c10_xpu.lib`。当前 `setup.py`
已经包含它们；出现错误通常意味着安装的不是 XPU Torch，或 Torch
`Lib\site-packages\torch\lib` 没有进入链接搜索路径。

### `lgrf_sdp.pyd` 找不到

Windows wheel 使用 ABI 后缀，例如
`lgrf_sdp.cp313-win_amd64.pyd`。当前 loader 会扫描 `lgrf_sdp*.pyd`；
固定查找无 ABI 后缀文件的旧 wheel/旧源码需要重新构建。

### wheel 可以安装但 import 失败

依次检查：

1. Python tag 是否一致，例如目标必须能使用 `cp313`；
2. Torch public version 是否为构建时的 `2.12.0`；
3. wheel target 是否与设备一致，例如 B70 使用 `bmg`；
4. `python_embeded\Library\bin`、`torch\lib` 和 oneDNN runtime DLL 是否可见；
5. 测试 cwd 是否离开源码 checkout。

## 10. Torch 2.13 后续阶段

Torch 2.13 不能直接复用本文的 `torch212.bmg` wheel。原 Portable 的
Torch 2.13 runtime 使用 2026 系列 SYCL/oneDNN，而本文已验证编译器是
2025.3.3。下一阶段应至少：

1. 在新的项目内目录创建独立 Torch 2.13 build environment；
2. 使用与 Torch 2.13 runtime 对齐的 2026 系列 DPC++ compiler；
3. 明确并校验对应 oneDNN header/library/runtime 版本；
4. 在 `_version.py` 和 packaging metadata 中增加 Torch 2.13 映射；
5. 重新构建 `torch213.bmg` wheel；
6. 重跑本文全部 kernel correctness 和 ComfyUI smoke。

在以上步骤完成前，不应把 Torch 2.13 标记为 Windows 已支持。
