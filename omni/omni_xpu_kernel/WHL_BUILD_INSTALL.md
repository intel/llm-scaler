# omni_xpu_kernel Windows WHL 构建与 Portable 安装

本文只描述当前 Windows 构建合同：

```text
Python 3.13.14
PyTorch 2.13.0+xpu
Intel oneAPI DPC++/C++ Compiler 2026.0
oneDNN 3.11.2 / package release 2026.0.0
Intel Arc Pro B70 / intel_gpu_bmg_g31
OMNI_XPU_DEVICE=bmg
Windows wheel tag: cp313-cp313-win_amd64
omni-xpu-kernel: 0.2.0b2+torch213.bmg
```

> [!IMPORTANT]
> Python ABI、Torch minor、oneDNN ABI 和 GPU AOT target 都属于 wheel
> 身份。目标 Portable 必须保留上面的 Torch XPU 组合；不得通过调整 Torch
> 版本、重命名 wheel 或跨 GPU target 安装来复用其他构建产物。

Portable 只用于安装和运行。原生扩展应在独立、可复用的项目构建环境中完成，
以便后续继续构建 Kitchen、AIMDO 和 kernel。

## 1. 当前依赖

| 组件 | 当前要求 |
|---|---|
| Windows | Windows 10/11 x64 |
| GPU | Intel Arc Pro B70，AOT target `bmg` |
| Visual Studio | Build Tools 2022，Desktop development with C++ |
| Windows SDK | Visual Studio 提供的 Windows 10/11 SDK，或第 4 节的项目内 fallback |
| Intel compiler | oneAPI DPC++/C++ Compiler 2026.0 |
| oneDNN development files | oneAPI oneDNN 2026.0，native ABI 3.11.2 |
| Python | 3.13.14 |
| Torch | `2.13.0+xpu` |
| sycl-tla | `2fc09973bfdf15755090fcb0e3b6ad236408a992` |

Torch 2.13 Windows 构建必须使用 matched oneDNN 3.11.2 headers、
`dnnl.lib` 和 `dnnl.dll`。`setup.py` 会检查 header/runtime ABI，并把
`dnnl.dll`、许可证和第三方 notices 放入 wheel。

## 2. Windows wheel 组成

当前 BMG CUTE/Sol-Attn wheel 包含：

```text
omni_xpu_kernel/_C.cp313-win_amd64.pyd
omni_xpu_kernel/lgrf_uni/lgrf_sdp.cp313-win_amd64.pyd
omni_xpu_kernel/cute/cute_fmha_torch.cp313-win_amd64.pyd
omni_xpu_kernel/.libs/dnnl.dll
omni_xpu_kernel/.libs/onednn/LICENSE
omni_xpu_kernel/.libs/onednn/THIRD-PARTY-PROGRAMS
omni_xpu_kernel/.libs/onednn/VERSION
```

- `_C` 提供 norm、FP8、GGUF、SVDQ、INT8、rotary 和 oneDNN 算子。
- `lgrf_sdp` 提供 ESIMD SDP sidecar。
- `cute_fmha_torch` 同时导出 CUTE FMHA 和 Sol-Attn。
- Windows 默认不构建 CUTE；当前完整 wheel 必须显式设置
  `OMNI_XPU_REQUIRE_CUTE=1`。
- 编译开关不会改变 ComfyUI runtime policy。运行时仍需显式设置
  `OMNI_ATTN_BACKEND=cute`。

## 3. 准备持久构建环境

以下 PowerShell 示例把构建环境放在仓库同级目录，避免污染 Portable，并可在
后续构建中继续复用：

```powershell
$repoRoot = (Resolve-Path "<llm-scaler-repository-root>").Path
$kernelRoot = Join-Path $repoRoot "omni\omni_xpu_kernel"
$workspaceRoot = Split-Path $repoRoot -Parent
$buildRoot = Join-Path $workspaceRoot ".omni-portable-build"
$venvRoot = Join-Path $buildRoot "venv"
$buildPython = Join-Path $venvRoot "Scripts\python.exe"

$env:UV_PYTHON_INSTALL_DIR = Join-Path $buildRoot "python"
$env:UV_CACHE_DIR = Join-Path $buildRoot "uv-cache"

New-Item -ItemType Directory -Force -Path $buildRoot | Out-Null

if (-not (Test-Path $buildPython)) {
    uv python install 3.13.14
    uv venv --seed --python 3.13.14 $venvRoot
}

& $buildPython -m pip install --upgrade pip setuptools wheel
& $buildPython -m pip install `
    "torch==2.13.0+xpu" `
    --index-url "https://download.pytorch.org/whl/xpu"
& $buildPython -m pip install pytest numpy
& $buildPython -m pip check
```

确认构建解释器没有引用 Portable 或其他项目环境：

```powershell
& $buildPython -c @"
import sys
import torch

print("python:", sys.executable)
print("torch:", torch.__version__)
print("torch XPU runtime:", torch.version.xpu)
print("XPU available:", torch.xpu.is_available())

assert torch.__version__ == "2.13.0+xpu"
"@
```

## 4. Windows SDK fallback

如果构建报错 `assert.h`、`windows.h` 或 UCRT 头文件缺失，首选通过 Visual
Studio Installer 添加 Windows 10/11 SDK。无法修改系统安装时，可以在
`$buildRoot` 中准备项目内 SDK：

```powershell
$sdkRoot = Join-Path $buildRoot "windows-sdk-nuget"
$sdkPackageVersion = "10.0.26100.3916"
$sdkPackages = @(
    "Microsoft.Windows.SDK.CPP",
    "Microsoft.Windows.SDK.CPP.x64",
    "Microsoft.Windows.SDK.BuildTools"
)

New-Item -ItemType Directory -Force -Path $sdkRoot | Out-Null

foreach ($package in $sdkPackages) {
    $archive = Join-Path $sdkRoot "$($package.ToLowerInvariant()).$sdkPackageVersion.nupkg"
    $destination = Join-Path $sdkRoot $package.ToLowerInvariant()
    Invoke-WebRequest `
        -Uri "https://www.nuget.org/api/v2/package/$package/$sdkPackageVersion" `
        -OutFile $archive
    New-Item -ItemType Directory -Force -Path $destination | Out-Null
    tar.exe -xf $archive -C $destination
}
```

包版本为 `10.0.26100.3916`，解压后的 SDK 文件目录为
`10.0.26100.0`。第 5 节的构建 shell 只在系统 SDK 不完整时需要添加这些
`INCLUDE`、`LIB` 和 `PATH`。

## 5. 准备 sycl-tla

使用干净 checkout，不能直接修改 sycl-tla。Windows LLP64、host scalar 和
BMG remainder-mask 修复由 kernel build 在临时 include overlay 中应用。

```powershell
$syclTlaRoot = Join-Path $buildRoot "sycl-tla"
$syclTlaCommit = "2fc09973bfdf15755090fcb0e3b6ad236408a992"

if (-not (Test-Path (Join-Path $syclTlaRoot ".git"))) {
    git clone --filter=blob:none --no-checkout `
        "https://github.com/intel/sycl-tla.git" `
        $syclTlaRoot
}

git -C $syclTlaRoot fetch --depth 1 origin $syclTlaCommit
git -C $syclTlaRoot checkout --detach $syclTlaCommit

if ((git -C $syclTlaRoot status --short)) {
    throw "sycl-tla checkout must be clean"
}
```

## 6. 构建 BMG CUTE/Sol-Attn wheel

在同一个 `cmd.exe` 中初始化 MSVC、oneAPI、oneDNN 和项目 venv。oneAPI
`setvars.bat` 在部分 Windows 安装上不能正确调用 component `vars.bat`，
因此下面直接设置当前 2026.0 安装布局：

```bat
@echo off

set "REPO_ROOT=<llm-scaler-repository-root>"
set "KERNEL_ROOT=%REPO_ROOT%\omni\omni_xpu_kernel"
set "BUILD_ROOT=<persistent-build-root>"
set "BUILD_PYTHON=%BUILD_ROOT%\venv\Scripts\python.exe"
set "CUTLASS_SYCL_ROOT=%BUILD_ROOT%\sycl-tla"

call "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=amd64 -host_arch=amd64
if errorlevel 1 exit /b 1

set "ONEAPI_ROOT=%ProgramFiles(x86)%\Intel\oneAPI"
set "ONEAPI_COMPILER_ROOT=%ONEAPI_ROOT%\compiler\2026.0"
set "DNNLROOT=%ONEAPI_ROOT%\dnnl\2026.0"

set "PATH=%ONEAPI_COMPILER_ROOT%\bin;%ONEAPI_COMPILER_ROOT%\bin\compiler;%ONEAPI_ROOT%\ocloc\2026.0\bin;%BUILD_ROOT%\venv\Library\bin;%BUILD_ROOT%\venv\Lib\site-packages\torch\lib;%DNNLROOT%\bin;%PATH%"
set "INCLUDE=%ONEAPI_COMPILER_ROOT%\include;%ONEAPI_COMPILER_ROOT%\include\sycl;%INCLUDE%"
set "LIB=%ONEAPI_COMPILER_ROOT%\lib;%ONEAPI_COMPILER_ROOT%\opt\compiler\lib;%BUILD_ROOT%\venv\Lib\site-packages\torch\lib;%DNNLROOT%\lib;%LIB%"

set "OMNI_XPU_DEVICE=bmg"
set "OMNI_XPU_REQUIRE_CUTE=1"
set "MAX_JOBS=8"

where cl
where icx
where llvm-foreach
where ocloc
"%BUILD_PYTHON%" -c "import torch; assert torch.__version__ == '2.13.0+xpu'; print(torch.__version__, torch.version.xpu)"
sycl-ls --verbose

cd /d "%KERNEL_ROOT%"
if not exist "%BUILD_ROOT%\wheels\kernel-solattn" mkdir "%BUILD_ROOT%\wheels\kernel-solattn"

"%BUILD_PYTHON%" -m pip wheel . ^
  --wheel-dir "%BUILD_ROOT%\wheels\kernel-solattn" ^
  --no-build-isolation ^
  --no-deps
```

B70 的 `sycl-ls --verbose` 应报告：

```text
Architecture: intel_gpu_bmg_g31
```

如果使用第 4 节的项目内 Windows SDK，在 `pip wheel` 前添加：

```bat
set "SDK_ROOT=%BUILD_ROOT%\windows-sdk-nuget"
set "SDK_VERSION=10.0.26100.0"
set "SDK_CPP=%SDK_ROOT%\microsoft.windows.sdk.cpp\c"
set "SDK_X64=%SDK_ROOT%\microsoft.windows.sdk.cpp.x64\c"
set "SDK_TOOLS=%SDK_ROOT%\microsoft.windows.sdk.buildtools"

set "INCLUDE=%SDK_CPP%\Include\%SDK_VERSION%\ucrt;%SDK_CPP%\Include\%SDK_VERSION%\shared;%SDK_CPP%\Include\%SDK_VERSION%\um;%SDK_CPP%\Include\%SDK_VERSION%\winrt;%SDK_CPP%\Include\%SDK_VERSION%\cppwinrt;%INCLUDE%"
set "LIB=%SDK_X64%\ucrt\x64;%SDK_X64%\um\x64;%LIB%"
set "PATH=%SDK_TOOLS%\bin\%SDK_VERSION%\x64;%PATH%"
```

`--no-build-isolation` 是必需的，构建必须读取当前 venv 中 Torch 2.13 XPU
的 headers、libraries 和 ABI。`--no-deps` 防止构建过程改变环境。

## 7. 检查 artifact

```powershell
$wheelRoot = Join-Path $buildRoot "wheels\kernel-solattn"
$kernelWheel = Get-ChildItem $wheelRoot `
    -Filter "omni_xpu_kernel-0.2.0b2+torch213.bmg-cp313-cp313-win_amd64.whl" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if (-not $kernelWheel) {
    throw "kernel wheel not found"
}

Get-FileHash -Algorithm SHA256 -LiteralPath $kernelWheel.FullName
& $buildPython -m zipfile -l $kernelWheel.FullName
```

当前验收 artifact：

```text
omni_xpu_kernel-0.2.0b2+torch213.bmg-cp313-cp313-win_amd64.whl
SHA256: 054B5AD9B7AC046153446A249ADBBAED56C95F00CC82FB40C5AF595F6345183A
```

zip listing 必须包含第 2 节列出的三个 `.pyd`、`dnnl.dll` 和 oneDNN
redistribution notices。缺少 CUTE `.pyd` 表示构建没有实际启用
`OMNI_XPU_REQUIRE_CUTE=1`。

## 8. 安装到 Portable

先关闭所有使用目标 `python_embeded` 的进程，然后验证基础包身份：

```powershell
$portableRoot = (Resolve-Path "<ComfyUI_windows_portable-root>").Path
$embeddedPython = Join-Path $portableRoot "python_embeded\python.exe"

& $embeddedPython -c @"
import torch
import torchvision
import torchaudio

assert torch.__version__ == "2.13.0+xpu"
assert torchvision.__version__ == "0.28.0+xpu"
assert torchaudio.__version__ == "2.11.0+xpu"
assert torch.xpu.is_available()
"@
```

安装时保留 `--no-deps`：

```powershell
& $embeddedPython -m pip install `
    --force-reinstall `
    --no-deps `
    $kernelWheel.FullName

& $embeddedPython -m pip check
```

## 9. 验收

所有安装态检查都应离开 kernel source checkout，避免本地源码遮蔽 Portable
中的 wheel：

```powershell
Set-Location $portableRoot

& $embeddedPython -c @"
from importlib import metadata
from pathlib import Path

import torch
import omni_xpu_kernel as omni
from omni_xpu_kernel import cute

print("torch:", torch.__version__)
print("kernel:", metadata.version("omni-xpu-kernel"))
print("module:", Path(omni.__file__).resolve())
print("target:", omni.__xpu_target__, omni.core_aot_target())
print("capabilities:", omni.native_capabilities())
print("CUTE:", cute.is_available())
print("Sol-Attn:", cute.supports_sol_attn())

assert torch.__version__ == "2.13.0+xpu"
assert metadata.version("omni-xpu-kernel") == "0.2.0b2+torch213.bmg"
assert omni.__xpu_target__ == "bmg"
assert omni.core_aot_target() == "bmg"
assert omni.is_available()
assert cute.is_available()
assert cute.supports_sol_attn()
"@
```

最小 D128 CUTE correctness：

```powershell
@'
import torch
from omni_xpu_kernel import cute

q = torch.randn(1, 256, 8, 128, device="xpu", dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)

actual = cute.sdp(q, k, v)
expected = torch.nn.functional.scaled_dot_product_attention(
    q.permute(0, 2, 1, 3),
    k.permute(0, 2, 1, 3),
    v.permute(0, 2, 1, 3),
).permute(0, 2, 1, 3)

torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
torch.xpu.synchronize()
print("Windows BMG CUTE D128: PASS")
'@ | & $embeddedPython -
```

源码侧当前回归入口：

```powershell
Set-Location $kernelRoot
& $buildPython -m pytest -q `
    tests\test_packaging.py `
    tests\test_cute_sol_attn_api.py
```

## 10. 常见错误

### 找不到 oneDNN 3.11.2

确认 `DNNLROOT` 指向完整 oneAPI oneDNN 2026.0 安装：

```bat
dir "%DNNLROOT%\include\oneapi\dnnl\dnnl.hpp"
dir "%DNNLROOT%\lib\dnnl.lib"
dir "%DNNLROOT%\bin\dnnl.dll"
```

不要混用不同安装根的 headers、import library 和 runtime DLL。自定义布局应
同时设置 `ONEDNN_INCLUDE`、`ONEDNN_LIB`、`ONEDNN_RUNTIME` 和
`ONEDNN_LICENSE_DIR`。

### `LNK1104: cannot open file 'libircmt.lib'`

把 oneAPI compiler library 加入同一个构建 shell：

```bat
set "LIB=%ONEAPI_COMPILER_ROOT%\lib;%ONEAPI_COMPILER_ROOT%\opt\compiler\lib;%LIB%"
```

### 找不到 `llvm-foreach.exe`、`sycl-post-link.exe` 或 `ocloc.exe`

确认 `PATH` 包含：

```text
compiler\2026.0\bin
compiler\2026.0\bin\compiler
ocloc\2026.0\bin
```

### sycl-tla overlay 校验失败

只使用第 5 节固定的干净 revision。overlay 是 fail-closed 的；源码内容不匹配
时必须更新 port 或 pin，不能跳过校验继续构建。

### CUTE `.pyd` 存在但无法加载

确认 Python tag 是 `cp313`，Torch 是 XPU build，wheel target 是 `bmg`，并且
Portable 的 `python_embeded\Library\bin` 和
`python_embeded\Lib\site-packages\torch\lib` 位于 `PATH`。

### 测试完成后 Python 进程不退出

部分 XPU 测试会在所有输出和断言完成后卡在 interpreter teardown。先确认测试
结果已经完整打印，再终止具体 PID；不要删除被进程锁定的 `.pyd`。

## 11. 当前边界

- Windows CUTE/Sol-Attn 当前只验收 BMG；PTL-H 需要独立构建与验收。
- 当前 ComfyUI CUTE route 以已验证 D128 contract 为主；不支持的
  dtype、layout、mask、head dimension 或 GQA contract 回退到 dense
  attention。
- CUTE 是 build-time 和 runtime 双重 opt-in。
- Sol-Attn 还需要固定版本的 ComfyUI custom node 和
  `SOL_ATTN_XPU_EXPERIMENTAL=1`。
