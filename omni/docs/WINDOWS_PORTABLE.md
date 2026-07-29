# Windows Intel XPU ComfyUI Portable 完整部署

本文说明如何在 Windows 上把本仓库的 Omni XPU 组件部署到官方
ComfyUI Intel XPU Portable。流程覆盖：

1. 在项目目录内创建独立构建环境；
2. 构建 `omni_xpu_kernel` Windows wheel；
3. 下载并检查官方 Intel XPU Portable；
4. 将 Portable 对齐到已经验证的 Torch XPU 版本；
5. 从 Intel Portable 的 requirements 中移除上游 `comfy-kitchen`
   依赖，由部署流程单独管理 XPU fork；
6. 构建并安装 XPU-enabled `comfy-kitchen`；
7. 安装 `omni_xpu_kernel` 和 `ComfyUI-OmniXPU` custom node；
8. 修改 Windows 启动脚本并完成分层验收；
9. 在更新 ComfyUI/Portable 后重放 Intel XPU 补丁。

这里的 Portable 只是最终运行环境，不参与原生扩展编译。所有源码、Python
工具链和构建缓存都留在 `llm-scaler` 项目目录中，不修改其他项目环境。

> [!IMPORTANT]
> 本文当前验证目标是 Python 3.13、Torch 2.12 和 Intel BMG。Python ABI、
> Torch minor 和 XPU AOT target 都是 native wheel 身份的一部分。不能把
> `cp313/torch212/bmg` wheel 重命名后用于 Python 3.14、Torch 2.13 或
> PTL-H。

## 1. 组件关系

Windows 部署保留 Dockerfile 中的核心分层，但不复制 Linux-only 部分：

| 层 | Windows 中的组件 | 责任 |
|---|---|---|
| XPU runtime | PyTorch XPU、oneDNN runtime | 设备、张量和 oneDNN/SYCL 运行库 |
| Native kernel | `omni_xpu_kernel` wheel | norm、FP8、INT8、SVDQ、rotary、ESIMD SDP 等 |
| Generic dispatch | `comfy-kitchen` XPU fork | 通用算子 API、capability、dispatch 和 eager fallback |
| ComfyUI adapter | `ComfyUI-OmniXPU` custom node | attention、norm、FP8 model bridge 和 fused INT8 FFN 接入 |
| Application | 官方 ComfyUI Intel Portable | UI、模型加载、workflow 和设备管理 |

Docker image 中的 CUTE FMHA、`sycl-tla`、Linux `.so`、`/dev/dri` 和
`LD_LIBRARY_PATH` 不适用于 Windows。Windows attention 默认保留 ComfyUI
的 PyTorch SDPA，不安装 attention patch。`omni_xpu_kernel` 中的 ESIMD
SDP 仍可通过 `OMNI_ATTN_BACKEND=esimd` 显式启用，但不会被自动选择。

## 2. 当前验证矩阵

以下是 2026-07-29 已经用于 Windows 构建和完整核心工作流测试的组合：

| 组件 | 已验证版本/修订 |
|---|---|
| Windows | Windows 11 Pro x64，build `10.0.26200` |
| Intel Arc Pro driver | `32.0.101.8515` |
| GPU | Intel Arc Pro B70，`intel_gpu_bmg_g31` |
| Visual Studio Build Tools | 2022 `17.14.36` |
| MSVC | v143 `14.42.34433`，`cl 19.42.34444` |
| Intel oneAPI DPC++ compiler | `2025.3.3` |
| Native oneDNN development API | `3.9.1`，来自 oneAPI oneDNN `2025.3` |
| Portable Python | `3.13.12` |
| torch | `2.12.0+xpu` |
| torchvision | `0.27.0+xpu` |
| torchaudio | 当前测试目录为 `2.11.0+xpu`；不是 Omni kernel 必需依赖 |
| onednn Python runtime | `2025.3.0` |
| omni-xpu-kernel | `0.1.0b9.dev0+torch212.bmg` |
| comfy-kitchen XPU fork | `0.2.18`，[`c7ae07e5...`](https://github.com/xiangyuT/comfy-kitchen-xpu/commit/c7ae07e5317d4a073562e278a41d98f05a4fe109) |
| ComfyUI | `0.28.0`，测试 commit `700821e1...` |
| llm-scaler | [`0c6cd57c...`](https://github.com/xiangyuT/llm-scaler/commit/0c6cd57c6034929c3f7ee6efc48d7aa0b72b7bed)（`feature/omni-0.1.0b9-preview`） |

`comfy-kitchen 0.2.18` 是 Dockerfile 当前固定的 XPU fork 版本。当前 ComfyUI
调用的 Kitchen API 已经在该 fork 中完成导入检查，XPU backend 也能在
Torch 2.12/B70 上注册。它应保留真实版本 `0.2.18`，不伪装为上游
ComfyUI requirements 中的其他版本。

`c7ae07e5...` 的祖先包含 Windows Triton 修复 `fead43b4...`。它在 Windows
上默认不注册 Triton backend，因为当前 Portable
路径没有验证 Triton JIT compiler/runtime。默认 dispatch 顺序因此是
`xpu -> eager`。只有已自行验证 Windows Triton toolchain 的环境才应在导入
Kitchen 前设置 `COMFY_KITCHEN_ENABLE_TRITON_WINDOWS=1`。

本轮核心验收不安装 GGUF 或 Nunchaku custom node/model。Kitchen/kernel 中
随 pin 提供的相关通用 capability 保留，第三方节点和端到端工作流放在下一
阶段单独验证。

## 3. 外部依赖

安装或下载：

- [已验证的 ComfyUI v0.28.0 Intel Portable](https://github.com/comfyanonymous/ComfyUI/releases/download/v0.28.0/ComfyUI_windows_portable_intel.7z)
- [ComfyUI v0.28.0 release](https://github.com/Comfy-Org/ComfyUI/releases/tag/v0.28.0)
- [ComfyUI releases](https://github.com/Comfy-Org/ComfyUI/releases)
- [7-Zip](https://7-zip.org/)
- [Intel Arc Pro Windows driver](https://www.intel.com/content/www/us/en/download/741626/intel-arc-pro-graphics-windows.html)
- [Microsoft Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)
- [Visual Studio C++ Build Tools](https://learn.microsoft.com/en-us/cpp/overview/acquire-msvc)
- [Intel oneAPI DPC++/C++ Compiler](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler-download.html)
- [Intel oneAPI Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/oneapi-toolkit.html)
- [uv installation](https://docs.astral.sh/uv/getting-started/installation/)
- [Python 3.13.12](https://www.python.org/downloads/release/python-31312/)
- [PyTorch Intel GPU guide](https://docs.pytorch.org/docs/main/notes/get_start_xpu.html)
- [PyTorch XPU wheel index](https://download.pytorch.org/whl/xpu)
- [`comfy-kitchen-xpu`](https://github.com/xiangyuT/comfy-kitchen-xpu)
- [`comfy-kitchen-xpu` pinned commit](https://github.com/xiangyuT/comfy-kitchen-xpu/commit/c7ae07e5317d4a073562e278a41d98f05a4fe109)

只运行已经构建好的 wheel 时不需要 Visual Studio 和 oneAPI compiler。
它们仅用于构建 `omni_xpu_kernel`。

## 4. 约定路径

下面的 PowerShell 示例使用这些变量。先把两个尖括号占位符替换为本机的
仓库根目录和 Portable 根目录：

```powershell
Set-Location "<llm-scaler-repository-root>"
$repoRoot = (Get-Location).Path
$omniRoot = Join-Path $repoRoot "omni"
$kernelRoot = Join-Path $omniRoot "omni_xpu_kernel"
$buildRoot = Join-Path $kernelRoot ".venv-win-py313-torch212"
$buildPython = Join-Path $buildRoot "venv\Scripts\python.exe"

$portableRoot = (Resolve-Path "<ComfyUI_windows_portable-root>").Path
$comfyRoot = Join-Path $portableRoot "ComfyUI"
$embeddedPython = Join-Path $portableRoot "python_embeded\python.exe"
```

建议把 Portable 解压到较短、无空格的路径。不要在任意命令中把系统 Python
或另一个项目的 venv 替换成 `$embeddedPython`。

## 5. 构建 omni_xpu_kernel

完整的工具链版本、Windows SDK fallback、构建参数、wheel 内容、hash 和
native correctness 测试见
[`omni_xpu_kernel/WHL_BUILD_INSTALL.md`](../omni_xpu_kernel/WHL_BUILD_INSTALL.md)。
本节只保留端到端部署所需的主路径。

### 5.1 创建项目内独立构建环境

在 PowerShell 中：

```powershell
$env:UV_PYTHON_INSTALL_DIR = Join-Path $buildRoot "python"
$env:UV_CACHE_DIR = Join-Path $buildRoot "cache"

Set-Location $kernelRoot

uv python install 3.13.12
uv venv --seed --python 3.13.12 (Join-Path $buildRoot "venv")

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

& $buildPython -m pip check
```

### 5.2 初始化编译器并构建

打开普通 `cmd.exe`，不要在 Portable Python 中构建：

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

where cl
where icx
sycl-ls --verbose

cd /d "%KERNEL_ROOT%"
if not exist "%BUILD_ROOT%\wheelhouse\patched" mkdir "%BUILD_ROOT%\wheelhouse\patched"

"%BUILD_PYTHON%" -m pip wheel . ^
  --wheel-dir "%BUILD_ROOT%\wheelhouse\patched" ^
  --no-build-isolation ^
  --no-deps
```

B70 的 `sycl-ls --verbose` 应包含：

```text
Architecture: intel_gpu_bmg_g31
```

当前已验证输出：

```text
omni_xpu_kernel-0.1.0b9.dev0+torch212.bmg-cp313-cp313-win_amd64.whl
size:   1,563,303 bytes
SHA256: 2F3E7363CB4E913031F125540BABDC1DA933E27928BEA408D1020E0A4CC4DC77
```

在 PowerShell 中取得实际 artifact，并检查 hash：

```powershell
$kernelWheel = Get-ChildItem `
    (Join-Path $buildRoot "wheelhouse\patched") `
    -Filter "omni_xpu_kernel-*.whl" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if (-not $kernelWheel) {
    throw "omni_xpu_kernel wheel not found"
}

Get-FileHash -Algorithm SHA256 -LiteralPath $kernelWheel.FullName
& $buildPython -m zipfile -l $kernelWheel.FullName
```

wheel 中必须至少存在：

```text
omni_xpu_kernel/_C.cp313-win_amd64.pyd
omni_xpu_kernel/lgrf_uni/lgrf_sdp.cp313-win_amd64.pyd
```

## 6. 下载并检查 Intel Portable

官方资产是 `.7z`，本文中的“Portable ZIP”泛指这个可解压的 Portable
发行包。

```powershell
$downloadRoot = "<download-directory>"
$comfyVersion = "v0.28.0"
$archive = Join-Path $downloadRoot "ComfyUI_windows_portable_intel-$comfyVersion.7z"
$extractRoot = "<extract-directory>"
$sevenZip = Join-Path $env:ProgramFiles "7-Zip\7z.exe"

New-Item -ItemType Directory -Force -Path $downloadRoot | Out-Null
New-Item -ItemType Directory -Force -Path $extractRoot | Out-Null

Invoke-WebRequest `
    -Uri "https://github.com/comfyanonymous/ComfyUI/releases/download/$comfyVersion/ComfyUI_windows_portable_intel.7z" `
    -OutFile $archive

& $sevenZip x $archive "-o$extractRoot"
```

本文固定下载已经完成端到端验证的 `v0.28.0`，不使用会随官方发布变化的
`latest`。该 release 中的 ComfyUI 应对应 commit
`700821e1364eaab0e8f21c538a2131719fec57bf`。解压后先核对 commit，并记录
实际 Python、Torch 和 XPU 环境：

```powershell
$expectedComfyCommit = "700821e1364eaab0e8f21c538a2131719fec57bf"
$actualComfyCommit = (git -C $comfyRoot rev-parse HEAD).Trim()

if ($actualComfyCommit -ne $expectedComfyCommit) {
    throw "Unexpected ComfyUI commit: $actualComfyCommit"
}

& $embeddedPython -c @"
import sys
import torch
print("python:", sys.version)
print("torch:", torch.__version__)
print("torch XPU runtime:", torch.version.xpu)
print("XPU available:", torch.xpu.is_available())
print("devices:", [torch.xpu.get_device_name(i) for i in range(torch.xpu.device_count())])
"@

Write-Host "ComfyUI commit: $actualComfyCommit"
& $embeddedPython -m pip list
```

升级 Portable 时，不要只把 URL 改回 `latest`。应重新完成本文的 kernel、
Kitchen、Custom Node 和启动验证，再同时更新 `$comfyVersion` 与
`$expectedComfyCommit`。

如果 Python 不是 3.13，不能使用本文的 `cp313` kernel wheel。如果 Torch
不是 2.12，可以按下一节对齐到 Torch 2.12，或为新的 Torch minor 重新构建
kernel wheel。

在开始修改前，保留原始压缩包，或者复制一份完整 Portable 目录作为回滚
点。不要直接在唯一副本上试验无法恢复的包组合。

## 7. 将 Portable 对齐到 Torch 2.12

先关闭所有使用该 `python_embeded` 的 ComfyUI/Python 进程。

对干净环境推荐安装匹配的 Torch 2.12 XPU 组合：

```powershell
& $embeddedPython -m pip install --force-reinstall `
    --index-url "https://download.pytorch.org/whl/xpu" `
    "torch==2.12.0+xpu" `
    "torchvision==0.27.0+xpu"

& $embeddedPython -m pip install "onednn==2025.3.0"
```

`omni_xpu_kernel` 本身不依赖 torchvision 或 torchaudio。2026-07-29
实际查询 XPU index 时，torchaudio 最新版仍是 `2.11.0+xpu`；当前 Portable
在 Torch 2.12 下的导入和 `pip check` 已通过。如果需要音频节点，可以保留或
单独安装该版本，同时禁止它解析和替换 Torch：

```powershell
& $embeddedPython -m pip install `
    --force-reinstall `
    --no-deps `
    --index-url "https://download.pytorch.org/whl/xpu" `
    "torchaudio==2.11.0+xpu"
```

如果不使用音频节点，可以不安装 torchaudio。

确认没有混入 CPU/CUDA wheel：

```powershell
& $embeddedPython -c @"
import torch
import torchvision
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("XPU:", torch.xpu.is_available())
assert torch.__version__ == "2.12.0+xpu"
assert torch.xpu.is_available()
"@
```

## 8. Patch ComfyUI requirements

### 8.1 管理策略

上游 ComfyUI 会精确固定官方 `comfy-kitchen`，但 Intel Portable 必须安装
包含 `xpu` backend 的 fork。这里不把 fork 伪装成上游固定版本，也不把
requirements 改成无版本的 `comfy-kitchen`：

- 保留 `comfy-kitchen` 行并固定版本，会在依赖同步时安装官方 wheel；
- 只去掉版本、保留 `comfy-kitchen`，仍会被
  `pip install --upgrade -r requirements.txt` 升级为官方 wheel；
- 因此 Intel XPU requirements 必须完全省略 `comfy-kitchen` 包依赖；
- Kitchen XPU wheel 由本部署流程按 commit 单独构建、安装和验收。

### 8.2 应用 patch

目标文件：

```text
ComfyUI_windows_portable\ComfyUI\requirements.txt
```

把其中任意形式的 `comfy-kitchen...` 依赖替换为紧邻说明：

```text
# Intel XPU portable builds install and update the XPU-enabled comfy-kitchen
# fork separately. It is intentionally omitted here: including even an
# unpinned requirement would let `pip install --upgrade -r requirements.txt`
# replace the XPU fork with the upstream wheel. After updating ComfyUI, validate
# its Kitchen API usage before updating the separately managed XPU fork.
```

可以用下面的 PowerShell 对当前上游版本执行一次 patch：

```powershell
$requirementsPath = Join-Path $comfyRoot "requirements.txt"
$kitchenPattern = '^\s*comfy-kitchen(?:\s*[<>=!~].*)?\s*$'
$sourceLines = [System.IO.File]::ReadAllLines($requirementsPath)
$outputLines = [System.Collections.Generic.List[string]]::new()
$patchedKitchen = $false

foreach ($line in $sourceLines) {
    if ($line -match $kitchenPattern) {
        if (-not $patchedKitchen) {
            $outputLines.Add("# Intel XPU portable builds install and update the XPU-enabled comfy-kitchen")
            $outputLines.Add("# fork separately. It is intentionally omitted here: including even an")
            $outputLines.Add("# unpinned requirement would let ``pip install --upgrade -r requirements.txt``")
            $outputLines.Add("# replace the XPU fork with the upstream wheel. After updating ComfyUI, validate")
            $outputLines.Add("# its Kitchen API usage before updating the separately managed XPU fork.")
        }
        $patchedKitchen = $true
        continue
    }
    $outputLines.Add($line)
}

if (-not $patchedKitchen) {
    throw "No comfy-kitchen requirement was found; inspect requirements.txt manually"
}

[System.IO.File]::WriteAllLines(
    $requirementsPath,
    $outputLines,
    [System.Text.UTF8Encoding]::new($false)
)
```

验证包依赖行已经完全移除：

```powershell
Select-String `
    -Path $requirementsPath `
    -Pattern '^\s*comfy-kitchen(?:\s*[<>=!~].*)?\s*$'
```

预期没有输出。注释中出现 `comfy-kitchen` 是正常的。

## 9. 构建并安装 comfy-kitchen XPU fork

Kitchen XPU wheel 是 pure-Python wheel，但仍应在项目内构建环境生成 artifact，
不要把 Portable 当作源码构建目录。

```powershell
$kitchenCommit = "c7ae07e5317d4a073562e278a41d98f05a4fe109"
$sourceRoot = Join-Path $buildRoot "sources"
$kitchenSource = Join-Path $sourceRoot "comfy-kitchen-xpu"
$kitchenWheelhouse = Join-Path $buildRoot "wheelhouse\kitchen"

New-Item -ItemType Directory -Force -Path $sourceRoot | Out-Null
New-Item -ItemType Directory -Force -Path $kitchenWheelhouse | Out-Null

git clone --filter=blob:none --no-checkout `
    "https://github.com/xiangyuT/comfy-kitchen-xpu.git" `
    $kitchenSource

git -C $kitchenSource fetch --depth 1 origin $kitchenCommit
git -C $kitchenSource checkout --detach FETCH_HEAD

& $buildPython -m pip wheel $kitchenSource `
    --wheel-dir $kitchenWheelhouse `
    --no-deps `
    --no-build-isolation
```

如果 `$kitchenSource` 已经存在，不要删除或覆盖一个来源不明的目录。先检查：

```powershell
git -C $kitchenSource remote -v
git -C $kitchenSource status --short
git -C $kitchenSource rev-parse HEAD
```

取得并检查 wheel：

```powershell
$kitchenWheel = Get-ChildItem $kitchenWheelhouse `
    -Filter "comfy_kitchen-0.2.18-*.whl" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if (-not $kitchenWheel) {
    throw "comfy-kitchen XPU wheel not found"
}

& $buildPython -m zipfile -l $kitchenWheel.FullName
```

wheel 应包含：

```text
comfy_kitchen/backends/xpu/
comfy_kitchen/backends/triton/
comfy_kitchen/backends/eager/
```

不应包含 `comfy_kitchen/backends/cuda/`。安装到 Portable：

当前 pin `c7ae07e5...` 的本机 Windows 构建得到：

```text
file:   comfy_kitchen-0.2.18-py3-none-any.whl
size:   113,212 bytes
SHA256: 5E6B5663C1757217D1D51FA098AAE10A119FEEF8513B83DABD0634A5BA1B0D64
```

该 hash 记录本机 artifact；fresh clone 的 ZIP timestamp 可能使 pure-Python
wheel hash 不同。跨机器验收身份仍应以 source commit、distribution version
和 wheel 内容为准。

```powershell
& $embeddedPython -m pip install `
    --force-reinstall `
    --no-deps `
    $kitchenWheel.FullName
```

`--no-deps` 防止 Kitchen 安装过程改变已经确认的 Torch XPU stack。

## 10. 安装 omni_xpu_kernel wheel

```powershell
& $embeddedPython -m pip install `
    --force-reinstall `
    --no-deps `
    $kernelWheel.FullName
```

安装后测试时应离开 `llm-scaler` 源码目录，避免源码 checkout 遮蔽真正安装的
wheel：

```powershell
Set-Location $portableRoot

& $embeddedPython -c @"
from pathlib import Path
import importlib.metadata as metadata
import torch
import omni_xpu_kernel as omni

print("torch:", torch.__version__)
print("kernel distribution:", metadata.version("omni-xpu-kernel"))
print("kernel module:", Path(omni.__file__).resolve())
print("metadata target:", omni.__xpu_target__)
print("core AOT target:", omni.core_aot_target())
print("capabilities:", omni.native_capabilities())

assert torch.__version__ == "2.12.0+xpu"
assert torch.xpu.is_available()
assert omni.__xpu_target__ == "bmg"
assert omni.core_aot_target() == "bmg"
assert omni.is_available()
"@
```

## 11. 安装 ComfyUI-OmniXPU custom node

Custom node 必须来自与 kernel/Kitchen 集成相匹配的 `llm-scaler` source
revision：

```powershell
$customNodeSource = Join-Path $omniRoot "ComfyUI-OmniXPU"
$customNodeRoot = Join-Path $comfyRoot "custom_nodes"
$customNodeTarget = Join-Path $customNodeRoot "ComfyUI-OmniXPU"
$backupRoot = Join-Path $portableRoot "backups"

if (-not (Test-Path $customNodeSource)) {
    throw "ComfyUI-OmniXPU source not found: $customNodeSource"
}

if (Test-Path $customNodeTarget) {
    $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    New-Item -ItemType Directory -Force -Path $backupRoot | Out-Null
    $backupTarget = Join-Path $backupRoot "ComfyUI-OmniXPU-$timestamp"
    Move-Item -LiteralPath $customNodeTarget -Destination $backupTarget
    Write-Host "Existing custom node moved to $backupTarget"
}

robocopy.exe `
    $customNodeSource `
    $customNodeTarget `
    /E `
    /XD __pycache__ `
    /XF *.pyc

if ($LASTEXITCODE -ge 8) {
    throw "Copying ComfyUI-OmniXPU failed: robocopy exit $LASTEXITCODE"
}
```

这里不创建目录 junction，也不做 editable install。Portable 应包含一份独立
副本，移动整个目录后仍能工作。旧节点备份不能留在 `custom_nodes` 中：
ComfyUI 会把任何子目录当作候选 custom node，再次导入旧 adapter。备份必须
放到 Portable 根目录下的 `backups` 或另一个不会被节点扫描的位置。

## 12. Windows 启动脚本

将 Portable 根目录中的 `run_intel_gpu.bat` 改为：

```bat
@echo off
setlocal

set "PORTABLE_ROOT=%~dp0"
set "PYTHON_DIR=%PORTABLE_ROOT%python_embeded"

set "PYTHONHOME="
set "PYTHONPATH="
set "PATH=%PYTHON_DIR%;%PYTHON_DIR%\Scripts;%PYTHON_DIR%\Library\bin;%PYTHON_DIR%\Lib\site-packages\torch\lib;%PATH%"

if not defined OMNIXPU_ENABLE set "OMNIXPU_ENABLE=1"
if not defined OMNI_XPU_REQUIRE_CUTE set "OMNI_XPU_REQUIRE_CUTE=0"
if not defined OMNI_ATTN_BACKEND set "OMNI_ATTN_BACKEND=torch"
if not defined OMNIXPU_INTERPOLATE_FIX set "OMNIXPU_INTERPOLATE_FIX=0"
if not defined OMNI_COMFYUI_RESERVE_VRAM_GB set "OMNI_COMFYUI_RESERVE_VRAM_GB=4"

cd /d "%PORTABLE_ROOT%ComfyUI"
"%PYTHON_DIR%\python.exe" -s main.py ^
  --windows-standalone-build ^
  --reserve-vram "%OMNI_COMFYUI_RESERVE_VRAM_GB%" ^
  %*

pause
```

说明：

- `Library\bin` 和 `torch\lib` 提供 Portable 内的 SYCL/oneDNN/Torch DLL；
- Windows 默认使用 `OMNI_ATTN_BACKEND=torch`，保留 ComfyUI 的 PyTorch
  SDPA，不安装 attention patch；
- ESIMD 保留为显式选项。需要单独诊断或对照时，在启动脚本之前设置
  `OMNI_ATTN_BACKEND=esimd`；它不会被默认或自动启用；
- 显式 ESIMD 不支持的 dtype/layout/shape 仍由 adapter 回退到原始
  PyTorch route；
- Docker 启动脚本默认保留 4 GiB VRAM，这里使用同一默认值；
- `OMNIXPU_INTERPOLATE_FIX` 和其他 legacy global fix 默认保持关闭；
- `%*` 允许在启动脚本后追加其他 ComfyUI 参数。

## 13. 分层验收

### 13.1 版本、DLL 和设备

```powershell
Set-Location $portableRoot

& $embeddedPython -c @"
from pathlib import Path
import importlib.metadata as metadata
import torch
import comfy_kitchen as ck
import omni_xpu_kernel as omni

print("torch:", torch.__version__)
print("XPU runtime:", torch.version.xpu)
print("devices:", [torch.xpu.get_device_name(i) for i in range(torch.xpu.device_count())])
print("kitchen distribution:", metadata.version("comfy-kitchen"))
print("kitchen module:", Path(ck.__file__).resolve())
print("kernel distribution:", metadata.version("omni-xpu-kernel"))
print("kernel module:", Path(omni.__file__).resolve())
print("kitchen backends:", ck.list_backends())

xpu_backend = ck.list_backends().get("xpu", {})
triton_backend = ck.list_backends().get("triton", {})
assert torch.__version__ == "2.12.0+xpu"
assert torch.xpu.is_available()
assert metadata.version("comfy-kitchen") == "0.2.18"
assert xpu_backend.get("available") is True
assert xpu_backend.get("disabled") is False
assert triton_backend.get("available") is False
assert "COMFY_KITCHEN_ENABLE_TRITON_WINDOWS=1" in (
    triton_backend.get("unavailable_reason") or ""
)
assert omni.is_available()
"@
```

### 13.2 Native kernel correctness

```powershell
@'
import torch
from omni_xpu_kernel import norm, sdp

x = torch.randn(8, 2048, device="xpu", dtype=torch.float16)
weight = torch.randn(2048, device="xpu", dtype=torch.float16)
actual = norm.rms_norm(weight, x, eps=1e-6)
x32 = x.float()
expected = (
    x32
    / torch.sqrt(torch.mean(x32 * x32, dim=-1, keepdim=True) + 1e-6)
    * weight.float()
).half()
torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)

q = torch.randn(1, 64, 8, 128, device="xpu", dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)
actual = sdp.sdp(q, k, v)
expected = torch.nn.functional.scaled_dot_product_attention(
    q.permute(0, 2, 1, 3).contiguous(),
    k.permute(0, 2, 1, 3).contiguous(),
    v.permute(0, 2, 1, 3).contiguous(),
).permute(0, 2, 1, 3).contiguous()
torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)

torch.xpu.synchronize()
print("native kernel smoke: PASS")
'@ | & $embeddedPython -
```

更完整的 FP16/BF16/FP32 和其他 kernel 测试见
[`WHL_BUILD_INSTALL.md`](../omni_xpu_kernel/WHL_BUILD_INSTALL.md#82-最小原生-kernel-correctness-smoke)。

当前 BMG core-only Windows wheel 验证的 standalone SDP 配置是
`head_dim=64/128` 的 FP16、BF16。Windows loader 已与 Linux loader
对齐，会解析 sidecar 已导出的 D64、D128 和 FP16 fast-path 符号。

### 13.3 ComfyUI/custom node 启动

```powershell
$env:OMNIXPU_ENABLE = "1"
$env:OMNI_XPU_REQUIRE_CUTE = "0"
$env:OMNI_ATTN_BACKEND = "torch"
$env:OMNIXPU_INTERPOLATE_FIX = "0"

& $embeddedPython (Join-Path $comfyRoot "main.py") `
    --windows-standalone-build `
    --disable-auto-launch `
    --quick-test-for-ci `
    --log-stdout `
    --verbose INFO
```

验收标准：

- 进程返回码为 `0`；
- 日志中的 PyTorch 版本是 `2.12.0+xpu`；
- 至少发现一张预期 Intel XPU；
- `comfy_kitchen` 的 `xpu` backend 为 available；
- `ComfyUI-OmniXPU` 被加载；
- kernel probe 报告预期模块；
- 日志包含 `attention_adapter: skipped`，reason 为
  `OMNI_ATTN_BACKEND=torch (using PyTorch SDPA, no patch)`；
- `rotary_adapter`、`norm_adapter`、`fp8_model_adapter` 和
  `int8_ffn_adapter` applied；
- Custom Node 只被导入一次，备份目录不出现在 import times 中；
- 不出现 custom node import failure 或 native DLL load failure。

启动 UI 后可以添加 **OmniXPU Status** node，检查：

- GPU 和 kernel capabilities；
- Kitchen XPU backend；
- attention/norm/FP8/INT8 adapter 的 apply 状态；
- dispatch 和 fallback 计数。

`--quick-test-for-ci` 只验证启动、导入和设备发现。最终发布还必须用目标模型
workflow 做一次结果正确性和显存行为验收。

### 13.4 Launcher 和 HTTP 服务

先从 Portable 目录之外验证 launcher 的路径解析和参数透传：

```powershell
& cmd.exe /d /c @"
echo.| call "$portableRoot\run_intel_gpu.bat" --disable-auto-launch --quick-test-for-ci --log-stdout --verbose INFO
"@
```

返回码应为 `0`。然后正常运行 launcher，或者在第一个终端显式启动测试
端口：

```powershell
& (Join-Path $portableRoot "run_intel_gpu.bat") `
    --disable-auto-launch `
    --listen 127.0.0.1 `
    --port 8199
```

在第二个终端检查 HTTP：

```powershell
curl.exe --noproxy "*" `
    --silent `
    --show-error `
    --output NUL `
    --write-out "%{http_code}" `
    "http://127.0.0.1:8199/"
```

预期返回 `200`。验收后在第一个终端按 `Ctrl+C` 关闭测试服务。

### 13.5 2026-07-29 实机部署记录

完成的变更：

- 从 `llm-scaler` `0c6cd57c...` 构建并安装 Torch 2.12/BMG Windows
  kernel wheel；
- 从 Kitchen `c7ae07e5...` 构建并安装 XPU fork wheel；
- 保留 requirements 中“完全省略 `comfy-kitchen`、单独管理 XPU fork”的
  说明；
- 把旧 Custom Node 移到 Portable 的 `backups` 目录，再复制匹配 revision
  的新节点；
- 将 launcher 的默认 attention policy 从 ESIMD 改为 PyTorch SDPA；
- GGUF/Nunchaku custom node 和模型不在本轮安装范围。

实际验收结果：

| 检查 | 结果 |
|---|---|
| `pip check` | `No broken requirements found.` |
| Kernel wheel | 1,563,303 bytes；SHA256 `2F3E7363...DC77` |
| Kitchen wheel | 113,212 bytes；SHA256 `5E6B5663...0D64` |
| Kitchen XPU backend | available，未 disabled |
| Kitchen Triton backend | Windows 默认 unavailable；保留显式环境变量 opt-in |
| Native target | `bmg` |
| Kitchen source suite | 445 passed，412 skipped |
| Kernel Windows 支持面 | 501 passed，36 skipped |
| Kernel packaging | 25 passed，4 skipped |
| Platform parity 定点测试 | Norm/Adaln 143 passed、4 skipped；SDP 91 passed、1 skipped；source dispatch 2 passed |
| Attention control flow | 53 passed |
| 双 XPU | 两张 B70 tensor、RMSNorm、D64/D128 ESIMD SDP correctness 通过 |
| 新 capability | Q4_1、GroupNorm、LTX direct RoPE、INT8 shared/prequantized pair 可见 |
| ComfyUI quick-test | 返回码 `0` |
| Launcher quick-test | 从 Portable 外部调用，返回码 `0` |
| Custom Node | 只导入一次；旧节点备份不在扫描目录 |

ComfyUI `0.28.0` 的实际 adapter 日志：

```text
[OmniXPU] omni_xpu_kernel 0.1.0b9.dev0+torch212.bmg - available: sdp, norm, rotary, linear_fp8, int8
[OmniXPU] attention_adapter: skipped (OMNI_ATTN_BACKEND=torch (using PyTorch SDPA, no patch))
[OmniXPU] rotary_adapter: applied
[OmniXPU] norm: H120 FP16 native route enabled (target=bmg)
[OmniXPU] norm: BMG GroupNorm route enabled (target=bmg)
[OmniXPU] norm_adapter: applied
[OmniXPU] fp8_model_adapter: applied
[OmniXPU] int8_ffn_adapter: applied
[OmniXPU] legacy_interpolate_fix: skipped (disabled by env)
[OmniXPU] legacy_median_fix: skipped (disabled by env)
```

相同 1024×1024、相同 workflow 和 seed 的更新前后端到端墙钟对照：

| Workflow | 更新前冷态 | 更新后冷态 | 更新前热态 | 更新后热态 |
|---|---:|---:|---:|---:|
| Z-Image BF16 | 29.666 s | 18.648 s | 9.106 s | 6.039 s |
| Z-Image INT8 ConvRot | 13.411 s | 8.732 s | 7.118 s | 5.042 s |
| Krea2 FP8 | 47.871 s | 22.994 s | 16.236 s | 10.262 s |
| Krea2 INT8 ConvRot | 21.920 s | 16.000 s | 13.379 s | 12.168 s |

四个更新后工作流都完成并保存有效的 1024×1024 RGB 图。相同 seed 的图像
允许因新融合路径产生数值差异；本次人工检查没有黑图、花图、NaN 或语义
损坏。更新后 Z-Image INT8 热态仍比 BF16 快约 16.5%。

当前还会出现以下非阻塞提示：

```text
Could not autodetect AIMDO implementation, assuming Nvidia
onednn_verbose,v1,primitive,error,gpu,jit::gemm,Insufficient registers in requested bundle
```

这是 `comfy-aimdo 0.4.10` 只识别 CUDA/ROCm 导致的探测提示。ComfyUI 随后用
`is_nvidia()` 守卫 DynamicVRAM 初始化，Intel XPU 路径不会启用 NVIDIA
AIMDO。显式添加 `--disable-dynamic-vram` 会触发 ComfyUI 自己的弃用警告，
因此当前 launcher 不添加该参数。`--reserve-vram 4` 仍由 ComfyUI 的常规
XPU memory management 处理。oneDNN JIT register 提示在更新前基线中同样
出现；oneDNN 随后选择可用实现，四个工作流均成功，因此不属于本轮回归。

## 14. 更新与防覆盖

这是 Intel XPU Portable 最容易被忽略的维护边界。

官方 `update\update.py` 会：

1. stash ComfyUI repo 中的本地修改；
2. 拉取/checkout 上游 ComfyUI；
3. 如果 requirements 变化，立即执行上游 requirements 安装。

`update_comfyui_and_python_dependencies.bat` 还会执行：

```text
pip install --upgrade ... -r ../ComfyUI/requirements.txt
```

因此上游更新可能同时：

- 恢复带固定版本的官方 `comfy-kitchen` requirement；
- 用官方 Kitchen wheel 覆盖 XPU fork；
- 升级 Torch minor，使现有 native kernel wheel 失效；
- 更新 ComfyUI API，使 custom node/Kitchen fork 需要重新验收。

在 Intel-aware updater 完成前，每次更新使用以下维护流程：

1. 复制整个 Portable 目录或保留可恢复快照；
2. 记录更新前 Torch、Kitchen、kernel、ComfyUI commit；
3. 运行官方更新；
4. 重新执行第 8 节 requirements patch；
5. 检查 Python ABI 和 Torch minor；
6. 如果仍使用 Python 3.13/Torch 2.12，重新安装已验收的 Kitchen 和 kernel
   wheels；
7. 如果 Torch minor 已变化，恢复 Torch 2.12，或者构建对应的新 kernel
   wheel，不能继续加载旧 wheel；
8. 重新复制匹配 revision 的 `ComfyUI-OmniXPU`；
9. 重跑第 13 节全部验收。

不要只看 `pip install` 是否成功。官方 Kitchen 和 XPU Kitchen 使用相同的
distribution/import name，必须检查：

```powershell
& $embeddedPython -c @"
import importlib.metadata as metadata
import comfy_kitchen as ck
print(metadata.version("comfy-kitchen"))
print(ck.__file__)
print(ck.list_backends().get("xpu"))
"@
```

## 15. 回滚

最可靠的回滚是关闭 ComfyUI，移走当前 Portable 目录，然后重新解压原始
官方 archive 或恢复完整目录快照。

如果只回滚 custom node，使用第 11 节创建的
`backups\ComfyUI-OmniXPU-<timestamp>`。先关闭 ComfyUI，把当前节点移出
`custom_nodes`，再把备份复制回 `custom_nodes\ComfyUI-OmniXPU`。不要把
备份本身留在 `custom_nodes`，也不要删除模型、output 或其他 custom node。

如果只回滚 Python 包，必须成组恢复 Torch XPU、Kitchen 和 kernel；不要把
旧 `omni_xpu_kernel` 留在不同 Torch minor 的环境中。

## 16. 已知限制

- Windows wheel 当前没有 CUTE FMHA；默认使用 PyTorch SDPA，ESIMD SDP
  只作为显式选项。
- Windows BMG ESIMD sidecar 已验证 `head_dim=64/128` 的 FP16、BF16；
  尚未覆盖的 dtype、layout、mask 和 GQA contract 仍由默认 SDPA 策略规避。
- 本文的 native artifact 只验证了 BMG；PTL-H 需要
  `OMNI_XPU_DEVICE=ptl-h` 独立构建和验收。
- Torch 2.13 不在当前 Windows 验证范围内。
- `comfy-kitchen 0.2.18` XPU fork 与未来 ComfyUI API 的兼容性必须在每次
  上游更新后重新测试。
- Dockerfile 中的可选第三方 custom node 集合不属于 Omni 核心依赖。先完成
  kernel、Kitchen、ComfyUI adapter 三层验收，再按 workflow 需求逐个安装。
- legacy interpolate/median global workaround 默认关闭，不应作为基础部署的一
  部分启用。
