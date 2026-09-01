# Windows Intel XPU ComfyUI Portable 部署

本文定义当前 ComfyUI Intel Portable 的安装、运行时策略和验收合同。

> [!IMPORTANT]
> 当前目标必须保留 Portable 自带的 Python 3.13.14、Torch 2.13.0+xpu、
> torchvision 0.28.0+xpu 和 torchaudio 2.11.0+xpu。安装 Omni、Kitchen、
> AIMDO 或 custom node 时不得让 pip 替换这组 XPU packages。

## 1. 当前部署合同

| 组件 | 当前版本或修订 |
|---|---|
| ComfyUI Intel Portable | `0.34.0` |
| ComfyUI source | `12d5279438bfefc058a269eae805ceab6047777f` |
| Python | `3.13.14` |
| torch | `2.13.0+xpu` |
| torchvision | `0.28.0+xpu` |
| torchaudio | `2.11.0+xpu` |
| GPU target | Intel Arc Pro B70 / `bmg` / `intel_gpu_bmg_g31` |
| oneAPI compiler | `2026.0` |
| oneDNN | `3.11.2` / package release `2026.0.0` |
| omni-xpu-kernel | `0.2.0b2+torch213.bmg` |
| sycl-tla | `2fc09973bfdf15755090fcb0e3b6ad236408a992` |
| official comfy-kitchen | `0.2.31` |
| comfy-kitchen XPU provider | `0.2.31` / `9eccb7fa42edf14bc4a4c41aafd645ff1f1dcb75` |
| official comfy-aimdo | `0.4.15` |
| comfy-aimdo XPU provider | `0.4.15` / `4972231de3141ccaf26cb44818a4977fcecf55ab` |
| ComfyUI-OmniXPU | 当前 `llm-scaler` source |
| ComfyUI-SolAttn | `5f1c4aac3ca32a00b0b4c15ddbb7cb53fa43344d` |

当前 kernel wheel：

```text
omni_xpu_kernel-0.2.0b2+torch213.bmg-cp313-cp313-win_amd64.whl
SHA256: 054B5AD9B7AC046153446A249ADBBAED56C95F00CC82FB40C5AF595F6345183A
```

该 wheel 包含 core kernels、ESIMD SDP、CUTE FMHA、Sol-Attn 和 matched
oneDNN runtime。

## 2. 组件关系

| 层 | Windows 组件 | 作用 |
|---|---|---|
| XPU runtime | Portable 的 PyTorch XPU | 设备、张量、SYCL/UR/OpenMP runtime |
| Native kernel | `omni_xpu_kernel` | norm、FP8、INT8、SVDQ、rotary、ESIMD、CUTE、Sol-Attn |
| Official API | `comfy-kitchen`、`comfy-aimdo` | ComfyUI 使用的官方 package contract |
| XPU providers | `comfy-kitchen-xpu-runtime`、`comfy-aimdo-xpu-runtime` | 与官方版本共存的 Intel XPU implementation |
| ComfyUI adapter | `ComfyUI-OmniXPU` | provider bootstrap、attention、norm、FP8 和 INT8 路由 |
| Sol-Attn dispatch | `ComfyUI-SolAttn` | workflow 级 sparse attention patch |
| Application | ComfyUI Intel Portable 0.34.0 | UI、workflow、模型和设备管理 |

Kitchen/AIMDO provider 不覆盖官方 package 文件。`ComfyUI-OmniXPU` 在
prestartup 阶段验证 official version、provider manifest、Torch XPU、
Windows platform 和 `bmg` target 后再激活 provider。

## 3. 路径约定

PowerShell 示例使用以下路径：

```powershell
$repoRoot = (Resolve-Path "<llm-scaler-repository-root>").Path
$omniRoot = Join-Path $repoRoot "omni"
$workspaceRoot = Split-Path $repoRoot -Parent
$buildRoot = Join-Path $workspaceRoot ".omni-portable-build"
$buildPython = Join-Path $buildRoot "venv\Scripts\python.exe"

$portableRoot = (Resolve-Path "<ComfyUI_windows_portable-root>").Path
$comfyRoot = Join-Path $portableRoot "ComfyUI"
$embeddedPython = Join-Path $portableRoot "python_embeded\python.exe"
$customNodesRoot = Join-Path $comfyRoot "custom_nodes"
```

构建环境保留在 `$buildRoot`，不要放进 Portable。Portable 只安装最终 wheel
和 custom nodes。

## 4. 验证 Portable 基础环境

关闭所有使用目标 Portable Python 的进程，然后运行：

```powershell
& $embeddedPython -c @"
from pathlib import Path
import torch
import torchvision
import torchaudio

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("torchaudio:", torchaudio.__version__)
print("XPU:", torch.xpu.is_available())
print("devices:", [torch.xpu.get_device_name(i) for i in range(torch.xpu.device_count())])

assert torch.__version__ == "2.13.0+xpu"
assert torchvision.__version__ == "0.28.0+xpu"
assert torchaudio.__version__ == "2.11.0+xpu"
assert torch.xpu.is_available()
"@

$actualComfyCommit = (git -C $comfyRoot rev-parse HEAD).Trim()
if ($actualComfyCommit -ne "12d5279438bfefc058a269eae805ceab6047777f") {
    throw "Unexpected ComfyUI revision: $actualComfyCommit"
}
```

如果版本不匹配，停止安装并重新取得目标 Portable。不要在此流程中通过 pip
修改 Torch stack。

## 5. 构建并安装 omni-xpu-kernel

按
[`omni_xpu_kernel/WHL_BUILD_INSTALL.md`](../omni_xpu_kernel/WHL_BUILD_INSTALL.md)
准备持久 venv、oneAPI 2026.0、oneDNN 3.11.2 和固定 sycl-tla，并设置：

```bat
set "OMNI_XPU_DEVICE=bmg"
set "CUTLASS_SYCL_ROOT=<persistent-build-root>\sycl-tla"
set "OMNI_XPU_REQUIRE_CUTE=1"
```

选择当前 wheel 并安装：

```powershell
$kernelWheel = Get-ChildItem `
    (Join-Path $buildRoot "wheels\kernel-solattn") `
    -Filter "omni_xpu_kernel-0.2.0b2+torch213.bmg-cp313-cp313-win_amd64.whl" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if (-not $kernelWheel) {
    throw "Current omni_xpu_kernel wheel not found"
}

$expectedKernelHash = "054B5AD9B7AC046153446A249ADBBAED56C95F00CC82FB40C5AF595F6345183A"
$actualKernelHash = (Get-FileHash -Algorithm SHA256 $kernelWheel.FullName).Hash
if ($actualKernelHash -ne $expectedKernelHash) {
    throw "Unexpected kernel wheel SHA256: $actualKernelHash"
}

& $embeddedPython -m pip install `
    --force-reinstall `
    --no-deps `
    $kernelWheel.FullName
```

`--no-deps` 是必须的；kernel metadata 不应触发 Portable Torch stack
重新解析。

## 6. 安装 Kitchen 和 AIMDO XPU providers

### 6.1 安装 official distributions

先安装与 provider contract 相同版本的官方 packages：

```powershell
& $embeddedPython -m pip install `
    --force-reinstall `
    --no-deps `
    "comfy-kitchen==0.2.31" `
    "comfy-aimdo==0.4.15"
```

### 6.2 Kitchen provider

准备固定 source：

```powershell
$kitchenSource = Join-Path $buildRoot "comfy-kitchen-xpu"
$kitchenCommit = "9eccb7fa42edf14bc4a4c41aafd645ff1f1dcb75"
$kitchenSourceWheels = Join-Path $buildRoot "wheels\kitchen-source"
$providerWheels = Join-Path $buildRoot "wheels\providers"

if (-not (Test-Path (Join-Path $kitchenSource ".git"))) {
    git clone --filter=blob:none --no-checkout `
        "https://github.com/xiangyuT/comfy-kitchen-xpu.git" `
        $kitchenSource
}

git -C $kitchenSource fetch --depth 1 origin $kitchenCommit
git -C $kitchenSource checkout --detach $kitchenCommit

New-Item -ItemType Directory -Force `
    -Path $kitchenSourceWheels, $providerWheels | Out-Null

& $buildPython -m pip wheel $kitchenSource `
    --wheel-dir $kitchenSourceWheels `
    --no-build-isolation `
    --no-deps

$kitchenSourceWheel = Get-ChildItem $kitchenSourceWheels `
    -Filter "comfy_kitchen-0.2.31-*.whl" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

& $buildPython `
    (Join-Path $kitchenSource "packaging\xpu_runtime_provider\build_wheel.py") `
    --source-wheel $kitchenSourceWheel.FullName `
    --output-dir $providerWheels `
    --source-revision $kitchenCommit `
    --torch-version "2.13.0+xpu" `
    --xpu-target bmg
```

### 6.3 AIMDO provider

Windows AIMDO provider 需要 Level Zero headers、Microsoft Detours、Visual
Studio Build Tools 和 oneAPI compiler。固定 source 后，按 source 中的
`docs\WINDOWS_XPU_BUILD_TEST_ACCEPTANCE.md` 准备依赖，并使用其 build
scripts：

```powershell
$aimdoSource = Join-Path $buildRoot "comfy-aimdo-xpu"
$aimdoCommit = "4972231de3141ccaf26cb44818a4977fcecf55ab"
$aimdoSourceWheels = Join-Path $buildRoot "wheels\aimdo-source-0415"

if (-not (Test-Path (Join-Path $aimdoSource ".git"))) {
    git clone --filter=blob:none --no-checkout `
        "https://github.com/xiangyuT/comfy-aimdo-xpu.git" `
        $aimdoSource
}

git -C $aimdoSource fetch origin $aimdoCommit
git -C $aimdoSource checkout --detach $aimdoCommit

Set-Location $aimdoSource
cmd /d /c scripts\build-windows-detours.cmd
if ($LASTEXITCODE -ne 0) {
    throw "AIMDO Detours build failed"
}

cmd /d /c scripts\build-windows-xpu.cmd
if ($LASTEXITCODE -ne 0) {
    throw "AIMDO Windows XPU build failed"
}

New-Item -ItemType Directory -Force -Path $aimdoSourceWheels | Out-Null

$previousPretendVersion = $env:SETUPTOOLS_SCM_PRETEND_VERSION
try {
    $env:SETUPTOOLS_SCM_PRETEND_VERSION = "0.4.15"
    & $buildPython -m pip wheel . `
        --wheel-dir $aimdoSourceWheels `
        --no-build-isolation `
        --no-deps
}
finally {
    if ($null -eq $previousPretendVersion) {
        Remove-Item Env:SETUPTOOLS_SCM_PRETEND_VERSION -ErrorAction SilentlyContinue
    }
    else {
        $env:SETUPTOOLS_SCM_PRETEND_VERSION = $previousPretendVersion
    }
}

$aimdoSourceWheel = Get-ChildItem $aimdoSourceWheels `
    -Filter "comfy_aimdo-0.4.15-*.whl" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

& $buildPython `
    (Join-Path $aimdoSource "packaging\xpu_runtime_provider\build_wheel.py") `
    --source-wheel $aimdoSourceWheel.FullName `
    --output-dir $providerWheels `
    --source-revision $aimdoCommit `
    --torch-version "2.13.0+xpu" `
    --xpu-target bmg
```

### 6.4 安装 providers

```powershell
$kitchenProvider = Get-ChildItem $providerWheels `
    -Filter "comfy_kitchen_xpu_runtime-0.2.31-*.whl" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

$aimdoProvider = Get-ChildItem $providerWheels `
    -Filter "comfy_aimdo_xpu_runtime-0.4.15-*.whl" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if (-not $kitchenProvider -or -not $aimdoProvider) {
    throw "Current Kitchen/AIMDO provider wheels were not found"
}

& $embeddedPython -m pip install `
    --force-reinstall `
    --no-deps `
    $kitchenProvider.FullName `
    $aimdoProvider.FullName
```

## 7. 安装 custom nodes

### 7.1 ComfyUI-OmniXPU

把本 branch 的 node 复制到 Portable。更新时先把目标移到
`$portableRoot\backups`，避免两个副本同时留在 `custom_nodes`：

```powershell
$omniNodeSource = Join-Path $omniRoot "ComfyUI-OmniXPU"
$omniNodeTarget = Join-Path $customNodesRoot "ComfyUI-OmniXPU"

if (Test-Path $omniNodeTarget) {
    $backupRoot = Join-Path $portableRoot "backups"
    New-Item -ItemType Directory -Force -Path $backupRoot | Out-Null
    $backupName = "ComfyUI-OmniXPU-" + (Get-Date -Format "yyyyMMdd-HHmmss")
    Move-Item $omniNodeTarget (Join-Path $backupRoot $backupName)
}

Copy-Item -Recurse -Force $omniNodeSource $omniNodeTarget
```

### 7.2 ComfyUI-SolAttn

Sol-Attn custom node 只负责 ComfyUI dispatch。不要安装它的 Triton/native
requirements；Windows native implementation 已位于 `omni_xpu_kernel`
CUTE sidecar。

```powershell
$solAttnTarget = Join-Path $customNodesRoot "ComfyUI-SolAttn"
$solAttnCommit = "5f1c4aac3ca32a00b0b4c15ddbb7cb53fa43344d"

if (Test-Path $solAttnTarget) {
    throw "Move the existing ComfyUI-SolAttn directory out of custom_nodes first"
}

git clone --filter=blob:none --no-checkout `
    "https://github.com/xiangyuT/ComfyUI-SolAttn_xpu.git" `
    $solAttnTarget
git -C $solAttnTarget fetch --depth 1 origin $solAttnCommit
git -C $solAttnTarget checkout --detach $solAttnCommit
```

Workflow 必须在 model loader 后添加 **Patch Sol-Attn** 才会启用 sparse
attention。未使用该 node 的 workflow 继续走 CUTE dense route。

### 7.3 其他 Dockerfile custom nodes

按 workflow 需要安装，不要让 requirements resolver 修改 Torch stack：

| Node | 当前 pin |
|---|---|
| ComfyUI-GGUF-XPU | `39671fe73117ba97de7011e7e06e32599dcda06d` |
| ComfyUI-nunchaku-XPU | `cc0f6236b6c329178ad4ef58452a874e774c7b8e` |

先在独立 venv 解析和检查 requirements，再用明确版本或本地 wheel 安装到
Portable。native package 安装继续使用 `--no-deps`。

## 8. 共享外部模型

不需要复制大体积权重。可在 Portable 根目录创建
`extra_model_paths_shared.yaml`：

```yaml
shared_models:
  base_path: <shared-ComfyUI-root>
  checkpoints: models/checkpoints
  configs: models/configs
  loras: models/loras
  vae: models/vae
  text_encoders: |
    models/text_encoders
    models/clip
  diffusion_models: |
    models/diffusion_models
    models/unet
  clip_vision: models/clip_vision
  embeddings: models/embeddings
  controlnet: models/controlnet
  upscale_models: models/upscale_models
  audio_encoders: models/audio_encoders
```

`base_path` 使用绝对 Windows 路径。未列出的 model type 继续使用当前
Portable 自己的 `ComfyUI\models`。

## 9. 启动脚本

当前 `run_intel_gpu.bat` 应包含以下 runtime policy：

```bat
@echo off
setlocal

set "PORTABLE_ROOT=%~dp0"
set "PYTHON_DIR=%PORTABLE_ROOT%python_embeded"
set "EXTRA_MODEL_PATHS=%PORTABLE_ROOT%extra_model_paths_shared.yaml"

set "PYTHONHOME="
set "PYTHONPATH="
set "PATH=%PYTHON_DIR%;%PYTHON_DIR%\Scripts;%PYTHON_DIR%\Library\bin;%PYTHON_DIR%\Lib\site-packages\torch\lib;%PATH%"

if not defined OMNIXPU_ENABLE set "OMNIXPU_ENABLE=1"
if not defined OMNIXPU_PROVIDER_BOOTSTRAP set "OMNIXPU_PROVIDER_BOOTSTRAP=required"
if not defined OMNI_IMAGE_XPU_TARGET set "OMNI_IMAGE_XPU_TARGET=bmg"
if not defined OMNI_ATTN_BACKEND set "OMNI_ATTN_BACKEND=cute"
if not defined SOL_ATTN_XPU_EXPERIMENTAL set "SOL_ATTN_XPU_EXPERIMENTAL=1"
if not defined OMNIXPU_INTERPOLATE_FIX set "OMNIXPU_INTERPOLATE_FIX=0"
if not defined OMNI_COMFYUI_RESERVE_VRAM_GB set "OMNI_COMFYUI_RESERVE_VRAM_GB=4"

cd /d "%PORTABLE_ROOT%ComfyUI"
"%PYTHON_DIR%\python.exe" -s main.py ^
  --windows-standalone-build ^
  --extra-model-paths-config "%EXTRA_MODEL_PATHS%" ^
  --enable-dynamic-vram ^
  --reserve-vram "%OMNI_COMFYUI_RESERVE_VRAM_GB%" ^
  %*

pause
```

关键策略：

- `OMNIXPU_PROVIDER_BOOTSTRAP=required`：Kitchen/AIMDO 任一 provider
  未激活就直接失败。
- `OMNI_ATTN_BACKEND=cute`：显式启用 Windows CUTE route。
- `SOL_ATTN_XPU_EXPERIMENTAL=1`：允许固定 custom node 使用 packaged
  Sol-Attn operators。
- `--enable-dynamic-vram`：AIMDO XPU provider 的必要条件。
- `--reserve-vram 4`：为模型切换保留 4 GiB，可按 workload 调整。

`OMNI_XPU_REQUIRE_CUTE` 只用于构建 wheel，不是 launcher runtime 开关。

## 10. 安装态验收

### 10.1 Package identity

```powershell
Set-Location $portableRoot

& $embeddedPython -c @"
from importlib import metadata
from pathlib import Path

import torch
import omni_xpu_kernel as omni
from omni_xpu_kernel import cute

expected = {
    "torch": "2.13.0+xpu",
    "torchvision": "0.28.0+xpu",
    "torchaudio": "2.11.0+xpu",
    "omni-xpu-kernel": "0.2.0b2+torch213.bmg",
    "comfy-kitchen": "0.2.31",
    "comfy-kitchen-xpu-runtime": "0.2.31",
    "comfy-aimdo": "0.4.15",
    "comfy-aimdo-xpu-runtime": "0.4.15",
}

for name, version in expected.items():
    actual = metadata.version(name)
    print(f"{name}: {actual}")
    assert actual == version

print("kernel module:", Path(omni.__file__).resolve())
print("target:", omni.__xpu_target__, omni.core_aot_target())
print("capabilities:", omni.native_capabilities())
print("CUTE:", cute.is_available())
print("Sol-Attn:", cute.supports_sol_attn())

assert torch.xpu.is_available()
assert omni.__xpu_target__ == "bmg"
assert omni.core_aot_target() == "bmg"
assert omni.is_available()
assert cute.is_available()
assert cute.supports_sol_attn()
"@

& $embeddedPython -m pip check
```

### 10.2 Source tests

```powershell
Set-Location (Join-Path $omniRoot "omni_xpu_kernel")
& $buildPython -m pytest -q `
    tests\test_packaging.py `
    tests\test_cute_sol_attn_api.py

Set-Location $omniRoot
& $buildPython -m pytest -q tests\test_comfyui_omnixpu_attention.py
```

Runtime-provider source tests 包含 Linux allocator contract，不作为 Windows
验收入口。Windows provider 状态由下一节的 ComfyUI `required` mode
prestartup 检查。

### 10.3 ComfyUI quick-start

```powershell
$env:OMNIXPU_ENABLE = "1"
$env:OMNIXPU_PROVIDER_BOOTSTRAP = "required"
$env:OMNI_IMAGE_XPU_TARGET = "bmg"
$env:OMNI_ATTN_BACKEND = "cute"
$env:SOL_ATTN_XPU_EXPERIMENTAL = "1"

& $embeddedPython (Join-Path $comfyRoot "main.py") `
    --windows-standalone-build `
    --enable-dynamic-vram `
    --reserve-vram 4 `
    --disable-auto-launch `
    --quick-test-for-ci `
    --database-url "sqlite:///:memory:" `
    --log-stdout `
    --verbose INFO
```

验收必须确认：

- ComfyUI 报告 `0.34.0`、Torch `2.13.0+xpu` 和 Arc Pro B70；
- `ComfyUI-OmniXPU` 只导入一次；
- Kitchen 和 AIMDO providers 都在 `required` mode 激活；
- CUTE `.pyd` 成功加载；
- `ComfyUI-SolAttn` 成功导入；
- 没有 fallback-shaped provider success、DLL load error 或 package mismatch。

当前 Windows 回归结论：

- kernel packaging 和 Sol-Attn API tests：36 passed，4 skipped；
- ComfyUI attention routing tests：160 passed；
- BMG D128 CUTE FP16 对 Torch SDPA 最大绝对误差
  `1.52587890625e-05`；
- MiniMax H3 adapter product smoke 命中
  `minimax_h3_h56_bf16_d128_qkv_bhld`，无 fallback，最大误差
  `0.001953125`；
- Sol-Attn kernel correctness、custom-node dispatch 和 ComfyUI
  quick-start 均通过。

## 11. 更新规则

更新 ComfyUI 或任一 official distribution 后，按以下顺序重新验证：

1. 确认 Python、Torch、torchvision 和 torchaudio 仍符合第 1 节；
2. 确认 ComfyUI source revision 和 requirements 与目标版本一致；
3. 重新安装相同版本的 official Kitchen/AIMDO 和 matching provider wheels；
4. 重新安装当前 kernel wheel；
5. 更新 `ComfyUI-OmniXPU` source copy；
6. 确认 `ComfyUI-SolAttn` 仍为固定 revision；
7. 运行第 10 节全部检查和实际使用的 workflow。

不要把 updater 产生的 package resolver 结果直接视为有效 XPU 环境。

## 12. 当前边界

- Windows CUTE/Sol-Attn 当前只验收 BMG；PTL-H 需要独立 port 和验收。
- CUTE 只覆盖已验证 tensor contracts；其他 dtype、layout、mask、head
  dimension 和 GQA 输入回退到 dense attention。
- Sol-Attn 不会全局自动启用，workflow 必须使用 **Patch Sol-Attn**。
- AIMDO provider 依赖 DynamicVRAM；关闭 DynamicVRAM 时
  `OMNIXPU_PROVIDER_BOOTSTRAP=required` 会拒绝启动。
- Kitchen Triton backend 不属于当前 Windows 必需路径。
- 部分 XPU 测试完成全部断言后可能卡在 interpreter teardown；确认输出完整
  后应终止具体 PID，避免 `.pyd` 长时间被锁定。
