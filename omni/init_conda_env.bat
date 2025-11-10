@echo off
setlocal enabledelayedexpansion

REM Configure Git for Windows compatibility
git config --global core.autocrlf false
git config --global core.filemode false

echo ============================================
echo Creating Conda Environment: omni_env
echo ============================================

REM Create conda environment
call conda create -n omni_env python=3.12 pip -y
if errorlevel 1 (
    echo ERROR: Failed to create conda environment
    pause
    exit /b 1
)

echo.
echo ============================================
echo Installing PyTorch and dependencies
echo ============================================

REM Install PyTorch and dependencies using conda run
call conda run -n omni_env pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/xpu
call conda run -n omni_env pip install oneccl_bind_pt==2.8.0+xpu --index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/cn/
call conda run -n omni_env pip install bigdl-core-xe-all==2.7.0b20250625

echo.
echo ============================================
echo Installing ComfyUI
echo ============================================

REM Install ComfyUI
if exist ComfyUI (
    echo ComfyUI directory already exists, skipping clone
) else (
    git clone https://github.com/comfyanonymous/ComfyUI.git
    if errorlevel 1 (
        echo ERROR: Failed to clone ComfyUI
        pause
        exit /b 1
    )
)

cd ComfyUI
git checkout 51696e3fdcdfad657cb15854345fbcbbe70eef8d

REM Apply patch using forward slashes for Git
git apply --reject --whitespace=fix --ignore-whitespace ../patches/comfyui_for_multi_arc.patch
if errorlevel 1 (
    echo WARNING: Patch may not have applied cleanly. Check for .rej files.
    echo Continuing installation...
)

call conda run -n omni_env pip install -r requirements.txt

if not exist custom_nodes (
    mkdir custom_nodes
)
cd custom_nodes

echo.
echo ============================================
echo Installing ComfyUI-Manager
echo ============================================
if exist comfyui-manager (
    echo ComfyUI-Manager already exists, skipping
) else (
    git clone https://github.com/ltdrdata/ComfyUI-Manager.git comfyui-manager
)

echo.
echo ============================================
echo Installing ComfyUI-VideoHelperSuite
echo ============================================
if exist comfyui-videohelpersuite (
    echo ComfyUI-VideoHelperSuite already exists, skipping
) else (
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git comfyui-videohelpersuite
)
cd comfyui-videohelpersuite
call conda run -n omni_env pip install -r requirements.txt
cd ..


echo.
echo ============================================
echo Installing ComfyUI-Easy-Use
echo ============================================
if exist comfyui-easy-use (
    echo ComfyUI-Easy-Use already exists, skipping
) else (
    git clone https://github.com/yolain/ComfyUI-Easy-Use.git comfyui-easy-use
)
cd comfyui-easy-use
call conda run -n omni_env pip install -r requirements.txt
cd ..

echo.
echo ============================================
echo Installing comfyui_controlnet_aux
echo ============================================
if exist comfyui_controlnet_aux (
    echo comfyui_controlnet_aux already exists, skipping
) else (
    git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git
)
cd comfyui_controlnet_aux
call conda run -n omni_env pip install -r requirements.txt
cd ..

echo.
echo ============================================
echo Installing ComfyUI-VoxCPM
echo ============================================
if exist comfyui-voxcpm (
    echo ComfyUI-VoxCPM already exists, skipping
) else (
    git clone https://github.com/wildminder/ComfyUI-VoxCPM.git comfyui-voxcpm
)
cd comfyui-voxcpm
git checkout 044dd93c0effc9090fb279117de5db4cd90242a0

REM Apply patch using forward slashes
git apply --reject --whitespace=fix --ignore-whitespace ../../../patches/comfyui_voxcpm_for_xpu.patch
if errorlevel 1 (
    echo WARNING: Patch may not have applied cleanly. Check for .rej files.
    echo Continuing installation...
)

call conda run -n omni_env pip install -r requirements.txt
cd ..

echo.
echo ============================================
echo Installation completed!
echo ============================================
echo.
echo To activate the environment, run:
echo   conda activate omni_env
echo.
echo If there were patch warnings, check for .rej files in the directories.
echo.
pause
