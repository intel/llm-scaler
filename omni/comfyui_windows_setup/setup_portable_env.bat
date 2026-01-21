@echo off
setlocal enabledelayedexpansion

REM ============================================
REM Windows Portable Python Environment Setup
REM For ComfyUI with Intel XPU Support
REM Git required for cloning repositories
REM ============================================

REM ============================================
REM Proxy Configuration (Modify as needed)
REM ============================================
REM Uncomment and modify the following lines if you need proxy
REM set "HTTP_PROXY=http://your-proxy-server:port"
REM set "HTTPS_PROXY=http://your-proxy-server:port"
REM set "NO_PROXY=localhost,127.0.0.1"

set "SCRIPT_DIR=%~dp0"
set "PATCHES_DIR=%SCRIPT_DIR%..\patches"
set "PYTHON_DIR=%SCRIPT_DIR%python_embeded"
set "PYTHON_EXE=%PYTHON_DIR%\python.exe"
set "PIP_EXE=%PYTHON_DIR%\Scripts\pip.exe"
set "PYTHON_VERSION=3.12.10"
set "PYTHON_EMBED_URL=https://www.python.org/ftp/python/%PYTHON_VERSION%/python-%PYTHON_VERSION%-embed-amd64.zip"
set "GET_PIP_URL=https://bootstrap.pypa.io/get-pip.py"

REM ComfyUI Git Configuration (matching Dockerfile)
set "COMFYUI_REPO=https://github.com/comfyanonymous/ComfyUI.git"
set "COMFYUI_COMMIT=acd0e536533cdf038bbaa32730cd12a438cc3a60"
set "COMFYUI_PATCH=%PATCHES_DIR%\comfyui_for_multi_arc.patch"

echo ============================================
echo  Windows Portable Python Environment Setup
echo  Target: %PYTHON_DIR%
echo  ComfyUI: git clone + patch
echo ============================================
echo.

REM ============================================
REM Check Dependencies
REM ============================================
echo Checking dependencies...

REM Check for curl (required for downloads)
where curl >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: curl is not installed or not in PATH
    echo Please install curl or add it to your PATH
    echo Windows 10+ should have curl built-in
    pause
    exit /b 1
)
echo [OK] curl found

REM Check for PowerShell (required for zip extraction)
where powershell >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: PowerShell is not installed or not in PATH
    echo Please install PowerShell
    pause
    exit /b 1
)
echo [OK] PowerShell found

REM Check for Git (required for cloning)
where git >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Git is not installed or not in PATH
    echo Please install Git from https://git-scm.com/download/win
    pause
    exit /b 1
)
echo [OK] Git found

echo.
echo Dependencies check complete.
echo.

REM ============================================
REM Step 1: Download and Extract Python Embeddable
REM ============================================
echo [Step 1/8] Setting up Python Embeddable Package...

if exist "%PYTHON_EXE%" (
    echo Python already exists at %PYTHON_DIR%
    echo Skipping download...
) else (
    echo Creating python_embeded directory...
    if not exist "%PYTHON_DIR%" mkdir "%PYTHON_DIR%"
    
    echo Downloading Python %PYTHON_VERSION% Embeddable...
    curl -L -o "%SCRIPT_DIR%python_embed.zip" "%PYTHON_EMBED_URL%"
    if errorlevel 1 (
        echo ERROR: Failed to download Python embeddable package
        echo Please download manually from: %PYTHON_EMBED_URL%
        pause
        exit /b 1
    )
    
    echo Extracting Python...
    powershell -Command "Expand-Archive -Path '%SCRIPT_DIR%python_embed.zip' -DestinationPath '%PYTHON_DIR%' -Force"
    if errorlevel 1 (
        echo ERROR: Failed to extract Python
        pause
        exit /b 1
    )
    
    del "%SCRIPT_DIR%python_embed.zip"
    echo Python extracted successfully.
)

REM ============================================
REM Step 2: Configure Python Path (Enable pip/site-packages)
REM ============================================
echo.
echo [Step 2/8] Configuring Python path...

set "PTH_FILE=%PYTHON_DIR%\python312._pth"
if exist "%PTH_FILE%" (
    echo Modifying %PTH_FILE% to enable site-packages...
    
    REM Create the correct _pth file content with ComfyUI paths
    (
        echo ../ComfyUI
        echo python312.zip
        echo .
        echo Lib\site-packages
        echo import site
    ) > "%PTH_FILE%"
    
    echo Path configuration updated.
) else (
    echo WARNING: _pth file not found. Creating new one...
    (
        echo ../ComfyUI
        echo python312.zip
        echo .
        echo Lib\site-packages
        echo import site
    ) > "%PTH_FILE%"
)

REM Create Lib\site-packages directory if not exists
if not exist "%PYTHON_DIR%\Lib\site-packages" (
    mkdir "%PYTHON_DIR%\Lib\site-packages"
)

REM ============================================
REM Step 3: Install pip
REM ============================================
echo.
echo [Step 3/8] Installing pip...

if exist "%PIP_EXE%" (
    echo pip already installed, upgrading...
    "%PYTHON_EXE%" -m pip install --upgrade pip
) else (
    echo Downloading get-pip.py...
    curl -L -o "%SCRIPT_DIR%get-pip.py" "%GET_PIP_URL%"
    if errorlevel 1 (
        echo ERROR: Failed to download get-pip.py
        pause
        exit /b 1
    )
    
    echo Installing pip...
    "%PYTHON_EXE%" "%SCRIPT_DIR%get-pip.py" --no-warn-script-location
    if errorlevel 1 (
        echo ERROR: Failed to install pip
        pause
        exit /b 1
    )
    
    del "%SCRIPT_DIR%get-pip.py"
    
    REM Upgrade pip
    "%PYTHON_EXE%" -m pip install --upgrade pip
)

echo pip installed successfully.

REM ============================================
REM Step 4: Install PyTorch with Intel XPU Support
REM ============================================
echo.
echo [Step 4/8] Installing PyTorch with Intel XPU support...

"%PYTHON_EXE%" -m pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/xpu
if errorlevel 1 (
    echo WARNING: Failed to install PyTorch XPU version
    echo Trying standard PyTorch...
    "%PYTHON_EXE%" -m pip install torch torchvision torchaudio
)

REM ============================================
REM Step 5: Clone and Setup ComfyUI from Git
REM ============================================
echo.
echo [Step 5/8] Setting up ComfyUI from Git...

cd /d "%SCRIPT_DIR%"

if exist ComfyUI (
    echo ComfyUI directory already exists.
    echo Updating to specified commit...
    cd ComfyUI
    git fetch origin
    git checkout %COMFYUI_COMMIT%
    cd ..
) else (
    echo Cloning ComfyUI from official repository...
    git clone %COMFYUI_REPO%
    if errorlevel 1 (
        echo ERROR: Failed to clone ComfyUI repository
        pause
        exit /b 1
    )
    
    cd ComfyUI
    echo Checking out commit %COMFYUI_COMMIT%...
    git checkout %COMFYUI_COMMIT%
    if errorlevel 1 (
        echo WARNING: Failed to checkout specific commit, using latest
    )
    cd ..
    echo ComfyUI cloned successfully.
)

cd ComfyUI

echo Installing ComfyUI requirements...
"%PYTHON_EXE%" -m pip install -r requirements.txt

REM ============================================
REM Step 6: Apply Intel XPU Patch to ComfyUI
REM ============================================
echo.
echo [Step 6/8] Applying Intel XPU patch to ComfyUI...

cd /d "%SCRIPT_DIR%\ComfyUI"

if exist "%COMFYUI_PATCH%" (
    echo Found patch file: %COMFYUI_PATCH%
    echo Applying Intel XPU patch...
    git apply "%COMFYUI_PATCH%"
    if errorlevel 1 (
        echo WARNING: Patch may have already been applied or failed.
        echo Trying with --check first...
        git apply --check "%COMFYUI_PATCH%" 2>nul
        if errorlevel 1 (
            echo Patch already applied or conflicts exist. Continuing...
        )
    ) else (
        echo Patch applied successfully.
    )
) else (
    echo WARNING: Patch file not found at %COMFYUI_PATCH%
    echo Skipping patch application...
    echo You may need to manually apply Intel XPU patches for full compatibility.
)

REM ============================================
REM Step 7: Install Custom Nodes
REM ============================================
echo.
echo [Step 7/8] Installing Custom Nodes...

cd /d "%SCRIPT_DIR%\ComfyUI\custom_nodes"

REM --- ComfyUI-Manager ---
echo.
echo Installing ComfyUI-Manager...
if exist comfyui-manager (
    echo ComfyUI-Manager already exists, skipping...
) else (
    git clone https://github.com/ltdrdata/ComfyUI-Manager.git comfyui-manager
    if errorlevel 1 (
        echo WARNING: Failed to clone ComfyUI-Manager
    )
)

REM --- ComfyUI-VideoHelperSuite ---
echo.
echo Installing ComfyUI-VideoHelperSuite...
if exist comfyui-videohelpersuite (
    echo ComfyUI-VideoHelperSuite already exists, skipping...
) else (
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git comfyui-videohelpersuite
    if errorlevel 1 (
        echo WARNING: Failed to clone ComfyUI-VideoHelperSuite
    ) else (
        cd comfyui-videohelpersuite
        "%PYTHON_EXE%" -m pip install -r requirements.txt
        cd ..
    )
)

REM --- ComfyUI-Easy-Use ---
echo.
echo Installing ComfyUI-Easy-Use...
if exist comfyui-easy-use (
    echo ComfyUI-Easy-Use already exists, skipping...
) else (
    git clone https://github.com/yolain/ComfyUI-Easy-Use.git comfyui-easy-use
    if errorlevel 1 (
        echo WARNING: Failed to clone ComfyUI-Easy-Use
    ) else (
        cd comfyui-easy-use
        "%PYTHON_EXE%" -m pip install -r requirements.txt
        cd ..
    )
)

REM --- comfyui_controlnet_aux ---
echo.
echo Installing comfyui_controlnet_aux...
if exist comfyui_controlnet_aux (
    echo comfyui_controlnet_aux already exists, skipping...
) else (
    git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git
    if errorlevel 1 (
        echo WARNING: Failed to clone comfyui_controlnet_aux
    ) else (
        cd comfyui_controlnet_aux
        "%PYTHON_EXE%" -m pip install -r requirements.txt
        cd ..
    )
)

REM --- ComfyUI-GGUF ---
echo.
echo Installing ComfyUI-GGUF...
set "GGUF_COMMIT=6ea2651e7df66d7585f6ffee804b20e92fb38b8a"
set "GGUF_PATCH=%PATCHES_DIR%\comfyui_gguf_xpu.patch"

if exist ComfyUI-GGUF (
    echo ComfyUI-GGUF already exists, skipping...
) else (
    git clone https://github.com/city96/ComfyUI-GGUF.git
    if errorlevel 1 (
        echo WARNING: Failed to clone ComfyUI-GGUF
    ) else (
        cd ComfyUI-GGUF
        git checkout %GGUF_COMMIT%
        if exist "%GGUF_PATCH%" (
            echo Applying XPU patch to ComfyUI-GGUF...
            git apply "%GGUF_PATCH%"
            if errorlevel 1 (
                echo WARNING: Failed to apply GGUF patch
            )
        )
        "%PYTHON_EXE%" -m pip install -r requirements.txt
        cd ..
    )
)

REM --- ComfyUI-KJNodes ---
echo.
echo Installing ComfyUI-KJNodes...
if exist ComfyUI-KJNodes (
    echo ComfyUI-KJNodes already exists, skipping...
) else (
    git clone https://github.com/kijai/ComfyUI-KJNodes.git
    if errorlevel 1 (
        echo WARNING: Failed to clone ComfyUI-KJNodes
    ) else (
        cd ComfyUI-KJNodes
        "%PYTHON_EXE%" -m pip install -r requirements.txt
        cd ..
    )
)

echo.
echo Custom nodes installation complete.

REM ============================================
REM Step 8: Create Launcher Scripts
REM ============================================
echo.
echo [Step 8/8] Creating launcher scripts...

cd /d "%SCRIPT_DIR%"

REM Create run_comfyui.bat
(
    echo @echo off
    echo setlocal
    echo set "SCRIPT_DIR=%%~dp0"
    echo set "PYTHON_EXE=%%SCRIPT_DIR%%python_embeded\python.exe"
    echo.
    echo cd /d "%%SCRIPT_DIR%%ComfyUI"
    echo "%%PYTHON_EXE%%" main.py %%*
    echo.
    echo pause
) > run_comfyui.bat

REM Create run_comfyui_lowvram.bat
(
    echo @echo off
    echo setlocal
    echo set "SCRIPT_DIR=%%~dp0"
    echo set "PYTHON_EXE=%%SCRIPT_DIR%%python_embeded\python.exe"
    echo.
    echo cd /d "%%SCRIPT_DIR%%ComfyUI"
    echo "%%PYTHON_EXE%%" main.py --lowvram %%*
    echo.
    echo pause
) > run_comfyui_lowvram.bat

REM Create run_comfyui_cpu.bat for CPU-only mode
(
    echo @echo off
    echo setlocal
    echo set "SCRIPT_DIR=%%~dp0"
    echo set "PYTHON_EXE=%%SCRIPT_DIR%%python_embeded\python.exe"
    echo.
    echo cd /d "%%SCRIPT_DIR%%ComfyUI"
    echo "%%PYTHON_EXE%%" main.py --cpu %%*
    echo.
    echo pause
) > run_comfyui_cpu.bat

echo Launcher scripts created.

REM ============================================
REM Verification
REM ============================================
echo.
echo Verifying installation...

echo.
echo Python version:
"%PYTHON_EXE%" --version

echo.
echo Pip version:
"%PYTHON_EXE%" -m pip --version

echo.
echo PyTorch verification:
"%PYTHON_EXE%" -c "import torch; print(f'PyTorch version: {torch.__version__}')"
"%PYTHON_EXE%" -c "import torch; print(f'XPU available: {torch.xpu.is_available() if hasattr(torch, \"xpu\") else \"N/A\"}')"

echo.
echo Installed Custom Nodes:
dir /b "%SCRIPT_DIR%ComfyUI\custom_nodes"

echo.
echo ============================================
echo  Installation Completed!
echo ============================================
echo.
echo Directory structure:
echo   %SCRIPT_DIR%
echo   +-- python_embeded/           (Python environment)
echo   +-- ComfyUI/                  (ComfyUI application)
echo   ^|   +-- custom_nodes/        (Custom nodes)
echo   ^|       +-- comfyui-manager
echo   ^|       +-- comfyui-videohelpersuite
echo   ^|       +-- comfyui-easy-use
echo   ^|       +-- comfyui_controlnet_aux
echo   ^|       +-- ComfyUI-GGUF
echo   ^|       +-- ComfyUI-KJNodes
echo   +-- run_comfyui.bat           (Launcher)
echo   +-- run_comfyui_lowvram.bat   (Low VRAM Launcher)
echo   +-- run_comfyui_cpu.bat       (CPU-only Launcher)
echo.
echo To start ComfyUI, run: run_comfyui.bat
echo.
echo NOTE: First launch will take longer time for initialization
echo       and dependency checking. Please be patient.
echo.
echo For low VRAM systems, run: run_comfyui_lowvram.bat
echo.
pause
