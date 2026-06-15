@echo off
REM Build script for omni_xpu_kernel on Windows
REM Usage:
REM   scripts\install_windows_onednn.bat          # Build and install
REM   scripts\install_windows_onednn.bat --dev    # Install in development mode

REM Parse arguments
:parse_args
if "%~1"=="" goto :done_parsing
if "%~1"=="--dev" (
    set "BUILD_DEV_MODE=1"
    shift
    goto :parse_args
)

echo Unknown option: %~1
echo Usage: %~nx0 [--dev]
exit /b 1

:done_parsing


REM Check if Intel oneAPI is available
where icx >nul 2>&1
if errorlevel 1 (
    echo.
    echo ERROR: Intel icx compiler not found!
    echo Please install Intel oneAPI Base Toolkit and run setvars.bat first:
    echo.
    echo   "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
    echo.
    echo Download Intel oneAPI from:
    echo   https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html
    echo.
    exit /b 1
)

REM Check for PyTorch
python -c "import torch; print(f'PyTorch version: {torch.__version__}')" 2>nul
if errorlevel 1 (
    echo.
    echo ERROR: PyTorch not found!
    echo Please install PyTorch with XPU support first.
    echo.
    exit /b 1
)

set OMNI_XPU_ONEAPI_VERSION=2025.3
set OMNI_XPU_ENABLE_ONEDNN=1



@REM Install It!
if "%BUILD_DEV_MODE%"=="1" (
    echo Installing in development mode...
    if not exist build_windows mkdir build_windows
    pip install -e . --no-build-isolation -vvv > build_windows/build_windows_log.txt 2>&1
) else (
    echo Installing...
    pip install . --no-build-isolation
)



if errorlevel 1 (
    echo.
    echo Build failed!
    exit /b 1
)

REM Verify installation
echo.
echo Verifying installation...
python -c "import omni_xpu_kernel; print(f'omni_xpu_kernel version: {omni_xpu_kernel.__version__}')"
if errorlevel 1 (
    echo.
    echo WARNING: Failed to import omni_xpu_kernel
    exit /b 1
)

echo.
echo Build successful!
echo.

endlocal