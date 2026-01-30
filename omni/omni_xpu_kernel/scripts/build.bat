@echo off
REM Build script for omni_xpu_kernel on Windows
REM
REM Usage:
REM   scripts\build.bat          # Build and install
REM   scripts\build.bat --dev    # Install in development mode
REM   scripts\build.bat --clean  # Clean and rebuild
REM
REM Prerequisites:
REM   1. Intel oneAPI Base Toolkit installed
REM   2. Run setvars.bat before building:
REM      "C:\Program Files\Intel\oneAPI\setvars.bat"
REM   3. PyTorch with XPU support installed

setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "PROJECT_DIR=%SCRIPT_DIR%.."

cd /d "%PROJECT_DIR%"

REM Parse arguments
set "DEV_MODE=0"
set "CLEAN=0"

:parse_args
if "%~1"=="" goto :done_parsing
if "%~1"=="--dev" (
    set "DEV_MODE=1"
    shift
    goto :parse_args
)
if "%~1"=="-e" (
    set "DEV_MODE=1"
    shift
    goto :parse_args
)
if "%~1"=="--clean" (
    set "CLEAN=1"
    shift
    goto :parse_args
)
echo Unknown option: %~1
echo Usage: %~nx0 [--dev^|-e] [--clean]
exit /b 1

:done_parsing

REM Check if Intel oneAPI is available
where icx >nul 2>&1
if errorlevel 1 (
    echo.
    echo ERROR: Intel icx compiler not found!
    echo Please install Intel oneAPI Base Toolkit and run setvars.bat first:
    echo.
    echo   "C:\Program Files\Intel\oneAPI\setvars.bat"
    echo.
    echo Download Intel oneAPI from:
    echo   https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html
    echo.
    exit /b 1
)

echo Intel oneAPI compiler found

REM Check for PyTorch
python -c "import torch; print(f'PyTorch version: {torch.__version__}')" 2>nul
if errorlevel 1 (
    echo.
    echo ERROR: PyTorch not found!
    echo Please install PyTorch with XPU support first.
    echo.
    exit /b 1
)

REM Clean if requested
if "%CLEAN%"=="1" (
    echo Cleaning build directories...
    if exist build rmdir /s /q build
    if exist dist rmdir /s /q dist
    if exist *.egg-info rmdir /s /q *.egg-info
    if exist omni_xpu_kernel\*.pyd del /q omni_xpu_kernel\*.pyd
    if exist omni_xpu_kernel\_C.*.pyd del /q omni_xpu_kernel\_C.*.pyd
)

REM Build and install
echo.
echo Building omni_xpu_kernel...
echo.

if "%DEV_MODE%"=="1" (
    echo Installing in development mode...
    pip install -e . --no-build-isolation
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
