@echo off
setlocal enabledelayedexpansion

set "WORKSPACE=C:\workspace"
set "LLM_SCALER_DIR=%WORKSPACE%\llm-scaler"
set "EMBED_PYTHON_DIR=%WORKSPACE%\omni\comfyui_windows_setup\python_embeded"
set "OUTPUT_DIR=%WORKSPACE%\llm_scaler_dist"
set "BUILD_LOG=%WORKSPACE%\build_log.txt"
set "CONDA_ENV=omni_env"
set "CONDA_ACTIVATE=C:\ProgramData\miniforge3\Scripts\activate.bat"
set "ONEAPI_SETVARS=C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
set "VS2022INSTALLDIR=C:\Program Files\Microsoft Visual Studio\18\Community"

if not exist "%ONEAPI_SETVARS%" (
	echo ERROR: oneAPI setvars.bat not found: !ONEAPI_SETVARS!
	exit /b 1
)

if not exist "%CONDA_ACTIVATE%" (
	echo ERROR: Conda activate.bat not found: !CONDA_ACTIVATE!
	exit /b 1
)

call "%ONEAPI_SETVARS%"
if errorlevel 1 exit /b %errorlevel%

call "%CONDA_ACTIVATE%" %CONDA_ENV%
if errorlevel 1 exit /b %errorlevel%

if not defined CONDA_PREFIX (
	echo ERROR: CONDA_PREFIX is not set after activating %CONDA_ENV%.
	exit /b 1
)

set "PYTHONNOUSERSITE=1"
set "PYTHONPATH="
set "CONDA_PYTHON=%CONDA_PREFIX%\python.exe"
set "TORCH_LIB=%CONDA_PREFIX%\Lib\site-packages\torch\lib"

if not exist "%CONDA_PYTHON%" (
	echo ERROR: Python not found in conda env: %CONDA_PYTHON%
	exit /b 1
)

if not exist "%TORCH_LIB%" (
	echo ERROR: PyTorch lib directory not found: %TORCH_LIB%
	exit /b 1
)

if not defined OMNI_XPU_DEVICE set "OMNI_XPU_DEVICE=bmg"

if not defined ONEDNN_INCLUDE if exist "%ProgramFiles(x86)%\Intel\oneAPI\dnnl\latest\include\oneapi\dnnl\dnnl.hpp" set "ONEDNN_INCLUDE=%ProgramFiles(x86)%\Intel\oneAPI\dnnl\latest\include"
if not defined ONEDNN_LIB if exist "%ProgramFiles(x86)%\Intel\oneAPI\dnnl\latest\lib\dnnl.lib" set "ONEDNN_LIB=%ProgramFiles(x86)%\Intel\oneAPI\dnnl\latest\lib"

set "PATH=%CONDA_PREFIX%;%CONDA_PREFIX%\Scripts;%CONDA_PREFIX%\Library\bin;%TORCH_LIB%;%ONEDNN_LIB%;%PATH%"

if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"
cd /d %LLM_SCALER_DIR%\omni\omni_xpu_kernel
if errorlevel 1 exit /b %errorlevel%

echo Build log: %BUILD_LOG%
echo Conda env: %CONDA_PREFIX%
"%CONDA_PYTHON%" -c "import pathlib, torch; p=pathlib.Path(torch.__file__).parent; h=p/'include'/'c10'/'xpu'/'impl'/'xpu_cmake_macros.h'; print('Python:', pathlib.Path(__import__('sys').executable)); print('Torch:', torch.__version__, torch.__file__); print('XPU header:', h, h.exists())"
if errorlevel 1 exit /b %errorlevel%

if exist build rmdir /s /q build
if exist omni_xpu_kernel.egg-info rmdir /s /q omni_xpu_kernel.egg-info
del /q "%BUILD_LOG%" 2>nul

"%CONDA_PYTHON%" -m pip wheel . -w "%OUTPUT_DIR%" --no-build-isolation --no-deps > "%BUILD_LOG%" 2>&1
if errorlevel 1 (
	echo ERROR: Wheel build failed. Last log lines:
	powershell -NoProfile -Command "Get-Content -Path '%BUILD_LOG%' -Tail 120"
	exit /b 1
)

echo Wheel build succeeded. Output directory: %OUTPUT_DIR%
dir /b "%OUTPUT_DIR%\omni_xpu_kernel-*.whl"
