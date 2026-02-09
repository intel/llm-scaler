@echo off
set "VS2022INSTALLDIR=C:\Program Files\Microsoft Visual Studio\18\Community"
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
call C:\ProgramData\miniforge3\Scripts\activate.bat omni_env
set "WORKSPACE=C:\workspace"
set "LLM_SCALER_DIR=%WORKSPACE%\llm-scaler"
set "EMBED_PYTHON_DIR=%WORKSPACE%\omni\comfyui_windows_setup\python_embeded"
set "OUTPUT_DIR=%WORKSPACE%\llm_scaler_dist"
set "PATH=%CONDA_PREFIX%\Library\bin;%CONDA_PREFIX%\Lib\site-packages\torch\lib;%PATH%"
cd /d %LLM_SCALER_DIR%\omni\omni_xpu_kernel
python -m pip wheel . -w %OUTPUT_DIR% --no-build-isolation --no-deps > %WORKSPACE%\build_log.txt 2>&1
