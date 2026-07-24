# ComfyUI portable setup for Intel XPU on Windows

`setup_portable_env.bat` creates a self-contained Windows ComfyUI environment.
It is a separate compatibility path from the Linux Docker image and currently
uses its own pinned dependency set:

- Python 3.12.10 embedded;
- PyTorch 2.9.0 XPU;
- ComfyUI commit `3dd10a59c00248d00f0cb0ab794ff1bb9fb00a5f`;
- the custom nodes listed in the setup script.

It does not build or install the target-specific Linux
`omni_xpu_kernel`/Comfy Kitchen wheel pair.

## Prerequisites

- Windows 10 or later;
- an Intel GPU supported by the installed graphics driver;
- Git, `curl`, and Windows PowerShell available in `PATH`;
- network access to the pinned Python, PyTorch, ComfyUI, and custom-node
  sources.

## Install

Clone the Intel repository and run the setup script:

```cmd
git clone https://github.com/intel/llm-scaler.git
cd llm-scaler\omni\comfyui_windows_setup
setup_portable_env.bat
```

Proxy variables can be configured at the top of the batch file before it is
run.

The script creates `python_embeded`, checks out the pinned ComfyUI revision,
applies `omni/patches/comfyui_for_multi_arc.patch`, installs its pinned custom
nodes, and generates three launchers:

```text
run_comfyui.bat
run_comfyui_lowvram.bat
run_comfyui_cpu.bat
```

## Verify and run

Verify XPU availability:

```cmd
python_embeded\python.exe -c "import torch; print(torch.__version__); print(torch.xpu.is_available())"
```

Start ComfyUI:

```cmd
run_comfyui.bat
```

Open `http://127.0.0.1:8188`. Additional ComfyUI arguments can be appended to
the launcher command.

Models belong in the standard subdirectories under `ComfyUI\models`.

## Recreate or remove

The setup is portable. Delete the generated `python_embeded`, `ComfyUI`, and
launcher files before rerunning the script when the pinned dependency set
changes. Removing the complete `comfyui_windows_setup` working copy uninstalls
the portable environment; the script does not register a system service.
