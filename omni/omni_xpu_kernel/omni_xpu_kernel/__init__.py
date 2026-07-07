"""
omni_xpu_kernel - High-performance Intel XPU ESIMD kernels for PyTorch.

Optimised SYCL/ESIMD kernels for Intel GPUs:

* **gguf** — GGUF dequantization (Q4_0, Q8_0, Q4_K, Q6_K)
* **norm** — RMSNorm, LayerNorm, fused Add+RMSNorm
* **svdq** — SVDQuant W4A4: ESIMD dequant, oneDNN INT4 GEMM, fused post-processing
* **rotary** — Fused rotary position embedding
* **sdp** — Standalone scaled dot-product attention
* **linear** — FP8 GEMM (oneDNN W8A16, E4M3/E5M2)

Usage::

    from omni_xpu_kernel import svdq, norm, rotary, gguf, sdp, linear
"""

import os
import sys
from pathlib import Path

__version__ = "0.1.0"
__author__ = "Intel"

# Lazy loading of native extension
_native_module = None
_dll_dir_handles = []
_dll_dir_paths = set()


def _add_windows_dll_directory(path: Path) -> None:
    """Register a DLL search path and keep the handle alive for process lifetime."""
    if not path.is_dir():
        return

    resolved = path.resolve()
    if resolved in _dll_dir_paths:
        return

    handle = os.add_dll_directory(str(resolved))
    _dll_dir_handles.append(handle)
    _dll_dir_paths.add(resolved)


def _configure_windows_dll_search_paths() -> None:
    """Ensure the active Python environment's runtime DLLs are discoverable."""
    if sys.platform != "win32":
        return

    python_root = Path(sys.executable).resolve().parent
    _add_windows_dll_directory(python_root / "Library" / "bin")
    _add_windows_dll_directory(python_root / "DLLs")

    program_files_x86 = Path(os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)"))
    oneapi_root = program_files_x86 / "Intel" / "oneAPI" / "compiler"
    preferred_version = os.environ.get("OMNI_XPU_ONEAPI_VERSION")
    locale_dirs = []
    if preferred_version:
        locale_dirs.append(oneapi_root / preferred_version / "bin" / "1033")
    if oneapi_root.is_dir():
        locale_dirs.extend(sorted(oneapi_root.glob("*\\bin\\1033"), reverse=True))
    for locale_dir in locale_dirs:
        _add_windows_dll_directory(locale_dir)

    try:
        import torch

        torch_root = Path(torch.__file__).resolve().parent
        _add_windows_dll_directory(torch_root / "lib")
    except Exception:
        pass

    extra_dirs = os.environ.get("OMNI_XPU_DLL_DIRS", "")
    for raw_path in extra_dirs.split(os.pathsep):
        if raw_path:
            _add_windows_dll_directory(Path(raw_path))

def _load_extension():
    """Load the native C++ extension module."""
    global _native_module
    if _native_module is not None:
        return _native_module
    
    try:
        _configure_windows_dll_search_paths()
        from omni_xpu_kernel import _C
        _native_module = _C
        return _native_module
    except ImportError as e:
        raise ImportError(
            f"Failed to load omni_xpu_kernel native extension. "
            f"Make sure you have Intel XPU support and the package is properly installed. "
            f"Error: {e}"
        ) from e


def is_available():
    """Check if omni_xpu_kernel is available and functional."""
    try:
        _load_extension()
        return True
    except ImportError:
        return False


# Submodule imports
from . import gguf
from . import norm
from . import svdq
from . import rotary
from . import sdp
from . import linear

# cute FMHA (CUTLASS-SYCL) is an optional backend — its AOT .so may be absent on
# non-XPU / header-less installs, so import defensively.
try:
    from . import cute
except Exception:  # pragma: no cover
    cute = None

__all__ = [
    "gguf",
    "norm",
    "svdq",
    "rotary",
    "sdp",
    "linear",
    "cute",
    "is_available",
    "__version__",
]
