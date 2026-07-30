"""
omni_xpu_kernel - High-performance Intel XPU ESIMD kernels for PyTorch.

Optimised SYCL/ESIMD kernels for Intel GPUs:

* **gguf** — GGUF dequantization (Q4_0, Q4_1, Q8_0, Q4_K, Q6_K)
* **norm** — RMSNorm, LayerNorm, fused Add+RMSNorm
* **svdq** — SVDQuant W4A4: ESIMD dequant, oneDNN INT4 GEMM, fused post-processing
* **rotary** — Fused rotary position embedding
* **sdp** — Standalone scaled dot-product attention
* **linear** — FP8 GEMM (oneDNN W8A16, E4M3/E5M2)
* **int8** — INT8 quantization, GEMM, and linear (oneDNN s8 matmul + ESIMD fusion)

Usage::

    from omni_xpu_kernel import svdq, norm, rotary, gguf, sdp, linear, int8
"""

import os
import sys
from pathlib import Path

from ._version import __torch_version__, __version__, __xpu_target__

__author__ = "Intel"

# Lazy loading of native extension
_native_module = None
_dll_dir_handles = []
_dll_dir_paths = set()


def _add_windows_dll_directory(path: Path) -> None:
    """Register a DLL directory and keep its handle alive."""
    if not path.is_dir():
        return

    resolved = path.resolve()
    if resolved in _dll_dir_paths:
        return

    _dll_dir_handles.append(os.add_dll_directory(str(resolved)))
    _dll_dir_paths.add(resolved)


def _configure_windows_dll_search_paths() -> None:
    """Register runtime locations used by Torch XPU, oneDNN, and oneAPI."""
    if sys.platform != "win32":
        return

    python_roots = {
        Path(sys.prefix),
        Path(sys.executable).resolve().parent,
    }
    for python_root in python_roots:
        _add_windows_dll_directory(python_root / "Library" / "bin")
        _add_windows_dll_directory(python_root / "DLLs")

    try:
        import torch

        _add_windows_dll_directory(Path(torch.__file__).resolve().parent / "lib")
    except Exception:
        pass

    preferred_version = os.environ.get("OMNI_XPU_ONEAPI_VERSION")
    program_roots = (
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")),
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")),
    )
    for program_root in program_roots:
        oneapi_root = program_root / "Intel" / "oneAPI"
        version_names = [preferred_version] if preferred_version else []
        version_names.append("latest")
        for version in version_names:
            if not version:
                continue
            for relative in (
                ("compiler", version, "bin"),
                ("compiler", version, "bin", "compiler"),
                ("compiler", version, "bin", "1033"),
                ("dnnl", version, "bin"),
                ("ocloc", version, "bin"),
            ):
                _add_windows_dll_directory(oneapi_root.joinpath(*relative))

    for raw_path in os.environ.get("OMNI_XPU_DLL_DIRS", "").split(os.pathsep):
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


def core_aot_target() -> str:
    """Return the GPU target embedded in the loaded core AOT image.

    Older artifacts and builds without core AOT return an empty string.  The
    value comes from ``_C`` rather than Python package metadata so a stale
    binary cannot accidentally advertise a capability it does not contain.
    """
    try:
        native = _load_extension()
    except ImportError:
        return ""
    return str(getattr(native, "__core_aot_target__", ""))


# Submodule imports
from . import gguf
from . import norm
from . import svdq
from . import rotary
from . import sdp
from . import linear
from . import int8
from . import fp8
from . import device

# cute FMHA (CUTLASS-SYCL) is required by default at build time. Import remains
# defensive for explicit core-only/Windows builds and older installed wheels.
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
    "int8",
    "fp8",
    "device",
    "cute",
    "core_aot_target",
    "is_available",
    "__torch_version__",
    "__xpu_target__",
    "__version__",
]


def native_capabilities() -> dict[str, tuple[str, ...]]:
    """Return loaded native symbols, or an empty mapping when unavailable."""
    try:
        native = _load_extension()
    except ImportError:
        return {}
    modules = (
        "device",
        "fp8",
        "gguf",
        "norm",
        "svdq",
        "rotary",
        "sdp",
        "linear",
        "int8",
    )
    return {
        name: tuple(sorted(item for item in dir(getattr(native, name)) if not item.startswith("_")))
        for name in modules
        if hasattr(native, name)
    }


__all__.append("native_capabilities")
