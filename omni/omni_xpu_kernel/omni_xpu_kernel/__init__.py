"""
omni_xpu_kernel - High-performance Intel XPU ESIMD kernels for PyTorch.

Optimised SYCL/ESIMD kernels for Intel GPUs:

* **gguf** — GGUF dequantization (Q4_0, Q8_0, Q4_K, Q6_K)
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

__version__ = "0.1.0"
__author__ = "Intel"

# Lazy loading of native extension
_native_module = None
_dll_directory_handles = []


def _add_windows_dll_directories():
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return

    candidates = [
        Path(sys.executable).parent,
        Path(sys.prefix) / "Library" / "bin",
    ]

    try:
        import torch
        candidates.append(Path(torch.__file__).parent / "lib")
    except Exception:
        pass

    program_files_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    candidates.extend([
        Path(program_files_x86) / "Intel" / "oneAPI" / "dnnl" / "latest" / "bin",
        Path(program_files_x86) / "Intel" / "oneAPI" / "compiler" / "latest" / "bin",
    ])

    for path in candidates:
        if path.is_dir():
            try:
                _dll_directory_handles.append(os.add_dll_directory(str(path)))
            except OSError:
                pass

def _load_extension():
    """Load the native C++ extension module."""
    global _native_module
    if _native_module is not None:
        return _native_module

    _add_windows_dll_directories()
    
    try:
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
from . import int8

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
    "int8",
    "cute",
    "is_available",
    "__version__",
]
