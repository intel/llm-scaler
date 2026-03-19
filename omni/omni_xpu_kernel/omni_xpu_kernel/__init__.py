"""
omni_xpu_kernel - High-performance Intel XPU ESIMD kernels for PyTorch.

Optimised SYCL/ESIMD kernels for Intel GPUs:

* **gguf** — GGUF dequantization (Q4_0, Q8_0, Q4_K, Q6_K)
* **norm** — RMSNorm, LayerNorm, fused Add+RMSNorm
* **svdq** — SVDQuant W4A4: ESIMD dequant, oneDNN INT4 GEMM, fused post-processing
* **rotary** — Fused rotary position embedding

Usage::

    from omni_xpu_kernel import svdq, norm, rotary, gguf
"""

import os
import sys

__version__ = "0.1.0"
__author__ = "Intel"

# Lazy loading of native extension
_native_module = None

def _load_extension():
    """Load the native C++ extension module."""
    global _native_module
    if _native_module is not None:
        return _native_module
    
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

__all__ = [
    "gguf",
    "norm",
    "svdq",
    "rotary",
    "is_available",
    "__version__",
]
