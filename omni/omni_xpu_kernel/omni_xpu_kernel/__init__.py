"""
omni_xpu_kernel - High-performance Intel XPU kernels

A collection of optimized SYCL/ESIMD kernels for Intel GPUs, including:
- GGUF Q4_0 dequantization (~340 GB/s on PVC)
- More kernels to come...

Usage:
    import omni_xpu_kernel
    from omni_xpu_kernel import gguf
    
    # Dequantize Q4_0 quantized tensor
    output = gguf.dequantize_q4_0(input_tensor, torch.float16)
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
from . import ops

__all__ = [
    "gguf",
    "norm",
    "ops",
    "is_available",
    "__version__",
]
