"""ESIMD Prefill Flash Attention kernel loader.

Loads the flash_attn_batch extension and registers torch.ops.flash_attn_batch.forward.
This module is imported by vLLM's flash_attn backend for XPU prefill.
"""
import torch  # must be first — loads libc10.so
import ctypes
import glob
import os
from pathlib import Path

# Find and load the .so file
_pkg_dir = Path(__file__).parent.parent.parent  # custom-esimd-kernels-vllm root
_search_paths = [
    _pkg_dir / "csrc" / "prefill_attn",  # dev build (build_ext --inplace)
    Path(__file__).parent,                 # installed in package
    # Hardcoded dev path as fallback
    Path("/llm/shaojun/code/llm-scaler/vllm/custom-esimd-kernels-vllm/csrc/prefill_attn"),
]

_loaded = False
for _dir in _search_paths:
    _so_files = glob.glob(str(_dir / "flash_attn_batch*.so"))
    if _so_files:
        ctypes.CDLL(_so_files[0])
        _loaded = True
        break

if not _loaded:
    raise ImportError(
        "flash_attn_batch kernel not found. Build it with:\n"
        "  cd custom-esimd-kernels-vllm/csrc/prefill_attn && "
        "USE_DOUBLE_GRF=1 KERNEL=flash_attn_batch TORCH_XPU_ARCH_LIST=bmg-g21 "
        "python3 setup_test.py build_ext --inplace"
    )

forward = torch.ops.flash_attn_batch.forward
