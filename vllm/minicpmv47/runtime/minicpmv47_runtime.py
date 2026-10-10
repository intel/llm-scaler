# SPDX-License-Identifier: Apache-2.0
"""Install the source/native overlays in each vLLM worker process."""

import importlib
import importlib.util
import os
import sys
from importlib.metadata import distribution
from pathlib import Path

_installed = False


def install() -> None:
    global _installed
    if _installed:
        return
    root = Path(os.environ.get("MINICPMV47_ROOT", "/opt/minicpmv47"))
    import vllm

    # Native and Rust binaries remain supplied by the upstream wheel.
    stock = str(distribution("vllm").locate_file("vllm"))
    if stock not in vllm.__path__:
        vllm.__path__.append(stock)
    import torch
    import vllm_xpu_kernels

    importlib.import_module("vllm_xpu_kernels._xpu_C")
    importlib.import_module("vllm_xpu_kernels._moe_C")
    torch.ops.load_library(str(root / "native/bf16-performance.so"))
    # Resolve additional Python modules while preserving the stock package first.
    kernel_source = str(root / "src/vllm-xpu-kernels/vllm_xpu_kernels")
    if kernel_source not in vllm_xpu_kernels.__path__:
        vllm_xpu_kernels.__path__.append(kernel_source)
    name = "vllm_xpu_kernels.moe_shared_fused_interface"
    path = root / "src/vllm-xpu-kernels/vllm_xpu_kernels/moe_shared_fused_interface.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    vllm_xpu_kernels.moe_shared_fused_interface = module
    importlib.import_module("vllm_xpu_kernels.flash_attn_interface")
    torch.ops.load_library(str(root / "native/_minicpm_dspark_fa2.abi3.so"))
    stock_attention = torch.ops._vllm_fa2_C.varlen_fwd
    dspark_attention = torch.ops._minicpm_dspark_fa2.varlen_fwd
    main_decode = os.environ.get("MINICPMV47_MAIN_DECODE", "0") == "1"

    def route(*args, **kwargs):
        # Keep unsupported cache formats and attention options on the stock path.
        if (
            not kwargs
            and len(args) == 29
            and args[0].shape[-1] == 256
            and args[1].ndim == 4
            and args[8] is not None
            and args[9] is None
            and args[17] is None
            and args[20] == -1
            and args[21] == -1
            and args[22] == 0
            and not args[23]
            and args[0].dtype in (torch.float16, torch.bfloat16)
            and args[1].dtype == args[0].dtype == args[2].dtype
        ):
            # A single decode query sees only current/past KV. The native API
            # dispatches it with causal=false and the tiled split-K reduction.
            if main_decode and tuple(args[0].shape) == (1, 8, 256):
                return dspark_attention(*args)
            if not args[19]:
                return dspark_attention(*args)
        return stock_attention(*args, **kwargs)

    torch.ops._vllm_fa2_C.varlen_fwd = route
    _installed = True
