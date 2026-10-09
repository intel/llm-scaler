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

    def route(*args, **kwargs):
        # Only the paged, noncausal, global-window head_dim=256 route is added.
        if (
            not kwargs
            and len(args) == 29
            and args[0].shape[-1] == 256
            and args[1].ndim == 4
            and args[8] is not None
            and args[17] is None
            and not args[19]
            and args[20] == -1
            and args[21] == -1
            and not args[23]
        ):
            return dspark_attention(*args)
        return stock_attention(*args, **kwargs)

    torch.ops._vllm_fa2_C.varlen_fwd = route
    _installed = True
