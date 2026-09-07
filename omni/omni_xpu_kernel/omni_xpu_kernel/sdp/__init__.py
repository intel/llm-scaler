"""Standalone scaled dot-product attention kernel wrapper."""

import torch

from .. import _compile_meta as _meta
from .._compile_ops import compile_op


def _get_native():
    from .. import _load_extension
    return _load_extension().sdp


@compile_op("sdp_sdp", _meta.unchanged, ordered=True)
def sdp(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    if torch.compiler.is_compiling():
        return torch.ops.omni_xpu.sdp_sdp(q, k, v)
    return _get_native().sdp(q, k, v)


__all__ = ["sdp"]
