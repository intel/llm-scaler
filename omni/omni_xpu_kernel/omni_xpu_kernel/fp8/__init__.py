"""FP8 quantization operations matching Comfy Kitchen semantics."""

import torch

from .. import _compile_meta as _meta
from .._compile_ops import compile_op


def _get_native():
    from .. import _load_extension

    return _load_extension().fp8


@compile_op("fp8_quantize_per_tensor", _meta.fp8_quantize)
def quantize_per_tensor(
    x: torch.Tensor,
    scale: torch.Tensor,
    out_dtype: torch.dtype = torch.float8_e4m3fn,
) -> torch.Tensor:
    if torch.compiler.is_compiling():
        return torch.ops.omni_xpu.fp8_quantize_per_tensor(x, scale, out_dtype)
    return _get_native().quantize_per_tensor(
        x.contiguous(), scale.contiguous(), out_dtype
    )


@compile_op("fp8_dequantize_per_tensor", _meta.fp8_dequantize)
def dequantize_per_tensor(
    x: torch.Tensor,
    scale: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    if torch.compiler.is_compiling():
        return torch.ops.omni_xpu.fp8_dequantize_per_tensor(x, scale, out_dtype)
    return _get_native().dequantize_per_tensor(
        x.contiguous(), scale.contiguous(), out_dtype
    )


@compile_op("fp8_stochastic_rounding", _meta.fp8_round)
def stochastic_rounding(
    x: torch.Tensor,
    rng: torch.Tensor,
    out_dtype: torch.dtype = torch.float8_e4m3fn,
) -> torch.Tensor:
    if torch.compiler.is_compiling():
        return torch.ops.omni_xpu.fp8_stochastic_rounding(x, rng, out_dtype)
    return _get_native().stochastic_rounding(
        x.contiguous(), rng.contiguous(), out_dtype
    )


__all__ = ["dequantize_per_tensor", "quantize_per_tensor", "stochastic_rounding"]
