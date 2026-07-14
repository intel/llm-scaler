"""FP8 quantization operations matching Comfy Kitchen semantics."""

import torch


def _get_native():
    from .. import _load_extension

    return _load_extension().fp8


def quantize_per_tensor(
    x: torch.Tensor,
    scale: torch.Tensor,
    out_dtype: torch.dtype = torch.float8_e4m3fn,
) -> torch.Tensor:
    return _get_native().quantize_per_tensor(
        x.contiguous(), scale.contiguous(), out_dtype
    )


def dequantize_per_tensor(
    x: torch.Tensor,
    scale: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    return _get_native().dequantize_per_tensor(
        x.contiguous(), scale.contiguous(), out_dtype
    )


def stochastic_rounding(
    x: torch.Tensor,
    rng: torch.Tensor,
    out_dtype: torch.dtype = torch.float8_e4m3fn,
) -> torch.Tensor:
    return _get_native().stochastic_rounding(
        x.contiguous(), rng.contiguous(), out_dtype
    )


__all__ = ["dequantize_per_tensor", "quantize_per_tensor", "stochastic_rounding"]
