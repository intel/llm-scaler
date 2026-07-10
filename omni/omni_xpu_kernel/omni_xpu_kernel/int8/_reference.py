"""
Pure PyTorch reference implementation of INT8 operations.

This module serves as:
1. Correctness baseline for validating native C++ kernels
2. Fallback when native extension is unavailable
3. POC for API design validation

All functions here are numerically correct but NOT optimized for performance.
The native C++ path (oneDNN + ESIMD) should always be preferred for inference.
"""

import math
from typing import Optional, Tuple

import torch

# =============================================================================
# Hadamard / ConvRot Helpers
# =============================================================================

_HADAMARD_CACHE: dict = {}


def _build_hadamard(
    size: int,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build a normalized REGULAR orthogonal Hadamard matrix (ConvRot).

    Uses recursive Kronecker product of H4 to build power-of-4 sizes.
    Results are cached by (size, device, dtype).

    Args:
        size: Matrix size (must be a power of 4: 4, 16, 64, 256, ...).
        device: Target device.
        dtype: Target dtype.

    Returns:
        Orthogonal Hadamard matrix [size, size] satisfying H @ H^T = I.

    Raises:
        ValueError: If size is not a power of 4.
    """
    cache_key = (size, str(device), dtype)
    if cache_key in _HADAMARD_CACHE:
        return _HADAMARD_CACHE[cache_key]

    if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
        raise ValueError(f"Regular Hadamard size must be a power of 4, got {size}")

    h4 = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        dtype=dtype,
        device=device,
    )

    h = h4
    current_size = 4
    while current_size < size:
        h = torch.kron(h, h4)
        current_size *= 4

    h_normalized = h / (size**0.5)
    _HADAMARD_CACHE[cache_key] = h_normalized
    return h_normalized


def _rotate_weight(
    weight: torch.Tensor,
    h: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Rotate weight matrix offline: W_rot = W @ H_block^T.

    Args:
        weight: Weight tensor [N, K].
        h: Normalized Hadamard matrix [group_size, group_size].
        group_size: Rotation group size.

    Returns:
        Rotated weight [N, K].
    """
    out_f, in_f = weight.shape
    if in_f % group_size != 0:
        raise ValueError(f"in_features {in_f} not divisible by group_size {group_size}")
    n_groups = in_f // group_size

    weight_grouped = weight.reshape(out_f, n_groups, group_size)
    h_t = h.T.to(dtype=weight.dtype, device=weight.device)
    weight_rotated = torch.matmul(weight_grouped, h_t)
    return weight_rotated.reshape(out_f, in_f)


def _rotate_activation(
    x: torch.Tensor,
    h: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Rotate activation online: x_rot = x @ H_block.

    Args:
        x: Activation tensor [..., K].
        h: Normalized Hadamard matrix [group_size, group_size].
        group_size: Rotation group size.

    Returns:
        Rotated activation [..., K].
    """
    orig_shape = x.shape
    features = orig_shape[-1]
    if features % group_size != 0:
        raise ValueError(f"features {features} not divisible by group_size {group_size}")
    n_groups = features // group_size

    x_grouped = x.reshape(-1, n_groups, group_size)
    h = h.to(dtype=x.dtype, device=x.device)
    x_rotated = torch.matmul(x_grouped, h)

    return x_rotated.reshape(orig_shape)


# =============================================================================
# Stochastic Rounding Helper
# =============================================================================


def _int8_stochastic_rng(x: torch.Tensor, seed: int) -> torch.Tensor:
    """Generate deterministic random noise for stochastic rounding."""
    generator = torch.Generator(device=x.device)
    generator.manual_seed(seed)
    return torch.rand(
        x.shape,
        dtype=x.dtype,
        layout=x.layout,
        device=x.device,
        generator=generator,
    )


def _round_int8(scaled: torch.Tensor, stochastic_rounding: int = 0) -> torch.Tensor:
    """Round to INT8 with optional stochastic rounding."""
    if stochastic_rounding > 0:
        rng = _int8_stochastic_rng(scaled, stochastic_rounding)
        scaled = scaled + rng
        return scaled.floor().clamp(-128.0, 127.0).to(torch.int8)
    return scaled.round().clamp(-128.0, 127.0).to(torch.int8)


# =============================================================================
# Quantization
# =============================================================================


def quantize_int8_tensorwise(
    x: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    stochastic_rounding: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize tensor to INT8 with single tensorwise scale.

    scale = absmax(x) / 127
    q = round(x / scale)

    Args:
        x: Input tensor of any shape.
        scale: Optional pre-computed scale. If None, computes from absmax.
        stochastic_rounding: Seed for stochastic rounding. Disabled when <= 0.

    Returns:
        Tuple of (quantized_int8, scale).
    """
    if scale is None:
        abs_max = x.abs().max()
        scale = (abs_max.float() / 127.0).clamp(min=1e-30)
    elif not isinstance(scale, torch.Tensor):
        scale = torch.tensor(scale, device=x.device, dtype=torch.float32)
    else:
        scale = scale.to(device=x.device, dtype=torch.float32)

    # Ensure scale is safe for division
    scale_for_div = scale.to(device=x.device, dtype=x.dtype)
    scale_min = torch.finfo(x.dtype).tiny
    scale_for_div = torch.where(
        scale_for_div == 0,
        torch.full_like(scale_for_div, scale_min),
        scale_for_div,
    )

    q = _round_int8(x / scale_for_div, stochastic_rounding=stochastic_rounding)
    return q, scale


def quantize_int8_rowwise(
    x: torch.Tensor,
    stochastic_rounding: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize tensor to INT8 with per-row scales.

    For each row: scale = absmax(row) / 127, q = round(row / scale)

    Args:
        x: Input tensor [..., K].
        stochastic_rounding: Seed for stochastic rounding. Disabled when <= 0.

    Returns:
        Tuple of (quantized_int8, scales [..., 1]).
    """
    abs_max = x.abs().amax(dim=-1, keepdim=True)
    scale = (abs_max.float() / 127.0).clamp(min=1e-30)

    scale_for_div = scale.to(device=x.device, dtype=x.dtype)
    scale_min = torch.finfo(x.dtype).tiny
    scale_for_div = torch.where(
        scale_for_div == 0,
        torch.full_like(scale_for_div, scale_min),
        scale_for_div,
    )

    q = _round_int8(x / scale_for_div, stochastic_rounding=stochastic_rounding)
    return q, scale


# =============================================================================
# Dequantization
# =============================================================================


def dequantize_int8_simple(
    q: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Dequantize INT8 tensor: result = q.float() * scale.

    Args:
        q: Quantized INT8 tensor.
        scale: Scale tensor (scalar or broadcastable).

    Returns:
        Dequantized float32 tensor.
    """
    return q.float() * scale


def dequantize_int8_simple_dtype(
    q: torch.Tensor,
    scale: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize INT8 tensor with output dtype conversion.

    Args:
        q: Quantized INT8 tensor.
        scale: Scale tensor (scalar or broadcastable).
        out_dtype: Target output dtype.

    Returns:
        Dequantized tensor in specified dtype.
    """
    return dequantize_int8_simple(q, scale).to(out_dtype)


# =============================================================================
# INT8 Matrix Multiplication
# =============================================================================


def _round_up(x: int, multiple: int) -> int:
    """Round x up to nearest multiple."""
    return ((x + multiple - 1) // multiple) * multiple


def mm_int8(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """INT8 matrix multiplication: C[M,N] = A[M,K] @ B[K,N].

    Uses torch._int_mm with necessary padding for alignment constraints.

    Args:
        a: INT8 tensor [M, K].
        b: INT8 tensor [K, N].

    Returns:
        INT32 tensor [M, N].
    """
    if a.dtype != torch.int8 or b.dtype != torch.int8:
        raise ValueError(f"a and b must be int8, got {a.dtype} and {b.dtype}")
    if a.dim() != 2 or b.dim() != 2:
        raise ValueError(f"a and b must be 2D, got {a.dim()}D and {b.dim()}D")
    if a.size(1) != b.size(0):
        raise ValueError(f"K dimension mismatch: a.size(1)={a.size(1)} vs b.size(0)={b.size(0)}")

    orig_m = a.size(0)
    orig_n = b.size(1)
    k = a.size(1)

    if orig_m == 0 or k == 0:
        return torch.zeros(orig_m, orig_n, dtype=torch.int32, device=a.device)

    # Pad K to multiple of 8 (required by _int_mm on most backends)
    padded_k = _round_up(k, 8)
    if padded_k != k:
        a_pad = torch.zeros((a.size(0), padded_k - k), device=a.device, dtype=a.dtype)
        b_pad = torch.zeros((padded_k - k, b.size(1)), device=b.device, dtype=b.dtype)
        a = torch.cat((a, a_pad), dim=1)
        b = torch.cat((b, b_pad), dim=0)

    # Pad N to required alignment
    # On XPU/CUDA, N typically needs to be multiple of 8 or 16
    n_align = 8
    padded_n = _round_up(orig_n, n_align)
    if padded_n != orig_n:
        b_pad = torch.zeros((b.size(0), padded_n - orig_n), device=b.device, dtype=b.dtype)
        b = torch.cat((b, b_pad), dim=1)

    # Pad M to at least 16 on some backends
    padded_m = _round_up(max(orig_m, 16), 16) if a.is_cuda or a.device.type == 'xpu' else orig_m
    if padded_m != orig_m:
        a_pad = torch.zeros((padded_m - orig_m, a.size(1)), device=a.device, dtype=a.dtype)
        a = torch.cat((a, a_pad), dim=0)

    # Execute INT8 matmul
    if hasattr(torch, 'int8_mm'):
        result = torch.int8_mm(a, b)
    else:
        result = torch._int_mm(a, b)

    # Trim padding
    if result.size(0) != orig_m or result.size(1) != orig_n:
        result = result[:orig_m, :orig_n]

    return result


# =============================================================================
# ConvRot Weight Quantization/Dequantization
# =============================================================================


def quantize_int8_convrot_weight(
    weight: torch.Tensor,
    group_size: int = 256,
    stochastic_rounding: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Offline ConvRot weight rotation followed by per-row INT8 quantization.

    Args:
        weight: Weight tensor [N, K].
        group_size: Hadamard rotation group size (power of 4).
        stochastic_rounding: Seed for stochastic rounding.

    Returns:
        Tuple of (rotated_quantized_int8 [N, K], per_row_scales [N, 1]).
    """
    h = _build_hadamard(group_size, device=weight.device, dtype=weight.dtype)
    weight_rot = _rotate_weight(weight, h, group_size)
    return quantize_int8_rowwise(weight_rot, stochastic_rounding=stochastic_rounding)


def dequantize_int8_convrot_weight(
    q: torch.Tensor,
    scale: torch.Tensor,
    group_size: int = 256,
) -> torch.Tensor:
    """Dequantize INT8 ConvRot weights and rotate back to original basis.

    Args:
        q: Quantized INT8 weight [N, K].
        scale: Per-row scales [N, 1].
        group_size: Hadamard rotation group size.

    Returns:
        Dequantized weight tensor in float32.
    """
    h = _build_hadamard(group_size, device=q.device, dtype=torch.float32)
    dq = dequantize_int8_simple(q, scale)
    return _rotate_weight(dq, h, group_size)


# =============================================================================
# INT8 Linear (Full Inference Path)
# =============================================================================


def int8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    convrot: bool = False,
    convrot_groupsize: int = 256,
) -> torch.Tensor:
    """INT8 linear layer with dynamic activation quantization.

    Pipeline:
        1. (Optional) ConvRot: rotate activation with Hadamard matrix
        2. Quantize activation per-row to INT8
        3. INT8 GEMM: x_int8 @ weight^T (s8×s8→s32)
        4. Rescale: result * (x_scale * weight_scale) → out_dtype
        5. (Optional) Add bias

    Args:
        x: Input [..., K] in fp16/bf16.
        weight: INT8 weight [N, K].
        weight_scale: Scalar or per-channel [N] or [N,1] weight scale.
        bias: Optional bias [N].
        out_dtype: Output dtype (defaults to x.dtype).
        convrot: If True, apply online activation rotation.
        convrot_groupsize: Group size for Hadamard rotation.

    Returns:
        Result tensor [..., N].
    """
    if out_dtype is None:
        out_dtype = x.dtype

    if x.shape[-1] != weight.shape[-1]:
        raise ValueError(
            f"Input and weight inner dimensions must match, "
            f"got {x.shape[-1]} and {weight.shape[-1]}"
        )

    weight = weight.to(device=x.device).contiguous()
    weight_scale = weight_scale.to(device=x.device, dtype=torch.float32).reshape(-1)
    if weight_scale.numel() not in (1, weight.shape[0]):
        raise ValueError(
            f"INT8 weight scale must be scalar or per-output-channel, "
            f"got {tuple(weight_scale.shape)} for weight shape {tuple(weight.shape)}"
        )

    # Step 1: Optional ConvRot rotation
    if convrot:
        if x.shape[-1] % convrot_groupsize != 0:
            raise ValueError(
                f"ConvRot group size {convrot_groupsize} does not divide "
                f"input features {x.shape[-1]}"
            )
        h = _build_hadamard(convrot_groupsize, device=x.device, dtype=x.dtype)
        x = _rotate_activation(x, h, convrot_groupsize)

    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])

    # Step 2: Quantize activation per-row
    x_8, x_scale = quantize_int8_rowwise(x_2d)

    # Step 3: INT8 GEMM — x_8 [M, K] @ weight^T [K, N]
    result = mm_int8(x_8, weight.T.contiguous())

    # Step 4: Rescale with chunked approach to avoid OOM on large models
    m, n = result.shape
    chunk_size = max(1, min(m, 256 * 1024 * 1024 // (n * 4)))

    weight_scale_row = weight_scale.reshape(1, -1)
    scaled_parts = []
    for i in range(0, m, chunk_size):
        end_i = min(i + chunk_size, m)
        chunk = result[i:end_i].float()

        # Combined scale: x_scale[i:end_i] * weight_scale
        chunk_scales = (
            x_scale[i:end_i].to(device=chunk.device, dtype=torch.float32)
            * weight_scale_row
        )
        chunk_scaled = chunk * chunk_scales
        chunk_scaled = chunk_scaled.to(out_dtype)
        scaled_parts.append(chunk_scaled)

    result = torch.cat(scaled_parts, dim=0)

    # Step 5: Optional bias
    if bias is not None:
        result = result + bias.to(device=result.device, dtype=result.dtype).reshape(1, -1)

    return result.reshape(*orig_shape[:-1], weight.shape[0])
