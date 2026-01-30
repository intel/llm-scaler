"""
Normalization Kernels

High-performance RMSNorm and LayerNorm using Intel ESIMD.

Example:
    import torch
    from omni_xpu_kernel import norm
    
    # RMSNorm
    output = norm.rms_norm(weight, input, eps=1e-6)
    
    # LayerNorm
    output = norm.layer_norm(input, weight, bias, eps=1e-5)
"""

import torch
from typing import Optional


def _get_native():
    """Get the native norm module."""
    from .. import _load_extension
    return _load_extension().norm


def rms_norm(
    weight: torch.Tensor,
    input: torch.Tensor,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    RMSNorm (Root Mean Square Normalization) using ESIMD optimization.
    
    RMSNorm normalizes input by the RMS of the elements:
        output = (input / sqrt(mean(input^2) + eps)) * weight
    
    Args:
        weight: Weight tensor of shape [hidden_size]
        input: Input tensor of shape [batch_size, hidden_size]
        eps: Small constant for numerical stability
    
    Returns:
        Normalized tensor of same shape as input
    
    Note:
        - Input must be 2D tensor [batch_size, hidden_size]
        - hidden_size must be <= 8192 and divisible by 32
        - Supports fp32, fp16, bf16
    """
    return _get_native().rms_norm(weight, input, eps)


def layer_norm(
    input: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5
) -> torch.Tensor:
    """
    LayerNorm using ESIMD optimization.
    
    LayerNorm normalizes input by mean and variance:
        output = ((input - mean) / sqrt(var + eps)) * weight + bias
    
    Args:
        input: Input tensor of shape [batch_size, hidden_size]
        weight: Optional weight tensor of shape [hidden_size]
        bias: Optional bias tensor of shape [hidden_size]
        eps: Small constant for numerical stability
    
    Returns:
        Normalized tensor of same shape as input
    
    Note:
        - Input must be 2D tensor [batch_size, hidden_size]
        - hidden_size must be <= 8192 and divisible by 32
        - Supports fp32, fp16, bf16
    """
    return _get_native().layer_norm(input, weight, bias, eps)


__all__ = [
    "rms_norm",
    "layer_norm",
]
