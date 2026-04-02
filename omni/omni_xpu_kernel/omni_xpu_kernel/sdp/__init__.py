"""Standalone scaled dot-product attention kernel wrapper."""

import torch


def _get_native():
    from .. import _load_extension
    return _load_extension().sdp


def sdp(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return _get_native().sdp(q, k, v)


__all__ = ["sdp"]
