"""Validated layout and materialization fusions for Intel XPU."""

import torch


def _get_native():
    from .. import _load_extension

    return _load_extension().layout


def supports_cat_pad_bmg() -> bool:
    """Return whether the native binary contains the BMG cat-pad route."""
    return bool(getattr(_get_native(), "__cat_pad_bmg__", False))


def cat_pad_bmg(
    prefix: torch.Tensor,
    input: torch.Tensor,
    spatial_pad: int = 1,
) -> torch.Tensor:
    """Concatenate a temporal prefix and apply symmetric spatial zero-pad.

    The native entry point is intentionally narrow: it accepts the validated
    BMG FP16 inference contracts and raises for all other inputs. Callers must
    retain their normal fallback.
    """
    return _get_native().cat_pad_bmg(prefix, input, spatial_pad)
