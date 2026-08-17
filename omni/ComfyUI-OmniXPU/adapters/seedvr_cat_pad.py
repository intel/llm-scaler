"""Route a validated causal-prefix/cat/pad contract to Omni XPU.

The native operator is model-independent. This adapter only bridges the
official SeedVR2 ``InflatedCausalConv3d.memory_limit_conv`` call site, whose
prefix is otherwise materialized by ``torch.cat`` and copied again by
``torch.nn.functional.pad``.
"""

from __future__ import annotations

import inspect
import logging
import math
import sys

import torch


log = logging.getLogger("ComfyUI-OmniXPU")

_PATCH_MARKER = "_omnixpu_seedvr_cat_pad_patched"
_layout = None
_logged_first_use = False
_shapes = {
    (1, 128, 4, 512, 512),
    (1, 256, 4, 256, 256),
    (1, 256, 4, 512, 512),
    (1, 512, 4, 256, 256),
    (1, 512, 2, 128, 128),
}
_source_contract = (
    "def memory_limit_conv(",
    "if memory_occupy < self.memory_limit or split_dim == x.ndim:",
    "x_concat = torch.cat([prev_cache, x], dim=split_dim - 1)",
    "padded = F.pad(x_concat, padding, mode='constant', value=0.0)",
    "with ignore_padding(self):",
    "return torch.nn.Conv3d.forward(self, padded)",
)


def _source_has_contract(function) -> bool:
    try:
        source = inspect.getsource(function)
    except (OSError, TypeError):
        return False
    return all(fragment in source for fragment in _source_contract)


def _is_temporal_major(input) -> bool:
    if tuple(input.shape) not in _shapes:
        return False
    _batch, channels, _temporal, height, width = input.shape
    spatial = height * width
    return tuple(input.stride()) == (
        channels * input.shape[2] * spatial,
        spatial,
        channels * spatial,
        width,
        1,
    )


def _is_prefix(prefix, input) -> bool:
    return bool(
        getattr(prefix, "is_xpu", False)
        and prefix.device == input.device
        and prefix.dtype == torch.float16
        and tuple(prefix.shape)
        == (1, input.shape[1], 2, input.shape[3], input.shape[4])
        and prefix.is_contiguous()
        and not prefix.requires_grad
    )


def _materialized_gib(input, prefix, padding) -> float:
    shape = list(input.shape)
    shape[2] += prefix.shape[2]
    for index, pad_sum in enumerate(
        (
            padding[4] + padding[5],
            padding[2] + padding[3],
            padding[0] + padding[1],
        )
    ):
        shape[-3 + index] += pad_sum
    return math.prod(shape) * input.element_size() / 1024**3


def _can_use(module, input, prefix, split_dim, padding) -> bool:
    return bool(
        _layout is not None
        and getattr(input, "is_xpu", False)
        and input.dtype == torch.float16
        and not input.requires_grad
        and _is_temporal_major(input)
        and _is_prefix(prefix, input)
        and split_dim == 3
        and tuple(padding) == (1, 1, 1, 1, 0, 0)
        and not math.isinf(module.memory_limit)
        and _materialized_gib(input, prefix, padding) < module.memory_limit
    )


def _patch(seedvr_vae):
    conv = getattr(seedvr_vae, "InflatedCausalConv3d", None)
    if conv is None:
        return False, "ComfyUI SeedVR2 InflatedCausalConv3d is unavailable"
    if getattr(conv, _PATCH_MARKER, False):
        return False, "SeedVR2 cat-pad adapter is already applied"

    original = getattr(conv, "memory_limit_conv", None)
    if original is None or not _source_has_contract(original):
        return False, "unsupported SeedVR2 memory_limit_conv contract"

    def memory_limit_conv(
        self,
        input,
        *,
        split_dim=3,
        padding=(0, 0, 0, 0, 0, 0),
        prev_cache=None,
    ):
        global _logged_first_use
        if prev_cache is not None and _can_use(
            self, input, prev_cache, split_dim, padding
        ):
            padded = _layout.cat_pad_bmg(prev_cache, input, 1)
            if not _logged_first_use:
                log.info(
                    "[OmniXPU] SeedVR BMG cat-pad: input=%s stride=%s "
                    "prefix=%s",
                    tuple(input.shape),
                    tuple(input.stride()),
                    tuple(prev_cache.shape),
                )
                _logged_first_use = True
            with seedvr_vae.ignore_padding(self):
                return torch.nn.Conv3d.forward(self, padded)
        return original(
            self,
            input,
            split_dim=split_dim,
            padding=padding,
            prev_cache=prev_cache,
        )

    setattr(memory_limit_conv, "_omnixpu_original", original)
    conv.memory_limit_conv = memory_limit_conv
    setattr(conv, _PATCH_MARKER, True)
    return True, None


def apply():
    global _layout
    probe = sys.modules.get("ComfyUI-OmniXPU.probe")
    if probe is None or getattr(probe, "layout", None) is None:
        return False, "omni_xpu_kernel layout not available"

    try:
        import omni_xpu_kernel
    except ImportError:
        return False, "omni_xpu_kernel not available"
    if getattr(omni_xpu_kernel, "__xpu_target__", "") != "bmg":
        return False, "SeedVR2 cat-pad route is BMG-only"
    supports = getattr(probe.layout, "supports_cat_pad_bmg", None)
    if not callable(supports) or not supports():
        return False, "omni_xpu_kernel BMG cat-pad capability unavailable"

    try:
        from comfy.ldm.seedvr import vae as seedvr_vae
    except ModuleNotFoundError as error:
        if error.name in {"comfy.ldm.seedvr", "comfy.ldm.seedvr.vae"}:
            return False, "ComfyUI SeedVR2 VAE is not available"
        raise

    _layout = probe.layout
    return _patch(seedvr_vae)
