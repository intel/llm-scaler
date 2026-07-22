import logging
import os

import torch
from torch import Tensor

from .debug import log_debug_event, trace_patch

log = logging.getLogger("ComfyUI-OmniXPU")

_omni_rotary = None
_logged_first = False
_logged_zimage_pair = 0
_logged_boogu_d120 = 0
_allow_ptl_zimage_pair = False
_allow_ptl_boogu_d120 = False


def _can_use(x):
    if _omni_rotary is None or not x.is_xpu:
        return False
    return x.shape[-1] in (64, 128)


def _torch_major_minor():
    try:
        components = torch.__version__.split("+", 1)[0].split(".")
        return int(components[0]), int(components[1])
    except (AttributeError, IndexError, ValueError):
        return None


def _use_ptl_zimage_pair(xq: Tensor, xk: Tensor, freqs_cis: Tensor):
    """Select the exact Z-Image pair route validated on PTL-H/Torch 2.11."""
    return (
        _allow_ptl_zimage_pair
        and hasattr(_omni_rotary, "apply_kitchen_rope")
        and xq.is_xpu
        and xk.is_xpu
        and freqs_cis.is_xpu
        and xq.device == xk.device == freqs_cis.device
        and xq.dtype == xk.dtype == torch.bfloat16
        and freqs_cis.dtype == torch.float32
        and xq.ndim == 4
        and xk.ndim == 4
        and freqs_cis.ndim == 6
        and xq.is_contiguous()
        and xk.is_contiguous()
        and freqs_cis.is_contiguous()
        and xq.shape == xk.shape
        and xq.shape[0] == 1
        and xq.shape[1] in (64, 1024, 1088)
        and xq.shape[2:] == (30, 128)
        and freqs_cis.shape == (1, xq.shape[1], 1, 64, 2, 2)
        and _torch_major_minor() == (2, 11)
    )


def _use_ptl_boogu_d120(x: Tensor, freqs_cis: Tensor):
    """Select exact Boogu D120 shapes validated with native Kitchen RoPE."""
    return (
        _allow_ptl_boogu_d120
        and hasattr(_omni_rotary, "apply_kitchen_rope1")
        and hasattr(_omni_rotary, "kitchen_rope_fast_supported")
        and x.is_xpu
        and freqs_cis.is_xpu
        and x.device == freqs_cis.device
        and x.dtype == torch.float16
        and freqs_cis.dtype == torch.float32
        and x.ndim == 4
        and freqs_cis.ndim == 6
        and x.is_contiguous()
        and freqs_cis.is_contiguous()
        and x.shape[0] == 1
        and x.shape[1] in (4096, 4205)
        and x.shape[2] in (7, 28)
        and x.shape[3] == 120
        and freqs_cis.shape == (1, x.shape[1], 1, 60, 2, 2)
        and _torch_major_minor() == (2, 11)
        and _omni_rotary.kitchen_rope_fast_supported(x, freqs_cis)
    )


def _omni_apply_boogu_d120(x: Tensor, freqs_cis: Tensor):
    global _logged_boogu_d120
    _logged_boogu_d120 += 1
    if _logged_boogu_d120 <= 3:
        log.info(
            "[OmniXPU] rope Boogu D120 #%d: seq=%d heads=%d dtype=%s",
            _logged_boogu_d120,
            x.shape[1],
            x.shape[2],
            x.dtype,
        )
    log_debug_event(
        "kernel",
        "rotary_emb",
        {"x": x, "freqs": freqs_cis},
        details={"backend": "kitchen", "route": "ptl_boogu_d120"},
    )
    return _omni_rotary.apply_kitchen_rope1(x, freqs_cis)


def _omni_apply_rope1(x: Tensor, freqs_cis: Tensor):
    global _logged_first
    B, H, S, D = x.shape
    S_freq = freqs_cis.shape[2]

    # Fallback if freq seq_len < x seq_len
    if S_freq < S:
        x_ = x.to(dtype=freqs_cis.dtype).reshape(*x.shape[:-1], -1, 1, 2)
        if x_.shape[2] != 1 and freqs_cis.shape[2] != 1 and x_.shape[2] != freqs_cis.shape[2]:
            freqs_cis = freqs_cis[:, :, :x_.shape[2]]
        x_out = freqs_cis[..., 0] * x_[..., 0]
        x_out.addcmul_(freqs_cis[..., 1], x_[..., 1])
        return x_out.reshape(*x.shape).type_as(x)

    if not _logged_first:
        _logged_first = True
        log.info("[OmniXPU] rope first use: ESIMD rotary_emb shape=%s", x.shape)

    cos_cache = freqs_cis[0, 0, :S, :, 0, 0].to(dtype=torch.float32).contiguous()
    sin_cache = freqs_cis[0, 0, :S, :, 1, 0].to(dtype=torch.float32).contiguous()
    x_flat = x.permute(0, 2, 1, 3).contiguous().reshape(B * S * H, D)
    log_debug_event(
        "kernel",
        "rotary_emb",
        {"x": x_flat, "cos": cos_cache, "sin": sin_cache},
        details={"backend": "esimd"},
    )
    out = _omni_rotary.rotary_emb(x_flat, cos_cache, sin_cache, S, H)
    return out.reshape(B, S, H, D).permute(0, 2, 1, 3).contiguous()


def apply():
    global _allow_ptl_boogu_d120, _allow_ptl_zimage_pair, _omni_rotary
    import sys
    probe = sys.modules.get("ComfyUI-OmniXPU.probe")
    if probe.rotary is None:
        return False, "omni_xpu_kernel rotary not available"
    _omni_rotary = probe.rotary
    try:
        import omni_xpu_kernel as _omni_package

        _allow_ptl_zimage_pair = (
            getattr(_omni_package, "__xpu_target__", "") == "ptl-h"
            and os.environ.get("OMNIXPU_ZIMAGE_ROPE_PAIR", "1") != "0"
        )
        fast_marker = getattr(_omni_rotary, "supports_kitchen_rope_fast", None)
        _allow_ptl_boogu_d120 = (
            getattr(_omni_package, "__xpu_target__", "") == "ptl-h"
            and callable(fast_marker)
            and fast_marker()
            and os.environ.get("OMNIXPU_BOOGU_D120_ROPE", "1") != "0"
        )
    except ImportError:
        _allow_ptl_zimage_pair = False
        _allow_ptl_boogu_d120 = False

    import comfy.ldm.flux.math as flux_math

    # Save originals
    _orig_apply_rope1 = flux_math._apply_rope1

    @trace_patch(
        "rope._apply_rope1",
        ("x", "freqs_cis"),
        stage="dispatch",
        verbose_only=True,
    )
    def _patched_apply_rope1(x: Tensor, freqs_cis: Tensor):
        if _use_ptl_boogu_d120(x, freqs_cis):
            return _omni_apply_boogu_d120(x, freqs_cis)
        if _can_use(x) and x.ndim == 4 and freqs_cis.ndim == 6:
            return _omni_apply_rope1(x, freqs_cis)
        return _orig_apply_rope1(x, freqs_cis)

    flux_math._apply_rope1 = _patched_apply_rope1

    # Patch apply_rope1 (the compiled/quant_ops version if present)
    _new_apply_rope1 = None
    if hasattr(flux_math, "apply_rope1"):
        _orig_compiled = flux_math.apply_rope1

        @trace_patch(
            "rope.apply_rope1",
            ("x", "freqs_cis"),
            stage="dispatch",
            verbose_only=True,
        )
        def _patched_compiled(x, freqs_cis):
            if _use_ptl_boogu_d120(x, freqs_cis):
                return _omni_apply_boogu_d120(x, freqs_cis)
            if _can_use(x) and x.ndim == 4 and freqs_cis.ndim == 6:
                return _omni_apply_rope1(x, freqs_cis)
            return _orig_compiled(x, freqs_cis)

        flux_math.apply_rope1 = _patched_compiled
        _new_apply_rope1 = _patched_compiled

    # ── Rebind by-value imports of apply_rope1 in already-loaded modules ─────
    # Several diffusion model files (wan, qwen_image, kandinsky5, sam3, ...)
    # do `from comfy.ldm.flux.math import apply_rope1` at module top-level.
    # Those bindings are frozen to the unpatched apply_rope1 by the time this
    # patch runs (after `import nodes`). Walk sys.modules and rebind.
    if _new_apply_rope1 is not None:
        rebound = 0
        for mod_name, mod in list(sys.modules.items()):
            if mod is None or mod is flux_math:
                continue
            try:
                cur = getattr(mod, "apply_rope1", None)
            except Exception:
                continue
            if cur is _orig_compiled:
                try:
                    setattr(mod, "apply_rope1", _new_apply_rope1)
                    rebound += 1
                except Exception:
                    pass
        log.info("[OmniXPU] rope: rebound %d by-value imports of apply_rope1", rebound)

    # ── General dual-tensor apply_rope ───────────────────────────────────────
    # comfy.ldm.flux.math.apply_rope(q, k, freqs) is the dual-tensor entry used
    # by flux (its own attention()), lumina, hidream, sam3, krea2, triposplat,
    # lens and pixeldit. Its inference branch forwards to comfy_kitchen
    # ck.apply_rope, which has no XPU backend (falls back to eager torch), so
    # none of these models reach the omni ESIMD rotary kernel via the
    # single-tensor apply_rope1 patch above. Patch apply_rope the same way:
    # route eligible tensors through the ESIMD kernel, fall back otherwise.
    # Numerically equivalent to the reference (verified: fp32 ~1e-7,
    # bf16 ~3e-3 rounding) — see krea2/verify_rope_equivalence.py.
    #
    # NOTE: this targets the flux.math.apply_rope symbol only. The
    # identically-named comfy.text_encoders.llama.apply_rope (used by the
    # llama/qwen35/sa3/gpt_oss text encoders and comfy.ldm.ideogram4) is a
    # different function with a different freq layout; the identity check in
    # the rebind walk below leaves it untouched.
    if hasattr(flux_math, "apply_rope"):
        _orig_apply_rope = flux_math.apply_rope

        @trace_patch(
            "rope.apply_rope",
            ("xq", "xk", "freqs_cis"),
            stage="dispatch",
            verbose_only=True,
        )
        def _patched_apply_rope(xq, xk, freqs_cis):
            global _logged_zimage_pair
            if _use_ptl_zimage_pair(xq, xk, freqs_cis):
                _logged_zimage_pair += 1
                if _logged_zimage_pair <= 3:
                    log.info(
                        "[OmniXPU] rope Kitchen pair #%d: seq=%d heads=%d "
                        "dtype=%s",
                        _logged_zimage_pair,
                        xq.shape[1],
                        xq.shape[2],
                        xq.dtype,
                    )
                log_debug_event(
                    "kernel",
                    "rotary_emb",
                    {"q": xq, "k": xk, "freqs": freqs_cis},
                    details={"backend": "kitchen_pair", "route": "ptl_zimage"},
                )
                return _omni_rotary.apply_kitchen_rope(xq, xk, freqs_cis)
            if (_can_use(xq) and _can_use(xk) and xq.ndim == 4 and xk.ndim == 4
                    and freqs_cis.ndim == 6):
                return _omni_apply_rope1(xq, freqs_cis), _omni_apply_rope1(xk, freqs_cis)
            return _orig_apply_rope(xq, xk, freqs_cis)

        flux_math.apply_rope = _patched_apply_rope

        # Rebind by-value imports of apply_rope in already-loaded modules
        # (lumina, hidream, sam3, krea2, triposplat, lens, pixeldit all do
        # `from comfy.ldm.flux.math import apply_rope`). The identity check
        # against the original flux.math.apply_rope excludes the llama-family
        # function of the same name.
        rebound_ar = 0
        for mod_name, mod in list(sys.modules.items()):
            if mod is None or mod is flux_math:
                continue
            try:
                cur = getattr(mod, "apply_rope", None)
            except Exception:
                continue
            if cur is _orig_apply_rope:
                try:
                    setattr(mod, "apply_rope", _patched_apply_rope)
                    rebound_ar += 1
                except Exception:
                    pass
        log.info("[OmniXPU] rope: patched apply_rope + rebound %d by-value imports", rebound_ar)

    return True, None
