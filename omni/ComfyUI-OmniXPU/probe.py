"""Centralized omni_xpu_kernel capability probe."""

import logging

log = logging.getLogger("ComfyUI-OmniXPU")

version = None
sdp = None
norm = None
rotary = None
linear_fp8 = None


def probe():
    """Probe omni_xpu_kernel and populate available submodules."""
    global version, sdp, norm, rotary, linear_fp8

    try:
        import omni_xpu_kernel as pkg
        version = getattr(pkg, "__version__", "unknown")
    except ImportError:
        log.info("[OmniXPU] omni_xpu_kernel not installed")
        return

    modules = {}

    try:
        from omni_xpu_kernel._C import sdp as _sdp
        sdp = _sdp
        modules["sdp"] = True
    except ImportError:
        modules["sdp"] = False

    try:
        from omni_xpu_kernel import norm as _norm
        norm = _norm
        modules["norm"] = True
    except ImportError:
        modules["norm"] = False

    try:
        from omni_xpu_kernel import rotary as _rotary
        rotary = _rotary
        modules["rotary"] = True
    except ImportError:
        modules["rotary"] = False

    try:
        from omni_xpu_kernel import linear as _linear
        linear_fp8 = _linear.onednn_w8a16_fp8
        modules["linear_fp8"] = True
    except (ImportError, AttributeError):
        modules["linear_fp8"] = False

    available = [k for k, v in modules.items() if v]
    missing = [k for k, v in modules.items() if not v]

    log.info("[OmniXPU] omni_xpu_kernel %s — available: %s%s",
             version, ", ".join(available) if available else "none",
             f" | missing: {', '.join(missing)}" if missing else "")


def summary():
    """Return a dict for diagnostics."""
    return {
        "version": version,
        "sdp": sdp is not None,
        "norm": norm is not None,
        "rotary": rotary is not None,
        "linear_fp8": linear_fp8 is not None,
    }
