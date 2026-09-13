"""Expose the XPU kernel's INT8 operators to ComfyUI's quantized path.

ComfyUI core hardcodes ``supports_int8_compute() == False`` for Intel XPU, so
``get_disabled_quant_formats()`` marks ``int8_tensorwise`` / ``convrot_w4a4`` /
``asym_w4a8_int8`` as emulated and every int8 checkpoint is dequantized to bf16
while it is loaded (visible as a doubled "Staged" size and ``quant_format=None``
in the int8 fast paths).

``omni_xpu_kernel`` ships the operators that quantized matmul needs, so reopen
the gate when they are present.  ``OMNIXPU_INT8_NATIVE=0`` restores stock
behaviour.
"""

from __future__ import annotations

import logging

log = logging.getLogger("ComfyUI-OmniXPU")

_PATCH_FLAG = "_omnixpu_int8_gate_open"
_REQUIRED_OPS = (
    "quantize_int8_rowwise",
    "dequantize_int8_simple_dtype",
    "mm_int8",
    "int8_linear",
)


def _int8_ops():
    import sys

    probe = sys.modules.get("ComfyUI-OmniXPU.probe")
    ops = getattr(probe, "int8", None) if probe is not None else None
    if ops is not None:
        return ops
    from omni_xpu_kernel import int8 as ops

    return ops


def apply():
    try:
        import comfy.model_management as comfy_mm
    except Exception as exc:  # pragma: no cover - startup guard
        return False, f"comfy.model_management unavailable: {exc}"

    if getattr(comfy_mm, _PATCH_FLAG, False):
        return False, "already patched"

    try:
        ops = _int8_ops()
    except Exception as exc:
        return False, f"omni_xpu_kernel.int8 unavailable: {exc}"

    missing = [name for name in _REQUIRED_OPS if not hasattr(ops, name)]
    if missing:
        return False, f"kernel is missing int8 operators: {missing}"

    original = comfy_mm.supports_int8_compute

    def supports_int8_compute(device=None):
        try:
            if comfy_mm.is_intel_xpu():
                return True
        except Exception:
            pass
        return original(device)

    comfy_mm.supports_int8_compute = supports_int8_compute
    setattr(comfy_mm, _PATCH_FLAG, True)
    log.info(
        "[OmniXPU] int8 compute gate opened for XPU; quantized matmul can now use "
        "the native omni_xpu_kernel int8 operators"
    )
    return True, ""
