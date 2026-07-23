"""Patch comfy.ops LayerNorm / RMSNorm and comfy.rmsnorm.rms_norm to use
omni_xpu_kernel accelerated norms on XPU.

Mirrors the omni_b7 branch (analytics-zoo/ComfyUI @ 4aa7b1c):
- forward_comfy_cast_weights: cast weights via cast_bias_weight(..., offloadable=True),
  use omni kernel when eligible, then uncast. Preserves offload_stream lifecycle.
- forward: only takes the omni fast path when NOT in comfy-cast-weights mode and
  when weight_function / bias_function hooks are empty.
"""

import logging
import os

import torch
import comfy.model_management

from ..patches.debug import log_debug_event, trace_patch

log = logging.getLogger("ComfyUI-OmniXPU")

_omni_norm = None
_logged_first_use = False
_allow_noncontiguous_rms = False
_allow_h120_rms = False


def _can_use_omni(x):
    if _omni_norm is None or not x.is_xpu:
        return False
    if not x.is_contiguous():
        return False
    if x.ndim < 2:
        return False
    h = x.shape[-1]
    return h <= 8192 and h % 32 == 0


def _rms_input_2d(x):
    """Return an eligible 2D RMSNorm input, materializing PTL split views."""
    if _omni_norm is None or not x.is_xpu or x.ndim < 2:
        return None
    h = x.shape[-1]
    if h > 8192 or (h % 32 != 0 and not (_allow_h120_rms and h == 120)):
        return None
    if not x.is_contiguous():
        # Lumina/Z-Image Q and K are views into a combined QKV projection. The
        # last dimension is dense, but every token has a gap after the selected
        # projection. Torch decomposes RMSNorm on this layout into several
        # large elementwise/reduction kernels. On PTL it is faster to make one
        # dense copy and then use the ESIMD RMSNorm kernel. Keep other targets
        # on their validated route until they receive their own workflow A/B.
        if (
            not _allow_noncontiguous_rms
            or x.ndim != 4
            or x.shape[0] != 1
            or x.shape[2] != 30
            or h != 128
            or x.stride(-1) != 1
            or x.stride(2) != h
            or x.stride(1) != 3 * x.shape[2] * h
            or x.stride(0) != x.shape[1] * x.stride(1)
        ):
            return None
        x = x.contiguous()
    return x.reshape(-1, h)


def _log_first(op, shape):
    global _logged_first_use
    if not _logged_first_use:
        _logged_first_use = True
        log.info("[OmniXPU] norm first use: %s shape=%s", op, shape)


def _run_layer_norm(x, weight, bias, eps):
    log_debug_event(
        "kernel",
        "layer_norm",
        {"input": x, "weight": weight, "bias": bias},
        details={"backend": "esimd"},
    )
    return _omni_norm.layer_norm(x, weight, bias, eps)


def _run_rms_norm(weight, x, eps):
    log_debug_event(
        "kernel",
        "rms_norm",
        {"input": x, "weight": weight},
        details={"backend": "esimd"},
    )
    return _omni_norm.rms_norm(weight, x, eps)


def apply():
    global _allow_h120_rms, _allow_noncontiguous_rms, _omni_norm
    import sys
    probe = sys.modules.get("ComfyUI-OmniXPU.probe")
    if probe.norm is None:
        return False, "omni_xpu_kernel norm not available"
    _omni_norm = probe.norm
    try:
        import omni_xpu_kernel as _omni_package

        target = getattr(_omni_package, "__xpu_target__", "")
        _allow_noncontiguous_rms = (
            target == "ptl-h"
            and os.environ.get("OMNIXPU_NONCONTIG_RMSNORM", "1") != "0"
        )
        supports_h120 = getattr(_omni_norm, "supports_h120_fp16", None)
        _allow_h120_rms = (
            target == "ptl-h"
            and callable(supports_h120)
            and supports_h120()
            and os.environ.get("OMNIXPU_H120_RMSNORM", "1") != "0"
        )
        log.info(
            "[OmniXPU] norm: H120 FP16 native route %s (target=%s)",
            "enabled" if _allow_h120_rms else "disabled",
            target or "unknown",
        )
    except ImportError:
        _allow_noncontiguous_rms = False
        _allow_h120_rms = False

    import comfy.ops as comfy_ops

    # cast_bias_weight / uncast_bias_weight must exist for us to preserve
    # the offload_stream lifecycle correctly.
    if not (hasattr(comfy_ops, "cast_bias_weight") and hasattr(comfy_ops, "uncast_bias_weight")):
        return False, "comfy.ops cast_bias_weight helpers not available"

    # --- LayerNorm ---
    LN = comfy_ops.disable_weight_init.LayerNorm
    _orig_ln_cast = LN.forward_comfy_cast_weights
    _orig_ln_fwd = LN.forward

    @trace_patch(
        "norm.LayerNorm.forward_comfy_cast_weights",
        ("self", "input"),
        stage="dispatch",
        verbose_only=True,
    )
    def _ln_cast(self, input):
        if self.weight is not None:
            weight, bias, offload_stream = comfy_ops.cast_bias_weight(self, input, offloadable=True)
        else:
            weight = None
            bias = None
            offload_stream = None
        if (_can_use_omni(input) and len(self.normalized_shape) == 1
                and (weight is None or weight.shape[0] == input.shape[-1])):
            _log_first("LayerNorm", input.shape)
            orig = input.shape
            x_2d = input.reshape(-1, orig[-1])
            x = _run_layer_norm(x_2d, weight, bias, self.eps).reshape(orig)
        else:
            x = torch.nn.functional.layer_norm(input, self.normalized_shape, weight, bias, self.eps)
        comfy_ops.uncast_bias_weight(self, weight, bias, offload_stream)
        return x

    @trace_patch(
        "norm.LayerNorm.forward",
        ("self", "input"),
        stage="dispatch",
        verbose_only=True,
    )
    def _ln_fwd(self, *args, **kwargs):
        # run_every_op() is called by the original forward; skip here to avoid
        # double-counting. Only use omni fast path when NOT in cast-weights mode.
        if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
            return _ln_cast(self, *args, **kwargs)
        input = args[0] if args else kwargs.get("input")
        if (input is not None and _can_use_omni(input) and len(self.normalized_shape) == 1
                and (self.weight is None or self.weight.shape[0] == input.shape[-1])):
            _log_first("LayerNorm", input.shape)
            orig = input.shape
            x_2d = input.reshape(-1, orig[-1])
            return _run_layer_norm(x_2d, self.weight, self.bias, self.eps).reshape(orig)
        return _orig_ln_fwd(self, *args, **kwargs)

    LN.forward_comfy_cast_weights = _ln_cast
    LN.forward = _ln_fwd

    # --- RMSNorm ---
    RN = comfy_ops.disable_weight_init.RMSNorm
    _orig_rn_cast = RN.forward_comfy_cast_weights
    _orig_rn_fwd = RN.forward

    @trace_patch(
        "norm.RMSNorm.forward_comfy_cast_weights",
        ("self", "input"),
        stage="dispatch",
        verbose_only=True,
    )
    def _rn_cast(self, input):
        if self.weight is not None:
            weight, bias, offload_stream = comfy_ops.cast_bias_weight(self, input, offloadable=True)
        else:
            weight = None
            bias = None
            offload_stream = None
        input_2d = _rms_input_2d(input)
        if input_2d is not None and weight is not None and weight.shape[0] == input.shape[-1]:
            _log_first("RMSNorm", input.shape)
            orig = input.shape
            eps = self.eps if self.eps is not None else 1e-6
            x = _run_rms_norm(weight, input_2d, eps).reshape(orig)
        else:
            x = torch.nn.functional.rms_norm(input, self.normalized_shape, weight, self.eps)
        comfy_ops.uncast_bias_weight(self, weight, bias, offload_stream)
        return x

    @trace_patch(
        "norm.RMSNorm.forward",
        ("self", "input"),
        stage="dispatch",
        verbose_only=True,
    )
    def _rn_fwd(self, *args, **kwargs):
        if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
            return _rn_cast(self, *args, **kwargs)
        input = args[0] if args else kwargs.get("input")
        input_2d = _rms_input_2d(input) if input is not None else None
        if (input_2d is not None and self.weight is not None
                and self.weight.shape[0] == input.shape[-1]):
            _log_first("RMSNorm", input.shape)
            orig = input.shape
            eps = self.eps if self.eps is not None else 1e-6
            return _run_rms_norm(self.weight, input_2d, eps).reshape(orig)
        return _orig_rn_fwd(self, *args, **kwargs)

    RN.forward_comfy_cast_weights = _rn_cast
    RN.forward = _rn_fwd

    # --- functional rms_norm ---
    try:
        import comfy.rmsnorm as comfy_rmsnorm
        _orig_rms_fn = comfy_rmsnorm.rms_norm

        @trace_patch(
            "norm.rms_norm",
            ("x", "weight", "eps"),
            stage="dispatch",
            verbose_only=True,
        )
        def _patched_rms_norm(x, weight=None, eps=1e-6):
            x_2d = _rms_input_2d(x)
            if x_2d is not None:
                _log_first("rms_norm_fn", x.shape)
                orig = x.shape
                if weight is not None:
                    w = comfy.model_management.cast_to(weight, dtype=x.dtype, device=x.device)
                else:
                    w = torch.ones(orig[-1], dtype=x.dtype, device=x.device)
                return _run_rms_norm(w, x_2d, eps).reshape(orig)
            return _orig_rms_fn(x, weight=weight, eps=eps)

        comfy_rmsnorm.rms_norm = _patched_rms_norm

        # ── Rebind by-value imports of rms_norm ──────────────────────────────
        # comfy.ldm.common_dit.py does `rms_norm = comfy.rmsnorm.rms_norm` at
        # import time, and many diffusion models (lightricks/LTX, genmo,
        # mmdit, llama text encoder) call `comfy.ldm.common_dit.rms_norm`.
        # Without this rebind, those call sites still hit the original
        # PyTorch implementation.
        rebound = 0
        for mod_name, mod in list(sys.modules.items()):
            if mod is None or mod is comfy_rmsnorm:
                continue
            try:
                cur = getattr(mod, "rms_norm", None)
            except Exception:
                continue
            if cur is _orig_rms_fn:
                try:
                    setattr(mod, "rms_norm", _patched_rms_norm)
                    rebound += 1
                except Exception:
                    pass
        log.info("[OmniXPU] norm: rebound %d by-value imports of rms_norm", rebound)
    except (ImportError, AttributeError):
        pass  # comfy.rmsnorm may not exist in all versions

    # --- Krea2 local RMSNorm (separate opt-in sub-switch) ---
    # Gated by OMNIXPU_KREA2_RMSNORM (default on). This is deliberately a
    # separate switch from the general norm patch: Krea2 is the only model that
    # defines its OWN RMSNorm class instead of using comfy.ops.RMSNorm / the
    # comfy rms_norm wrapper, so this hook is Krea2-specific and may become
    # unnecessary if upstream refactors Krea2 onto the shared wrapper.
    #
    # comfy.ldm.krea2.model defines its OWN RMSNorm(nn.Module) that calls
    # torch F.rms_norm directly, using the (1 + scale) weight convention and
    # fp32 accumulation. It uses neither comfy.ops.RMSNorm nor
    # comfy.rmsnorm.rms_norm, so the patches above never reach it. Hijack its
    # forward to use the omni ESIMD kernel (which also supports fp32), while
    # preserving the exact numerics: weight = scale.float() + 1.0, fp32 compute,
    # cast back to the input dtype.
    if os.environ.get("OMNIXPU_KREA2_RMSNORM", "1") == "0":
        log.info("[OmniXPU] norm: Krea2 RMSNorm patch disabled (OMNIXPU_KREA2_RMSNORM=0)")
    else:
        try:
            import comfy.ldm.krea2.model as _krea2_model

            _KreaRMS = _krea2_model.RMSNorm

            @trace_patch(
                "norm.Krea2RMSNorm.forward",
                ("self", "x"),
                stage="dispatch",
                verbose_only=True,
            )
            def _krea2_rms_forward(self, x):
                dtype = x.dtype
                weight = comfy.model_management.cast_to(
                    self.scale, dtype=torch.float32, device=x.device) + 1.0
                h = x.shape[-1]
                if (_omni_norm is not None and x.is_xpu and x.ndim >= 2
                        and h <= 8192 and h % 32 == 0):
                    _log_first("Krea2RMSNorm", x.shape)
                    orig = x.shape
                    x_2d = x.float().reshape(-1, h).contiguous()
                    return _run_rms_norm(weight, x_2d, self.eps).reshape(orig).to(dtype)
                return torch.nn.functional.rms_norm(
                    x.float(), (h,), weight=weight, eps=self.eps).to(dtype)

            _KreaRMS.forward = _krea2_rms_forward
            log.info("[OmniXPU] norm: patched Krea2 local RMSNorm.forward")
        except (ImportError, AttributeError):
            pass  # krea2 model not present in this ComfyUI version

    return True, None
