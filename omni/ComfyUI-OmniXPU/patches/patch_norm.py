"""Patch comfy.ops LayerNorm / RMSNorm and comfy.rmsnorm.rms_norm to use
omni_xpu_kernel accelerated norms on XPU.

Mirrors the omni_b7 branch (analytics-zoo/ComfyUI @ 4aa7b1c):
- forward_comfy_cast_weights: cast weights via cast_bias_weight(..., offloadable=True),
  use omni kernel when eligible, then uncast. Preserves offload_stream lifecycle.
- forward: only takes the omni fast path when NOT in comfy-cast-weights mode and
  when weight_function / bias_function hooks are empty.
"""

import logging

import torch
import comfy.model_management

log = logging.getLogger("ComfyUI-OmniXPU")

_omni_norm = None
_logged_first_use = False


def _can_use_omni(x):
    if _omni_norm is None or not x.is_xpu:
        return False
    if not x.is_contiguous():
        return False
    if x.ndim < 2:
        return False
    h = x.shape[-1]
    return h <= 8192 and h % 32 == 0


def _log_first(op, shape):
    global _logged_first_use
    if not _logged_first_use:
        _logged_first_use = True
        log.info("[OmniXPU] norm first use: %s shape=%s", op, shape)


def apply():
    global _omni_norm
    import sys
    probe = sys.modules.get("ComfyUI-OmniXPU.probe")
    if probe.norm is None:
        return False, "omni_xpu_kernel norm not available"
    _omni_norm = probe.norm

    import comfy.ops as comfy_ops

    # cast_bias_weight / uncast_bias_weight must exist for us to preserve
    # the offload_stream lifecycle correctly.
    if not (hasattr(comfy_ops, "cast_bias_weight") and hasattr(comfy_ops, "uncast_bias_weight")):
        return False, "comfy.ops cast_bias_weight helpers not available"

    # --- LayerNorm ---
    LN = comfy_ops.disable_weight_init.LayerNorm
    _orig_ln_cast = LN.forward_comfy_cast_weights
    _orig_ln_fwd = LN.forward

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
            x = _omni_norm.layer_norm(x_2d, weight, bias, self.eps).reshape(orig)
        else:
            x = torch.nn.functional.layer_norm(input, self.normalized_shape, weight, bias, self.eps)
        comfy_ops.uncast_bias_weight(self, weight, bias, offload_stream)
        return x

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
            return _omni_norm.layer_norm(x_2d, self.weight, self.bias, self.eps).reshape(orig)
        return _orig_ln_fwd(self, *args, **kwargs)

    LN.forward_comfy_cast_weights = _ln_cast
    LN.forward = _ln_fwd

    # --- RMSNorm ---
    RN = comfy_ops.disable_weight_init.RMSNorm
    _orig_rn_cast = RN.forward_comfy_cast_weights
    _orig_rn_fwd = RN.forward

    def _rn_cast(self, input):
        if self.weight is not None:
            weight, bias, offload_stream = comfy_ops.cast_bias_weight(self, input, offloadable=True)
        else:
            weight = None
            bias = None
            offload_stream = None
        if _can_use_omni(input) and weight is not None and weight.shape[0] == input.shape[-1]:
            _log_first("RMSNorm", input.shape)
            orig = input.shape
            x_2d = input.reshape(-1, orig[-1])
            eps = self.eps if self.eps is not None else 1e-6
            x = _omni_norm.rms_norm(weight, x_2d, eps).reshape(orig)
        else:
            x = torch.nn.functional.rms_norm(input, self.normalized_shape, weight, self.eps)
        comfy_ops.uncast_bias_weight(self, weight, bias, offload_stream)
        return x

    def _rn_fwd(self, *args, **kwargs):
        if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
            return _rn_cast(self, *args, **kwargs)
        input = args[0] if args else kwargs.get("input")
        if (input is not None and _can_use_omni(input)
                and self.weight is not None and self.weight.shape[0] == input.shape[-1]):
            _log_first("RMSNorm", input.shape)
            orig = input.shape
            x_2d = input.reshape(-1, orig[-1])
            eps = self.eps if self.eps is not None else 1e-6
            return _omni_norm.rms_norm(self.weight, x_2d, eps).reshape(orig)
        return _orig_rn_fwd(self, *args, **kwargs)

    RN.forward_comfy_cast_weights = _rn_cast
    RN.forward = _rn_fwd

    # --- functional rms_norm ---
    try:
        import comfy.rmsnorm as comfy_rmsnorm
        _orig_rms_fn = comfy_rmsnorm.rms_norm

        def _patched_rms_norm(x, weight=None, eps=1e-6):
            if _can_use_omni(x):
                _log_first("rms_norm_fn", x.shape)
                orig = x.shape
                x_2d = x.reshape(-1, orig[-1])
                if weight is not None:
                    w = comfy.model_management.cast_to(weight, dtype=x.dtype, device=x.device)
                else:
                    w = torch.ones(orig[-1], dtype=x.dtype, device=x.device)
                return _omni_norm.rms_norm(w, x_2d, eps).reshape(orig)
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

    return True, None
