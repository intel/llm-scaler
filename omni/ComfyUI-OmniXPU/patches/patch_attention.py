import logging

import torch

log = logging.getLogger("ComfyUI-OmniXPU")

_esimd_sdp = None
_esimd_call_count = 0
_esimd_fallback_count = 0
_esimd_fallback_reasons = {}


def get_stats():
    return {
        "esimd": _esimd_call_count,
        "fallback": _esimd_fallback_count,
        "reasons": dict(_esimd_fallback_reasons),
    }


def apply():
    global _esimd_sdp
    import sys
    probe = sys.modules.get("ComfyUI-OmniXPU.probe")
    if probe.sdp is None:
        return False, "omni_xpu_kernel sdp not available"
    _esimd_sdp = probe.sdp

    import comfy.ldm.modules.attention as attn_mod

    if not hasattr(attn_mod, "attention_pytorch"):
        return False, "attention_pytorch not found"

    _pytorch_fallback = attn_mod.attention_pytorch
    wrap_attn = attn_mod.wrap_attn

    @wrap_attn
    def attention_esimd(q, k, v, heads, mask=None, attn_precision=None,
                        skip_reshape=False, skip_output_reshape=False, **kwargs):
        global _esimd_call_count, _esimd_fallback_count

        if skip_reshape:
            b, _, _, dim_head = q.shape
        else:
            b, _, dim_head = q.shape
            dim_head //= heads

        # Constraint check
        reasons = []
        if b != 1:
            reasons.append(f"batch={b}")
        if mask is not None:
            reasons.append(f"mask={mask.shape}")
        if dim_head not in (64, 128):
            reasons.append(f"dim_head={dim_head}")
        if q.device.type != "xpu":
            reasons.append(f"device={q.device.type}")
        if q.dtype not in (torch.float16, torch.bfloat16):
            reasons.append(f"dtype={q.dtype}")

        if reasons:
            _esimd_fallback_count += 1
            key = ",".join(reasons)
            _esimd_fallback_reasons[key] = _esimd_fallback_reasons.get(key, 0) + 1
            if _esimd_fallback_count <= 5:
                seq = q.shape[1] if not skip_reshape else q.shape[2]
                log.info("[OmniXPU] attention fallback: %s (seq=%d)", key, seq)
            return _pytorch_fallback(q, k, v, heads, mask=mask, attn_precision=attn_precision,
                                     skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape, **kwargs)

        _esimd_call_count += 1
        if _esimd_call_count <= 3:
            seq = q.shape[1] if not skip_reshape else q.shape[2]
            log.info("[OmniXPU] attention ESIMD #%d: heads=%d seq=%d dtype=%s", _esimd_call_count, heads, seq, q.dtype)

        if skip_reshape:
            q_blhd = q.permute(0, 2, 1, 3).contiguous()
            k_blhd = k.permute(0, 2, 1, 3).contiguous()
            v_blhd = v.permute(0, 2, 1, 3).contiguous()
        else:
            q_blhd = q.view(b, -1, heads, dim_head).contiguous()
            k_blhd = k.view(b, -1, heads, dim_head).contiguous()
            v_blhd = v.view(b, -1, heads, dim_head).contiguous()

        out = _esimd_sdp.sdp(q_blhd, k_blhd, v_blhd)

        # FP16 NaN safety
        if q.dtype == torch.float16 and (out != out).any():
            _esimd_fallback_count += 1
            _esimd_fallback_reasons["output_non_finite"] = _esimd_fallback_reasons.get("output_non_finite", 0) + 1
            if _esimd_fallback_reasons["output_non_finite"] <= 3:
                log.warning("[OmniXPU] FP16 overflow in ESIMD, falling back to SDPA")
            return _pytorch_fallback(q, k, v, heads, mask=mask, attn_precision=attn_precision,
                                     skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape, **kwargs)

        if skip_output_reshape:
            return out.permute(0, 2, 1, 3)
        return out.reshape(b, -1, heads * dim_head)

    # Capture the originals BEFORE rebinding so we can detect by-value imports
    # in already-loaded modules.
    _originals = {
        attn_mod.attention_basic,
        attn_mod.attention_pytorch,
        attn_mod.optimized_attention,
        attn_mod.optimized_attention_masked,
    }

    # Patch module-level variables
    attn_mod.optimized_attention = attention_esimd
    attn_mod.optimized_attention_masked = attention_esimd

    # ── Rebind by-value imports in already-loaded modules ────────────────────
    # Many comfy.ldm.* and custom_nodes do `from comfy.ldm.modules.attention
    # import optimized_attention[_masked]` at module top-level. Those bindings
    # are frozen to attention_basic/attention_pytorch by the time this patch
    # runs (after `import nodes` has already pulled in model_base → all
    # ldm.*.model files). We must walk sys.modules and rebind each one.
    NAMES = ("optimized_attention", "optimized_attention_masked")
    rebound = 0
    for mod_name, mod in list(sys.modules.items()):
        if mod is None or mod is attn_mod:
            continue
        for name in NAMES:
            try:
                cur = getattr(mod, name, None)
            except Exception:
                continue
            if cur is not None and cur in _originals:
                try:
                    setattr(mod, name, attention_esimd)
                    rebound += 1
                except Exception:
                    pass
    log.info("[OmniXPU] attention: rebound %d by-value imports across sys.modules", rebound)

    # Also register via the official API
    if hasattr(attn_mod, "register_attention_function"):
        attn_mod.register_attention_function("esimd", attention_esimd)

    return True, None
