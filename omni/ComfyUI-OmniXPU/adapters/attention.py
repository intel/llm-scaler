import logging
import os

import torch

from ..patches.debug import log_debug_event

log = logging.getLogger("ComfyUI-OmniXPU")

_fallback_esimd_sdp = None
_esimd_call_count = 0
_esimd_fallback_count = 0
_esimd_fallback_reasons = {}

# ── Attention backend selection ──────────────────────────────────────────────
# OMNI_ATTN_BACKEND selects which attention routing policy the patched ComfyUI
# path uses:
#   auto   (default) — use platform/workflow-tuned routes where validated, then
#                      cute for d128 self-attention, ESIMD for supported
#                      d64/cross-attention, and finally PyTorch fallback.
#   cute             — CUTLASS-SYCL FMHA (omni_xpu_kernel.cute). fp32 accumulation,
#                      so it does NOT overflow on large activations (Qwen-Image etc.)
#                      where the ESIMD fp16-accumulator kernel can. Unsupported
#                      shapes fall back to PyTorch rather than switching backend.
#   esimd            — omni_xpu_kernel.sdp (hand-written ESIMD flash attention;
#                      ~6% faster on large self-attn but fp16 accumulator).
#   torch            — no cute/esimd; always fall back to PyTorch SDPA.
# The cute backend prefers the packaged omni_xpu_kernel.cute module and falls back
# to a raw .so (OMNI_CUTE_FMHA_SO overrides the path).
_backend = os.environ.get("OMNI_ATTN_BACKEND", "auto").lower()
_backend_name = _backend  # for logging
_backend_sdp = None  # callable(q_blhd, k_blhd, v_blhd) -> out_blhd
_torch_sdpa_count = 0


def _torch_major_minor():
    try:
        components = torch.__version__.split("+", 1)[0].split(".")
        return int(components[0]), int(components[1])
    except (AttributeError, IndexError, ValueError):
        return None


def _omni_xpu_target():
    try:
        import omni_xpu_kernel as pkg

        return getattr(pkg, "__xpu_target__", None)
    except ImportError:
        return None


def _use_ptl_torch_sdpa(
    q,
    heads,
    dim_head,
    q_len,
    kv_len,
    skip_reshape,
    skip_output_reshape,
):
    """Select only workflow shapes validated on PTL-H with Torch 2.11."""
    is_zimage = heads == 30 and q_len in (64, 1024, 1088)
    is_krea2 = heads == 48 and q_len == 4192
    return (
        _backend == "auto"
        and _backend_name == "cute"
        and _omni_xpu_target() == "ptl-h"
        and _torch_major_minor() == (2, 11)
        and q.dtype == torch.bfloat16
        and dim_head == 128
        and q_len == kv_len
        and (is_zimage or is_krea2)
        and skip_reshape
        and not skip_output_reshape
    )


def _is_dense_d120_bhld(tensor, heads, seq, dim_head):
    try:
        shape = tuple(tensor.shape)
        strides = tensor.stride()
    except (AttributeError, TypeError):
        return False
    if shape != (1, heads, seq, dim_head):
        return False
    if len(strides) != 4 or strides[3] != 1:
        return False
    if strides[0] != heads * seq * dim_head:
        return False
    packed_bhld = strides[1] == seq * dim_head and strides[2] == dim_head
    blhd_backed = strides[1] == dim_head and strides[2] == heads * dim_head
    return packed_bhld or blhd_backed


def _use_ptl_cute_d120(
    q,
    k,
    v,
    heads,
    dim_head,
    q_len,
    kv_len,
    skip_reshape,
    skip_output_reshape,
):
    capability = getattr(_backend_sdp, "supports_d120_bhld", None)
    return (
        _backend == "auto"
        and _backend_name == "cute"
        and _omni_xpu_target() == "ptl-h"
        and _torch_major_minor() == (2, 11)
        and callable(capability)
        and capability()
        and q.dtype == torch.float16
        and k.dtype == q.dtype
        and v.dtype == q.dtype
        and q.device.type == "xpu"
        and k.device.type == "xpu"
        and v.device.type == "xpu"
        and heads == 28
        and dim_head == 120
        and q_len == kv_len
        and q_len in (4096, 4205)
        and skip_reshape
        and not skip_output_reshape
        and _is_dense_d120_bhld(q, heads, q_len, dim_head)
        and _is_dense_d120_bhld(k, heads, kv_len, dim_head)
        and _is_dense_d120_bhld(v, heads, kv_len, dim_head)
    )


def _default_cute_so():
    # Ship next to the omni_xpu_kernel package by default.
    try:
        import omni_xpu_kernel as pkg

        d = os.path.dirname(os.path.abspath(pkg.__file__))
        return os.path.join(d, "cute", "cute_fmha_torch.so")
    except Exception:
        return ""


def _load_cute_backend():
    # Preferred: the packaged submodule (handles .so location + torch op load).
    try:
        from omni_xpu_kernel import cute as _cute

        if _cute is not None and _cute.is_available():
            return _cute, None
    except Exception:
        pass
    # Fallback: load a raw .so directly (dev / override via OMNI_CUTE_FMHA_SO).
    so = os.environ.get("OMNI_CUTE_FMHA_SO", "") or _default_cute_so()
    if not so or not os.path.exists(so):
        return None, f"cute backend unavailable (.so not found: {so})"
    try:
        torch.ops.load_library(so)
        fn = torch.ops.cute_fmha.sdp

        class _Wrap:
            @staticmethod
            def sdp(q, k, v):
                return fn(q, k, v)

        return _Wrap, None
    except Exception as e:
        return None, f"cute load failed: {e}"


def get_stats():
    return {
        "policy": _backend,
        "backend": _backend_name,
        "esimd": _esimd_call_count,
        "torch_sdpa": _torch_sdpa_count,
        "fallback": _esimd_fallback_count,
        "reasons": dict(_esimd_fallback_reasons),
    }


def apply():
    global _fallback_esimd_sdp, _backend_sdp, _backend_name
    import sys

    probe = sys.modules.get("ComfyUI-OmniXPU.probe")

    # Resolve the requested backend.
    if _backend not in {"auto", "cute", "esimd", "torch"}:
        return False, f"invalid OMNI_ATTN_BACKEND={_backend!r}"
    if _backend == "torch":
        # Force PyTorch SDPA everywhere: do not patch at all.
        return False, "OMNI_ATTN_BACKEND=torch (using PyTorch SDPA, no patch)"
    elif _backend in {"auto", "cute"}:
        wrap, err = _load_cute_backend()
        if wrap is not None:
            _backend_sdp = wrap
            _backend_name = "cute"
        elif _backend == "auto" and probe is not None and probe.sdp is not None:
            # cute requested but unavailable — degrade to esimd rather than SDPA.
            log.warning(
                "[OmniXPU] cute backend unavailable (%s); falling back to esimd", err
            )
            _backend_sdp = probe.sdp
            _backend_name = "esimd"
        else:
            return False, err
    else:  # esimd
        if probe is None or probe.sdp is None:
            return False, "omni_xpu_kernel sdp not available"
        _backend_sdp = probe.sdp
        _backend_name = "esimd"

    _fallback_esimd_sdp = probe.sdp if probe is not None else None

    import comfy.ldm.modules.attention as attn_mod

    if not hasattr(attn_mod, "attention_pytorch"):
        return False, "attention_pytorch not found"

    _pytorch_fallback = attn_mod.attention_pytorch
    wrap_attn = attn_mod.wrap_attn

    @wrap_attn
    def attention_esimd(
        q,
        k,
        v,
        heads,
        mask=None,
        attn_precision=None,
        skip_reshape=False,
        skip_output_reshape=False,
        **kwargs,
    ):
        global _esimd_call_count, _esimd_fallback_count, _torch_sdpa_count

        log_debug_event(
            "dispatch",
            "attention",
            {"q": q, "k": k, "v": v, "mask": mask},
            details={"policy": _backend},
            verbose_only=True,
        )

        if skip_reshape:
            b, _, _, dim_head = q.shape
        else:
            b, _, dim_head = q.shape
            dim_head //= heads

        if skip_reshape:
            q_len, kv_len = q.shape[2], k.shape[2]
        else:
            q_len, kv_len = q.shape[1], k.shape[1]

        use_ptl_cute_d120 = _use_ptl_cute_d120(
            q,
            k,
            v,
            heads,
            dim_head,
            q_len,
            kv_len,
            skip_reshape,
            skip_output_reshape,
        )

        # Constraint check
        reasons = []
        if b != 1:
            reasons.append(f"batch={b}")
        if mask is not None:
            reasons.append(f"mask={mask.shape}")
        if dim_head not in (64, 128) and not use_ptl_cute_d120:
            reasons.append(f"dim_head={dim_head}")
        if q.device.type != "xpu":
            reasons.append(f"device={q.device.type}")
        if q.dtype not in (torch.float16, torch.bfloat16):
            reasons.append(f"dtype={q.dtype}")
        if kwargs.get("enable_gqa", False):
            reasons.append("enable_gqa")
        if "scale" in kwargs:
            reasons.append("custom_scale")

        selected_sdp = _backend_sdp
        selected_backend = _backend_name
        # Torch 2.11 SDPA is faster end-to-end for the measured PTL-H D128
        # workflow shapes. Keep this route narrower than the generic d128 CUTE
        # domain: explicit `cute`, other platforms/versions, dtypes, head
        # counts, and sequence lengths retain the existing policy.
        if not reasons and _use_ptl_torch_sdpa(
            q,
            heads,
            dim_head,
            q_len,
            kv_len,
            skip_reshape,
            skip_output_reshape,
        ):
            _torch_sdpa_count += 1
            if _torch_sdpa_count <= 3:
                log.info(
                    "[OmniXPU] attention TORCH #%d: heads=%d seq=%d dtype=%s",
                    _torch_sdpa_count,
                    heads,
                    q_len,
                    q.dtype,
                )
            log_debug_event(
                "kernel",
                "attention",
                {"q": q, "k": k, "v": v},
                details={"backend": "torch", "route": "ptl_torch211_workflow"},
            )
            return _pytorch_fallback(
                q,
                k,
                v,
                heads,
                mask=mask,
                attn_precision=attn_precision,
                skip_reshape=skip_reshape,
                skip_output_reshape=skip_output_reshape,
                **kwargs,
            )

        # Boogu's PTL-H D120 route consumes the exact BHLD input strides and
        # returns a BLHD-backed BHLD view.  The final transpose+reshape is a
        # metadata-only view, avoiding all layout copies.  This remains an
        # auto-only, Torch-2.11, workflow-shape route; unsupported wheels and
        # layouts retain the unmodified Torch fallback.
        if not reasons and use_ptl_cute_d120:
            _esimd_call_count += 1
            if _esimd_call_count <= 3:
                log.info(
                    "[OmniXPU] attention CUTE_D120 #%d: heads=%d seq=%d dtype=%s",
                    _esimd_call_count,
                    heads,
                    q_len,
                    q.dtype,
                )
            log_debug_event(
                "kernel",
                "attention",
                {"q": q, "k": k, "v": v},
                details={"backend": "cute", "route": "ptl_cute_d120_bhld"},
            )
            out = _backend_sdp.sdp_bhld_d120(q, k, v)
            return out.transpose(1, 2).reshape(b, -1, heads * dim_head)

        # cute is currently accepted only for d128 self-attention. Auto keeps
        # d64 and cross-attention on ESIMD; explicit cute never silently changes
        # to another fused backend and instead uses the safe PyTorch fallback.
        cute_needs_esimd = _backend_name == "cute" and (
            dim_head != 128 or q_len != kv_len
        )
        if not reasons and cute_needs_esimd:
            if _backend == "auto" and _fallback_esimd_sdp is not None:
                selected_sdp = _fallback_esimd_sdp
                selected_backend = "esimd"
            else:
                reasons.append(f"cute_unsupported=dim{dim_head},q{q_len},kv{kv_len}")

        if reasons:
            _esimd_fallback_count += 1
            key = ",".join(reasons)
            _esimd_fallback_reasons[key] = _esimd_fallback_reasons.get(key, 0) + 1
            if _esimd_fallback_count <= 5:
                seq = q.shape[1] if not skip_reshape else q.shape[2]
                log.info("[OmniXPU] attention fallback: %s (seq=%d)", key, seq)
            return _pytorch_fallback(
                q,
                k,
                v,
                heads,
                mask=mask,
                attn_precision=attn_precision,
                skip_reshape=skip_reshape,
                skip_output_reshape=skip_output_reshape,
                **kwargs,
            )

        _esimd_call_count += 1
        if _esimd_call_count <= 3:
            seq = q.shape[1] if not skip_reshape else q.shape[2]
            log.info(
                "[OmniXPU] attention %s #%d: heads=%d seq=%d dtype=%s",
                selected_backend.upper(),
                _esimd_call_count,
                heads,
                seq,
                q.dtype,
            )

        if skip_reshape:
            q_blhd = q.permute(0, 2, 1, 3).contiguous()
            k_blhd = k.permute(0, 2, 1, 3).contiguous()
            v_blhd = v.permute(0, 2, 1, 3).contiguous()
        else:
            q_blhd = q.view(b, -1, heads, dim_head).contiguous()
            k_blhd = k.view(b, -1, heads, dim_head).contiguous()
            v_blhd = v.view(b, -1, heads, dim_head).contiguous()

        log_debug_event(
            "kernel",
            "attention",
            {"q": q_blhd, "k": k_blhd, "v": v_blhd},
            details={"backend": selected_backend},
        )
        out = selected_sdp.sdp(q_blhd, k_blhd, v_blhd)

        # FP16 NaN safety
        if q.dtype == torch.float16 and (out != out).any():
            _esimd_fallback_count += 1
            _esimd_fallback_reasons["output_non_finite"] = (
                _esimd_fallback_reasons.get("output_non_finite", 0) + 1
            )
            if _esimd_fallback_reasons["output_non_finite"] <= 3:
                log.warning(
                    "[OmniXPU] FP16 overflow in %s, falling back to SDPA",
                    selected_backend.upper(),
                )
            return _pytorch_fallback(
                q,
                k,
                v,
                heads,
                mask=mask,
                attn_precision=attn_precision,
                skip_reshape=skip_reshape,
                skip_output_reshape=skip_output_reshape,
                **kwargs,
            )

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
    log.info(
        "[OmniXPU] attention[%s]: rebound %d by-value imports across sys.modules",
        _backend_name,
        rebound,
    )

    # Also register via the official API
    if hasattr(attn_mod, "register_attention_function"):
        attn_mod.register_attention_function("esimd", attention_esimd)

    return True, None
