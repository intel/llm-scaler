import glob
import logging
import os
import sys
from importlib.machinery import EXTENSION_SUFFIXES

import torch

from ..patches.debug import log_debug_event
from .errors import is_fatal_accelerator_error

log = logging.getLogger("ComfyUI-OmniXPU")

_cute_call_count = 0
_esimd_call_count = 0
_attention_fallback_count = 0
_attention_fallback_reasons = {}
_attention_route_counts = {}
_attention_failed_contracts = set()
_attention_experimental_warning_emitted = False
_attention_traced_contracts = set()

_MINIMAX_H3_H56_CUTE_MIN_SEQUENCE = 31
_MINIMAX_H3_VAE_D64_CUTE_MIN_SEQUENCE = 6
_VALIDATE_OUTPUT_ENV = "OMNIXPU_VALIDATE_ATTENTION_OUTPUT"

# ── Attention backend selection ──────────────────────────────────────────────
# OMNI_ATTN_BACKEND selects which attention routing policy the patched ComfyUI
# path uses:
#   auto   (default outside Windows) — use platform/workflow-tuned routes where
#                      validated, then
#                      cute for d128 self-attention and exact safe cross routes,
#                      and PyTorch for every remaining attention contract.
#                      Auto never selects the ESIMD attention backend.
#   cute             — CUTLASS-SYCL FMHA (omni_xpu_kernel.cute). fp32 accumulation,
#                      so it does NOT overflow on large activations (Qwen-Image etc.)
#                      where the ESIMD fp16-accumulator kernel can. Unsupported
#                      shapes fall back to PyTorch rather than switching backend.
#   esimd            — omni_xpu_kernel.sdp (hand-written ESIMD flash attention;
#                      ~6% faster on large self-attn but fp16 accumulator).
#   torch            — no cute/esimd; always fall back to PyTorch SDPA.
# The cute backend prefers the packaged omni_xpu_kernel.cute module and falls
# back to a raw native extension (OMNI_CUTE_FMHA_SO overrides the path).
#
# Windows defaults to the upstream PyTorch SDPA path. ESIMD remains available
# only through an explicit OMNI_ATTN_BACKEND=esimd opt-in.
_default_backend = "torch" if sys.platform == "win32" else "auto"
_backend = os.environ.get("OMNI_ATTN_BACKEND", _default_backend).lower()
_backend_name = _backend  # for logging
_backend_sdp = None  # callable(q_blhd, k_blhd, v_blhd) -> out_blhd
_torch_sdpa_count = 0


def _record_attention_route(route):
    count = _attention_route_counts.get(route, 0) + 1
    _attention_route_counts[route] = count
    return count


def _validate_attention_output():
    value = os.environ.get(_VALIDATE_OUTPUT_ENV, "0").strip().lower()
    return value not in ("", "0", "false", "no", "off")


def _attention_tensor_contract(tensor):
    return (
        tuple(tensor.shape),
        tuple(tensor.stride()),
        str(tensor.dtype),
        str(tensor.device),
    )


def _attention_layout(tensor):
    shape = tuple(tensor.shape)
    stride = tuple(tensor.stride())
    if len(shape) != 4 or len(stride) != 4 or stride[3] != 1:
        return "other"
    _, heads, seq, dim_head = shape
    if stride[1] == seq * dim_head and stride[2] == dim_head:
        return "packed_bhld"
    if stride[1] == dim_head and stride[2] == heads * dim_head:
        return "blhd_backed"
    return "strided_bhld"


def _attention_caller():
    import inspect

    frame = inspect.currentframe()
    try:
        frame = frame.f_back if frame is not None else None
        while frame is not None:
            module = frame.f_globals.get("__name__", "")
            if (
                module != __name__
                and module != "comfy.ldm.modules.attention"
                and (
                    module.startswith("comfy.")
                    or module.startswith("custom_nodes.")
                )
            ):
                return f"{module}:{frame.f_code.co_name}:{frame.f_lineno}"
            frame = frame.f_back
    finally:
        del frame
    return "unknown"


def _trace_attention_contract(
    q,
    k,
    v,
    heads,
    mask,
    skip_reshape,
    skip_output_reshape,
):
    if os.environ.get("OMNI_ATTN_TRACE_CONTRACTS", "0") not in {
        "1",
        "true",
        "yes",
    }:
        return
    contract = (
        _attention_tensor_contract(q),
        _attention_tensor_contract(k),
        _attention_tensor_contract(v),
        int(heads),
        mask is not None,
        bool(skip_reshape),
        bool(skip_output_reshape),
    )
    if contract in _attention_traced_contracts:
        return
    _attention_traced_contracts.add(contract)
    log.info(
        "[OmniXPU] attention contract: caller=%s heads=%d "
        "q=%s/%s/%s k=%s/%s/%s v=%s/%s/%s "
        "dtype=%s device=%s mask=%s skip_reshape=%s "
        "skip_output_reshape=%s",
        _attention_caller(),
        heads,
        tuple(q.shape),
        tuple(q.stride()),
        _attention_layout(q) if skip_reshape else "bld",
        tuple(k.shape),
        tuple(k.stride()),
        _attention_layout(k) if skip_reshape else "bld",
        tuple(v.shape),
        tuple(v.stride()),
        _attention_layout(v) if skip_reshape else "bld",
        q.dtype,
        q.device,
        mask is not None,
        skip_reshape,
        skip_output_reshape,
    )


def _torch_major_minor():
    try:
        components = torch.__version__.split("+", 1)[0].split(".")
        return int(components[0]), int(components[1])
    except (AttributeError, IndexError, ValueError):
        return None


# Keep this closed: a kernel capability bit does not validate a future Torch
# minor or a different device target for these workflow-specific routes.
_VALIDATED_ROUTED_TORCH_BY_TARGET = {
    "ptl-h": frozenset({(2, 11), (2, 12)}),
    "bmg": frozenset({(2, 11), (2, 12), (2, 13)}),
}


def _torch_supports_versioned_routes():
    version = _torch_major_minor()
    target = _omni_xpu_target()
    return version in _VALIDATED_ROUTED_TORCH_BY_TARGET.get(target, ())


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
    """Select the guarded PTL-H workflow shapes on validated Torch minors."""
    is_zimage = heads == 30 and q_len in (64, 1024, 1088)
    is_krea2 = heads == 48 and q_len == 4192
    return (
        _backend == "auto"
        and _backend_name == "cute"
        and _omni_xpu_target() == "ptl-h"
        and _torch_supports_versioned_routes()
        and q.dtype == torch.bfloat16
        and dim_head == 128
        and q_len == kv_len
        and (is_zimage or is_krea2)
        and skip_reshape
        and not skip_output_reshape
    )


def _is_dense_bhld(tensor, batch, heads, seq, dim_head):
    try:
        shape = tuple(tensor.shape)
        strides = tensor.stride()
    except (AttributeError, TypeError):
        return False
    if shape != (batch, heads, seq, dim_head):
        return False
    if len(strides) != 4 or strides[3] != 1:
        return False
    if strides[0] != heads * seq * dim_head:
        return False
    packed_bhld = strides[1] == seq * dim_head and strides[2] == dim_head
    blhd_backed = strides[1] == dim_head and strides[2] == heads * dim_head
    return packed_bhld or blhd_backed


def _is_minimax_h3_h56_bhld(tensor, seq):
    try:
        shape = tuple(tensor.shape)
        strides = tuple(tensor.stride())
    except (AttributeError, TypeError):
        return False
    head_width = 56 * 128
    return (
        shape == (1, 56, seq, 128)
        and len(strides) == 4
        and strides[0] > 0
        and strides[1] == 128
        and strides[2] in (head_width, 3 * head_width)
        and strides[3] == 1
    )


def _is_minimax_h3_vae_d64_bhld(tensor, seq, *, value):
    try:
        shape = tuple(tensor.shape)
        strides = tuple(tensor.stride())
    except (AttributeError, TypeError):
        return False
    sequence_stride = 32 * (3 * 64 if value else 64)
    head_stride = 3 * 64 if value else 64
    return (
        shape == (1, 32, seq, 64)
        and strides
        == (seq * sequence_stride, head_stride, sequence_stride, 1)
    )


def _use_bmg_minimax_h3_vae_d64(
    q,
    k,
    v,
    b,
    heads,
    dim_head,
    q_len,
    kv_len,
    skip_reshape,
    skip_output_reshape,
):
    capability = getattr(
        _backend_sdp, "supports_minimax_h3_vae_d64", None
    )
    return (
        _backend_name == "cute"
        and _omni_xpu_target() == "bmg"
        and _torch_supports_versioned_routes()
        and callable(capability)
        and capability()
        and q.dtype == torch.float16
        and k.dtype == q.dtype
        and v.dtype == q.dtype
        and q.device.type == "xpu"
        and k.device == q.device
        and v.device == q.device
        and b == 1
        and heads == 32
        and dim_head == 64
        and q_len == kv_len
        and q_len >= _MINIMAX_H3_VAE_D64_CUTE_MIN_SEQUENCE
        and skip_reshape
        and not skip_output_reshape
        and _is_minimax_h3_vae_d64_bhld(q, q_len, value=False)
        and _is_minimax_h3_vae_d64_bhld(k, q_len, value=False)
        and _is_minimax_h3_vae_d64_bhld(v, q_len, value=True)
    )


def _use_workflow_cute_d120(
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
        and _omni_xpu_target() in ("ptl-h", "bmg")
        and _torch_supports_versioned_routes()
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
        and _is_dense_bhld(q, 1, heads, q_len, dim_head)
        and _is_dense_bhld(k, 1, heads, kv_len, dim_head)
        and _is_dense_bhld(v, 1, heads, kv_len, dim_head)
    )


def _is_dense_bld(tensor, batch, seq, width):
    try:
        shape = tuple(tensor.shape)
        strides = tensor.stride()
    except (AttributeError, TypeError):
        return False
    return (
        shape == (batch, seq, width)
        and len(strides) == 3
        and strides == (seq * width, width, 1)
    )


_ANIMATE2_CROSS_ENV = "OMNIXPU_ANIMATE2_CROSS"
_ANIMATE2_SHAPES_ENV = "OMNIXPU_ANIMATE2_SHAPES"
_ANIMATE2_HEADS = 40


def _animate2_cross_enabled():
    value = os.environ.get(_ANIMATE2_CROSS_ENV, "1").strip().lower()
    return value not in ("", "0", "false", "no", "off")


def _animate2_shape_allowed(q_len, kv_len):
    spec = os.environ.get(_ANIMATE2_SHAPES_ENV, "all").strip().lower()
    if spec in ("", "all", "*"):
        return True
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        q_spec, sep, kv_spec = item.partition(":")
        if not sep:
            continue
        if q_spec.strip() in ("*", str(q_len)) and kv_spec.strip() in (
            "*",
            str(kv_len),
        ):
            return True
    return False


def _is_animate2_cute_shape(b, heads, dim_head, q_len, kv_len, dtype):
    """Admission for Wan Animate 2 (14B, heads=40) cross-attention.

    The BMG D128 BHLD CUTE kernel was measured on an idle GPU to accept every
    Animate 2 cross shape (q44550/kv512, q44550/kv257, q42525/kv512,
    q42525/kv257, q2025/kv46575, q2025/kv44550) at heads=40 in both FP16 and
    BF16, with the same fp32-relative error as Torch SDPA and 1.35-1.54x the
    throughput. A sweep over realistic geometries (512x512 through 1920x1088,
    21-121 frames) and over shape edges (prime, odd, q>>kv, kv>>q, up to
    q=262144 / kv=269280) found no shape sensitivity. The bf16-only /
    heads==32 / kv_len==1024 restrictions above are therefore an admission-list
    limitation, not a kernel capability limit. The route is enabled by default;
    set OMNIXPU_ANIMATE2_CROSS=0 to restore the previous routing for diagnosis.

    Wan 2.2 14B T2V is also heads=40 / d128 / fp16, and its q75600/kv512 cross
    contract has a dedicated validated route (_use_bmg_wan22_cute_cross) that
    is checked *after* this one in the dispatcher. That exact shape is excluded
    here so the default route never diverts another model off its validated
    kernel. Animate 2 can reach the same shape (hw=3600 at 1280x720 with 77
    frames), in which case it simply takes the Wan 2.2 kernel instead -- same
    attention math, still a CUTE route, just marginally slower. The exclusion
    is FP16-only because sdp_wan22_cross rejects BF16 outright ("Wan 2.2 cross
    attention requires FP16 Q/K/V"), so excluding BF16 there would hand the
    shape to Torch rather than to a faster kernel.
    """
    if dtype == torch.float16 and q_len == 75600 and kv_len == 512:
        return False
    return (
        _animate2_cross_enabled()
        and b == 1
        and heads == _ANIMATE2_HEADS
        and dim_head == 128
        and dtype in (torch.float16, torch.bfloat16)
        and q_len != kv_len
        and q_len >= 256
        and kv_len >= 128
        and _animate2_shape_allowed(q_len, kv_len)
    )


def _prepare_bmg_d128_bhld_cute(
    q,
    k,
    v,
    b,
    heads,
    dim_head,
    q_len,
    kv_len,
    skip_reshape,
    skip_output_reshape,
):
    capability = getattr(_backend_sdp, "supports_d128_bhld", None)
    self_attention = q_len == kv_len and q_len >= 768
    cross_attention = kv_len == 1024 and q_len >= 1024
    minimax_h3_attention = (
        b == 1
        and heads == 56
        and dim_head == 128
        and q_len == kv_len
        and q_len >= _MINIMAX_H3_H56_CUTE_MIN_SEQUENCE
        and skip_reshape
        and not skip_output_reshape
    )
    animate2_attention = _is_animate2_cute_shape(
        b, heads, dim_head, q_len, kv_len, q.dtype
    )
    supported_batch_kind = (
        (b == 2 and (self_attention or cross_attention))
        or (b == 1 and cross_attention)
        or minimax_h3_attention
        or animate2_attention
    )
    if not (
        _backend_name == "cute"
        and _omni_xpu_target() == "bmg"
        and _torch_supports_versioned_routes()
        and callable(capability)
        and capability()
        and (
            q.dtype == torch.bfloat16
            or (animate2_attention and q.dtype == torch.float16)
        )
        and k.dtype == q.dtype
        and v.dtype == q.dtype
        and q.device.type == "xpu"
        and k.device == q.device
        and v.device == q.device
        and (heads == 32 or minimax_h3_attention or animate2_attention)
        and dim_head == 128
        and supported_batch_kind
    ):
        return None

    if skip_reshape:
        q_bhld, k_bhld, v_bhld = q, k, v
    else:
        width = heads * dim_head
        if not (
            _is_dense_bld(q, b, q_len, width)
            and _is_dense_bld(k, b, kv_len, width)
            and _is_dense_bld(v, b, kv_len, width)
        ):
            return None
        q_bhld = q.view(b, q_len, heads, dim_head).transpose(1, 2)
        k_bhld = k.view(b, kv_len, heads, dim_head).transpose(1, 2)
        v_bhld = v.view(b, kv_len, heads, dim_head).transpose(1, 2)

    dense_layouts = (
        _is_dense_bhld(q_bhld, b, heads, q_len, dim_head)
        and _is_dense_bhld(k_bhld, b, heads, kv_len, dim_head)
        and _is_dense_bhld(v_bhld, b, heads, kv_len, dim_head)
    )
    minimax_h3_layouts = minimax_h3_attention and all(
        _is_minimax_h3_h56_bhld(tensor, q_len)
        for tensor in (q_bhld, k_bhld, v_bhld)
    )
    if not (dense_layouts or minimax_h3_layouts):
        return None

    contract = (
        str(q.dtype),
        b,
        heads,
        q_len,
        kv_len,
        tuple(q_bhld.stride()),
        tuple(k_bhld.stride()),
        tuple(v_bhld.stride()),
        skip_output_reshape,
    )
    return contract, q_bhld, k_bhld, v_bhld


def _use_bmg_wan22_cute_cross(
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
    capability = getattr(_backend_sdp, "supports_wan22_cross", None)
    return (
        _backend_name == "cute"
        and _omni_xpu_target() == "bmg"
        and _torch_supports_versioned_routes()
        and callable(capability)
        and capability()
        and q.dtype == torch.float16
        and k.dtype == q.dtype
        and v.dtype == q.dtype
        and q.device.type == "xpu"
        and k.device.type == "xpu"
        and v.device.type == "xpu"
        and heads == 40
        and dim_head == 128
        and q_len == 75600
        and kv_len == 512
        and not skip_reshape
        and not skip_output_reshape
    )


def _default_cute_extension():
    # Ship next to the omni_xpu_kernel package by default.
    try:
        import omni_xpu_kernel as pkg

        d = os.path.dirname(os.path.abspath(pkg.__file__))
        root = os.path.join(d, "cute", "cute_fmha_torch")
        for suffix in (*EXTENSION_SUFFIXES, ".pyd", ".so"):
            exact = root + suffix
            if os.path.isfile(exact):
                return exact
            matches = sorted(glob.glob(root + "*" + suffix))
            if matches:
                return matches[0]
        return root + (".pyd" if sys.platform == "win32" else ".so")
    except Exception:
        return ""


def _load_cute_backend():
    # Preferred: the packaged submodule handles native extension discovery.
    try:
        from omni_xpu_kernel import cute as _cute

        if _cute is not None and _cute.is_available():
            return _cute, None
    except Exception:
        pass
    # Fallback: load a raw extension directly (development/path override).
    extension = (
        os.environ.get("OMNI_CUTE_FMHA_SO", "")
        or _default_cute_extension()
    )
    if not extension or not os.path.exists(extension):
        return None, (
            "cute backend unavailable "
            f"(native extension not found: {extension})"
        )
    try:
        torch.ops.load_library(extension)
        fn = torch.ops.cute_fmha.sdp

        class _Wrap:
            @staticmethod
            def sdp(q, k, v):
                return fn(q, k, v)

            @staticmethod
            def supports_wan22_cross():
                return hasattr(
                    torch.ops.cute_fmha,
                    "sdp_wan22_cross",
                )

            @staticmethod
            def sdp_wan22_cross(q, k, v):
                return torch.ops.cute_fmha.sdp_wan22_cross(q, k, v)

            @staticmethod
            def supports_d128_bhld():
                return hasattr(
                    torch.ops.cute_fmha,
                    "sdp_bhld_d128",
                )

            @staticmethod
            def sdp_bhld_d128(q, k, v):
                return torch.ops.cute_fmha.sdp_bhld_d128(q, k, v)

            @staticmethod
            def supports_minimax_h3_vae_d64():
                return hasattr(
                    torch.ops.cute_fmha,
                    "sdp_minimax_h3_vae_d64",
                )

            @staticmethod
            def sdp_minimax_h3_vae_d64(q, k, v):
                return torch.ops.cute_fmha.sdp_minimax_h3_vae_d64(q, k, v)

        return _Wrap, None
    except Exception as e:
        return None, f"cute load failed: {e}"


def get_stats():
    return {
        "policy": _backend,
        "backend": _backend_name,
        "cute": _cute_call_count,
        "esimd": _esimd_call_count,
        "torch_sdpa": _torch_sdpa_count,
        "fallback": _attention_fallback_count,
        "reasons": dict(_attention_fallback_reasons),
        "routes": dict(_attention_route_counts),
        "quarantined_contracts": len(_attention_failed_contracts),
    }


def apply():
    global _backend_sdp, _backend_name
    import sys

    probe = sys.modules.get("ComfyUI-OmniXPU.probe")
    target = _omni_xpu_target()

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
        elif _backend == "auto":
            return (
                False,
                "cute backend unavailable "
                f"({err}); auto keeps PyTorch SDPA",
            )
        else:
            return False, err
    else:  # esimd
        if probe is None or probe.sdp is None:
            return False, "omni_xpu_kernel sdp not available"
        _backend_sdp = probe.sdp
        _backend_name = "esimd"

    import comfy.ldm.modules.attention as attn_mod

    if not hasattr(attn_mod, "attention_pytorch"):
        return False, "attention_pytorch not found"

    _pytorch_fallback = attn_mod.attention_pytorch
    wrap_attn = attn_mod.wrap_attn

    @wrap_attn
    def attention_omni(
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
        global _cute_call_count, _esimd_call_count
        global _attention_fallback_count, _torch_sdpa_count
        global _attention_experimental_warning_emitted

        log_debug_event(
            "dispatch",
            "attention",
            {"q": q, "k": k, "v": v, "mask": mask},
            details={"policy": _backend},
            verbose_only=True,
        )
        _trace_attention_contract(
            q,
            k,
            v,
            heads,
            mask,
            skip_reshape,
            skip_output_reshape,
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

        use_workflow_cute_d120 = _use_workflow_cute_d120(
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
        # Keep target-specific capability probing out of other platforms'
        # hot attention dispatchers.
        bmg_d128_prepared = (
            _prepare_bmg_d128_bhld_cute(
                q,
                k,
                v,
                b,
                heads,
                dim_head,
                q_len,
                kv_len,
                skip_reshape,
                skip_output_reshape,
            )
            if target == "bmg"
            else None
        )
        bmg_d128_contract = (
            bmg_d128_prepared[0]
            if bmg_d128_prepared is not None
            else None
        )
        use_bmg_d128_bhld_cute = (
            bmg_d128_contract is not None
            and bmg_d128_contract not in _attention_failed_contracts
        )
        use_bmg_wan22_cute_cross = (
            _use_bmg_wan22_cute_cross(
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
            if target == "bmg"
            else False
        )
        use_bmg_minimax_h3_vae_d64 = (
            _use_bmg_minimax_h3_vae_d64(
                q,
                k,
                v,
                b,
                heads,
                dim_head,
                q_len,
                kv_len,
                skip_reshape,
                skip_output_reshape,
            )
            if target == "bmg"
            else False
        )
        bmg_minimax_h3_vae_d64_contract = (
            (
                "minimax_h3_vae_d64",
                str(q.dtype),
                b,
                heads,
                q_len,
                tuple(q.stride()),
                tuple(k.stride()),
                tuple(v.stride()),
            )
            if use_bmg_minimax_h3_vae_d64
            else None
        )
        use_bmg_minimax_h3_vae_d64 = (
            use_bmg_minimax_h3_vae_d64
            and bmg_minimax_h3_vae_d64_contract not in _attention_failed_contracts
        )

        # Constraint check
        reasons = []
        if b != 1 and bmg_d128_prepared is None:
            reasons.append(f"batch={b}")
        if (
            bmg_d128_contract is not None
            and bmg_d128_contract in _attention_failed_contracts
        ):
            reasons.append("cute_runtime_quarantined")
        if mask is not None:
            reasons.append(f"mask={mask.shape}")
        if dim_head not in (64, 128) and not use_workflow_cute_d120:
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
        # Torch SDPA is faster end-to-end for the measured PTL-H D128 workflow
        # shapes. Keep this route narrower than the generic d128 CUTE domain:
        # explicit `cute`, versions outside the validated target matrix, other
        # platforms, dtypes, head counts, and sequence lengths retain the
        # existing policy.
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
                details={"backend": "torch", "route": "ptl_torch_workflow"},
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

        # Boogu's PTL-H/BMG D120 route consumes the exact BHLD input strides and
        # returns a BLHD-backed BHLD view.  The final transpose+reshape is a
        # metadata-only view, avoiding all layout copies.  This remains an
        # auto-only route for validated target/Torch pairs and workflow shapes;
        # unsupported wheels and layouts retain the unmodified Torch fallback.
        if not reasons and use_workflow_cute_d120:
            _cute_call_count += 1
            route_call_count = _record_attention_route(
                "boogu_cute_d120_bhld"
            )
            if route_call_count <= 3:
                log.info(
                    "[OmniXPU] attention CUTE_D120 #%d: heads=%d seq=%d dtype=%s",
                    route_call_count,
                    heads,
                    q_len,
                    q.dtype,
                )
            log_debug_event(
                "kernel",
                "attention",
                {"q": q, "k": k, "v": v},
                details={"backend": "cute", "route": "boogu_cute_d120_bhld"},
            )
            out = _backend_sdp.sdp_bhld_d120(q, k, v)
            return out.transpose(1, 2).reshape(b, -1, heads * dim_head)

        # LTX reaches optimized_attention with dense BLD tensors. MiniMax H3
        # reaches it with B1/H56 BHLD views backed by one QKV buffer. Prepare
        # the former as metadata-only views and pass the latter directly so
        # the public D128 op avoids three 216 MiB layout copies per H3 block.
        if not reasons and use_bmg_d128_bhld_cute:
            _cute_call_count += 1
            if heads == 56:
                route = "minimax_h3_h56_bf16_d128_qkv_bhld"
            elif q_len == kv_len:
                route = "bmg_b2_bf16_d128_self"
            elif heads == _ANIMATE2_HEADS:
                tag = "fp16" if q.dtype == torch.float16 else "bf16"
                route = (
                    f"animate2_b{b}_{tag}_d128_q{q_len}_kv{kv_len}_cross"
                )
            else:
                route = f"bmg_b{b}_bf16_d128_kv1024_cross"
            route_call_count = _record_attention_route(route)
            if not _attention_experimental_warning_emitted:
                log.warning(
                    "[OmniXPU] experimental capability-driven BMG D128 "
                    "CUTE route enabled; set OMNI_ATTN_BACKEND=torch "
                    "to disable it if this workflow shows an issue"
                )
                _attention_experimental_warning_emitted = True
            if route_call_count <= 3:
                log.info(
                    "[OmniXPU] attention CUTE_BHLD_D128 #%d: "
                    "heads=%d q=%d kv=%d dtype=%s",
                    route_call_count,
                    heads,
                    q_len,
                    kv_len,
                    q.dtype,
                )
            log_debug_event(
                "kernel",
                "attention",
                {
                    "q": bmg_d128_prepared[1],
                    "k": bmg_d128_prepared[2],
                    "v": bmg_d128_prepared[3],
                },
                details={
                    "backend": "cute",
                    "route": route,
                },
            )
            _, q_bhld, k_bhld, v_bhld = bmg_d128_prepared
            try:
                out = _backend_sdp.sdp_bhld_d128(
                    q_bhld, k_bhld, v_bhld
                )
                if skip_output_reshape:
                    return out
                return out.transpose(1, 2).reshape(
                    b, -1, heads * dim_head
                )
            except Exception as error:
                if is_fatal_accelerator_error(error):
                    raise
                _attention_failed_contracts.add(bmg_d128_contract)
                _attention_fallback_count += 1
                key = "cute_runtime_error"
                _attention_fallback_reasons[key] = (
                    _attention_fallback_reasons.get(key, 0) + 1
                )
                log.warning(
                    "[OmniXPU] BMG D128 CUTE failed for "
                    "B=%d H=%d Q=%d KV=%d; this contract is disabled "
                    "for the process and the call is falling back to "
                    "PyTorch. Set OMNI_ATTN_BACKEND=torch to disable "
                    "the experimental route globally. Error: %s",
                    b,
                    heads,
                    q_len,
                    kv_len,
                    error,
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

        # The official Wan 2.2 14B T2V Turbo 720p workflow has an exact FP16
        # rectangular contract. Use the dedicated BMG CUTE entry point so the
        # cross-attention accumulates in FP32 instead of the ESIMD FP16 path.
        if not reasons and use_bmg_wan22_cute_cross:
            _cute_call_count += 1
            route_call_count = _record_attention_route(
                "wan22_t2v_turbo_720p_cross"
            )
            if route_call_count <= 3:
                log.info(
                    "[OmniXPU] attention CUTE_WAN22 #%d: "
                    "heads=%d q=%d kv=%d dtype=%s",
                    route_call_count,
                    heads,
                    q_len,
                    kv_len,
                    q.dtype,
                )
            q_blhd = q.view(b, q_len, heads, dim_head).contiguous()
            k_blhd = k.view(b, kv_len, heads, dim_head).contiguous()
            v_blhd = v.view(b, kv_len, heads, dim_head).contiguous()
            log_debug_event(
                "kernel",
                "attention",
                {"q": q_blhd, "k": k_blhd, "v": v_blhd},
                details={
                    "backend": "cute",
                    "route": "wan22_t2v_turbo_720p_cross",
                },
            )
            out = _backend_sdp.sdp_wan22_cross(
                q_blhd, k_blhd, v_blhd
            )
            if _validate_attention_output() and (out != out).any():
                _attention_fallback_count += 1
                _attention_fallback_reasons["output_non_finite"] = (
                    _attention_fallback_reasons.get(
                        "output_non_finite", 0
                    )
                    + 1
                )
                del out, q_blhd, k_blhd, v_blhd
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
            return out.reshape(b, q_len, heads * dim_head)

        # MiniMax H3's tiled VideoVAE preserves one structural FP16 D64
        # contract while the tile token count changes with temporal/spatial
        # extent. Derive all strides from the runtime sequence length and keep
        # unrelated D64 layouts on Torch SDPA.
        if not reasons and use_bmg_minimax_h3_vae_d64:
            _cute_call_count += 1
            route = "minimax_h3_video_vae_fp16_d64"
            route_call_count = _record_attention_route(route)
            if route_call_count <= 3:
                log.info(
                    "[OmniXPU] attention CUTE_H3_VAE_D64 #%d: "
                    "heads=%d seq=%d dtype=%s",
                    route_call_count,
                    heads,
                    q_len,
                    q.dtype,
                )
            log_debug_event(
                "kernel",
                "attention",
                {"q": q, "k": k, "v": v},
                details={"backend": "cute", "route": route},
            )
            try:
                out = _backend_sdp.sdp_minimax_h3_vae_d64(q, k, v)
                return out.transpose(1, 2).reshape(
                    b, q_len, heads * dim_head
                )
            except Exception as error:
                if is_fatal_accelerator_error(error):
                    raise
                _attention_failed_contracts.add(
                    bmg_minimax_h3_vae_d64_contract
                )
                _attention_fallback_count += 1
                key = "cute_h3_vae_d64_runtime_error"
                _attention_fallback_reasons[key] = (
                    _attention_fallback_reasons.get(key, 0) + 1
                )
                log.warning(
                    "[OmniXPU] MiniMax H3 VideoVAE D64 CUTE failed for "
                    "S=%d; this structural contract is disabled for the "
                    "process and the call is falling back to PyTorch. "
                    "Error: %s",
                    q_len,
                    error,
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

        # CUTE is accepted only for its validated contracts. Both auto and
        # explicit cute use the safe PyTorch fallback for everything else;
        # auto never silently changes to ESIMD.
        cute_unsupported = _backend_name == "cute" and (
            dim_head != 128
            or (q_len != kv_len and not use_bmg_wan22_cute_cross)
        )
        if not reasons and cute_unsupported:
            reasons.append(
                f"cute_unsupported=dim{dim_head},q{q_len},kv{kv_len}"
            )

        if reasons:
            _attention_fallback_count += 1
            key = ",".join(reasons)
            _attention_fallback_reasons[key] = (
                _attention_fallback_reasons.get(key, 0) + 1
            )
            if _attention_fallback_count <= 5:
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

        if selected_backend == "cute":
            _cute_call_count += 1
            selected_call_count = _cute_call_count
        else:
            _esimd_call_count += 1
            selected_call_count = _esimd_call_count
        if selected_call_count <= 3:
            seq = q.shape[1] if not skip_reshape else q.shape[2]
            log.info(
                "[OmniXPU] attention %s #%d: heads=%d seq=%d dtype=%s",
                selected_backend.upper(),
                selected_call_count,
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

        # ESIMD accumulates in FP16, so keep its overflow safety check. CUTE
        # accumulates in FP32 and its validated routes avoid this per-call full
        # output scan by default; enable the diagnostic switch to restore it.
        validate_output = (
            selected_backend == "esimd" or _validate_attention_output()
        )
        if (
            q.dtype == torch.float16
            and validate_output
            and (out != out).any()
        ):
            _attention_fallback_count += 1
            _attention_fallback_reasons["output_non_finite"] = (
                _attention_fallback_reasons.get("output_non_finite", 0) + 1
            )
            if _attention_fallback_reasons["output_non_finite"] <= 3:
                log.warning(
                    "[OmniXPU] FP16 overflow in %s, falling back to SDPA",
                    selected_backend.upper(),
                )
            del out, q_blhd, k_blhd, v_blhd
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
    attn_mod.optimized_attention = attention_omni
    attn_mod.optimized_attention_masked = attention_omni

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
                    setattr(mod, name, attention_omni)
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
        attn_mod.register_attention_function("omnixpu", attention_omni)

    return True, None
