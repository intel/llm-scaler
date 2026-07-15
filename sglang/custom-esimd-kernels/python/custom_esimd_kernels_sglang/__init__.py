"""custom_esimd_kernels_sglang — PTL iGPU (XeLPG) tolerant __init__.

Some compiled extensions require XMX (DPAS) intrinsics which are not
available on XeLPG (Panther Lake iGPU). On such hosts we build only a
subset of the extensions; the remaining ones are absent. Everything Qwen3.5
dense production needs (``esimd_qkv_split_norm_rope`` from
``custom_esimd_kernels`` and ``eagle_page_attn_decode`` from ``eagle_ops``)
lives in DPAS-free extensions.

Each ext import and each op re-export is wrapped in a best-effort block so
a missing extension does not prevent the package from loading. Missing ops
surface as AttributeError only at call time.
"""
import logging

import torch

_log = logging.getLogger(__name__)
_MISSING_EXTS = []


def _try_import_ext(name):
    """Import a compiled extension and also re-dlopen it with RTLD_GLOBAL.

    Without RTLD_GLOBAL the TORCH_LIBRARY static initializers in some of
    these .so files do not register their ops into torch.ops.<namespace>.
    Confirmed via nm: symbols TORCH_LIBRARY_FRAGMENT_static_init_<ns> exist
    but are not invoked under the default Python ext loader's RTLD_LOCAL.
    """
    import ctypes
    import os
    try:
        mod = __import__(f"custom_esimd_kernels_sglang.{name}")
        sub = getattr(mod, name)
        so_path = getattr(sub, "__file__", None)
        if so_path and os.path.exists(so_path):
            # idempotent: CDLL caches by path internally
            ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)
        return True
    except ImportError as e:
        _MISSING_EXTS.append((name, str(e)))
        return False


# Compiled extension modules (each registers ops into torch.ops).
_try_import_ext("custom_esimd_kernels")        # always expected (Qwen3.5 path)
_try_import_ext("custom_esimd_kernels_lgrf")   # always expected (Qwen3.5 path)
_try_import_ext("custom_esimd_kernels_moe")    # XeLPG: skipped (DPAS-dep)
_try_import_ext("custom_esimd_kernels_gemm")   # XeLPG: skipped (DPAS-dep)
_try_import_ext("eagle_ops")                   # Qwen3.5 decode uses this
_try_import_ext("moe_ops")                     # XeLPG: skipped (DPAS-dep)
_try_import_ext("moe_int4_ops")                # XeLPG: skipped (DPAS-dep)
_try_import_ext("moe_fp8_prefill_ops")         # v2-merged: FP8 MoE prefill
_try_import_ext("custom_esimd_kernels_attn")   # v2-merged: decode SDPA (pybind)

if _MISSING_EXTS:
    _log.info(
        "custom_esimd_kernels_sglang: extensions unavailable on this host: %s. "
        "Ops from those extensions will raise AttributeError at call time.",
        ", ".join(n for n, _ in _MISSING_EXTS),
    )

# Python wrappers. ops.py does not eagerly call any kernel; each function
# only touches torch.ops.custom_esimd_kernels_sglang.<name> when invoked.
# Re-export them individually so a missing op (e.g. an MoE helper that the
# skipped extension would have registered) doesn't block loading the rest.
from custom_esimd_kernels_sglang import ops as _ops_mod  # noqa: E402

_EXPORTS = [
    # Core ESIMD ops (custom_esimd_kernels)
    "esimd_gemv_fp8_pern",
    "esimd_gemv_fp8_pern_fused2",
    "esimd_gemv_fp8_pern_fused3",
    "esimd_gemv_fp8_pert",
    "esimd_gemv_fp8_pert_fused2",
    "esimd_gemv_fp8_pert_fused3",
    # INT4 GEMV ops
    "esimd_gemv_int4",
    "esimd_gemv_int4_fused2",
    "esimd_gemv_q4_0",
    "esimd_gemv_q8_0",
    "esimd_gemv_q8_0_m",
    "esimd_gemv_q4_k",
    "esimd_gemv_q5_k",
    "esimd_gemv_q6_k",
    "esimd_gemv_q6_k_m",
    "esimd_moe_up_q4k",
    "esimd_moe_down_q5k",
    "esimd_moe_down_q6k",
    "esimd_gemm_q4_0",
    "esimd_qkv_split_norm_rope",
    "esimd_gdn_conv_fused",
    "esimd_fused_add_rms_norm",
    "esimd_rms_norm_gated",
    "esimd_fused_add_rms_norm_batched",
    "esimd_resadd_norm_gemv_fp8_pert",
    "esimd_resadd_norm_gemv_int4_pert",
    "esimd_resadd_norm_gemv2_fp8_pert",
    "esimd_norm_gemv_fp8_pert",
    "esimd_norm_gemv_int4_pert",
    "esimd_gdn_conv_fused_seq",
    "esimd_moe_topk",
    "esimd_moe_scatter_fused",
    "esimd_moe_silu_mul",
    "esimd_moe_gather",
    "esimd_moe_gemm_fp8",
    "esimd_moe_gemm_fp8_pert",
    "esimd_gemm_fp8_pert",
    "esimd_gemm_int4_pgrp",
    # Gemma4 dense decode ops
    "esimd_gemv_fp16",
    "esimd_norm_add_norm",
    "esimd_kv_scatter",
    "esimd_rmsnorm_residual_scalar",
    "splitk_decode_attention",
    # v2-merged: FP8 MoE prefill + decode SDPA
    "moe_prefill_full_fp8",
    "sglang_decode_attn",
    "sglang_decode_attn_temp_size",
    # Eagle ops
    "eagle_gdn",
    "eagle_page_attn_decode",
    # MoE Batch ops
    "moe_router_forward",
    "moe_batch_topk",
    "moe_up_forward",
    "moe_down_forward",
    "moe_accumulate",
    "moe_forward_fused",
    "moe_forward_full",
    # MoE INT4 Batch ops
    "moe_router_forward_int4",
    "moe_router_topk_int4",
    "moe_forward_full_int4",
    "moe_topk_int4",
    "to_cutlass_nmajor_int4",
    "cutlass_nmajor_int4_to_signed",
    "prepare_cutlass_nmajor_int4_weight",
    "precompute_moe_route",
    "moe_silu_mul_int4",
    "moe_route_gather_int4",
    "moe_forward_routed_cutlass_nmajor_int4",
    "moe_forward_full_cutlass_nmajor_int4",
    "moe_forward_full_cutlass_nmajor_int4_with_router",
    "moe_forward_tiny_cutlass_nmajor_int4",
    "moe_forward_tiny_cutlass_nmajor_int4_full_fp16_shared",
    "moe_forward_tiny_cutlass_nmajor_int4_full_fp16_shared_from_logits",
    "moe_tiny_cutlass_nmajor_int4_up",
    "moe_tiny_cutlass_nmajor_int4_down",
    "moe_tiny_fp16_shared_up",
    "moe_tiny_fp16_shared_finalize",
]

_MISSING_OPS = []
for _name in _EXPORTS:
    try:
        globals()[_name] = getattr(_ops_mod, _name)
    except AttributeError:
        _MISSING_OPS.append(_name)

if _MISSING_OPS:
    _log.info(
        "custom_esimd_kernels_sglang: %d python ops not available on this host "
        "(e.g. %s%s)",
        len(_MISSING_OPS),
        ", ".join(_MISSING_OPS[:3]),
        " ..." if len(_MISSING_OPS) > 3 else "",
    )
