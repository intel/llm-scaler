"""custom_esimd_kernels v2 - migrated from llm-scaler/vllm.

Provides FP8 GEMM (M>=2) and FP8 MoE (silu, e4m3) optimized for BMG via DPAS.

Modules:
  - custom_esimd_kernels_gemm:       esimd_gemm_fp8_pert + esimd_gemm_int4_pgrp
  - custom_esimd_kernels_moe_batch:  moe_forward_full_silu_routed +
                                     building blocks (moe_topk, moe_up_forward,
                                     moe_down_forward, moe_accumulate, ...)
"""
import torch  # noqa: F401

# Importing the compiled .so as Python extensions runs both the PyMODINIT_FUNC
# stub *and* (via the C++ static initialiser ordering inside the .so) the
# TORCH_LIBRARY/TORCH_LIBRARY_IMPL registration blocks. ctypes.CDLL alone is
# NOT enough on this toolchain because the dynamic loader ignores .ctors
# from a flat dlopen — only Python's import_module path triggers them.
from custom_esimd_kernels import custom_esimd_kernels_gemm  # noqa: F401
from custom_esimd_kernels import custom_esimd_kernels_moe_batch  # noqa: F401
from custom_esimd_kernels import custom_esimd_kernels_attn as _attn_mod  # noqa: F401

from custom_esimd_kernels.ops import (
    esimd_gemm_fp8_pert,
    esimd_gemm_int4_pgrp,
)


def sglang_decode_attn(
    q, k_buffer, v_buffer, kv_indptr, kv_indices, out, sm_scale,
    temp_p=None, max_seq_len=-1,
):
    """ESIMD decode SDPA on sglang flat-NHD KV layout.

    head_dim=256 fixed; q/out fp16; k/v fp16 or bf16; kv_indptr/indices int32.

    temp_p:      optional pre-allocated float32 scratch buffer holding the
                 m/l/o partial-stats from phase 1. Size must be at least
                 ``batches*Hq*n_splits*(1+1+256)``. Required for XPUGraph
                 capture/replay so the kernel sees a stable data_ptr.
    max_seq_len: optional upper bound on KV length so n_splits is computed
                 host-side without copying kv_indptr off device. Pass <=0
                 (default) to keep the legacy device->host scan.
    """
    return _attn_mod.sglang_decode_attn(
        q, k_buffer, v_buffer, kv_indptr, kv_indices, out, float(sm_scale),
        temp_p, int(max_seq_len),
    )


def sglang_decode_attn_temp_size(batches: int, num_q_heads: int, max_seq_len: int) -> int:
    """Number of float32 elements needed for the ``temp_p`` scratch buffer
    of :func:`sglang_decode_attn` at a given max_seq_len.

    Mirrors the C++ sizing (phase-1 partial accumulators):
        m_part: batches * Hq * n_splits
        l_part: batches * Hq * n_splits
        o_part: batches * Hq * n_splits * 256
    where ``n_splits = ceil_div(max_seq_len, SPLIT_TILE=64)``.
    """
    SPLIT_TILE = 64
    n_splits = max((max_seq_len + SPLIT_TILE - 1) // SPLIT_TILE, 1)
    per_partial = batches * num_q_heads * n_splits
    return per_partial * (1 + 1 + 256)


__all__ = [
    "esimd_gemm_fp8_pert",
    "esimd_gemm_int4_pgrp",
    "sglang_decode_attn",
    "sglang_decode_attn_temp_size",
]
