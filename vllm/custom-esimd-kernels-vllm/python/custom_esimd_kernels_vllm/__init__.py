import torch

# Core ESIMD kernels (4 compiled modules)
from custom_esimd_kernels_vllm import custom_esimd_kernels
from custom_esimd_kernels_vllm import custom_esimd_kernels_lgrf
from custom_esimd_kernels_vllm import custom_esimd_kernels_moe
from custom_esimd_kernels_vllm import custom_esimd_kernels_gemm

# Eagle kernels — registers torch.ops.eagle_ops.*
from custom_esimd_kernels_vllm import eagle_ops

# MoE Batch kernels — registers torch.ops.moe_ops.*
from custom_esimd_kernels_vllm import moe_ops

# MoE INT4 Batch kernels — registers torch.ops.moe_int4_ops.*
from custom_esimd_kernels_vllm import moe_int4_ops

from custom_esimd_kernels_vllm.ops import (
    # Core ESIMD ops
    esimd_gemv_fp8_pern,
    esimd_gemv_fp8_pern_fused2,
    esimd_gemv_fp8_pern_fused3,
    esimd_gemv_fp8_pert,
    esimd_gemv_fp8_pert_fused2,
    esimd_gemv_fp8_pert_fused3,
    # INT4 GEMV ops
    esimd_gemv_int4,
    esimd_gemv_int4_fused2,
    esimd_qkv_split_norm_rope,
    esimd_gdn_conv_fused,
    esimd_fused_add_rms_norm,
    esimd_rms_norm_gated,
    esimd_fused_add_rms_norm_batched,
    esimd_resadd_norm_gemv_fp8_pert,
    esimd_resadd_norm_gemv2_fp8_pert,
    esimd_norm_gemv_fp8_pert,
    esimd_gdn_conv_fused_seq,
    esimd_moe_topk,
    esimd_moe_scatter_fused,
    esimd_moe_silu_mul,
    esimd_moe_gather,
    esimd_moe_gemm_fp8,
    esimd_moe_gemm_fp8_pert,
    esimd_gemm_fp8_pert,
    # Eagle ops
    eagle_gdn,
    eagle_page_attn_decode,
    # MoE Batch ops
    moe_router_forward,
    moe_batch_topk,
    moe_up_forward,
    moe_down_forward,
    moe_accumulate,
    moe_forward_fused,
    moe_forward_full,
    # MoE INT4 Batch ops
    moe_router_forward_int4,
    moe_forward_full_int4,
)
