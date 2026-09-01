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

# Qwen3.8 QSA — direct pybind API: qsa_ops.sparse_paged_attention(...)
from custom_esimd_kernels_vllm import qsa_ops

from custom_esimd_kernels_vllm.ops import (
    # Core ESIMD ops
    QWEN38_NGRAM_OFFSETS,
    QWEN38_NGRAM_VOCAB_SIZES,
    esimd_qwen38_ngram_ids_decode,
    esimd_qwen38_ngram_ids_decode_out,
    esimd_qwen38_ngram_embedding_gather,
    esimd_qwen38_ngram_embedding_gather_out,
    ple_ngram_ids,
    ple_embedding_gather,
    ple_grouped_norm,
    ple_score_gate,
    ple_gated_value,
    ple_gated_value_grouped_norm,
    ple_gated_value_norm,
    ple_embedding_assemble,
    ple_projection_int4,
    ple_projection_fp16,
    ple_staged,
    ple_staged_full,
    ple_short_conv_mixed,
    ple_short_conv_mixed_three_way,
    ple_residual_add,
    ple_short_conv_decode,
    ple_short_conv_prefill,
    ple_short_conv_spec,
    esimd_gemv_fp8_pern,
    esimd_gemv_fp8_pern_fused2,
    esimd_gemv_fp8_pern_fused3,
    esimd_gemv_fp8_pert,
    esimd_gemv_fp16,
    esimd_gemv_fp16_gelu_mul,
    esimd_gemv_fp8_pert_fused2,
    esimd_gemv_fp8_blockscale_fused2,
    esimd_gemv_fp8_blockscale_fp16_fused2,
    esimd_gemv_fp8_pert_fused3,
    # INT4 GEMV ops
    esimd_gemv_int4,
    esimd_gemv_int4_fused2,
    esimd_gemm_int4_pgrp,
    esimd_qkv_split_norm_rope,
    esimd_qkv_split_norm_rope_v,
    esimd_qkv_split_norm_rope_muse_glimmer,
    esimd_qkv_split_norm_rope_muse_glimmer_neox,
    esimd_gdn_conv_fused,
    esimd_fused_add_rms_norm,
    esimd_norm_gemv_norm_fp16,
    esimd_scaled_resadd_norm_gemv_fp8_pert,
    esimd_norm_add_norm,
    esimd_accum_norm_add_norm,
    esimd_gemv_fp8_pert_bmg,
    esimd_rms_norm,
    esimd_fused_scaled_add_rms_norm,
    esimd_rms_norm_gated,
    esimd_fused_add_rms_norm_batched,
    esimd_resadd_norm_gemv_fp8_pert,
    esimd_resadd_norm_gemv_int4_pert,
    esimd_resadd_norm_gemv2_fp8_pert,
    esimd_norm_gemv_fp8_pert,
    esimd_norm_gemv_fp8_blockscale,
    esimd_norm_gemv_int4_pert,
    esimd_gdn_conv_fused_seq,
    esimd_gdn_conv_fused_seq_spec,
    esimd_moe_topk,
    esimd_moe_scatter_fused,
    esimd_moe_silu_mul,
    esimd_moe_gelu_tanh_mul,
    moe_forward_full_gelu_tanh,
    esimd_moe_gather,
    esimd_moe_gemm_fp8,
    esimd_moe_gemm_fp8_blockscale,
    esimd_moe_gemm_fp8_pert,
    esimd_gemm_fp8_pert,
    esimd_gemm_fp8_blockscale,
    # Eagle ops
    eagle_gdn,
    eagle_page_attn_decode,
    eagle_page_attn_decode_separate,
    # MoE Batch ops
    moe_router_forward,
    moe_batch_topk,
    moe_up_forward,
    moe_down_forward,
    moe_accumulate,
    moe_forward_fused,
    moe_forward_full,
    moe_forward_full_fp8_grouped,
    moe_forward_full_fp8_block,
    # MoE INT4 Batch ops
    moe_router_forward_int4,
    moe_router_topk_int4,
    moe_forward_full_int4,
    moe_topk_int4,
    to_cutlass_nmajor_int4,
    cutlass_nmajor_int4_to_signed,
    prepare_cutlass_nmajor_int4_weight,
    precompute_moe_route,
    moe_silu_mul_int4,
    moe_route_gather_int4,
    moe_forward_routed_cutlass_nmajor_int4,
    moe_forward_full_cutlass_nmajor_int4,
    moe_forward_full_cutlass_nmajor_int4_with_router,
    moe_forward_tiny_cutlass_nmajor_int4,
    moe_forward_tiny_cutlass_nmajor_int4_full_fp16_shared,
    moe_forward_tiny_cutlass_nmajor_int4_full_fp16_shared_from_logits,
    moe_forward_m1_cutlass_nmajor_int4_fp16_shared_asymmetric_out_v1,
    moe_tiny_cutlass_nmajor_int4_up,
    moe_tiny_cutlass_nmajor_int4_down,
    moe_tiny_fp16_shared_up,
    moe_tiny_fp16_shared_finalize,
)
