// ============================================================================
// Validated BMG SKU-local kernel policies
// ============================================================================
#pragma once

#include "bmg_device_policy.h"

namespace omni_xpu {
namespace device {

// B70 values preserve the 09c4cdd public BMG behavior.  B60 values are the
// independently measured E210 policy; E211 intentionally shares this policy.
// Runtime dispatch remains exact-device guarded in each public entry point so
// unsupported shapes retain the B70/generic implementation.
struct B60KernelPolicy {
    static constexpr int adaln_block_size = 512;
    static constexpr int adaln_work_group_size = 1;

    static constexpr int int8_dequant_fp32_elements = 256;
    static constexpr int int8_dequant_fp32_work_group_size = 64;
    static constexpr int int8_dequant_fp16_elements = 32;
    static constexpr int int8_dequant_fp16_work_group_size = 64;
    static constexpr int int8_dequant_bf16_elements = 128;
    static constexpr int int8_dequant_bf16_work_group_size = 16;

    static constexpr int int8_scaleback_elements = 256;
    static constexpr int int8_scaleback_work_group_rows = 4;
    static constexpr int int8_scaleback_work_group_cols = 8;

    static constexpr int convrot_g16_groups_per_dpas = 7;
    static constexpr int convrot_g16_work_items_per_row = 30;

    static constexpr int fp8_stochastic_elements = 7;

    static constexpr int svdq_dequant_groups = 60;
    static constexpr int svdq_dequant_work_group_size = 1;
    static constexpr int svdq_quant_groups = 60;
    static constexpr int svdq_quant_work_group_size = 1;
    static constexpr int svdq_smooth_elements = 256;
    static constexpr int svdq_smooth_work_group_size = 1;
    static constexpr int svdq_convert_add_elements = 128;

    static constexpr int kitchen_rope_pairs_per_work_item = 1;
    static constexpr int kitchen_rope_work_group_size = 64;

    static constexpr int d120_l4205_v_tile = 64;
};

struct B70KernelPolicy {
    static constexpr int adaln_block_size = 32;
    static constexpr int adaln_work_group_size = 64;

    static constexpr int int8_dequant_fp32_elements = 32;
    static constexpr int int8_dequant_fp32_work_group_size = 64;
    static constexpr int int8_dequant_fp16_elements = 32;
    static constexpr int int8_dequant_fp16_work_group_size = 64;
    static constexpr int int8_dequant_bf16_elements = 32;
    static constexpr int int8_dequant_bf16_work_group_size = 64;

    static constexpr int int8_scaleback_elements = 32;
    static constexpr int int8_scaleback_work_group_rows = 4;
    static constexpr int int8_scaleback_work_group_cols = 8;

    static constexpr int convrot_g16_groups_per_dpas = 8;
    static constexpr int convrot_g16_work_items_per_row = 27;

    static constexpr int fp8_stochastic_elements = 6;

    static constexpr int svdq_dequant_groups = 60;
    static constexpr int svdq_dequant_work_group_size = 64;
    static constexpr int svdq_quant_groups = 60;
    static constexpr int svdq_quant_work_group_size = 64;
    static constexpr int svdq_smooth_elements = 256;
    static constexpr int svdq_smooth_work_group_size = 64;
    static constexpr int svdq_convert_add_elements = 32;

    static constexpr int kitchen_rope_pairs_per_work_item = 0;
    static constexpr int kitchen_rope_work_group_size = 0;

    static constexpr int d120_l4205_v_tile = 32;
};

// Recognized-but-unvalidated and unknown BMG IDs preserve the previously
// shipped B70-compatible values, but own an independent policy type. Future
// B70 tuning must therefore opt in to changing generic fallback behavior.
struct GenericBmgKernelPolicy {
    static constexpr int adaln_block_size = 32;
    static constexpr int adaln_work_group_size = 64;

    static constexpr int int8_dequant_fp32_elements = 32;
    static constexpr int int8_dequant_fp32_work_group_size = 64;
    static constexpr int int8_dequant_fp16_elements = 32;
    static constexpr int int8_dequant_fp16_work_group_size = 64;
    static constexpr int int8_dequant_bf16_elements = 32;
    static constexpr int int8_dequant_bf16_work_group_size = 64;

    static constexpr int int8_scaleback_elements = 32;
    static constexpr int int8_scaleback_work_group_rows = 4;
    static constexpr int int8_scaleback_work_group_cols = 8;

    static constexpr int convrot_g16_groups_per_dpas = 8;
    static constexpr int convrot_g16_work_items_per_row = 27;

    static constexpr int fp8_stochastic_elements = 6;

    static constexpr int svdq_dequant_groups = 60;
    static constexpr int svdq_dequant_work_group_size = 64;
    static constexpr int svdq_quant_groups = 60;
    static constexpr int svdq_quant_work_group_size = 64;
    static constexpr int svdq_smooth_elements = 256;
    static constexpr int svdq_smooth_work_group_size = 64;
    static constexpr int svdq_convert_add_elements = 32;

    static constexpr int kitchen_rope_pairs_per_work_item = 0;
    static constexpr int kitchen_rope_work_group_size = 0;

    static constexpr int d120_l4205_v_tile = 32;
};

// A candidate type differs from GenericBmgKernelPolicy in exactly one legal
// policy axis. This keeps matched B580 A/B runs attributable while compiling
// every candidate into one BMG development image.
template <B580PolicyCandidate Candidate>
struct B580CandidateKernelPolicy {
    static constexpr int adaln_block_size =
        Candidate == B580PolicyCandidate::adaln
        ? B60KernelPolicy::adaln_block_size
        : GenericBmgKernelPolicy::adaln_block_size;
    static constexpr int adaln_work_group_size =
        Candidate == B580PolicyCandidate::adaln
        ? B60KernelPolicy::adaln_work_group_size
        : GenericBmgKernelPolicy::adaln_work_group_size;

    static constexpr int int8_dequant_fp32_elements =
        Candidate == B580PolicyCandidate::int8_dequant_fp32
        ? B60KernelPolicy::int8_dequant_fp32_elements
        : GenericBmgKernelPolicy::int8_dequant_fp32_elements;
    static constexpr int int8_dequant_fp32_work_group_size =
        Candidate == B580PolicyCandidate::int8_dequant_fp32
        ? B60KernelPolicy::int8_dequant_fp32_work_group_size
        : GenericBmgKernelPolicy::int8_dequant_fp32_work_group_size;
    static constexpr int int8_dequant_fp16_elements =
        GenericBmgKernelPolicy::int8_dequant_fp16_elements;
    static constexpr int int8_dequant_fp16_work_group_size =
        GenericBmgKernelPolicy::int8_dequant_fp16_work_group_size;
    static constexpr int int8_dequant_bf16_elements =
        Candidate == B580PolicyCandidate::int8_dequant_bf16
        ? B60KernelPolicy::int8_dequant_bf16_elements
        : GenericBmgKernelPolicy::int8_dequant_bf16_elements;
    static constexpr int int8_dequant_bf16_work_group_size =
        Candidate == B580PolicyCandidate::int8_dequant_bf16
        ? B60KernelPolicy::int8_dequant_bf16_work_group_size
        : GenericBmgKernelPolicy::int8_dequant_bf16_work_group_size;

    static constexpr int int8_scaleback_elements =
        Candidate == B580PolicyCandidate::int8_scaleback
        ? B60KernelPolicy::int8_scaleback_elements
        : GenericBmgKernelPolicy::int8_scaleback_elements;
    static constexpr int int8_scaleback_work_group_rows =
        GenericBmgKernelPolicy::int8_scaleback_work_group_rows;
    static constexpr int int8_scaleback_work_group_cols =
        GenericBmgKernelPolicy::int8_scaleback_work_group_cols;

    static constexpr int convrot_g16_groups_per_dpas =
        Candidate == B580PolicyCandidate::convrot_g16
        ? B60KernelPolicy::convrot_g16_groups_per_dpas
        : GenericBmgKernelPolicy::convrot_g16_groups_per_dpas;
    static constexpr int convrot_g16_work_items_per_row =
        Candidate == B580PolicyCandidate::convrot_g16
        ? B60KernelPolicy::convrot_g16_work_items_per_row
        : GenericBmgKernelPolicy::convrot_g16_work_items_per_row;

    static constexpr int fp8_stochastic_elements =
        Candidate == B580PolicyCandidate::fp8_stochastic
        ? B60KernelPolicy::fp8_stochastic_elements
        : GenericBmgKernelPolicy::fp8_stochastic_elements;

    static constexpr int svdq_dequant_groups =
        GenericBmgKernelPolicy::svdq_dequant_groups;
    static constexpr int svdq_dequant_work_group_size =
        Candidate == B580PolicyCandidate::svdq_dequant
        ? B60KernelPolicy::svdq_dequant_work_group_size
        : GenericBmgKernelPolicy::svdq_dequant_work_group_size;
    static constexpr int svdq_quant_groups =
        GenericBmgKernelPolicy::svdq_quant_groups;
    static constexpr int svdq_quant_work_group_size =
        Candidate == B580PolicyCandidate::svdq_quant
        ? B60KernelPolicy::svdq_quant_work_group_size
        : GenericBmgKernelPolicy::svdq_quant_work_group_size;
    static constexpr int svdq_smooth_elements =
        GenericBmgKernelPolicy::svdq_smooth_elements;
    static constexpr int svdq_smooth_work_group_size =
        Candidate == B580PolicyCandidate::svdq_smooth
        ? B60KernelPolicy::svdq_smooth_work_group_size
        : GenericBmgKernelPolicy::svdq_smooth_work_group_size;
    static constexpr int svdq_convert_add_elements =
        Candidate == B580PolicyCandidate::svdq_convert_add
        ? B60KernelPolicy::svdq_convert_add_elements
        : GenericBmgKernelPolicy::svdq_convert_add_elements;

    static constexpr int kitchen_rope_pairs_per_work_item =
        Candidate == B580PolicyCandidate::kitchen_rope
        ? B60KernelPolicy::kitchen_rope_pairs_per_work_item
        : GenericBmgKernelPolicy::kitchen_rope_pairs_per_work_item;
    static constexpr int kitchen_rope_work_group_size =
        Candidate == B580PolicyCandidate::kitchen_rope
        ? B60KernelPolicy::kitchen_rope_work_group_size
        : GenericBmgKernelPolicy::kitchen_rope_work_group_size;

    static constexpr int d120_l4205_v_tile =
        Candidate == B580PolicyCandidate::d120_l4205_v_tile
        ? B60KernelPolicy::d120_l4205_v_tile
        : GenericBmgKernelPolicy::d120_l4205_v_tile;
};

using B580AdalnCandidatePolicy =
    B580CandidateKernelPolicy<B580PolicyCandidate::adaln>;
using B580Int8DequantFp32CandidatePolicy =
    B580CandidateKernelPolicy<B580PolicyCandidate::int8_dequant_fp32>;
using B580Int8DequantBf16CandidatePolicy =
    B580CandidateKernelPolicy<B580PolicyCandidate::int8_dequant_bf16>;
using B580Int8ScalebackCandidatePolicy =
    B580CandidateKernelPolicy<B580PolicyCandidate::int8_scaleback>;
using B580ConvrotG16CandidatePolicy =
    B580CandidateKernelPolicy<B580PolicyCandidate::convrot_g16>;
using B580Fp8StochasticCandidatePolicy =
    B580CandidateKernelPolicy<B580PolicyCandidate::fp8_stochastic>;
using B580SvdqDequantCandidatePolicy =
    B580CandidateKernelPolicy<B580PolicyCandidate::svdq_dequant>;
using B580SvdqQuantCandidatePolicy =
    B580CandidateKernelPolicy<B580PolicyCandidate::svdq_quant>;
using B580SvdqSmoothCandidatePolicy =
    B580CandidateKernelPolicy<B580PolicyCandidate::svdq_smooth>;
using B580SvdqConvertAddCandidatePolicy =
    B580CandidateKernelPolicy<B580PolicyCandidate::svdq_convert_add>;
using B580KitchenRopeCandidatePolicy =
    B580CandidateKernelPolicy<B580PolicyCandidate::kitchen_rope>;
using B580D120L4205CandidatePolicy =
    B580CandidateKernelPolicy<B580PolicyCandidate::d120_l4205_v_tile>;

}  // namespace device
}  // namespace omni_xpu
