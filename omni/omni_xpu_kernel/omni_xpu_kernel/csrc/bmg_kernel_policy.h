// ============================================================================
// Validated BMG SKU-local kernel policies
// ============================================================================
#pragma once

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

    static constexpr bool kitchen_rope_exact_row = true;
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

    static constexpr bool kitchen_rope_exact_row = false;
    static constexpr int kitchen_rope_pairs_per_work_item = 0;
    static constexpr int kitchen_rope_work_group_size = 0;

    static constexpr int d120_l4205_v_tile = 32;
};

// Unknown BMG IDs preserve the previously shipped BMG implementation.
using GenericBmgKernelPolicy = B70KernelPolicy;

}  // namespace device
}  // namespace omni_xpu
