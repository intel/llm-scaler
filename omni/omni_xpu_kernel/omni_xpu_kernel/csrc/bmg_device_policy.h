// ============================================================================
// Device-independent BMG identity and kernel-profile selection
// ============================================================================
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

namespace omni_xpu {
namespace device {

// E210 is a G21 platform ID rather than the public B60 product ID, but it has
// been validated with the same local B60 kernel policy and intentionally keeps
// the existing b60 compatibility identity.
enum class BmgSku : uint8_t {
    unknown = 0,
    b580 = 1,
    b50 = 2,
    b60 = 3,
    b70 = 4,
};

enum class BmgKernelProfile : uint8_t {
    generic_bmg = 0,
    b580 = 1,
    b60 = 2,
    b70 = 3,
};

// Development-only B580 candidate axes. Fields that form one legal template
// shape are intentionally selected together (for example AdaLN block and work
// group sizes). The selector never changes another physical SKU's policy.
enum class B580PolicyCandidate : uint8_t {
    none = 0,
    adaln = 1,
    int8_dequant_fp32 = 2,
    int8_dequant_bf16 = 3,
    int8_scaleback = 4,
    convrot_g16 = 5,
    fp8_stochastic = 6,
    svdq_dequant = 7,
    svdq_quant = 8,
    svdq_smooth = 9,
    svdq_convert_add = 10,
    kitchen_rope = 11,
    d120_l4205_v_tile = 12,
    h3_vae_d64_s1797_kv_tile = 13,
};

constexpr uint32_t kBmgE210 = 0xE210;
constexpr uint32_t kArcB580 = 0xE20B;
constexpr uint32_t kArcProB60 = 0xE211;
constexpr uint32_t kArcProB50 = 0xE212;
constexpr uint32_t kArcProB70 = 0xE223;

constexpr BmgSku classify_bmg_device_id(uint32_t device_id) {
    switch (device_id) {
        case kBmgE210:
        case kArcProB60:
            return BmgSku::b60;
        case kArcB580:
            return BmgSku::b580;
        case kArcProB50:
            return BmgSku::b50;
        case kArcProB70:
            return BmgSku::b70;
        default:
            return BmgSku::unknown;
    }
}

constexpr std::string_view bmg_sku_name(BmgSku sku) {
    switch (sku) {
        case BmgSku::b580:
            return "b580";
        case BmgSku::b50:
            return "b50";
        case BmgSku::b60:
            return "b60";
        case BmgSku::b70:
            return "b70";
        default:
            return "unknown";
    }
}

constexpr BmgKernelProfile bmg_kernel_profile(BmgSku sku) {
    switch (sku) {
        case BmgSku::b580:
            return BmgKernelProfile::b580;
        case BmgSku::b60:
            return BmgKernelProfile::b60;
        case BmgSku::b70:
            return BmgKernelProfile::b70;
        default:
            return BmgKernelProfile::generic_bmg;
    }
}

constexpr std::string_view bmg_kernel_profile_name(BmgKernelProfile profile) {
    switch (profile) {
        case BmgKernelProfile::b580:
            return "b580";
        case BmgKernelProfile::b60:
            return "b60";
        case BmgKernelProfile::b70:
            return "b70";
        default:
            return "generic-bmg";
    }
}

constexpr std::string_view b580_policy_candidate_name(
        B580PolicyCandidate candidate) {
    switch (candidate) {
        case B580PolicyCandidate::adaln:
            return "adaln";
        case B580PolicyCandidate::int8_dequant_fp32:
            return "int8-dequant-fp32";
        case B580PolicyCandidate::int8_dequant_bf16:
            return "int8-dequant-bf16";
        case B580PolicyCandidate::int8_scaleback:
            return "int8-scaleback";
        case B580PolicyCandidate::convrot_g16:
            return "convrot-g16";
        case B580PolicyCandidate::fp8_stochastic:
            return "fp8-stochastic";
        case B580PolicyCandidate::svdq_dequant:
            return "svdq-dequant";
        case B580PolicyCandidate::svdq_quant:
            return "svdq-quant";
        case B580PolicyCandidate::svdq_smooth:
            return "svdq-smooth";
        case B580PolicyCandidate::svdq_convert_add:
            return "svdq-convert-add";
        case B580PolicyCandidate::kitchen_rope:
            return "kitchen-rope";
        case B580PolicyCandidate::d120_l4205_v_tile:
            return "d120-l4205-v-tile";
        case B580PolicyCandidate::h3_vae_d64_s1797_kv_tile:
            return "h3-vae-d64-s1797-kv-tile";
        default:
            return "none";
    }
}

struct BmgSelection {
    BmgSku physical_sku;
    BmgSku effective_sku;
    BmgKernelProfile kernel_profile;
    bool forced;
    B580PolicyCandidate b580_policy_candidate;
};

inline BmgSelection resolve_bmg_selection(
        uint32_t device_id,
        const char* forced_sku,
        const char* b580_candidate = nullptr) {
    const BmgSku physical = classify_bmg_device_id(device_id);
    const bool forced = forced_sku != nullptr && forced_sku[0] != '\0';
    BmgSku effective = physical;
    if (forced) {
        const std::string value(forced_sku);
        effective = BmgSku::unknown;
        if (value == "b580") {
            effective = BmgSku::b580;
        } else if (value == "b50") {
            effective = BmgSku::b50;
        } else if (value == "b60") {
            effective = BmgSku::b60;
        } else if (value == "b70") {
            effective = BmgSku::b70;
        } else if (value != "generic" && value != "generic-bmg") {
            throw std::runtime_error(
                "invalid OMNI_XPU_FORCE_SKU='" + value +
                "'; expected b580, b50, b60, b70, generic, or generic-bmg");
        }
    }

    B580PolicyCandidate candidate = B580PolicyCandidate::none;
    if (b580_candidate != nullptr && b580_candidate[0] != '\0' &&
            std::string_view(b580_candidate) != "none") {
        if (physical != BmgSku::b580) {
            throw std::runtime_error(
                "OMNI_XPU_B580_POLICY_CANDIDATE requires physical SKU b580");
        }
        if (forced) {
            throw std::runtime_error(
                "OMNI_XPU_B580_POLICY_CANDIDATE cannot be combined with "
                "OMNI_XPU_FORCE_SKU");
        }
        const std::string value(b580_candidate);
        if (value == "adaln") {
            candidate = B580PolicyCandidate::adaln;
        } else if (value == "int8-dequant-fp32") {
            candidate = B580PolicyCandidate::int8_dequant_fp32;
        } else if (value == "int8-dequant-bf16") {
            candidate = B580PolicyCandidate::int8_dequant_bf16;
        } else if (value == "int8-scaleback") {
            candidate = B580PolicyCandidate::int8_scaleback;
        } else if (value == "convrot-g16") {
            candidate = B580PolicyCandidate::convrot_g16;
        } else if (value == "fp8-stochastic") {
            candidate = B580PolicyCandidate::fp8_stochastic;
        } else if (value == "svdq-dequant") {
            candidate = B580PolicyCandidate::svdq_dequant;
        } else if (value == "svdq-quant") {
            candidate = B580PolicyCandidate::svdq_quant;
        } else if (value == "svdq-smooth") {
            candidate = B580PolicyCandidate::svdq_smooth;
        } else if (value == "svdq-convert-add") {
            candidate = B580PolicyCandidate::svdq_convert_add;
        } else if (value == "kitchen-rope") {
            candidate = B580PolicyCandidate::kitchen_rope;
        } else if (value == "d120-l4205-v-tile") {
            candidate = B580PolicyCandidate::d120_l4205_v_tile;
        } else if (value == "h3-vae-d64-s1797-kv-tile") {
            candidate = B580PolicyCandidate::h3_vae_d64_s1797_kv_tile;
        } else {
            throw std::runtime_error(
                "invalid OMNI_XPU_B580_POLICY_CANDIDATE='" + value +
                "'; expected adaln, int8-dequant-fp32, "
                "int8-dequant-bf16, int8-scaleback, convrot-g16, "
                "fp8-stochastic, svdq-dequant, svdq-quant, svdq-smooth, "
                "svdq-convert-add, kitchen-rope, d120-l4205-v-tile, "
                "h3-vae-d64-s1797-kv-tile, or none");
        }
    }
    return {
        physical,
        effective,
        bmg_kernel_profile(effective),
        forced,
        candidate};
}

}  // namespace device
}  // namespace omni_xpu
