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
    b60 = 1,
    b70 = 2,
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
        case BmgKernelProfile::b60:
            return "b60";
        case BmgKernelProfile::b70:
            return "b70";
        default:
            return "generic-bmg";
    }
}

struct BmgSelection {
    BmgSku physical_sku;
    BmgSku effective_sku;
    BmgKernelProfile kernel_profile;
    bool forced;
};

inline BmgSelection resolve_bmg_selection(
        uint32_t device_id, const char* forced_sku) {
    const BmgSku physical = classify_bmg_device_id(device_id);
    if (forced_sku == nullptr || forced_sku[0] == '\0') {
        return {physical, physical, bmg_kernel_profile(physical), false};
    }

    const std::string value(forced_sku);
    BmgSku effective = BmgSku::unknown;
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
    return {physical, effective, bmg_kernel_profile(effective), true};
}

}  // namespace device
}  // namespace omni_xpu
