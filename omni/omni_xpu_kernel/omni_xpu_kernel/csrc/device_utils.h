// ============================================================================
// Exact BMG device identity and runtime kernel-profile selection
// ============================================================================
#pragma once

#include <cstdint>
#include <string_view>

#include <sycl/sycl.hpp>

namespace omni_xpu {
namespace device {

// Keep product identity separate from the kernel profile.  E210 is a G21
// platform ID rather than the public B60 product ID, but it has been validated
// with the same local B60 kernel policy and is intentionally routed there.
enum class BmgSku : uint8_t {
    unknown = 0,
    b60 = 1,
    b70 = 2,
};

constexpr uint32_t kBmgE210 = 0xE210;
constexpr uint32_t kArcProB60 = 0xE211;
constexpr uint32_t kArcProB70 = 0xE223;

constexpr BmgSku classify_bmg_device_id(uint32_t device_id) {
    switch (device_id) {
        case kBmgE210:
        case kArcProB60:
            return BmgSku::b60;
        case kArcProB70:
            return BmgSku::b70;
        default:
            return BmgSku::unknown;
    }
}

constexpr std::string_view bmg_sku_name(BmgSku sku) {
    switch (sku) {
        case BmgSku::b60:
            return "b60";
        case BmgSku::b70:
            return "b70";
        default:
            return "unknown";
    }
}

inline uint32_t get_device_id(const sycl::device& sycl_device) {
    if (!sycl_device.has(sycl::aspect::ext_intel_device_id)) {
        return 0;
    }
    return sycl_device.get_info<
        sycl::ext::intel::info::device::device_id>();
}

inline uint32_t get_device_id(const sycl::queue& queue) {
    return get_device_id(queue.get_device());
}

inline BmgSku get_bmg_sku(const sycl::device& sycl_device) {
    return classify_bmg_device_id(get_device_id(sycl_device));
}

inline BmgSku get_bmg_sku(const sycl::queue& queue) {
    return get_bmg_sku(queue.get_device());
}

inline bool use_b60_kernel_profile(const sycl::queue& queue) {
    return get_bmg_sku(queue) == BmgSku::b60;
}

}  // namespace device
}  // namespace omni_xpu
