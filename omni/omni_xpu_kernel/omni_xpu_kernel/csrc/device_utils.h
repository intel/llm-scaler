// ============================================================================
// Exact BMG device identity and runtime kernel-profile selection
// ============================================================================
#pragma once

#include <cstdlib>

#include <sycl/sycl.hpp>

#include "bmg_device_warning.h"

namespace omni_xpu {
namespace device {

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

// The core _C extension owns process-visible warnings. Native sidecars use
// this selector and route their public Python entry points through _C first,
// so loading more than one DSO cannot duplicate a fallback warning.
inline BmgSelection get_bmg_selection_unwarned(
        const sycl::device& sycl_device) {
    return resolve_bmg_selection(
        get_device_id(sycl_device),
        std::getenv("OMNI_XPU_FORCE_SKU"),
        std::getenv("OMNI_XPU_B580_POLICY_CANDIDATE"));
}

inline BmgSelection get_bmg_selection_unwarned(const sycl::queue& queue) {
    return get_bmg_selection_unwarned(queue.get_device());
}

inline BmgSelection get_bmg_selection(const sycl::device& sycl_device) {
    const uint32_t device_id = get_device_id(sycl_device);
    const BmgSelection selection = get_bmg_selection_unwarned(sycl_device);
#if defined(OMNI_XPU_ARCH_BMG)
    warn_bmg_selection_once(device_id, selection);
#endif
    return selection;
}

inline BmgSelection get_bmg_selection(const sycl::queue& queue) {
    return get_bmg_selection(queue.get_device());
}

inline BmgSku get_bmg_sku(const sycl::device& sycl_device) {
    return get_bmg_selection(sycl_device).effective_sku;
}

inline BmgSku get_bmg_sku(const sycl::queue& queue) {
    return get_bmg_sku(queue.get_device());
}

inline bool use_b60_kernel_profile(const sycl::queue& queue) {
    return get_bmg_selection(queue).kernel_profile == BmgKernelProfile::b60;
}

inline bool use_b580_policy_candidate(
        const sycl::queue& queue, B580PolicyCandidate candidate) {
    return get_bmg_selection(queue).b580_policy_candidate == candidate;
}

}  // namespace device
}  // namespace omni_xpu
