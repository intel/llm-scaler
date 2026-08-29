// ============================================================================
// Process-visible warnings for BMG debug and generic policy selection
// ============================================================================
#pragma once

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <mutex>

#include "bmg_device_policy.h"

namespace omni_xpu {
namespace device {

inline void warn_bmg_selection_once(
        uint32_t device_id, const BmgSelection& selection) {
    if (selection.forced) {
        static std::once_flag forced_warning;
        std::call_once(forced_warning, [device_id, selection]() {
            std::cerr
                << "[omni_xpu::device] warning: OMNI_XPU_FORCE_SKU overrides "
                << "physical Device ID 0x" << std::hex << device_id << std::dec
                << " (physical_sku=" << bmg_sku_name(selection.physical_sku)
                << ", effective_sku=" << bmg_sku_name(selection.effective_sku)
                << ", kernel_profile="
                << bmg_kernel_profile_name(selection.kernel_profile)
                << "); debug/prescreen only, performance_claim=false\n";
        });
    } else if (selection.kernel_profile == BmgKernelProfile::generic_bmg) {
        static std::once_flag generic_warning;
        std::call_once(generic_warning, [device_id, selection]() {
            std::cerr
                << "[omni_xpu::device] warning: physical Device ID 0x"
                << std::hex << device_id << std::dec
                << " (physical_sku=" << bmg_sku_name(selection.physical_sku)
                << ") uses kernel_profile=generic-bmg; "
                << "policy=GenericBmgKernelPolicy; "
                << "SKU-specific kernel policy is unvalidated\n";
        });
    }
}

}  // namespace device
}  // namespace omni_xpu
