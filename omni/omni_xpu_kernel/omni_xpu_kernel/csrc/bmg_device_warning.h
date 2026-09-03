// ============================================================================
// Process-visible warnings for BMG debug and generic policy selection
// ============================================================================
#pragma once

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <mutex>

#include "bmg_kernel_policy.h"

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
    } else if (
            selection.b580_policy_candidate != B580PolicyCandidate::none) {
        static std::once_flag candidate_warning;
        std::call_once(candidate_warning, [device_id, selection]() {
            std::cerr
                << "[omni_xpu::device] warning: physical Device ID 0x"
                << std::hex << device_id << std::dec
                << " (physical_sku=b580) enables "
                << "OMNI_XPU_B580_POLICY_CANDIDATE="
                << b580_policy_candidate_name(
                       selection.b580_policy_candidate)
                << "; development A/B only, performance_claim=false\n";
        });
    } else if (!kernel_policy_performance_claim_allowed(
                   selection.kernel_profile)) {
        static std::once_flag experimental_warning;
        std::call_once(experimental_warning, [device_id, selection]() {
            std::cerr
                << "[omni_xpu::device] warning: physical Device ID 0x"
                << std::hex << device_id << std::dec
                << " (physical_sku=" << bmg_sku_name(selection.physical_sku)
                << ") uses kernel_profile="
                << bmg_kernel_profile_name(selection.kernel_profile)
                << ", policy_id=" << kernel_policy_id(selection.kernel_profile)
                << ", support="
                << kernel_policy_status(selection.kernel_profile)
                << "; functional support may be accepted independently, "
                << "performance_claim=false\n";
        });
    }
}

}  // namespace device
}  // namespace omni_xpu
