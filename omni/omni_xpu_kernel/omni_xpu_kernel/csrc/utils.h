// ============================================================================
// Intel XPU Utilities - Cross-version compatibility layer
// ============================================================================
#pragma once

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <torch/extension.h>

// PyTorch XPU backend compatibility
#if TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR >= 3
#include <c10/xpu/XPUStream.h>
#else
#include <ipex.h>
#endif

namespace omni_xpu {

// ============================================================================
// Debug logging — controlled by OMNI_XPU_DEBUG environment variable
// ============================================================================
// Usage:
//   OMNI_XPU_DEBUG=1      — enable all modules
//   OMNI_XPU_DEBUG=sdp    — enable SDP module only
//   OMNI_XPU_DEBUG=fp8    — enable FP8 module only
//   OMNI_XPU_DEBUG=sdp,fp8 — enable SDP and FP8
//   (unset or empty)      — disabled (default)
//
// In code: OMNI_DEBUG("sdp", "V_max=%.1f needs_scaling=%d", v_max, needs);
// ============================================================================

namespace debug {

inline bool is_enabled(const char* module) {
    static const char* env = std::getenv("OMNI_XPU_DEBUG");
    if (!env || env[0] == '\0') return false;
    // "1" or "all" enables everything
    if (env[0] == '1' || std::strstr(env, "all")) return true;
    // Check if module name is in the comma-separated list
    return std::strstr(env, module) != nullptr;
}

}  // namespace debug

// Printf-style debug macro — compiled away to nothing when not enabled
#define OMNI_DEBUG(module, fmt, ...) \
    do { \
        if (::omni_xpu::debug::is_enabled(module)) { \
            std::fprintf(stderr, "[omni_xpu::%s] " fmt "\n", module, ##__VA_ARGS__); \
        } \
    } while (0)

namespace utils {

// Get SYCL queue from PyTorch XPU device
inline sycl::queue& get_queue(const torch::Device& device) {
#if TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR >= 3
    return c10::xpu::getCurrentXPUStream(device.index()).queue();
#else
    c10::impl::VirtualGuardImpl impl(device.type());
    c10::Stream c10_stream = impl.getStream(c10::Device(device));
    return xpu::get_queue_from_stream(c10_stream);
#endif
}

// Submit SYCL kernel with profiling support
// Uses template to avoid std::function heap allocation and virtual dispatch overhead
template<typename KernelFunc>
inline sycl::event submit_kernel(
    KernelFunc&& kernel,
    const at::Device& device,
    [[maybe_unused]] const char* desc
) {
    sycl::queue& queue = get_queue(device);
    sycl::event event = queue.submit(std::forward<KernelFunc>(kernel));
#if TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR >= 3
    // Profiler support for newer versions
    // xpu::profiler_record(desc, event);
#else
    xpu::profiler_record(desc, event);
#endif
    return event;
}

// Synchronize XPU device
inline void synchronize(const torch::Device& device) {
    get_queue(device).wait();
}

}  // namespace utils
}  // namespace omni_xpu
