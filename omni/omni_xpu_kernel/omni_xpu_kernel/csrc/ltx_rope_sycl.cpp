#include <torch/extension.h>
#include <sycl/sycl.hpp>

#include <cstdint>
#include <limits>
#include <type_traits>

#include "utils.h"

using fp16 = sycl::half;
using bf16 = sycl::ext::oneapi::bfloat16;

namespace omni_xpu {
namespace rotary {

namespace {

#if defined(OMNI_XPU_ARCH_BMG)
template <typename T>
T force_dtype_round(T value) {
    using Bits = std::conditional_t<sizeof(T) == 2, uint16_t, uint32_t>;
    const volatile Bits stored = sycl::bit_cast<Bits>(value);
    const Bits loaded = stored;
    return sycl::bit_cast<T>(loaded);
}

bool supported(
    const torch::Tensor& input,
    const torch::Tensor& cos,
    const torch::Tensor& sin) {
    if (!input.device().is_xpu() || !cos.device().is_xpu() ||
        !sin.device().is_xpu()) {
        return false;
    }
    if (input.device() != cos.device() || input.device() != sin.device()) {
        return false;
    }
    if (input.scalar_type() != cos.scalar_type() ||
        input.scalar_type() != sin.scalar_type()) {
        return false;
    }
    if (input.scalar_type() != torch::kFloat16 &&
        input.scalar_type() != torch::kBFloat16) {
        return false;
    }
    if (input.dim() != 3 || !input.is_contiguous() ||
        cos.dim() != 4 || sin.dim() != 4) {
        return false;
    }
    if (cos.sizes() != sin.sizes() || cos.strides() != sin.strides()) {
        return false;
    }
    if (cos.size(0) != input.size(0) ||
        cos.size(2) != input.size(1) ||
        cos.size(1) <= 0 ||
        (cos.size(3) != 32 && cos.size(3) != 64)) {
        return false;
    }
    if (input.size(2) != cos.size(1) * cos.size(3) * 2) {
        return false;
    }
    if (cos.stride(3) != 1 || sin.stride(3) != 1) {
        return false;
    }
    for (int64_t dim = 0; dim < cos.dim(); ++dim) {
        if (cos.stride(dim) < 0 || sin.stride(dim) < 0) {
            return false;
        }
    }
    return input.numel() <= std::numeric_limits<uint32_t>::max() &&
           cos.numel() <= std::numeric_limits<uint32_t>::max();
}

template <typename T>
void launch(
    const torch::Tensor& input,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    torch::Tensor& output) {
    const auto* input_ptr = reinterpret_cast<const T*>(input.data_ptr());
    const auto* cos_ptr = reinterpret_cast<const T*>(cos.data_ptr());
    const auto* sin_ptr = reinterpret_cast<const T*>(sin.data_ptr());
    auto* output_ptr = reinterpret_cast<T*>(output.data_ptr());

    const uint32_t batch = static_cast<uint32_t>(input.size(0));
    const uint32_t tokens = static_cast<uint32_t>(input.size(1));
    const uint32_t heads = static_cast<uint32_t>(cos.size(1));
    const uint32_t half = static_cast<uint32_t>(cos.size(3));
    const uint32_t head_dim = half * 2;
    const uint32_t rows = batch * tokens * heads;
    const int64_t cos_s0 = cos.stride(0);
    const int64_t cos_s1 = cos.stride(1);
    const int64_t cos_s2 = cos.stride(2);
    const int64_t sin_s0 = sin.stride(0);
    const int64_t sin_s1 = sin.stride(1);
    const int64_t sin_s2 = sin.stride(2);

    auto cgf = [&](sycl::handler& handler) {
        handler.parallel_for(
            sycl::nd_range<1>(
                sycl::range<1>(static_cast<size_t>(rows) * half),
                sycl::range<1>(half)),
            [=](sycl::nd_item<1> item) {
                uint32_t row = static_cast<uint32_t>(item.get_group(0));
                const uint32_t pair =
                    static_cast<uint32_t>(item.get_local_id(0));
                const uint32_t head = row % heads;
                row /= heads;
                const uint32_t token = row % tokens;
                const uint32_t b = row / tokens;

                const uint32_t input_base =
                    ((b * tokens + token) * heads + head) * head_dim;
                const int64_t cos_offset =
                    static_cast<int64_t>(b) * cos_s0 +
                    static_cast<int64_t>(head) * cos_s1 +
                    static_cast<int64_t>(token) * cos_s2 + pair;
                const int64_t sin_offset =
                    static_cast<int64_t>(b) * sin_s0 +
                    static_cast<int64_t>(head) * sin_s1 +
                    static_cast<int64_t>(token) * sin_s2 + pair;

                const T first = input_ptr[input_base + pair];
                const T second = input_ptr[input_base + half + pair];
                const T cosine = cos_ptr[cos_offset];
                const T sine = sin_ptr[sin_offset];
                const T first_product =
                    force_dtype_round<T>(cosine * first);
                const T second_product =
                    force_dtype_round<T>(cosine * second);
                const float first_output = sycl::fma(
                    -static_cast<float>(sine),
                    static_cast<float>(second),
                    static_cast<float>(first_product));
                const float second_output = sycl::fma(
                    static_cast<float>(sine),
                    static_cast<float>(first),
                    static_cast<float>(second_product));
                output_ptr[input_base + pair] = force_dtype_round<T>(
                    static_cast<T>(first_output));
                output_ptr[input_base + half + pair] = force_dtype_round<T>(
                    static_cast<T>(second_output));
            });
    };
    utils::submit_kernel(
        cgf, input.device(), "ltx_split_rope_direct_bmg");
}
#endif

}  // namespace

bool ltx_split_rope_direct_supported(
    const torch::Tensor& input,
    const torch::Tensor& cos,
    const torch::Tensor& sin) {
#if defined(OMNI_XPU_ARCH_BMG)
    return supported(input, cos, sin);
#else
    return false;
#endif
}

torch::Tensor apply_ltx_split_rope_direct(
    const torch::Tensor& input,
    const torch::Tensor& cos,
    const torch::Tensor& sin) {
#if defined(OMNI_XPU_ARCH_BMG)
    TORCH_CHECK(
        supported(input, cos, sin),
        "unsupported direct LTX split-half RoPE contract");
    auto output = torch::empty_like(input);
    if (input.numel() == 0) return output;
    if (input.scalar_type() == torch::kFloat16) {
        launch<fp16>(input, cos, sin, output);
    } else {
        launch<bf16>(input, cos, sin, output);
    }
    return output;
#else
    TORCH_CHECK(
        false,
        "direct LTX split-half RoPE is only available in a BMG core");
#endif
}

}  // namespace rotary
}  // namespace omni_xpu
