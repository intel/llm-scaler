// ============================================================================
// BMG GroupNorm for the captured Boogu Image Turbo activation contracts
// ============================================================================

#include <cstdint>

#include <torch/extension.h>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

#include "utils.h"

#if defined(OMNI_XPU_ARCH_BMG)

using bf16 = sycl::ext::oneapi::bfloat16;
using namespace sycl::ext::intel::esimd;

namespace omni_xpu {
namespace norm {
namespace {

constexpr int64_t kGroups = 32;
constexpr int64_t kTile = 32768;
constexpr int kReduceVector = 32;

class GroupNormPartialMomentsBMGKernel;
class GroupNormFinalizeMomentsBMGKernel;
class GroupNormFusedParamsBMGKernel;
class GroupNormNormalizeBMGKernel;

bool is_valid_boogu_shape(const torch::Tensor& input) {
    const int64_t channels = input.size(1);
    const int64_t height = input.size(2);
    const int64_t width = input.size(3);
    return
        (channels == 512 && height == 128 && width == 128) ||
        (channels == 512 && height == 256 && width == 256) ||
        (channels == 512 && height == 512 && width == 512) ||
        (channels == 256 && height == 512 && width == 512) ||
        (channels == 256 && height == 1024 && width == 1024) ||
        (channels == 128 && height == 1024 && width == 1024);
}

void launch_group_norm_bmg(
    const bf16* __restrict__ input,
    const bf16* __restrict__ weight,
    const bf16* __restrict__ bias,
    bf16* __restrict__ output,
    float* __restrict__ partial_sums,
    float* __restrict__ partial_squares,
    float* __restrict__ means,
    float* __restrict__ rstds,
    float* __restrict__ scales,
    float* __restrict__ shifts,
    int64_t channels,
    int64_t spatial,
    float eps,
    const at::Device& device) {
    const int64_t channels_per_group = channels / kGroups;
    const int64_t group_elements = channels_per_group * spatial;
    const int64_t partials_per_group = group_elements / kTile;
    const int64_t partial_count = kGroups * partials_per_group;

    auto partial_cgf = [&](sycl::handler& handler) {
        handler.parallel_for<GroupNormPartialMomentsBMGKernel>(
            sycl::range<1>(static_cast<size_t>(partial_count)),
            [=](sycl::item<1> item) SYCL_ESIMD_KERNEL {
                const int64_t partial = item.get_id(0);
                const int64_t group = partial / partials_per_group;
                const int64_t tile_index =
                    partial - group * partials_per_group;
                const bf16* tile_input =
                    input + group * group_elements + tile_index * kTile;

                simd<float, kReduceVector> sums = 0.0f;
                simd<float, kReduceVector> squares = 0.0f;
#pragma unroll
                for (int offset = 0; offset < kTile;
                     offset += kReduceVector) {
                    simd<bf16, kReduceVector> values_bf16 =
                        block_load<bf16, kReduceVector>(
                            tile_input + offset);
                    simd<float, kReduceVector> values = values_bf16;
                    sums += values;
                    squares += values * values;
                }
                partial_sums[partial] =
                    sycl::ext::intel::esimd::detail::sum<
                        float, float, kReduceVector>(sums);
                partial_squares[partial] =
                    sycl::ext::intel::esimd::detail::sum<
                        float, float, kReduceVector>(squares);
            });
    };
    utils::submit_kernel(
        partial_cgf, device, "group_norm_bmg_partial_moments");

    auto finalize_cgf = [&](sycl::handler& handler) {
        handler.parallel_for<GroupNormFinalizeMomentsBMGKernel>(
            sycl::range<1>(static_cast<size_t>(kGroups)),
            [=](sycl::item<1> item) {
                const int64_t group = item.get_id(0);
                const int64_t start = group * partials_per_group;
                float total = 0.0f;
                float total_square = 0.0f;
                for (int64_t partial = 0;
                     partial < partials_per_group;
                     ++partial) {
                    total += partial_sums[start + partial];
                    total_square += partial_squares[start + partial];
                }
                const float inverse_count =
                    1.0f / static_cast<float>(group_elements);
                const float mean = total * inverse_count;
                const float raw_variance =
                    total_square * inverse_count - mean * mean;
                const float variance =
                    raw_variance > 0.0f ? raw_variance : 0.0f;
                means[group] = mean;
                rstds[group] = 1.0f / sycl::sqrt(variance + eps);
            });
    };
    utils::submit_kernel(
        finalize_cgf, device, "group_norm_bmg_finalize_moments");

    auto fused_params_cgf = [&](sycl::handler& handler) {
        handler.parallel_for<GroupNormFusedParamsBMGKernel>(
            sycl::range<1>(static_cast<size_t>(channels)),
            [=](sycl::item<1> item) {
                const int64_t channel = item.get_id(0);
                const int64_t group = channel / channels_per_group;
                const float scale =
                    rstds[group] * static_cast<float>(weight[channel]);
                scales[channel] = scale;
                shifts[channel] =
                    static_cast<float>(bias[channel]) -
                    means[group] * scale;
            });
    };
    utils::submit_kernel(
        fused_params_cgf, device, "group_norm_bmg_fused_params");

    auto normalize_cgf = [&](sycl::handler& handler) {
        handler.parallel_for<GroupNormNormalizeBMGKernel>(
            sycl::range<2>(
                static_cast<size_t>(channels),
                static_cast<size_t>(spatial)),
            [=](sycl::item<2> item) {
                const int64_t channel = item.get_id(0);
                const int64_t position = item.get_id(1);
                const int64_t offset = channel * spatial + position;
                output[offset] = static_cast<bf16>(
                    static_cast<float>(input[offset]) * scales[channel] +
                    shifts[channel]);
            });
    };
    utils::submit_kernel(
        normalize_cgf, device, "group_norm_bmg_normalize");
}

}  // namespace

torch::Tensor group_norm_bmg(
    torch::Tensor input,
    int64_t groups,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps) {
    TORCH_CHECK(input.device().is_xpu(), "input must be on XPU");
    TORCH_CHECK(
        weight.device() == input.device() &&
            bias.device() == input.device(),
        "weight and bias must be on the input XPU device");
    TORCH_CHECK(
        input.dim() == 4 && input.size(0) == 1,
        "input must be [1,C,H,W]");
    TORCH_CHECK(
        groups == kGroups,
        "BMG GroupNorm requires 32 groups");
    TORCH_CHECK(
        input.scalar_type() == torch::kBFloat16 &&
            weight.scalar_type() == torch::kBFloat16 &&
            bias.scalar_type() == torch::kBFloat16,
        "BMG GroupNorm requires BF16 input, weight, and bias");
    TORCH_CHECK(
        input.is_contiguous() && weight.is_contiguous() &&
            bias.is_contiguous(),
        "BMG GroupNorm requires contiguous tensors");
    TORCH_CHECK(
        weight.dim() == 1 && weight.numel() == input.size(1),
        "weight must match the channel dimension");
    TORCH_CHECK(
        bias.dim() == 1 && bias.numel() == input.size(1),
        "bias must match the channel dimension");
    TORCH_CHECK(
        eps == 1e-6,
        "BMG GroupNorm requires eps=1e-6");
    TORCH_CHECK(
        is_valid_boogu_shape(input),
        "unsupported BMG GroupNorm shape");

    const int64_t channels = input.size(1);
    const int64_t spatial = input.size(2) * input.size(3);
    const int64_t group_elements = channels / kGroups * spatial;
    const int64_t partials_per_group = group_elements / kTile;
    auto float_options = input.options().dtype(torch::kFloat32);
    auto output = torch::empty_like(input);
    auto partial_sums =
        torch::empty({kGroups, partials_per_group}, float_options);
    auto partial_squares =
        torch::empty({kGroups, partials_per_group}, float_options);
    auto means = torch::empty({kGroups}, float_options);
    auto rstds = torch::empty({kGroups}, float_options);
    auto scales = torch::empty({channels}, float_options);
    auto shifts = torch::empty({channels}, float_options);

    launch_group_norm_bmg(
        reinterpret_cast<const bf16*>(input.data_ptr()),
        reinterpret_cast<const bf16*>(weight.data_ptr()),
        reinterpret_cast<const bf16*>(bias.data_ptr()),
        reinterpret_cast<bf16*>(output.data_ptr()),
        partial_sums.data_ptr<float>(),
        partial_squares.data_ptr<float>(),
        means.data_ptr<float>(),
        rstds.data_ptr<float>(),
        scales.data_ptr<float>(),
        shifts.data_ptr<float>(),
        channels,
        spatial,
        static_cast<float>(eps),
        input.device());
    return output;
}

}  // namespace norm
}  // namespace omni_xpu

#endif
