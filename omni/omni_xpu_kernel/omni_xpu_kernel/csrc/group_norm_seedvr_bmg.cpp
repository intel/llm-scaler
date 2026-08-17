// ============================================================================
// BMG GroupNorm for captured SeedVR2 temporal-interleaved FP16 activations
// ============================================================================

#include <cstdint>

#include <torch/extension.h>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

#include "utils.h"

#if defined(OMNI_XPU_ARCH_BMG)

using fp16 = sycl::half;
using namespace sycl::ext::intel::esimd;

namespace omni_xpu {
namespace norm {
namespace {

constexpr int64_t kSeedVRGroups = 32;
constexpr int64_t kSeedVRTileSmall = 16384;
constexpr int64_t kSeedVRTileLarge = 32768;
constexpr int kSeedVRReduceVector = 32;

template <int Tile>
class SeedVRGroupNormPartialMomentsBMGKernel;

template <int Tile>
class SeedVRGroupNormFinalizeMomentsBMGKernel;

template <int Tile>
class SeedVRGroupNormFusedParamsBMGKernel;

template <int Tile>
class SeedVRGroupNormNormalizeBMGKernel;

bool is_valid_seedvr_shape(const torch::Tensor& input) {
    const int64_t batch = input.size(0);
    const int64_t channels = input.size(1);
    const int64_t height = input.size(2);
    const int64_t width = input.size(3);
    return
        (batch == 4 && channels == 128 && height == 512 && width == 512) ||
        (batch == 4 && channels == 256 && height == 256 && width == 256) ||
        (batch == 4 && channels == 256 && height == 512 && width == 512) ||
        (batch == 4 && channels == 512 && height == 256 && width == 256) ||
        (batch == 2 && channels == 512 && height == 128 && width == 128);
}

template <int Tile>
void launch_group_norm_seedvr_bmg(
    const fp16* __restrict__ input,
    const fp16* __restrict__ weight,
    const fp16* __restrict__ bias,
    fp16* __restrict__ output,
    float* __restrict__ partial_means,
    float* __restrict__ partial_m2s,
    float* __restrict__ means,
    float* __restrict__ rstds,
    float* __restrict__ scales,
    float* __restrict__ shifts,
    int64_t batch,
    int64_t channels,
    int64_t spatial,
    int64_t input_batch_stride,
    int64_t input_channel_stride,
    int64_t groups,
    float eps,
    const at::Device& device) {
    const int64_t channels_per_group = channels / groups;
    const int64_t group_elements = channels_per_group * spatial;
    const int64_t partials_per_group = group_elements / Tile;
    const int64_t total_groups = batch * groups;
    const int64_t partial_count = total_groups * partials_per_group;
    const int64_t total_channels = batch * channels;
    const int64_t numel = total_channels * spatial;

    auto partial_cgf = [&](sycl::handler& handler) {
        handler.parallel_for<SeedVRGroupNormPartialMomentsBMGKernel<Tile>>(
            sycl::range<1>(static_cast<size_t>(partial_count)),
            [=](sycl::item<1> item) SYCL_ESIMD_KERNEL {
                const int64_t partial = item.get_id(0);
                const int64_t group = partial / partials_per_group;
                const int64_t batch_index = group / groups;
                const int64_t group_index = group % groups;
                const int64_t tile_index =
                    partial - group * partials_per_group;
                const int64_t logical_offset = tile_index * Tile;
                const int64_t channel_in_group = logical_offset / spatial;
                const int64_t position = logical_offset % spatial;
                const fp16* tile_input =
                    input + batch_index * input_batch_stride +
                    (group_index * channels_per_group + channel_in_group) *
                        input_channel_stride +
                    position;

                simd<fp16, kSeedVRReduceVector> anchor_values_fp16 =
                    block_load<fp16, kSeedVRReduceVector>(tile_input);
                simd<float, kSeedVRReduceVector> anchor_values =
                    anchor_values_fp16;
                const float anchor =
                    sycl::ext::intel::esimd::detail::sum<
                        float, float, kSeedVRReduceVector>(anchor_values) /
                    static_cast<float>(kSeedVRReduceVector);
                simd<float, kSeedVRReduceVector> centered_sums = 0.0f;
                simd<float, kSeedVRReduceVector> centered_squares = 0.0f;
#pragma unroll
                for (int offset = 0; offset < Tile;
                     offset += kSeedVRReduceVector) {
                    simd<fp16, kSeedVRReduceVector> values_fp16 =
                        block_load<fp16, kSeedVRReduceVector>(
                            tile_input + offset);
                    simd<float, kSeedVRReduceVector> values = values_fp16;
                    simd<float, kSeedVRReduceVector> centered =
                        values - anchor;
                    centered_sums += centered;
                    centered_squares += centered * centered;
                }
                const float centered_sum =
                    sycl::ext::intel::esimd::detail::sum<
                        float, float, kSeedVRReduceVector>(centered_sums);
                const float centered_square_sum =
                    sycl::ext::intel::esimd::detail::sum<
                        float, float, kSeedVRReduceVector>(centered_squares);
                const float mean =
                    anchor + centered_sum / static_cast<float>(Tile);
                const float raw_m2 = centered_square_sum -
                    centered_sum * centered_sum / static_cast<float>(Tile);
                partial_means[partial] = mean;
                partial_m2s[partial] = raw_m2 < 0.0f ? 0.0f : raw_m2;
            });
    };
    utils::submit_kernel(
        partial_cgf, device, "group_norm_seedvr_bmg_partial_moments");

    auto finalize_cgf = [&](sycl::handler& handler) {
        handler.parallel_for<SeedVRGroupNormFinalizeMomentsBMGKernel<Tile>>(
            sycl::range<1>(static_cast<size_t>(total_groups)),
            [=](sycl::item<1> item) {
                const int64_t group = item.get_id(0);
                const int64_t start = group * partials_per_group;
                float mean = partial_means[start];
                float m2 = partial_m2s[start];
                float merged_count = static_cast<float>(Tile);
                for (int64_t partial = 1;
                     partial < partials_per_group;
                     ++partial) {
                    const float next_count =
                        merged_count + static_cast<float>(Tile);
                    const float delta =
                        partial_means[start + partial] - mean;
                    mean += delta * static_cast<float>(Tile) / next_count;
                    m2 += partial_m2s[start + partial] + delta * delta *
                        merged_count * static_cast<float>(Tile) / next_count;
                    merged_count = next_count;
                }
                const float variance =
                    m2 / static_cast<float>(group_elements);
                means[group] = mean;
                rstds[group] = 1.0f / sycl::sqrt(variance + eps);
            });
    };
    utils::submit_kernel(
        finalize_cgf, device, "group_norm_seedvr_bmg_finalize_moments");

    auto fused_params_cgf = [&](sycl::handler& handler) {
        handler.parallel_for<SeedVRGroupNormFusedParamsBMGKernel<Tile>>(
            sycl::range<1>(static_cast<size_t>(total_channels)),
            [=](sycl::item<1> item) {
                const int64_t global_channel = item.get_id(0);
                const int64_t batch_index = global_channel / channels;
                const int64_t channel = global_channel % channels;
                const int64_t group =
                    batch_index * groups + channel / channels_per_group;
                const float scale =
                    rstds[group] * static_cast<float>(weight[channel]);
                scales[global_channel] = scale;
                shifts[global_channel] =
                    static_cast<float>(bias[channel]) -
                    means[group] * scale;
            });
    };
    utils::submit_kernel(
        fused_params_cgf, device, "group_norm_seedvr_bmg_fused_params");

    auto normalize_cgf = [&](sycl::handler& handler) {
        handler.parallel_for<SeedVRGroupNormNormalizeBMGKernel<Tile>>(
            sycl::range<2>(
                static_cast<size_t>(total_channels),
                static_cast<size_t>(spatial)),
            [=](sycl::item<2> item) {
                const int64_t global_channel = item.get_id(0);
                const int64_t batch_index = global_channel / channels;
                const int64_t channel = global_channel % channels;
                const int64_t position = item.get_id(1);
                const int64_t input_offset =
                    batch_index * input_batch_stride +
                    channel * input_channel_stride + position;
                const int64_t output_offset =
                    global_channel * spatial + position;
                output[output_offset] = static_cast<fp16>(
                    static_cast<float>(input[input_offset]) *
                        scales[global_channel] +
                    shifts[global_channel]);
            });
    };
    utils::submit_kernel(
        normalize_cgf, device, "group_norm_seedvr_bmg_normalize");
}

}  // namespace

torch::Tensor group_norm_seedvr_bmg(
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
        input.dim() == 4,
        "SeedVR BMG GroupNorm input must be [T,C,H,W]");
    TORCH_CHECK(
        groups == kSeedVRGroups,
        "SeedVR BMG GroupNorm requires 32 groups");
    TORCH_CHECK(
        input.scalar_type() == torch::kFloat16 &&
            weight.scalar_type() == torch::kFloat16 &&
            bias.scalar_type() == torch::kFloat16,
        "SeedVR BMG GroupNorm requires FP16 input, weight, and bias");
    TORCH_CHECK(
        weight.is_contiguous() && bias.is_contiguous(),
        "SeedVR BMG GroupNorm requires contiguous weight and bias");
    TORCH_CHECK(
        weight.dim() == 1 && weight.numel() == input.size(1),
        "weight must match the channel dimension");
    TORCH_CHECK(
        bias.dim() == 1 && bias.numel() == input.size(1),
        "bias must match the channel dimension");
    TORCH_CHECK(
        eps == 1e-6,
        "SeedVR BMG GroupNorm requires eps=1e-6");
    TORCH_CHECK(
        is_valid_seedvr_shape(input),
        "unsupported SeedVR BMG GroupNorm shape");

    const int64_t batch = input.size(0);
    const int64_t channels = input.size(1);
    const int64_t spatial = input.size(2) * input.size(3);
    TORCH_CHECK(
        input.stride(3) == 1 && input.stride(2) == input.size(3) &&
            input.stride(0) == spatial &&
            input.stride(1) == batch * spatial,
        "SeedVR BMG GroupNorm requires temporal-interleaved N/C strides");
    const int64_t tile =
        spatial >= kSeedVRTileLarge ? kSeedVRTileLarge : kSeedVRTileSmall;
    const int64_t group_elements = channels / groups * spatial;
    TORCH_CHECK(
        group_elements % tile == 0 && spatial % tile == 0,
        "SeedVR BMG GroupNorm shape is not divisible by its tile");

    const int64_t partials_per_group = group_elements / tile;
    auto float_options = input.options().dtype(torch::kFloat32);
    auto output = torch::empty(input.sizes(), input.options());
    auto partial_means =
        torch::empty({batch, groups, partials_per_group}, float_options);
    auto partial_m2s =
        torch::empty({batch, groups, partials_per_group}, float_options);
    auto means = torch::empty({batch, groups}, float_options);
    auto rstds = torch::empty({batch, groups}, float_options);
    auto scales = torch::empty({batch, channels}, float_options);
    auto shifts = torch::empty({batch, channels}, float_options);

    if (tile == kSeedVRTileLarge) {
        launch_group_norm_seedvr_bmg<kSeedVRTileLarge>(
            reinterpret_cast<const fp16*>(input.data_ptr()),
            reinterpret_cast<const fp16*>(weight.data_ptr()),
            reinterpret_cast<const fp16*>(bias.data_ptr()),
            reinterpret_cast<fp16*>(output.data_ptr()),
            partial_means.data_ptr<float>(),
            partial_m2s.data_ptr<float>(),
            means.data_ptr<float>(),
            rstds.data_ptr<float>(),
            scales.data_ptr<float>(),
            shifts.data_ptr<float>(),
            batch,
            channels,
            spatial,
            input.stride(0),
            input.stride(1),
            groups,
            static_cast<float>(eps),
            input.device());
    } else {
        launch_group_norm_seedvr_bmg<kSeedVRTileSmall>(
            reinterpret_cast<const fp16*>(input.data_ptr()),
            reinterpret_cast<const fp16*>(weight.data_ptr()),
            reinterpret_cast<const fp16*>(bias.data_ptr()),
            reinterpret_cast<fp16*>(output.data_ptr()),
            partial_means.data_ptr<float>(),
            partial_m2s.data_ptr<float>(),
            means.data_ptr<float>(),
            rstds.data_ptr<float>(),
            scales.data_ptr<float>(),
            shifts.data_ptr<float>(),
            batch,
            channels,
            spatial,
            input.stride(0),
            input.stride(1),
            groups,
            static_cast<float>(eps),
            input.device());
    }
    return output;
}

}  // namespace norm
}  // namespace omni_xpu

#endif
