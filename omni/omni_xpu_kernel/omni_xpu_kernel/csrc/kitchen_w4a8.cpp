#include <cstdint>
#include <optional>

#include <torch/extension.h>
#include <c10/util/Float8_e4m3fn.h>
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/dot_product.hpp>

#include "utils.h"

namespace omni_xpu::int8_ops {

namespace {

template <typename Scale, int TileRows>
class KitchenW4A8Gemv;

// Keep the AWQ GEMV workgroup/reduction structure, but decode onto Kitchen's
// INT8 grid before multiplying. Summation in uint32 preserves s8 GEMM's int32
// wraparound without C++ signed-overflow undefined behavior.
template <typename Scale, int TileRows>
void launch_w4a8(
    const torch::Tensor& input, const torch::Tensor& packed,
    const torch::Tensor& relative_scale,
    const std::optional<torch::Tensor>& codebook,
    int64_t group_size, torch::Tensor& output) {
    constexpr size_t workgroup = 128;
    const int64_t rows = input.size(0);
    const int64_t width = input.size(1);
    const int64_t columns = packed.size(0);
    const int64_t groups = width / group_size;
    const auto* x = input.data_ptr<int8_t>();
    const auto* weights = packed.data_ptr<int8_t>();
    const auto* scales = relative_scale.data_ptr<Scale>();
    const auto* codes = codebook.has_value() ? codebook->data_ptr<float>() : nullptr;
    auto* out = output.data_ptr<int32_t>();
    auto submit = [&](sycl::handler& handler) {
        handler.parallel_for<KitchenW4A8Gemv<Scale, TileRows>>(
            sycl::nd_range<2>(
                sycl::range<2>((rows + TileRows - 1) / TileRows, columns * workgroup),
                sycl::range<2>(1, workgroup)),
            [=](sycl::nd_item<2> item) {
                const int64_t first_row = item.get_group(0) * TileRows;
                const int64_t column = item.get_group(1);
                const int64_t lane = item.get_local_id(1);
                uint32_t sums[TileRows] = {};
                for (int64_t group = lane; group < groups; group += workgroup) {
                    const float scale = static_cast<float>(scales[column * groups + group]);
                    const int64_t first = group * group_size;
                    if constexpr (TileRows == 1) {
                        for (int64_t offset = 0; offset < group_size; offset += 2) {
                            const int64_t k = first + offset;
                            const uint8_t byte = static_cast<uint8_t>(
                                weights[column * (width / 2) + k / 2]);
#pragma unroll
                            for (int half = 0; half < 2; ++half) {
                                const int code = (byte >> (4 * half)) & 15;
                                const float value = (codes ? codes[code] : float(code - 8)) * scale;
                                const float rounded = sycl::rint(value);
                                const int32_t weight = sycl::isnan(rounded) ? 0 :
                                    static_cast<int32_t>(sycl::clamp(rounded, -127.0f, 127.0f));
                                sums[0] += static_cast<uint32_t>(static_cast<int32_t>(
                                    x[first_row * width + k + half]) * weight);
                            }
                        }
                    } else {
                        for (int64_t offset = 0; offset < group_size; offset += 4) {
                            const int64_t k = first + offset;
                            const uint8_t lo = static_cast<uint8_t>(
                                weights[column * (width / 2) + k / 2]);
                            const uint8_t hi = static_cast<uint8_t>(
                                weights[column * (width / 2) + k / 2 + 1]);
                            sycl::vec<int8_t, 4> decoded;
#pragma unroll
                            for (int i = 0; i < 4; ++i) {
                                const int code = ((i < 2 ? lo : hi) >> (4 * (i & 1))) & 15;
                                const float value = (codes ? codes[code] : float(code - 8)) * scale;
                                const float rounded = sycl::rint(value);
                                decoded[i] = sycl::isnan(rounded) ? 0 : static_cast<int8_t>(
                                    sycl::clamp(rounded, -127.0f, 127.0f));
                            }
#pragma unroll
                            for (int r = 0; r < TileRows; ++r) {
                                if (first_row + r < rows) {
                                    sycl::vec<int8_t, 4> activation;
                                    activation.load(0, x + (first_row + r) * width + k);
                                    // Four signed-byte products cannot overflow INT32.
                                    // Accumulate their dot modulo 2^32 as in s8 GEMM.
                                    const int32_t dot = sycl::ext::oneapi::dot_acc(activation, decoded, 0);
                                    sums[r] += static_cast<uint32_t>(dot);
                                }
                            }
                        }
                    }
                }
#pragma unroll
                for (int r = 0; r < TileRows; ++r) {
                    if (first_row + r < rows) {
                        const uint32_t sum = sycl::reduce_over_group(
                            item.get_group(), sums[r], sycl::plus<uint32_t>());
                        if (lane == 0) {
                            out[(first_row + r) * columns + column] = sycl::bit_cast<int32_t>(sum);
                        }
                    }
                }
            });
    };
    utils::submit_kernel(submit, input.device(), "kitchen_w4a8_packed_gemv");
}

template <typename Scale>
void dispatch_w4a8(
    const torch::Tensor& input, const torch::Tensor& packed,
    const torch::Tensor& relative_scale, const std::optional<torch::Tensor>& codebook,
    int64_t group_size, torch::Tensor& output) {
    const int64_t rows = input.size(0);
    // Decode each weight once per row tile. MTP verification reuses it across
    // its rows; independent integer accumulators preserve the INT8 dot contract.
    if (rows == 1 || group_size % 4 != 0) launch_w4a8<Scale, 1>(input, packed, relative_scale, codebook, group_size, output);
    else if (rows == 2) launch_w4a8<Scale, 2>(input, packed, relative_scale, codebook, group_size, output);
    else if (rows <= 4) launch_w4a8<Scale, 4>(input, packed, relative_scale, codebook, group_size, output);
    else launch_w4a8<Scale, 8>(input, packed, relative_scale, codebook, group_size, output);
}

}  // namespace

torch::Tensor mm_w4a8(
    torch::Tensor input, torch::Tensor packed, torch::Tensor relative_scale,
    std::optional<torch::Tensor> codebook, int64_t group_size) {
    TORCH_CHECK(input.device().is_xpu() && input.scalar_type() == at::kChar &&
                    input.dim() == 2 && input.is_contiguous(),
                "W4A8 activation must be contiguous XPU INT8 [M, K]");
    const int64_t width = input.size(1);
    TORCH_CHECK(width > 0 && width % 2 == 0 && group_size >= 4 &&
                    group_size % 2 == 0 && width % group_size == 0,
                "W4A8 requires positive even K and an even group size >=4 dividing K");
    TORCH_CHECK(packed.device() == input.device() && packed.scalar_type() == at::kChar &&
                    packed.dim() == 2 && packed.is_contiguous() &&
                    packed.size(1) == width / 2,
                "W4A8 packed weight must be contiguous same-device INT8 [N, K/2]");
    const int64_t columns = packed.size(0);
    TORCH_CHECK(relative_scale.device() == input.device() && relative_scale.is_contiguous() &&
                    relative_scale.dim() == 2 &&
                    relative_scale.sizes() == at::IntArrayRef({columns, width / group_size}) &&
                    (relative_scale.scalar_type() == at::kFloat ||
                     relative_scale.scalar_type() == at::kFloat8_e4m3fn),
                "W4A8 group scales must be contiguous same-device FP32/FP8 E4M3 [N, K/G]");
    if (codebook.has_value()) {
        TORCH_CHECK(codebook->device() == input.device() &&
                        codebook->scalar_type() == at::kFloat && codebook->is_contiguous() &&
                        codebook->dim() == 1 && codebook->numel() == 16,
                    "W4A8 codebook must be contiguous same-device FP32 [16]");
    }
    auto output = torch::empty({input.size(0), columns}, input.options().dtype(at::kInt));
    if (input.size(0) == 0 || columns == 0) return output;
    if (relative_scale.scalar_type() == at::kFloat) {
        dispatch_w4a8<float>(input, packed, relative_scale, codebook, group_size, output);
    } else {
        dispatch_w4a8<c10::Float8_e4m3fn>(input, packed, relative_scale, codebook, group_size, output);
    }
    return output;
}


}  // namespace omni_xpu::int8_ops
