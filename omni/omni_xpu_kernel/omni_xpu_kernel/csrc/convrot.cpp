#include <torch/extension.h>
#include <cmath>
#include <map>
#include <mutex>
#include <tuple>

namespace omni_xpu {
namespace int8_ops {

std::tuple<torch::Tensor, torch::Tensor> quantize_int8_rowwise(
    torch::Tensor x,
    int64_t stochastic_rounding);
std::tuple<torch::Tensor, torch::Tensor> quantize_int8_rowwise_fused(
    torch::Tensor x);
torch::Tensor dequantize_int8_convrot_fused(
    torch::Tensor input,
    torch::Tensor scale,
    int64_t group_size);

namespace {

using HadamardKey = std::tuple<int, int, int64_t>;

std::map<HadamardKey, torch::Tensor>& hadamard_cache() {
    static std::map<HadamardKey, torch::Tensor> cache;
    return cache;
}

std::mutex& hadamard_mutex() {
    static std::mutex mutex;
    return mutex;
}

int validate_group_size(int64_t group_size) {
    TORCH_CHECK(group_size >= 4, "ConvRot group size must be a power of 4, got ", group_size);
    int stages = 0;
    for (int64_t size = group_size; size > 1; size /= 4) {
        TORCH_CHECK(size % 4 == 0, "ConvRot group size must be a power of 4, got ", group_size);
        ++stages;
    }
    return stages;
}

torch::Tensor get_hadamard(
    int64_t group_size,
    torch::ScalarType dtype,
    const torch::Device& device) {
    const HadamardKey key{device.index(), static_cast<int>(dtype), group_size};
    std::lock_guard<std::mutex> lock(hadamard_mutex());
    auto& cache = hadamard_cache();
    auto found = cache.find(key);
    if (found != cache.end()) return found->second;

    auto options = torch::TensorOptions().dtype(dtype).device(device);
    auto h4 = torch::tensor(
                  {1.0, 1.0, 1.0, -1.0,
                   1.0, 1.0, -1.0, 1.0,
                   1.0, -1.0, 1.0, 1.0,
                   -1.0, 1.0, 1.0, 1.0},
                  options)
                  .reshape({4, 4});
    auto h = h4;
    int64_t size = 4;
    while (size < group_size) {
        h = (h.unsqueeze(1).unsqueeze(3) * h4.unsqueeze(0).unsqueeze(2))
                .reshape({size * 4, size * 4});
        size *= 4;
    }
    h = (h / std::sqrt(static_cast<double>(group_size))).contiguous();
    cache.emplace(key, h);
    return h;
}

}  // namespace

torch::Tensor rotate_convrot(torch::Tensor input, int64_t group_size) {
    TORCH_CHECK(input.device().is_xpu(), "input must be on XPU");
    TORCH_CHECK(input.is_floating_point(), "input must be floating point");
    TORCH_CHECK(input.dim() >= 2, "input must have at least two dimensions");
    TORCH_CHECK(input.size(-1) % group_size == 0,
                "input features ", input.size(-1),
                " not divisible by group_size ", group_size);
    validate_group_size(group_size);

    const auto original_shape = input.sizes().vec();
    const int64_t groups = input.size(-1) / group_size;
    auto grouped = input.contiguous().reshape({-1, groups, group_size});
    auto h = get_hadamard(group_size, input.scalar_type(), input.device());
    return torch::matmul(grouped, h).reshape(original_shape);
}

std::tuple<torch::Tensor, torch::Tensor> quantize_int8_convrot_weight(
    torch::Tensor weight,
    int64_t group_size,
    int64_t stochastic_rounding) {
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D");
    auto rotated = rotate_convrot(weight, group_size);
    if (stochastic_rounding <= 0 &&
        (rotated.scalar_type() == torch::kBFloat16 ||
         rotated.scalar_type() == torch::kHalf)) {
        return quantize_int8_rowwise_fused(rotated);
    }
    return quantize_int8_rowwise(rotated, stochastic_rounding);
}

torch::Tensor dequantize_int8_convrot_weight(
    torch::Tensor q,
    torch::Tensor scale,
    int64_t group_size) {
    TORCH_CHECK(q.device().is_xpu(), "q must be on XPU");
    TORCH_CHECK(q.scalar_type() == torch::kInt8, "q must be int8");
#if defined(OMNI_XPU_ARCH_PTL_H)
    if (q.dim() == 2 && (group_size == 64 || group_size == 256) &&
        q.size(1) % group_size == 0 && scale.numel() == q.size(0)) {
        return dequantize_int8_convrot_fused(q, scale, group_size);
    }
#endif
    auto dequantized =
        q.to(torch::kFloat32) * scale.to(q.device()).to(torch::kFloat32);
    return rotate_convrot(dequantized, group_size);
}

}  // namespace int8_ops
}  // namespace omni_xpu
