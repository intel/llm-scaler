#include <torch/extension.h>

namespace omni_xpu {
namespace rotary {

bool kitchen_rope_fast_supported(const torch::Tensor& x, const torch::Tensor& freqs);
torch::Tensor apply_kitchen_rope1_fast(
    const torch::Tensor& x, const torch::Tensor& freqs, bool split_half);
std::tuple<torch::Tensor, torch::Tensor> apply_kitchen_rope_fast(
    const torch::Tensor& xq,
    const torch::Tensor& xk,
    const torch::Tensor& freqs,
    bool split_half);

namespace {

void check_inputs(const torch::Tensor& x, const torch::Tensor& freqs_cis) {
    TORCH_CHECK(x.device().is_xpu(), "x must be on XPU");
    TORCH_CHECK(freqs_cis.device().is_xpu(), "freqs_cis must be on XPU");
    TORCH_CHECK(x.device() == freqs_cis.device(), "x and freqs_cis must be on the same XPU");
    TORCH_CHECK(x.dim() >= 3, "x must have at least three dimensions");
    TORCH_CHECK(x.size(-1) % 2 == 0, "x's last dimension must be even");
    TORCH_CHECK(freqs_cis.dim() == x.dim() + 2, "freqs_cis must have x.dim() + 2 dimensions");
    TORCH_CHECK(freqs_cis.size(-1) == 2 && freqs_cis.size(-2) == 2,
                "freqs_cis must end in a 2x2 transform");
}

std::vector<int64_t> paired_shape(const torch::Tensor& x) {
    auto shape = x.sizes().vec();
    shape.pop_back();
    shape.push_back(-1);
    shape.push_back(1);
    shape.push_back(2);
    return shape;
}

std::vector<int64_t> split_shape(const torch::Tensor& x) {
    auto shape = x.sizes().vec();
    shape.pop_back();
    shape.push_back(2);
    shape.push_back(-1);
    return shape;
}

}  // namespace

torch::Tensor apply_kitchen_rope1(
    const torch::Tensor& x,
    const torch::Tensor& freqs_cis) {
    check_inputs(x, freqs_cis);
    if (kitchen_rope_fast_supported(x, freqs_cis)) {
        return apply_kitchen_rope1_fast(x, freqs_cis, false);
    }
    auto paired = x.to(freqs_cis.scalar_type()).reshape(paired_shape(x));
    auto freqs = freqs_cis;
    if (paired.size(2) != 1 && freqs.size(2) != 1 && paired.size(2) != freqs.size(2)) {
        TORCH_CHECK(freqs.size(2) >= paired.size(2),
                    "freqs_cis dimension 2 is shorter than the input");
        freqs = freqs.slice(2, 0, paired.size(2));
    }
    auto output = freqs.select(-1, 0) * paired.select(-1, 0);
    output.addcmul_(freqs.select(-1, 1), paired.select(-1, 1));
    return output.reshape(x.sizes()).to(x.scalar_type());
}

std::tuple<torch::Tensor, torch::Tensor> apply_kitchen_rope(
    const torch::Tensor& xq,
    const torch::Tensor& xk,
    const torch::Tensor& freqs_cis) {
    check_inputs(xq, freqs_cis);
    check_inputs(xk, freqs_cis);
    if (xq.scalar_type() == xk.scalar_type() &&
        kitchen_rope_fast_supported(xq, freqs_cis) &&
        kitchen_rope_fast_supported(xk, freqs_cis)) {
        return apply_kitchen_rope_fast(xq, xk, freqs_cis, false);
    }
    return {
        apply_kitchen_rope1(xq, freqs_cis),
        apply_kitchen_rope1(xk, freqs_cis),
    };
}

torch::Tensor apply_kitchen_rope_split_half1(
    const torch::Tensor& x,
    const torch::Tensor& freqs_cis) {
    check_inputs(x, freqs_cis);
    if (kitchen_rope_fast_supported(x, freqs_cis)) {
        return apply_kitchen_rope1_fast(x, freqs_cis, true);
    }
    auto split = x.reshape(split_shape(x))
                     .movedim(-2, -1)
                     .unsqueeze(-2)
                     .to(freqs_cis.scalar_type());
    auto output = freqs_cis.select(-1, 0) * split.select(-1, 0) +
                  freqs_cis.select(-1, 1) * split.select(-1, 1);
    return output.movedim(-1, -2).reshape(x.sizes()).to(x.scalar_type());
}

std::tuple<torch::Tensor, torch::Tensor> apply_kitchen_rope_split_half(
    const torch::Tensor& xq,
    const torch::Tensor& xk,
    const torch::Tensor& freqs_cis) {
    check_inputs(xq, freqs_cis);
    check_inputs(xk, freqs_cis);
    if (xq.scalar_type() == xk.scalar_type() &&
        kitchen_rope_fast_supported(xq, freqs_cis) &&
        kitchen_rope_fast_supported(xk, freqs_cis)) {
        return apply_kitchen_rope_fast(xq, xk, freqs_cis, true);
    }
    return {
        apply_kitchen_rope_split_half1(xq, freqs_cis),
        apply_kitchen_rope_split_half1(xk, freqs_cis),
    };
}

}  // namespace rotary
}  // namespace omni_xpu
