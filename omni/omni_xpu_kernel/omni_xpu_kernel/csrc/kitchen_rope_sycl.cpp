#include <torch/extension.h>
#include <sycl/sycl.hpp>

#include <cstdint>
#include <type_traits>

#include "utils.h"

using fp16 = sycl::half;
using bf16 = sycl::ext::oneapi::bfloat16;

namespace omni_xpu {
namespace rotary {

namespace {

bool broadcastable_dim(int64_t source, int64_t target, bool allow_longer) {
    return source == 1 || source == target || (allow_longer && source >= target);
}

bool supported_shape(const torch::Tensor& x, const torch::Tensor& freqs) {
    if (!x.device().is_xpu() || !freqs.device().is_xpu() || x.device() != freqs.device()) {
        return false;
    }
    if (x.dim() != 4 || freqs.dim() != 6 || !x.is_contiguous() || !freqs.is_contiguous()) {
        return false;
    }
    if (x.size(3) % 2 != 0 || freqs.size(4) != 2 || freqs.size(5) != 2) return false;
    const int64_t pairs = x.size(3) / 2;
    return broadcastable_dim(freqs.size(0), x.size(0), false) &&
           broadcastable_dim(freqs.size(1), x.size(1), false) &&
           broadcastable_dim(freqs.size(2), x.size(2), true) &&
           broadcastable_dim(freqs.size(3), pairs, false);
}

template <typename T>
T force_dtype_round(T value) {
    using Bits = std::conditional_t<sizeof(T) == 2, uint16_t, uint32_t>;
    const volatile Bits stored = sycl::bit_cast<Bits>(value);
    const Bits loaded = stored;
    return sycl::bit_cast<T>(loaded);
}

template <typename InputT, typename FreqT, bool SplitHalf, bool Pair>
void launch_rope(
    const torch::Tensor& xq,
    const torch::Tensor& xk,
    const torch::Tensor& freqs,
    torch::Tensor& out_q,
    torch::Tensor& out_k) {
    const auto* q_ptr = reinterpret_cast<const InputT*>(xq.data_ptr());
    const auto* k_ptr = Pair ? reinterpret_cast<const InputT*>(xk.data_ptr()) : nullptr;
    const auto* f_ptr = reinterpret_cast<const FreqT*>(freqs.data_ptr());
    auto* oq_ptr = reinterpret_cast<InputT*>(out_q.data_ptr());
    auto* ok_ptr = Pair ? reinterpret_cast<InputT*>(out_k.data_ptr()) : nullptr;

    const int64_t q0 = xq.size(0), q1 = xq.size(1), q2 = xq.size(2), qd = xq.size(3);
    const int64_t k0 = Pair ? xk.size(0) : 0;
    const int64_t k1 = Pair ? xk.size(1) : 0;
    const int64_t k2 = Pair ? xk.size(2) : 0;
    const int64_t kd = Pair ? xk.size(3) : 0;
    const int64_t qpairs = xq.numel() / 2;
    const int64_t kpairs = Pair ? xk.numel() / 2 : 0;
    const int64_t f0 = freqs.size(0), f1 = freqs.size(1), f2 = freqs.size(2);
    const int64_t fpairs = freqs.size(3);
    const int64_t total = qpairs + kpairs;
    constexpr int64_t WG = 256;
    const int64_t padded = (total + WG - 1) / WG * WG;

    auto cgf = [&](sycl::handler& handler) {
        handler.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(padded), sycl::range<1>(WG)),
            [=](sycl::nd_item<1> item) {
                const int64_t gid = item.get_global_id(0);
                if (gid >= total) return;
                const bool key = Pair && gid >= qpairs;
                int64_t logical = key ? gid - qpairs : gid;
                const int64_t xd = key ? kd : qd;
                const int64_t pairs = xd / 2;
                const int64_t x2 = key ? k2 : q2;
                const int64_t x1 = key ? k1 : q1;
                const int64_t pair = logical % pairs;
                logical /= pairs;
                const int64_t i2 = logical % x2;
                logical /= x2;
                const int64_t i1 = logical % x1;
                const int64_t i0 = logical / x1;
                const int64_t base = ((i0 * x1 + i1) * x2 + i2) * xd;
                const int64_t xoff0 = base + (SplitHalf ? pair : pair * 2);
                const int64_t xoff1 = base + (SplitHalf ? pairs + pair : pair * 2 + 1);
                const InputT* src = key ? k_ptr : q_ptr;
                InputT* dst = key ? ok_ptr : oq_ptr;

                const int64_t fi0 = f0 == 1 ? 0 : i0;
                const int64_t fi1 = f1 == 1 ? 0 : i1;
                const int64_t fi2 = f2 == 1 ? 0 : i2;
                const int64_t fpair = fpairs == 1 ? 0 : pair;
                const int64_t fbase = (((fi0 * f1 + fi1) * f2 + fi2) * fpairs + fpair) * 4;
                // Kitchen casts the input to freqs.dtype before applying the
                // transform. Adjacent RoPE uses addcmul_ (fused semantics),
                // while split-half uses two pointwise multiplies followed by
                // an add. Preserve that distinction in reduced precision.
                const FreqT xv0 = static_cast<FreqT>(src[xoff0]);
                const FreqT xv1 = static_cast<FreqT>(src[xoff1]);
                if constexpr (SplitHalf) {
                    const FreqT p00 = force_dtype_round<FreqT>(f_ptr[fbase] * xv0);
                    const FreqT p01 = force_dtype_round<FreqT>(f_ptr[fbase + 1] * xv1);
                    const FreqT p10 = force_dtype_round<FreqT>(f_ptr[fbase + 2] * xv0);
                    const FreqT p11 = force_dtype_round<FreqT>(f_ptr[fbase + 3] * xv1);
                    dst[xoff0] = static_cast<InputT>(p00 + p01);
                    dst[xoff1] = static_cast<InputT>(p10 + p11);
                } else {
                    const FreqT p00 = force_dtype_round<FreqT>(f_ptr[fbase] * xv0);
                    const FreqT p10 = force_dtype_round<FreqT>(f_ptr[fbase + 2] * xv0);
                    const float y0 = sycl::fma(static_cast<float>(f_ptr[fbase + 1]),
                                               static_cast<float>(xv1),
                                               static_cast<float>(p00));
                    const float y1 = sycl::fma(static_cast<float>(f_ptr[fbase + 3]),
                                               static_cast<float>(xv1),
                                               static_cast<float>(p10));
                    const FreqT rounded_y0 = force_dtype_round<FreqT>(static_cast<FreqT>(y0));
                    const FreqT rounded_y1 = force_dtype_round<FreqT>(static_cast<FreqT>(y1));
                    dst[xoff0] = static_cast<InputT>(rounded_y0);
                    dst[xoff1] = static_cast<InputT>(rounded_y1);
                }
            });
    };
    utils::submit_kernel(cgf, xq.device(), Pair ? "kitchen_rope_pair_sycl" : "kitchen_rope_sycl");
}

template <typename InputT, bool SplitHalf, bool Pair>
void dispatch_freq(
    const torch::Tensor& xq,
    const torch::Tensor& xk,
    const torch::Tensor& freqs,
    torch::Tensor& out_q,
    torch::Tensor& out_k) {
    switch (freqs.scalar_type()) {
        case torch::kFloat32:
            launch_rope<InputT, float, SplitHalf, Pair>(xq, xk, freqs, out_q, out_k);
            break;
        case torch::kFloat16:
            launch_rope<InputT, fp16, SplitHalf, Pair>(xq, xk, freqs, out_q, out_k);
            break;
        case torch::kBFloat16:
            launch_rope<InputT, bf16, SplitHalf, Pair>(xq, xk, freqs, out_q, out_k);
            break;
        default:
            TORCH_CHECK(false, "unsupported freqs dtype");
    }
}

template <bool SplitHalf, bool Pair>
void dispatch_input(
    const torch::Tensor& xq,
    const torch::Tensor& xk,
    const torch::Tensor& freqs,
    torch::Tensor& out_q,
    torch::Tensor& out_k) {
    switch (xq.scalar_type()) {
        case torch::kFloat32:
            dispatch_freq<float, SplitHalf, Pair>(xq, xk, freqs, out_q, out_k);
            break;
        case torch::kFloat16:
            dispatch_freq<fp16, SplitHalf, Pair>(xq, xk, freqs, out_q, out_k);
            break;
        case torch::kBFloat16:
            dispatch_freq<bf16, SplitHalf, Pair>(xq, xk, freqs, out_q, out_k);
            break;
        default:
            TORCH_CHECK(false, "unsupported input dtype");
    }
}

}  // namespace

bool kitchen_rope_fast_supported(const torch::Tensor& x, const torch::Tensor& freqs) {
    return supported_shape(x, freqs);
}

torch::Tensor apply_kitchen_rope1_fast(
    const torch::Tensor& x,
    const torch::Tensor& freqs,
    bool split_half) {
    TORCH_CHECK(supported_shape(x, freqs), "unsupported fast RoPE shape");
    auto output = torch::empty_like(x);
    auto unused = torch::Tensor();
    if (split_half) {
        dispatch_input<true, false>(x, unused, freqs, output, unused);
    } else {
        dispatch_input<false, false>(x, unused, freqs, output, unused);
    }
    return output;
}

std::tuple<torch::Tensor, torch::Tensor> apply_kitchen_rope_fast(
    const torch::Tensor& xq,
    const torch::Tensor& xk,
    const torch::Tensor& freqs,
    bool split_half) {
    TORCH_CHECK(supported_shape(xq, freqs) && supported_shape(xk, freqs),
                "unsupported fast RoPE pair shape");
    TORCH_CHECK(xq.scalar_type() == xk.scalar_type(), "query and key dtypes must match");
    auto out_q = torch::empty_like(xq);
    auto out_k = torch::empty_like(xk);
    if (split_half) {
        dispatch_input<true, true>(xq, xk, freqs, out_q, out_k);
    } else {
        dispatch_input<false, true>(xq, xk, freqs, out_q, out_k);
    }
    return {out_q, out_k};
}

}  // namespace rotary
}  // namespace omni_xpu
