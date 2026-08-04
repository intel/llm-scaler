#include <torch/extension.h>
#include <sycl/sycl.hpp>

#include <cstdint>
#include <limits>
#include <type_traits>

#if defined(OMNI_XPU_ARCH_BMG)
#include "bmg_kernel_policy.h"
#include "device_utils.h"
#endif
#include "utils.h"

using fp16 = sycl::half;
using bf16 = sycl::ext::oneapi::bfloat16;

namespace omni_xpu {
namespace rotary {

namespace {

#ifndef OMNI_KITCHEN_ROPE_PAIR_SAME_SHAPE
#if defined(OMNI_XPU_ARCH_PTL_H)
#define OMNI_KITCHEN_ROPE_PAIR_SAME_SHAPE 1
#elif defined(OMNI_XPU_ARCH_BMG)
#define OMNI_KITCHEN_ROPE_PAIR_SAME_SHAPE 1
#else
#error "Define OMNI_XPU_ARCH_PTL_H or OMNI_XPU_ARCH_BMG"
#endif
#endif

#ifndef OMNI_KITCHEN_ROPE_PAIR_WG_SIZE
#if defined(OMNI_XPU_ARCH_PTL_H)
#define OMNI_KITCHEN_ROPE_PAIR_WG_SIZE 128
#elif defined(OMNI_XPU_ARCH_BMG)
#define OMNI_KITCHEN_ROPE_PAIR_WG_SIZE 32
#else
#error "Define OMNI_XPU_ARCH_PTL_H or OMNI_XPU_ARCH_BMG"
#endif
#endif

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

#if defined(OMNI_XPU_ARCH_BMG)
bool d120_bmg_single_supported(
    const torch::Tensor& x,
    const torch::Tensor& freqs) {
    if (!x.device().is_xpu() || !freqs.device().is_xpu() ||
        x.device() != freqs.device()) {
        return false;
    }
    if (x.scalar_type() != torch::kFloat16 ||
        freqs.scalar_type() != torch::kFloat32) {
        return false;
    }
    if (!x.is_contiguous() || !freqs.is_contiguous()) return false;
    if (x.dim() != 4 || freqs.dim() != 6) return false;
    if (x.size(0) != 1 || x.size(3) != 120) return false;
    if (x.size(2) != 7 && x.size(2) != 28) return false;
    if (x.numel() > std::numeric_limits<uint32_t>::max() ||
        freqs.numel() > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    return freqs.size(0) == 1 &&
           freqs.size(1) == x.size(1) &&
           freqs.size(2) == 1 &&
           freqs.size(3) == 60 &&
           freqs.size(4) == 2 &&
           freqs.size(5) == 2;
}

bool krea2_bmg_pair_supported(
    const torch::Tensor& xq,
    const torch::Tensor& xk,
    const torch::Tensor& freqs) {
    return xq.device().is_xpu() && xk.device().is_xpu() &&
           freqs.device().is_xpu() &&
           xq.device() == xk.device() && xq.device() == freqs.device() &&
           xq.scalar_type() == torch::kBFloat16 &&
           xk.scalar_type() == torch::kBFloat16 &&
           freqs.scalar_type() == torch::kFloat32 &&
           xq.is_contiguous() && xk.is_contiguous() &&
           freqs.is_contiguous() &&
           xq.dim() == 4 && xk.dim() == 4 && freqs.dim() == 6 &&
           xq.sizes() == torch::IntArrayRef({1, 48, 4192, 128}) &&
           xk.sizes() == torch::IntArrayRef({1, 12, 4192, 128}) &&
           freqs.sizes() ==
               torch::IntArrayRef({1, 1, 4192, 64, 2, 2});
}

inline fp16 apply_d120_component(
    float f0,
    float f1,
    fp16 x0,
    fp16 x1) {
    const float product = f0 * static_cast<float>(x0);
    return static_cast<fp16>(
        sycl::fma(f1, static_cast<float>(x1), product));
}

// Boogu's single-input D120 route reuses each token/pair frequency matrix
// across all heads. One work-group per token removes the generic kernel's
// repeated 64-bit index decomposition and redundant frequency loads.
template <uint32_t Heads>
void launch_rope_d120_bmg(
    const torch::Tensor& x,
    const torch::Tensor& freqs,
    torch::Tensor& output) {
    constexpr uint32_t HeadDim = 120;
    constexpr uint32_t Pairs = HeadDim / 2;
    constexpr uint32_t FreqValuesPerToken = Pairs * 4;
    constexpr uint32_t WG = 64;

    const auto* x_ptr = reinterpret_cast<const fp16*>(x.data_ptr());
    const auto* f_ptr = freqs.data_ptr<float>();
    auto* out_ptr = reinterpret_cast<fp16*>(output.data_ptr());
    const uint32_t tokens = static_cast<uint32_t>(x.size(1));

    auto cgf = [&](sycl::handler& handler) {
        handler.parallel_for(
            sycl::nd_range<1>(
                sycl::range<1>(static_cast<size_t>(tokens) * WG),
                sycl::range<1>(WG)),
            [=](sycl::nd_item<1> item) {
                const uint32_t token =
                    static_cast<uint32_t>(item.get_group(0));
                const uint32_t pair =
                    static_cast<uint32_t>(item.get_local_id(0));
                if (pair >= Pairs) return;

                const uint32_t token_x_base = token * Heads * HeadDim;
                const uint32_t f_offset =
                    token * FreqValuesPerToken + pair * 4;
                const float f00 = f_ptr[f_offset];
                const float f01 = f_ptr[f_offset + 1];
                const float f10 = f_ptr[f_offset + 2];
                const float f11 = f_ptr[f_offset + 3];

#pragma unroll
                for (uint32_t head = 0; head < Heads; ++head) {
                    const uint32_t x_offset =
                        token_x_base + head * HeadDim + pair * 2;
                    const fp16 x0 = x_ptr[x_offset];
                    const fp16 x1 = x_ptr[x_offset + 1];
                    out_ptr[x_offset] =
                        apply_d120_component(f00, f01, x0, x1);
                    out_ptr[x_offset + 1] =
                        apply_d120_component(f10, f11, x0, x1);
                }
            });
    };
    utils::submit_kernel(cgf, x.device(), "kitchen_rope_d120_bmg");
}

void dispatch_rope_d120_bmg(
    const torch::Tensor& x,
    const torch::Tensor& freqs,
    torch::Tensor& output) {
    if (x.size(2) == 7) {
        launch_rope_d120_bmg<7>(x, freqs, output);
    } else {
        launch_rope_d120_bmg<28>(x, freqs, output);
    }
}
#endif

template <typename T>
T force_dtype_round(T value) {
    using Bits = std::conditional_t<sizeof(T) == 2, uint16_t, uint32_t>;
    const volatile Bits stored = sycl::bit_cast<Bits>(value);
    const Bits loaded = stored;
    return sycl::bit_cast<T>(loaded);
}

#if defined(OMNI_XPU_ARCH_BMG)
bool b60_exact_row_supported(
    const torch::Tensor& input,
    const torch::Tensor& freqs,
    int64_t sequence) {
    return input.device().is_xpu() && freqs.device().is_xpu() &&
           input.device() == freqs.device() &&
           input.scalar_type() == torch::kBFloat16 &&
           freqs.scalar_type() == torch::kFloat32 &&
           input.is_contiguous() && freqs.is_contiguous() &&
           input.sizes() ==
               torch::IntArrayRef({1, 24, sequence, 128}) &&
           freqs.sizes() ==
               torch::IntArrayRef({1, 1, sequence, 64, 2, 2});
}

inline void apply_b60_exact_row_pair(
    const bf16* source,
    bf16* destination,
    uint32_t offset0,
    uint32_t offset1,
    const float* freq,
    bool split_half) {
    const float x0 = static_cast<float>(source[offset0]);
    const float x1 = static_cast<float>(source[offset1]);
    if (split_half) {
        const float p00 =
            force_dtype_round<float>(freq[0] * x0);
        const float p01 =
            force_dtype_round<float>(freq[1] * x1);
        const float p10 =
            force_dtype_round<float>(freq[2] * x0);
        const float p11 =
            force_dtype_round<float>(freq[3] * x1);
        destination[offset0] = static_cast<bf16>(p00 + p01);
        destination[offset1] = static_cast<bf16>(p10 + p11);
    } else {
        const float p00 =
            force_dtype_round<float>(freq[0] * x0);
        const float p10 =
            force_dtype_round<float>(freq[2] * x0);
        const float y0 = sycl::fma(freq[1], x1, p00);
        const float y1 = sycl::fma(freq[3], x1, p10);
        destination[offset0] =
            static_cast<bf16>(force_dtype_round<float>(y0));
        destination[offset1] =
            static_cast<bf16>(force_dtype_round<float>(y1));
    }
}

template<
    uint32_t Sequence,
    bool Pair,
    uint32_t PairsPerWorkItem,
    uint32_t WorkGroupSize>
void launch_b60_exact_row(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& freqs,
    torch::Tensor& out_query,
    torch::Tensor& out_key,
    bool split_half) {
    constexpr uint32_t Heads = 24;
    constexpr uint32_t HeadDim = 128;
    constexpr uint32_t Pairs = HeadDim / 2;
    constexpr uint32_t Rows = Heads * Sequence;
    static_assert(PairsPerWorkItem > 0);
    static_assert(Pairs % PairsPerWorkItem == 0);
    static_assert(WorkGroupSize == Pairs / PairsPerWorkItem);

    const auto* query_ptr =
        reinterpret_cast<const bf16*>(query.data_ptr());
    const auto* key_ptr = Pair
        ? reinterpret_cast<const bf16*>(key.data_ptr())
        : nullptr;
    const auto* freq_ptr = freqs.data_ptr<float>();
    auto* out_query_ptr =
        reinterpret_cast<bf16*>(out_query.data_ptr());
    auto* out_key_ptr = Pair
        ? reinterpret_cast<bf16*>(out_key.data_ptr())
        : nullptr;

    auto cgf = [&](sycl::handler& handler) {
        handler.parallel_for(
            sycl::nd_range<1>(
                sycl::range<1>(
                    static_cast<size_t>(Rows) * WorkGroupSize),
                sycl::range<1>(WorkGroupSize)),
            [=](sycl::nd_item<1> item) {
                const uint32_t row =
                    static_cast<uint32_t>(item.get_group(0));
                const uint32_t token = row % Sequence;
                const uint32_t first_pair =
                    static_cast<uint32_t>(item.get_local_id(0)) *
                    PairsPerWorkItem;
                const uint32_t base = row * HeadDim;

#pragma unroll
                for (uint32_t lane = 0;
                     lane < PairsPerWorkItem;
                     ++lane) {
                    const uint32_t pair = first_pair + lane;
                    const uint32_t offset0 =
                        base + (split_half ? pair : pair * 2);
                    const uint32_t offset1 =
                        base + (
                            split_half
                            ? Pairs + pair
                            : pair * 2 + 1);
                    const uint32_t freq_offset =
                        (token * Pairs + pair) * 4;
                    apply_b60_exact_row_pair(
                        query_ptr,
                        out_query_ptr,
                        offset0,
                        offset1,
                        freq_ptr + freq_offset,
                        split_half);
                    if constexpr (Pair) {
                        apply_b60_exact_row_pair(
                            key_ptr,
                            out_key_ptr,
                            offset0,
                            offset1,
                            freq_ptr + freq_offset,
                            split_half);
                    }
                }
            });
    };
    utils::submit_kernel(
        cgf, query.device(), "kitchen_rope_b60_exact_row");
}

inline void apply_krea2_bmg_pair(
    const bf16* source,
    bf16* destination,
    uint32_t offset,
    float f00,
    float f01,
    float f10,
    float f11) {
    const float x0 = static_cast<float>(source[offset]);
    const float x1 = static_cast<float>(source[offset + 1]);
    const float p00 = force_dtype_round<float>(f00 * x0);
    const float p10 = force_dtype_round<float>(f10 * x0);
    const float y0 = sycl::fma(f01, x1, p00);
    const float y1 = sycl::fma(f11, x1, p10);
    destination[offset] =
        static_cast<bf16>(force_dtype_round<float>(y0));
    destination[offset + 1] =
        static_cast<bf16>(force_dtype_round<float>(y1));
}

// Krea2's Q/K use [1,H,S,D] with H48/H12 and broadcast one frequency
// matrix over the heads. Align one WG64 with every token/head row to remove
// the generic flat dispatch's repeated 64-bit index decomposition.
void launch_rope_krea2_bmg(
    const torch::Tensor& xq,
    const torch::Tensor& xk,
    const torch::Tensor& freqs,
    torch::Tensor& out_q,
    torch::Tensor& out_k) {
    constexpr uint32_t Sequence = 4192;
    constexpr uint32_t QueryHeads = 48;
    constexpr uint32_t KeyHeads = 12;
    constexpr uint32_t HeadDim = 128;
    constexpr uint32_t Pairs = HeadDim / 2;
    constexpr uint32_t WG = Pairs;

    const auto* q_ptr = reinterpret_cast<const bf16*>(xq.data_ptr());
    const auto* k_ptr = reinterpret_cast<const bf16*>(xk.data_ptr());
    const auto* f_ptr = freqs.data_ptr<float>();
    auto* oq_ptr = reinterpret_cast<bf16*>(out_q.data_ptr());
    auto* ok_ptr = reinterpret_cast<bf16*>(out_k.data_ptr());

    auto cgf = [&](sycl::handler& handler) {
        handler.parallel_for(
            sycl::nd_range<1>(
                sycl::range<1>(
                    static_cast<size_t>(Sequence) *
                    (QueryHeads + KeyHeads) * WG),
                sycl::range<1>(WG)),
            [=](sycl::nd_item<1> item) {
                const uint32_t group =
                    static_cast<uint32_t>(item.get_group(0));
                const uint32_t token =
                    group / (QueryHeads + KeyHeads);
                const uint32_t logical_head =
                    group % (QueryHeads + KeyHeads);
                const uint32_t pair =
                    static_cast<uint32_t>(item.get_local_id(0));
                const uint32_t fbase = (token * Pairs + pair) * 4;
                const float f00 = f_ptr[fbase];
                const float f01 = f_ptr[fbase + 1];
                const float f10 = f_ptr[fbase + 2];
                const float f11 = f_ptr[fbase + 3];

                if (logical_head < QueryHeads) {
                    const uint32_t offset =
                        ((logical_head * Sequence + token) * HeadDim) +
                        pair * 2;
                    apply_krea2_bmg_pair(
                        q_ptr, oq_ptr, offset, f00, f01, f10, f11);
                } else {
                    const uint32_t key_head =
                        logical_head - QueryHeads;
                    const uint32_t offset =
                        ((key_head * Sequence + token) * HeadDim) +
                        pair * 2;
                    apply_krea2_bmg_pair(
                        k_ptr, ok_ptr, offset, f00, f01, f10, f11);
                }
            });
    };
    utils::submit_kernel(
        cgf, xq.device(), "kitchen_rope_krea2_bmg");
}
#endif

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

// Same-shape Q/K pairs can share the expensive logical-index calculation and
// each 2x2 frequency matrix.  Keeping one pair per work-item avoids the
// register-pressure collapse seen when several pairs are unrolled together.
template <typename InputT, typename FreqT, bool SplitHalf>
void launch_rope_pair_same_shape(
    const torch::Tensor& xq,
    const torch::Tensor& xk,
    const torch::Tensor& freqs,
    torch::Tensor& out_q,
    torch::Tensor& out_k) {
    const auto* q_ptr = reinterpret_cast<const InputT*>(xq.data_ptr());
    const auto* k_ptr = reinterpret_cast<const InputT*>(xk.data_ptr());
    const auto* f_ptr = reinterpret_cast<const FreqT*>(freqs.data_ptr());
    auto* oq_ptr = reinterpret_cast<InputT*>(out_q.data_ptr());
    auto* ok_ptr = reinterpret_cast<InputT*>(out_k.data_ptr());

    const int64_t x1 = xq.size(1), x2 = xq.size(2), xd = xq.size(3);
    const int64_t pairs = xd / 2;
    const int64_t total = xq.numel() / 2;
    const int64_t f0 = freqs.size(0), f1 = freqs.size(1), f2 = freqs.size(2);
    const int64_t fpairs = freqs.size(3);
    constexpr int64_t WG = OMNI_KITCHEN_ROPE_PAIR_WG_SIZE;
    const int64_t padded = (total + WG - 1) / WG * WG;

    auto cgf = [&](sycl::handler& handler) {
        handler.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(padded), sycl::range<1>(WG)),
            [=](sycl::nd_item<1> item) {
                int64_t logical = item.get_global_id(0);
                if (logical >= total) return;
                const int64_t pair = logical % pairs;
                logical /= pairs;
                const int64_t i2 = logical % x2;
                logical /= x2;
                const int64_t i1 = logical % x1;
                const int64_t i0 = logical / x1;
                const int64_t base = ((i0 * x1 + i1) * x2 + i2) * xd;
                const int64_t xoff0 = base + (SplitHalf ? pair : pair * 2);
                const int64_t xoff1 =
                    base + (SplitHalf ? pairs + pair : pair * 2 + 1);

                const int64_t fi0 = f0 == 1 ? 0 : i0;
                const int64_t fi1 = f1 == 1 ? 0 : i1;
                const int64_t fi2 = f2 == 1 ? 0 : i2;
                const int64_t fpair = fpairs == 1 ? 0 : pair;
                const int64_t fbase =
                    (((fi0 * f1 + fi1) * f2 + fi2) * fpairs + fpair) * 4;

                const FreqT f00 = f_ptr[fbase];
                const FreqT f01 = f_ptr[fbase + 1];
                const FreqT f10 = f_ptr[fbase + 2];
                const FreqT f11 = f_ptr[fbase + 3];
                const FreqT q0 = static_cast<FreqT>(q_ptr[xoff0]);
                const FreqT q1 = static_cast<FreqT>(q_ptr[xoff1]);
                const FreqT k0 = static_cast<FreqT>(k_ptr[xoff0]);
                const FreqT k1 = static_cast<FreqT>(k_ptr[xoff1]);

                if constexpr (SplitHalf) {
                    const FreqT q00 = force_dtype_round<FreqT>(f00 * q0);
                    const FreqT q01 = force_dtype_round<FreqT>(f01 * q1);
                    const FreqT q10 = force_dtype_round<FreqT>(f10 * q0);
                    const FreqT q11 = force_dtype_round<FreqT>(f11 * q1);
                    const FreqT k00 = force_dtype_round<FreqT>(f00 * k0);
                    const FreqT k01 = force_dtype_round<FreqT>(f01 * k1);
                    const FreqT k10 = force_dtype_round<FreqT>(f10 * k0);
                    const FreqT k11 = force_dtype_round<FreqT>(f11 * k1);
                    oq_ptr[xoff0] = static_cast<InputT>(q00 + q01);
                    oq_ptr[xoff1] = static_cast<InputT>(q10 + q11);
                    ok_ptr[xoff0] = static_cast<InputT>(k00 + k01);
                    ok_ptr[xoff1] = static_cast<InputT>(k10 + k11);
                } else {
                    const FreqT q00 = force_dtype_round<FreqT>(f00 * q0);
                    const FreqT q10 = force_dtype_round<FreqT>(f10 * q0);
                    const FreqT k00 = force_dtype_round<FreqT>(f00 * k0);
                    const FreqT k10 = force_dtype_round<FreqT>(f10 * k0);
                    const float qy0 = sycl::fma(
                        static_cast<float>(f01), static_cast<float>(q1),
                        static_cast<float>(q00));
                    const float qy1 = sycl::fma(
                        static_cast<float>(f11), static_cast<float>(q1),
                        static_cast<float>(q10));
                    const float ky0 = sycl::fma(
                        static_cast<float>(f01), static_cast<float>(k1),
                        static_cast<float>(k00));
                    const float ky1 = sycl::fma(
                        static_cast<float>(f11), static_cast<float>(k1),
                        static_cast<float>(k10));
                    oq_ptr[xoff0] = static_cast<InputT>(
                        force_dtype_round<FreqT>(static_cast<FreqT>(qy0)));
                    oq_ptr[xoff1] = static_cast<InputT>(
                        force_dtype_round<FreqT>(static_cast<FreqT>(qy1)));
                    ok_ptr[xoff0] = static_cast<InputT>(
                        force_dtype_round<FreqT>(static_cast<FreqT>(ky0)));
                    ok_ptr[xoff1] = static_cast<InputT>(
                        force_dtype_round<FreqT>(static_cast<FreqT>(ky1)));
                }
            });
    };
    utils::submit_kernel(cgf, xq.device(), "kitchen_rope_pair_same_shape_sycl");
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

template <typename InputT, bool SplitHalf>
void dispatch_freq_pair_same_shape(
    const torch::Tensor& xq,
    const torch::Tensor& xk,
    const torch::Tensor& freqs,
    torch::Tensor& out_q,
    torch::Tensor& out_k) {
    switch (freqs.scalar_type()) {
        case torch::kFloat32:
            launch_rope_pair_same_shape<InputT, float, SplitHalf>(
                xq, xk, freqs, out_q, out_k);
            break;
        case torch::kFloat16:
            launch_rope_pair_same_shape<InputT, fp16, SplitHalf>(
                xq, xk, freqs, out_q, out_k);
            break;
        case torch::kBFloat16:
            launch_rope_pair_same_shape<InputT, bf16, SplitHalf>(
                xq, xk, freqs, out_q, out_k);
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

template <bool SplitHalf>
void dispatch_input_pair_same_shape(
    const torch::Tensor& xq,
    const torch::Tensor& xk,
    const torch::Tensor& freqs,
    torch::Tensor& out_q,
    torch::Tensor& out_k) {
    switch (xq.scalar_type()) {
        case torch::kFloat32:
            dispatch_freq_pair_same_shape<float, SplitHalf>(
                xq, xk, freqs, out_q, out_k);
            break;
        case torch::kFloat16:
            dispatch_freq_pair_same_shape<fp16, SplitHalf>(
                xq, xk, freqs, out_q, out_k);
            break;
        case torch::kBFloat16:
            dispatch_freq_pair_same_shape<bf16, SplitHalf>(
                xq, xk, freqs, out_q, out_k);
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
#if defined(OMNI_XPU_ARCH_BMG)
    auto& queue = utils::get_queue(x.device());
    if (device::use_b60_kernel_profile(queue) &&
        b60_exact_row_supported(x, freqs, 4352)) {
        launch_b60_exact_row<
            4352,
            false,
            device::B60KernelPolicy::kitchen_rope_pairs_per_work_item,
            device::B60KernelPolicy::kitchen_rope_work_group_size>(
                x,
                unused,
                freqs,
                output,
                unused,
                split_half);
        return output;
    }
    if (!split_half && d120_bmg_single_supported(x, freqs)) {
        if (output.numel() != 0) {
            dispatch_rope_d120_bmg(x, freqs, output);
        }
        return output;
    }
#endif
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
#if defined(OMNI_XPU_ARCH_BMG)
    auto& queue = utils::get_queue(xq.device());
    if (device::use_b60_kernel_profile(queue) &&
        b60_exact_row_supported(xq, freqs, 4096) &&
        b60_exact_row_supported(xk, freqs, 4096)) {
        launch_b60_exact_row<
            4096,
            true,
            device::B60KernelPolicy::kitchen_rope_pairs_per_work_item,
            device::B60KernelPolicy::kitchen_rope_work_group_size>(
                xq,
                xk,
                freqs,
                out_q,
                out_k,
                split_half);
        return {out_q, out_k};
    }
    if (!split_half && krea2_bmg_pair_supported(xq, xk, freqs)) {
        launch_rope_krea2_bmg(xq, xk, freqs, out_q, out_k);
        return {out_q, out_k};
    }
#endif
#if OMNI_KITCHEN_ROPE_PAIR_SAME_SHAPE
    if (xq.sizes() == xk.sizes()) {
        if (split_half) {
            dispatch_input_pair_same_shape<true>(
                xq, xk, freqs, out_q, out_k);
        } else {
            dispatch_input_pair_same_shape<false>(
                xq, xk, freqs, out_q, out_k);
        }
        return {out_q, out_k};
    }
#endif
    if (split_half) {
        dispatch_input<true, true>(xq, xk, freqs, out_q, out_k);
    } else {
        dispatch_input<false, true>(xq, xk, freqs, out_q, out_k);
    }
    return {out_q, out_k};
}

}  // namespace rotary
}  // namespace omni_xpu
