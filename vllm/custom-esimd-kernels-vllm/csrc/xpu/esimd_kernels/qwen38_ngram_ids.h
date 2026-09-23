#pragma once

#include <cstdint>
#include <sycl/ext/intel/experimental/esimd/math.hpp>

#include "utils.h"

namespace qwen38_ngram_ids {

class DecodeKernel;

ESIMD_INLINE simd<std::uint64_t, 16> multiply_high_u64(
    simd<std::uint64_t, 16> lhs,
    simd<std::uint64_t, 16> rhs) {
    namespace experimental_esimd = sycl::ext::intel::experimental::esimd;

    auto lhs_dw = lhs.bit_cast_view<std::uint32_t>();
    auto rhs_dw = rhs.bit_cast_view<std::uint32_t>();
    const simd<std::uint32_t, 16> lhs_lo = lhs_dw.select<16, 2>(0);
    const simd<std::uint32_t, 16> lhs_hi = lhs_dw.select<16, 2>(1);
    const simd<std::uint32_t, 16> rhs_lo = rhs_dw.select<16, 2>(0);
    const simd<std::uint32_t, 16> rhs_hi = rhs_dw.select<16, 2>(1);

    simd<std::uint32_t, 16> p00_lo;
    simd<std::uint32_t, 16> p01_lo;
    simd<std::uint32_t, 16> p10_lo;
    simd<std::uint32_t, 16> p11_lo;
    const simd<std::uint32_t, 16> p00_hi =
        experimental_esimd::imul(p00_lo, lhs_lo, rhs_lo);
    const simd<std::uint32_t, 16> p01_hi =
        experimental_esimd::imul(p01_lo, lhs_lo, rhs_hi);
    const simd<std::uint32_t, 16> p10_hi =
        experimental_esimd::imul(p10_lo, lhs_hi, rhs_lo);
    const simd<std::uint32_t, 16> p11_hi =
        experimental_esimd::imul(p11_lo, lhs_hi, rhs_hi);

    const simd<std::uint64_t, 16> middle =
        simd<std::uint64_t, 16>(p00_hi) +
        simd<std::uint64_t, 16>(p01_lo) +
        simd<std::uint64_t, 16>(p10_lo);
    simd<std::uint64_t, 16> high = simd<std::uint64_t, 16>(p11_lo);
    high += simd<std::uint64_t, 16>(p11_hi) << 32;
    high += simd<std::uint64_t, 16>(p01_hi);
    high += simd<std::uint64_t, 16>(p10_hi);
    high += middle >> 32;
    return high;
}

ESIMD_INLINE simd<std::uint64_t, 16> exact_constant_remainder(
    simd<std::uint64_t, 16> dividend) {
    const simd<std::uint64_t, 16> vocab({
        20000003ULL, 20000023ULL, 20000033ULL, 20000047ULL,
        20000059ULL, 20000063ULL, 20000069ULL, 20000077ULL,
        20000081ULL, 20000093ULL, 20000107ULL, 20000147ULL,
        20000153ULL, 20000159ULL, 20000161ULL, 20000171ULL});
    const simd<std::uint64_t, 16> magic({
        0xad7f2572edd5e756ULL, 0xad7f094d2dd3860cULL,
        0xad7efb3a4f348292ULL, 0xad7ee7864c48f7c0ULL,
        0xad7ed6a2dd81b3bfULL, 0xad7ed101b8e02bbbULL,
        0xad7ec8900234b56bULL, 0xad7ebd4db9d4513dULL,
        0xad7eb7ac95dcca18ULL, 0xad7ea6c92ad8e05eULL,
        0xad7e93152facb510ULL, 0xad7e5ac9d9b6d43aULL,
        0xad7e525827b16249ULL, 0xad7e49e67600f083ULL,
        0xad7e4715e583ae2dULL, 0xad7e3903139f0d61ULL});

    simd<std::uint64_t, 16> quotient = multiply_high_u64(dividend, magic);
    quotient = (quotient + ((dividend - quotient) >> 1)) >> 24;
    return dividend - quotient * vocab;
}

inline void decode_host(
    const std::int64_t* input_ids,
    const std::int64_t* ngram_context,
    const std::int64_t* layer_multipliers,
    std::int64_t* ngram_ids,
    sycl::queue& q) {
    q.submit([&](sycl::handler& h) {
        h.parallel_for<DecodeKernel>(
            sycl::range<1>(1), [=](sycl::id<1>) SYCL_ESIMD_KERNEL {
                constexpr int kHeads = 16;
                constexpr int kBigramHeads = 8;

                const auto* input_u =
                    reinterpret_cast<const std::uint64_t*>(input_ids);
                const auto* context_u =
                    reinterpret_cast<const std::uint64_t*>(ngram_context);
                const auto* multipliers_u =
                    reinterpret_cast<const std::uint64_t*>(layer_multipliers);
                auto* output_u = reinterpret_cast<std::uint64_t*>(ngram_ids);

                const simd<std::uint64_t, 1> current_v =
                    block_load<std::uint64_t, 1>(input_u);
                const simd<std::uint64_t, 2> context_v =
                    block_load<std::uint64_t, 2>(context_u);
                const simd<std::uint64_t, 3> multipliers_v =
                    block_load<std::uint64_t, 3>(multipliers_u);

                const std::uint64_t current = current_v[0];
                const std::uint64_t previous_2 = context_v[0];
                const std::uint64_t previous = context_v[1];
                const std::uint64_t bigram =
                    current * multipliers_v[0] ^ previous * multipliers_v[1];
                const std::uint64_t trigram =
                    bigram ^ previous_2 * multipliers_v[2];

                simd<std::uint64_t, kHeads> mixed(bigram);
                const simd<std::uint32_t, kHeads> lanes(0, 1);
                mixed.merge(simd<std::uint64_t, kHeads>(trigram),
                            lanes >= kBigramHeads);

                const simd_mask<kHeads> negative = (mixed >> 63) != 0;
                const simd<std::uint64_t, kHeads> magnitude = (~mixed) + 1;
                simd<std::uint64_t, kHeads> dividend = mixed;
                dividend.merge(magnitude, negative);

                const simd<std::uint64_t, kHeads> vocab({
                    20000003ULL, 20000023ULL, 20000033ULL, 20000047ULL,
                    20000059ULL, 20000063ULL, 20000069ULL, 20000077ULL,
                    20000081ULL, 20000093ULL, 20000107ULL, 20000147ULL,
                    20000153ULL, 20000159ULL, 20000161ULL, 20000171ULL});
                const simd<std::uint64_t, kHeads> offsets({
                    0ULL, 20000003ULL, 40000026ULL, 60000059ULL,
                    80000106ULL, 100000165ULL, 120000228ULL, 140000297ULL,
                    160000374ULL, 180000455ULL, 200000548ULL, 220000655ULL,
                    240000802ULL, 260000955ULL, 280001114ULL, 300001275ULL});

                const simd<std::uint64_t, kHeads> magnitude_remainder =
                    exact_constant_remainder(dividend);
                simd<std::uint64_t, kHeads> negative_remainder =
                    vocab - magnitude_remainder;
                negative_remainder.merge(simd<std::uint64_t, kHeads>(0),
                                         magnitude_remainder == 0);

                simd<std::uint64_t, kHeads> remainder = magnitude_remainder;
                remainder.merge(negative_remainder, negative);
                block_store<std::uint64_t, kHeads>(
                    output_u, remainder + offsets);
            });
    });
}

}  // namespace qwen38_ngram_ids
