// FP32 fragments carried through CuTe's TF32 DPAS layout without narrowing
// global Q/K/V or the online-softmax probabilities to a 16-bit type.
#pragma once

#include <cute/arch/mma_xe.hpp>
#include <sycl/sycl.hpp>
#include <type_traits>

namespace cute {

// Three products retain the cross terms of a two-component TF32 expansion.
// This is a numerical approximation to FP32 GEMM, not bitwise IEEE FP32 FMA.
// The surrounding FMHA reductions and output remain FP32.
template <int M>
struct XE_DPAS_TT<M, float, float, float, float>
    : XE_DPAS_TT_Base<M, float, float, float, float> {
  using Base = XE_DPAS_TT_Base<M, float, float, float, float>;
  using Native = XE_DPAS_TT<M, float, tfloat32_t, tfloat32_t, float>;
  using typename Base::AVector;
  using typename Base::BVector;
  using typename Base::CVector;
  using typename Base::DVector;

  CUTE_DEVICE static float high(float value) {
    const uint32_t bits = sycl::bit_cast<uint32_t>(value);
    if ((bits & 0x7f800000u) == 0x7f800000u) return value;
    uint32_t rounded = (bits + 0xfffu + ((bits >> 13) & 1u)) & ~0x1fffu;
    // Rounding the largest finite inputs must not introduce infinity.
    if ((rounded & 0x7f800000u) == 0x7f800000u) rounded = bits & ~0x1fffu;
    return sycl::bit_cast<float>(rounded);
  }

  template <class Vector>
  CUTE_DEVICE static Vector high_vector(Vector source) {
    if constexpr (std::is_same_v<Vector, float>) {
      return high(source);
    } else {
      // Use whole register vectors to avoid addressable element temporaries
      // in the SYCL lowering of this atom.
#if defined(__SYCL_DEVICE_ONLY__)
      using Bits = intel::vector_t<uint32_t, sizeof(Vector) / sizeof(float)>;
      const auto bits = sycl::bit_cast<Bits>(source);
      auto rounded = (bits + 0xfffu + ((bits >> 13) & 1u)) & ~0x1fffu;
      rounded = ((rounded & 0x7f800000u) == 0x7f800000u)
                    ? (bits & ~0x1fffu) : rounded;
      rounded = ((bits & 0x7f800000u) == 0x7f800000u) ? bits : rounded;
      return sycl::bit_cast<Vector>(rounded);
#else
      // CuTe uses marray for the non-executed host instantiation.
      Vector rounded;
      for (int i = 0; i < sizeof(Vector) / sizeof(float); ++i)
        rounded[i] = high(source[i]);
      return rounded;
#endif
    }
  }

  template <class Accumulator = CVector>
  CUTE_DEVICE static void fma(DVector& d, AVector const& a,
                              BVector const& b, Accumulator const& c) {
    const AVector ah = high_vector(a);
    const BVector bh = high_vector(b);
    const AVector al = high_vector(a - ah);
    const BVector bl = high_vector(b - bh);
    const auto ah_native = sycl::bit_cast<typename Native::AVector>(ah);
    const auto al_native = sycl::bit_cast<typename Native::AVector>(al);
    const auto bh_native = sycl::bit_cast<typename Native::BVector>(bh);
    const auto bl_native = sycl::bit_cast<typename Native::BVector>(bl);
    Native::fma(d, al_native, bh_native, c);
    Native::fma(d, ah_native, bl_native, d);
    Native::fma(d, ah_native, bh_native, d);
  }
};

}  // namespace cute
