// FP32 fragments carried through CuTe's TF32 DPAS layout without narrowing
// global Q/K/V or the online-softmax probabilities to a 16-bit type.
#pragma once

#include <cute/arch/mma_xe.hpp>
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
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

  // Largest float whose ties-to-even TF32 rounding is still finite (0x7f7fefff).
  static constexpr float kHighLimit = 0x1.ffdffep+127f;

  // Ties-to-even TF32 rounding of the clamped value: one hardware conversion
  // on BMG (SPV_INTEL_tensor_float32_conversion). The clamp keeps the high part
  // finite; Inf and NaN inputs still reach the result through a - high.
  CUTE_DEVICE static float high(float value) {
    return sycl::ext::oneapi::experimental::matrix::round_to_tf32(
        sycl::fmin(sycl::fmax(value, -kHighLimit), kHighLimit));
  }

  template <class Vector>
  CUTE_DEVICE static Vector high_vector(Vector source) {
    if constexpr (std::is_same_v<Vector, float>) {
      return high(source);
    } else {
#if defined(__SYCL_DEVICE_ONLY__)
      // Clamp whole register vectors; the conversion is per element.
      const Vector clamped = __builtin_elementwise_min(
          __builtin_elementwise_max(source, Vector(-kHighLimit)), Vector(kHighLimit));
      Vector rounded;
      CUTE_UNROLL
      for (int i = 0; i < int(sizeof(Vector) / sizeof(float)); ++i)
        rounded[i] = sycl::ext::oneapi::experimental::matrix::round_to_tf32(clamped[i]);
      return rounded;
#else
      // CuTe uses marray for the non-executed host instantiation.
      Vector rounded;
      for (int i = 0; i < sizeof(Vector) / sizeof(float); ++i)
        rounded[i] = high(source[i]);
      return rounded;
#endif
    }
  }

  // The low part a - high is exact and is not rounded again: the DPAS reads
  // only its TF32 bits, which drops at most 2^-21 of |a|, the same order as the
  // omitted low*low product.
  template <class Accumulator = CVector>
  CUTE_DEVICE static void fma(DVector& d, AVector const& a,
                              BVector const& b, Accumulator const& c) {
    const AVector ah = high_vector(a);
    const BVector bh = high_vector(b);
    const AVector al = a - ah;
    const BVector bl = b - bh;
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
