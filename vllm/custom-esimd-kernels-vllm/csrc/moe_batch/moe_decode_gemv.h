// ============================================================================
// MoE decode-only expert GEMV (M==1 fast path)
//
// Replaces the DPAS GEMM kernels for decode. Root cause fixed: DPAS path uses
// lsc_load_2d<uint8_t,16,16,1> (16B-wide 2D tiles, 1/4 of BMG's 64B cacheline),
// capping MoE decode at ~316 GB/s vs dense GEMV's ~574. Expert weight is plain
// row-major inside each expert (gate_up [E,2*inter,hidden], down [E,hidden,inter]),
// so 1D block_load<uint8_t,VL> along K reads contiguously like fp8_GEMV_bmg.
//
// One work-item computes one output element via VL-strided 1D loads + tail.
// KS=1 (no K-split) — first version focuses on the load-width fix.
// ============================================================================
#pragma once
#include <sycl/ext/intel/esimd.hpp>

// Forward decl: defined in moe.sycl above the include point's use site.
template<int N>
SYCL_ESIMD_FUNCTION simd<sycl::half, N> fp8e4m3_to_half(simd<uint8_t, N> raw);

// Fast E4M3→half: uint16 bit-twiddle for normals + correct subnormal handling.
// e4m3 bias=7, fp16 bias=15 → exp_fp16 = exp_e4m3 + 8. mant 3b → fp16 mant top.
// Subnormal (e==0): value = mant * 2^-9 (representable as normal fp16).
template<int N>
SYCL_ESIMD_FUNCTION simd<float, N> fp8e4m3_dequant_fast(simd<uint8_t, N> raw) {
    using namespace sycl::ext::intel::esimd;
    simd<uint16_t, N> u = convert<uint16_t>(raw);
    simd<uint16_t, N> sign = (u >> 7) & 1;
    simd<uint16_t, N> e = (u >> 3) & 0xF;
    simd<uint16_t, N> m = u & 0x7;
    // Normal path
    simd<uint16_t, N> norm_bits = (sign << 15) | ((e + 8) << 10) | (m << 7);
    simd<fp16, N> hn = norm_bits.template bit_cast_view<fp16>();
    // Subnormal path: m * 2^-9, with sign
    simd<fp16, N> hs = convert<fp16>(m) * fp16(1.0f / 512.0f);
    simd<fp16, N> hs_signed = hs;
    hs_signed.merge(-hs, sign == 1);
    // Select subnormal where e==0
    simd<fp16, N> out = hn;
    out.merge(hs_signed, e == 0);
    return simd<float, N>(out);
}


// VL_BIG full chunks + one VL_TAIL chunk (VL_TAIL may be 0). KS=1.
// Up + gelu_tanh. gate_up_weight [E, 2*inter, hidden], K = hidden.
template<int VL, int VL_TAIL>
struct MoeUpDecodeGeluTanh {
    const fp16*    x;
    const uint8_t* gate_up_weight;
    const float*   gate_up_scale;
    const int*     selected_experts;
    fp16*          intermediates;   // [top_k, inter]
    int hidden, inter, top_k, fp8_mode;

    void operator()(sycl::nd_item<2> item) const SYCL_ESIMD_KERNEL {
        using namespace sycl::ext::intel::esimd;
        const int route = (int)item.get_global_id(0);
        const int n     = (int)item.get_global_id(1);
        if (n >= inter) return;

        const int two_inter = 2 * inter;
        const int eid = selected_experts[route];
        const uint8_t* wbase = gate_up_weight + (size_t)eid * two_inter * hidden;
        const uint8_t* w_gate = wbase + (size_t)n * hidden;
        const uint8_t* w_up   = wbase + (size_t)(inter + n) * hidden;

        const int kp_full = (hidden / VL) * VL;
        simd<float, VL> g_acc(0.f), u_acc(0.f);
        for (int k = 0; k < kp_full; k += VL) {
            simd<fp16, VL> xv = block_load<fp16, VL>(x + k);
            simd<float, VL> xf = xv;
            g_acc += xf * fp8e4m3_dequant_fast<VL>((block_load<uint8_t, VL>(w_gate + k)));
            u_acc += xf * fp8e4m3_dequant_fast<VL>((block_load<uint8_t, VL>(w_up + k)));
        }
        float g_sum = reduce<float>(g_acc, std::plus<>());
        float u_sum = reduce<float>(u_acc, std::plus<>());
        if constexpr (VL_TAIL > 0) {
            int kt = kp_full;
            simd<fp16, VL_TAIL> xv = block_load<fp16, VL_TAIL>(x + kt);
            simd<float, VL_TAIL> xf = xv;
            g_sum += reduce<float>(xf * fp8e4m3_dequant_fast<VL_TAIL>((block_load<uint8_t, VL_TAIL>(w_gate + kt))), std::plus<>());
            u_sum += reduce<float>(xf * fp8e4m3_dequant_fast<VL_TAIL>((block_load<uint8_t, VL_TAIL>(w_up + kt))), std::plus<>());
        }

        float scale = gate_up_scale[eid];
        float gs = g_sum * scale, us = u_sum * scale;
        constexpr float sqrt_2_over_pi = 0.7978845608f, coeff = 0.044715f;
        float gs3 = gs*gs*gs;
        float inner = sqrt_2_over_pi * (gs + coeff*gs3);
        float e2 = sycl::exp(2.0f*inner);
        float tanh_v = (e2 - 1.0f)/(e2 + 1.0f);
        float gelu = 0.5f*gs*(1.0f + tanh_v);
        intermediates[(size_t)route*inter + n] = fp16(gelu * us);
    }
};

// Down. down_weight [E, hidden, inter], K = inter.
template<int VL, int VL_TAIL>
struct MoeDownDecode {
    const fp16*    intermediates;   // [top_k, inter]
    const uint8_t* down_weight;
    const float*   down_scale;
    const fp16*    routing_weights;
    const int*     selected_experts;
    fp16*          output;          // [top_k, hidden]
    int hidden, inter, top_k, fp8_mode;

    void operator()(sycl::nd_item<2> item) const SYCL_ESIMD_KERNEL {
        using namespace sycl::ext::intel::esimd;
        const int route = (int)item.get_global_id(0);
        const int h     = (int)item.get_global_id(1);
        if (h >= hidden) return;

        const int eid = selected_experts[route];
        const uint8_t* wrow = down_weight + (size_t)eid * hidden * inter + (size_t)h * inter;
        const fp16* hi = intermediates + (size_t)route * inter;

        const int kp_full = (inter / VL) * VL;
        simd<float, VL> acc(0.f);
        for (int k = 0; k < kp_full; k += VL) {
            simd<fp16, VL> hv = block_load<fp16, VL>(hi + k);
            simd<float, VL> hf = hv;
            acc += hf * fp8e4m3_dequant_fast<VL>((block_load<uint8_t, VL>(wrow + k)));
        }
        float s = reduce<float>(acc, std::plus<>());
        if constexpr (VL_TAIL > 0) {
            int kt = kp_full;
            simd<fp16, VL_TAIL> hv = block_load<fp16, VL_TAIL>(hi + kt);
            simd<float, VL_TAIL> hf = hv;
            s += reduce<float>(hf * fp8e4m3_dequant_fast<VL_TAIL>((block_load<uint8_t, VL_TAIL>(wrow + kt))), std::plus<>());
        }

        float w = (float)routing_weights[route];
        float ds = down_scale[eid];
        output[(size_t)route*hidden + h] = fp16(s * w * ds);
    }
};
