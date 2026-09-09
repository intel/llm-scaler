/* norm_add_norm_gemv_q4k_gelu.h — fused (RMSNorm + residual-add + RMSNorm +
 * q4_K gate_up GEMV + GeluAndMul) for the gemma-4 dense-MLP GGUF decode path.
 *
 *   inv1     = rsqrt(mean(h^2) + eps1)
 *   nr[t,:]  = fp16( fp16(h[t,:] * inv1 * w1) + res[t,:] )   (new residual)
 *   inv2     = rsqrt(mean(nr^2) + eps2)
 *   xn       = fp16( nr * inv2 * w2 )
 *   g[j]     = fp16( xn @ dequant_q4_K(W)[j]^T )            j in [0, I)
 *   u[j]     = fp16( xn @ dequant_q4_K(W)[I + j]^T )
 *   y[t,j]   = fp16( gelu_tanh(float(g[j])) * float(u[j]) )
 *
 * This replaces three dispatches per gemma-4 decoder layer (esimd_norm_add_norm,
 * one esimd_gemv_q4_k over the merged [2I, K] gate_up matrix, and
 * gelu_tanh_and_mul) with one. gemma-4-31B decode is launch-bound, so on 60
 * layers that is 120 of ~1000 dispatches per token.
 *
 * Relation to the two kernels it descends from:
 *
 *   - norm_add_norm_gemv_gelu.h (fp8) has the same math but assigns ONE
 *     work-item per output column and re-derives both norms inside that
 *     work-item. That costs 3 full re-reads of the activation per output
 *     column; at gemma-4-31B's I=10752 per rank it would dwarf the weight
 *     traffic. It is only viable at gemma-3n's tiny hidden_size=2816.
 *   - resadd_norm_gemv_q4k_silu.h (qwen3.6 GGUF) has the right structure: a
 *     work-group of RNS_WGS work-items cooperatively computes the norm once
 *     and stages the normed activation in SLM, so the activation is read O(1)
 *     times per work-group rather than once per column. This file keeps that
 *     structure and swaps in gemma-4's two-norm prologue and gelu_tanh tail.
 *
 * The merged gate_up weight is row-concatenated in output order (rows [0, I)
 * gate, rows [I, 2I) up), so one work-item owns column j and walks both row j
 * and row I + j against the same SLM-resident activation.
 *
 * Numerics: every fp16 rounding of the unfused path is reproduced — `nr` is
 * rounded before the second variance is taken, `xn` is rounded before the dot
 * products, and g/u are rounded before the activation — so the result is
 * bit-comparable with norm_add_norm + esimd_gemv_q4_k + gelu_tanh_and_mul.
 *
 * Included into esimd_kernel.sycl (utils.h provides fp16 + esimd namespace).
 */
#pragma once

// Phase-2 K-tile. q4_K carries no pre-shuffled high-bit plane, so this is free
// to be chosen for register pressure alone; 256 keeps one activation tile plus
// one weight tile at ~2 KB while two rows are walked per work-item. It must
// also divide every supported hidden size, which 256 does (the GGUF super-block
// is 256, so K % 256 == 0 holds for any k-quant tensor).
static constexpr int NNG_VL   = 256;
static constexpr int NNG_WGS  = 32;   // work-items per work-group
static constexpr int NNG_ROWS = 32;   // output columns per work-group

template <int KT>
struct Norm_add_norm_gemv_q4k_gelu_kernel {
    const fp16*    h;      // [M, KT] attention output
    const fp16*    res;    // [M, KT] residual in
    fp16*          nr;     // [M, KT] residual out (may alias res)
    const fp16*    w1;     // [KT] post_attention_layernorm weight
    const fp16*    w2;     // [KT] pre_feedforward_layernorm weight

    const uint8_t* q4_w;   // [2I, KT/2] nibble-interleaved
    const fp16*    q4_sc;  // [2I, KT/32]
    const fp16*    q4_mn;  // [2I, KT/32]
    fp16*          y;      // [M, I]

    float eps1, eps2;
    int M, I;
    int blocks;

    static constexpr int SLM_X   = KT * (int)sizeof(fp16);
    static constexpr int SLM_RED = NNG_WGS * (int)sizeof(float);

    // One q4_K row dotted against the SLM-resident normed activation.
    inline float row_dot(int row) const {
        constexpr int VL      = NNG_VL;
        constexpr int VL_H    = VL / 2;
        constexpr int VL_G32  = VL / 32;
        constexpr int K_ITERS = KT / VL;

        simd<float, 8> acc(0.0f);
        int ai = 0;
        for (int iter = 0; iter < K_ITERS; iter++) {
            const int k = iter * VL;
            simd<float, VL> act = slm_block_load<fp16, VL>(k * sizeof(fp16));
            simd<float, VL> weight_f;

            simd<uint8_t, VL_H> w_data = block_load<uint8_t, VL_H>(
                q4_w + (size_t)row * (KT / 2) + k / 2);
            simd<float, VL_G32> sc_f =
                block_load<fp16, VL_G32>(q4_sc + (size_t)row * (KT / 32) + k / 32);
            simd<float, VL_G32> mn_f =
                block_load<fp16, VL_G32>(q4_mn + (size_t)row * (KT / 32) + k / 32);
            #pragma unroll
            for (int c = 0; c < VL_H / 64; c++) {
                auto p = w_data.template select<64, 1>(c * 64);
                simd<float, 64> lo = p & 0x0F;
                simd<float, 64> hi = (p >> 4) & 0x0F;
                weight_f.template select<64, 2>(c * 128) = lo;
                weight_f.template select<64, 2>(c * 128 + 1) = hi;
            }
            #pragma unroll
            for (int sb = 0; sb < VL_G32; sb++) {
                const float s = sc_f[sb], m = mn_f[sb];
                weight_f.template select<32, 1>(sb * 32) =
                    weight_f.template select<32, 1>(sb * 32) * s - m;
            }

            acc[ai] += esimd_detail::sum<float, float, VL>(weight_f * act);
            ai = (ai + 1) & 7;
        }
        return esimd_detail::sum<float, float, 8>(acc);
    }

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        slm_init<SLM_X + SLM_RED>();

        const int gid   = (int)item.get_group(0);
        const int lid   = (int)item.get_local_id(0);
        const int token = gid / blocks;
        const int blk   = gid % blocks;

        constexpr int NTILE = KT / NNG_VL;
        const fp16* hr = h + (size_t)token * KT;
        const fp16* rr = res + (size_t)token * KT;

        // ---- phase 1a: variance of the raw attention output ----------------
        float part = 0.0f;
        for (int t = lid; t < NTILE; t += NNG_WGS) {
            const int k = t * NNG_VL;
            simd<float, NNG_VL> v = block_load<fp16, NNG_VL>(hr + k);
            part += esimd_detail::sum<float, float, NNG_VL>(v * v);
        }
        slm_block_store<float, 1>(SLM_X + lid * sizeof(float), simd<float, 1>(part));
        barrier();
        simd<float, NNG_WGS> p1 = slm_block_load<float, NNG_WGS>(SLM_X);
        const float inv1 = 1.0f / sycl::sqrt(
            esimd_detail::sum<float, float, NNG_WGS>(p1) / (float)KT + eps1);
        barrier();  // p1 consumed before phase 1b overwrites the scratch

        // ---- phase 1b: new residual (fp16) + its variance, staged in SLM ----
        part = 0.0f;
        for (int t = lid; t < NTILE; t += NNG_WGS) {
            const int k = t * NNG_VL;
            // Two fp16 roundings, both present in the unfused path: RMSNorm
            // returns fp16, and the residual add is then done in fp16. Doing
            // the whole expression in fp32 would change the second variance.
            simd<fp16, NNG_VL> nrm = convert<fp16>(
                simd<float, NNG_VL>(block_load<fp16, NNG_VL>(hr + k)) * inv1 *
                simd<float, NNG_VL>(block_load<fp16, NNG_VL>(w1 + k)));
            simd<fp16, NNG_VL> upd = nrm + block_load<fp16, NNG_VL>(rr + k);
            simd<float, NNG_VL> v = upd;
            part += esimd_detail::sum<float, float, NNG_VL>(v * v);
            slm_block_store<fp16, NNG_VL>(k * sizeof(fp16), upd);
            if (blk == 0) block_store<fp16, NNG_VL>(nr + (size_t)token * KT + k, upd);
        }
        slm_block_store<float, 1>(SLM_X + lid * sizeof(float), simd<float, 1>(part));
        barrier();
        simd<float, NNG_WGS> p2 = slm_block_load<float, NNG_WGS>(SLM_X);
        const float inv2 = 1.0f / sycl::sqrt(
            esimd_detail::sum<float, float, NNG_WGS>(p2) / (float)KT + eps2);

        // ---- phase 1c: second norm, in place in SLM ------------------------
        for (int t = lid; t < NTILE; t += NNG_WGS) {
            const int k = t * NNG_VL;
            simd<float, NNG_VL> v = slm_block_load<fp16, NNG_VL>(k * sizeof(fp16));
            slm_block_store<fp16, NNG_VL>(k * sizeof(fp16), convert<fp16>(
                v * inv2 * simd<float, NNG_VL>(block_load<fp16, NNG_VL>(w2 + k))));
        }
        barrier();

        // ---- phase 2: one output column per work-item ----------------------
        const int j = blk * NNG_ROWS + lid;   // NNG_ROWS == NNG_WGS
        if (j >= I) return;

        const fp16 gh = (fp16)row_dot(j);
        const fp16 uh = (fp16)row_dot(I + j);

        // gelu_tanh, evaluated exactly as sgl_kernel::gelu_tanh_and_mul does:
        // 0.5*x*(1+tanh(sqrt(2/pi)*(x + 0.044715 x^3))), with tanh expanded
        // through a clamped exp so the fp8 sibling and this kernel agree.
        const float g = (float)gh;
        constexpr float kSqrt2OverPi = 0.7978845608f;
        constexpr float kCoeff       = 0.044715f;
        float two_inner = 2.0f * kSqrt2OverPi * (g + kCoeff * g * g * g);
        two_inner = two_inner > 30.0f ? 30.0f : two_inner;
        two_inner = two_inner < -30.0f ? -30.0f : two_inner;
        const float e = sycl::exp(two_inner);
        const float gelu = 0.5f * g * (1.0f + (e - 1.0f) / (e + 1.0f));
        y[(size_t)token * I + j] = (fp16)(gelu * (float)uh);
    }
};

// Returns false when the hidden size has no compile-time instantiation, so the
// caller can fall back to the separate norm_add_norm + GEMV + gelu path.
inline bool norm_add_norm_gemv_q4k_gelu_host(
    const fp16* h, const fp16* res, fp16* nr, const fp16* w1, const fp16* w2,
    const uint8_t* q4_w, const fp16* q4_sc, const fp16* q4_mn, fp16* y,
    float eps1, float eps2, int M, int K, int I, sycl::queue& q) {
    if (K % NNG_VL != 0 || I <= 0) return false;
    // `nr` must not alias `res`: phase 1b has every work-group read the whole
    // residual row, but only block 0 writes the updated one, so an in-place
    // buffer would let block 0's stores race the other blocks' loads.
    if (nr == res) return false;

    const int blocks = (I + NNG_ROWS - 1) / NNG_ROWS;

#define LAUNCH_NNG(KT)                                                        \
    q.submit([&](sycl::handler& hd) {                                         \
        hd.parallel_for(                                                      \
            sycl::nd_range<1>((size_t)M * blocks * NNG_WGS, NNG_WGS),         \
            Norm_add_norm_gemv_q4k_gelu_kernel<KT>{                           \
                h, res, nr, w1, w2, q4_w, q4_sc, q4_mn, y,                    \
                eps1, eps2, M, I, blocks});                                   \
    });                                                                       \
    return true;

    switch (K) {
        case 2048: LAUNCH_NNG(2048)
        case 2816: LAUNCH_NNG(2816)
        case 3072: LAUNCH_NNG(3072)
        case 4096: LAUNCH_NNG(4096)
        case 5120: LAUNCH_NNG(5120)
        case 5376: LAUNCH_NNG(5376)
        case 6144: LAUNCH_NNG(6144)
        default: return false;
    }
#undef LAUNCH_NNG
}
