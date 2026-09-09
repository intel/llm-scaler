/* resadd_norm_gemv_q4k_silu.h — fused (residual-add + GemmaRMSNorm + q4_K
 * gate_up GEMV + SiluAndMul) for the GGUF dense-MLP decode path (ESIMD).
 *
 *   nr[t,:] = fp16( h[t,:] + res[t,:] )                   (new residual)
 *   v       = float(nr[t,:])
 *   xn      = fp16( v * rsqrt(mean(v^2) + eps) * nw )     (nw = 1 + gemma w)
 *   g[j]    = fp16( xn @ dequant_q4_K(w)[j]^T )           j in [0, I)
 *   u[j]    = fp16( xn @ dequant_q4_K(w)[I + j]^T )
 *   y[t,j]  = fp16( silu(float(g[j])) * float(u[j]) )
 *
 * This replaces three dispatches per layer (gemma_fused_add_rmsnorm, one
 * esimd_gemv_q4_k over the merged [2I, K] gate_up matrix, and silu_and_mul)
 * with one. On a launch-bound GGUF decode step that is the entire win; the
 * arithmetic is unchanged.
 *
 * The merged gate_up weight is row-concatenated in output order, i.e. rows
 * [0, I) are gate and rows [I, 2I) are up (MergedColumnParallelLinear with
 * output_sizes [I, I]). One work-item therefore owns output column j and walks
 * BOTH row j and row I + j, reading the shared normed activation from SLM once
 * per K-tile instead of twice.
 *
 * Phase 1 (residual add + norm into SLM) is identical to resadd_norm_gemv_kq.h;
 * see that file for why the normed vector is staged in SLM rather than
 * recomputed per work-group.
 *
 * Numerics: g and u are rounded to fp16 before the activation, matching the
 * unfused path where silu_and_mul consumes the fp16 GEMV output. The silu
 * itself is evaluated in fp32, as SiluAndMul does.
 *
 * Included into esimd_kernel.sycl (utils.h provides fp16 + esimd namespace).
 */
#pragma once

// Phase-2 K-tile. Unlike the q6_K path there is no pre-shuffle constraint here
// (q4_K carries no high-bit plane), so this is chosen purely for register
// pressure: each work-item holds one activation tile plus one weight tile, and
// 256 keeps that at 2 KB rather than 4 KB now that two rows are walked per
// work-item.
static constexpr int RNS_VL   = 256;
static constexpr int RNS_WGS  = 32;   // work-items per work-group
static constexpr int RNS_ROWS = 32;   // output columns per work-group

template <int KT>
struct Resadd_norm_gemv_q4k_silu_kernel {
    const fp16*    h;      // [M, KT]
    const fp16*    res;    // [M, KT]
    fp16*          nr;     // [M, KT] new residual (must not alias res)
    const fp16*    nw;     // [KT]

    const uint8_t* q4_w;   // [2I, KT/2] nibble-interleaved
    const fp16*    q4_sc;  // [2I, KT/32]
    const fp16*    q4_mn;  // [2I, KT/32]
    fp16*          y;      // [M, I]

    float eps;
    int M, I;
    int blocks;

    static constexpr int SLM_X   = KT * (int)sizeof(fp16);
    static constexpr int SLM_RED = RNS_WGS * (int)sizeof(float);

    // One q4_K row dotted against the SLM-resident normed activation.
    inline float row_dot(int row) const {
        constexpr int VL     = RNS_VL;
        constexpr int VL_H   = VL / 2;
        constexpr int VL_G32 = VL / 32;
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

        // ---- phase 1: cooperative residual-add + RMSNorm into SLM ----------
        constexpr int NTILE = KT / RNS_VL;
        const fp16* hr = h + (size_t)token * KT;
        const fp16* rr = res + (size_t)token * KT;

        float part = 0.0f;
        for (int t = lid; t < NTILE; t += RNS_WGS) {
            const int k = t * RNS_VL;
            // fp16 add first: the reference rounds the residual to fp16 BEFORE
            // taking the fp32 variance, so a fp32 accumulate would not match.
            simd<fp16, RNS_VL> vh = block_load<fp16, RNS_VL>(hr + k) +
                                    block_load<fp16, RNS_VL>(rr + k);
            simd<float, RNS_VL> v = vh;
            part += esimd_detail::sum<float, float, RNS_VL>(v * v);
            slm_block_store<fp16, RNS_VL>(k * sizeof(fp16), vh);
            if (blk == 0) block_store<fp16, RNS_VL>(nr + (size_t)token * KT + k, vh);
        }

        slm_block_store<float, 1>(SLM_X + lid * sizeof(float), simd<float, 1>(part));
        barrier();
        simd<float, RNS_WGS> parts = slm_block_load<float, RNS_WGS>(SLM_X);
        const float rstd = 1.0f / sycl::sqrt(
            esimd_detail::sum<float, float, RNS_WGS>(parts) / (float)KT + eps);

        for (int t = lid; t < NTILE; t += RNS_WGS) {
            const int k = t * RNS_VL;
            simd<float, RNS_VL> v = slm_block_load<fp16, RNS_VL>(k * sizeof(fp16));
            slm_block_store<fp16, RNS_VL>(k * sizeof(fp16), convert<fp16>(
                v * rstd * simd<float, RNS_VL>(block_load<fp16, RNS_VL>(nw + k))));
        }
        barrier();

        // ---- phase 2: one output column per work-item ----------------------
        const int j = blk * RNS_ROWS + lid;   // RNS_ROWS == RNS_WGS
        if (j >= I) return;

        const fp16 gh = (fp16)row_dot(j);
        const fp16 uh = (fp16)row_dot(I + j);
        const float gf = (float)gh;
        y[(size_t)token * I + j] =
            (fp16)(gf / (1.0f + sycl::exp(-gf)) * (float)uh);
    }
};

// Returns false when the shape is unsupported so the caller can fall back to
// the separate norm + GEMV + silu_and_mul path.
inline bool resadd_norm_gemv_q4k_silu_host(
    const fp16* h, const fp16* res, fp16* nr, const fp16* nw,
    const uint8_t* q4_w, const fp16* q4_sc, const fp16* q4_mn, fp16* y,
    float eps, int M, int K, int I, sycl::queue& q) {
    if (K % RNS_VL != 0 || I <= 0) return false;

    const int blocks = (I + RNS_ROWS - 1) / RNS_ROWS;

#define LAUNCH_RNS(KT)                                                        \
    q.submit([&](sycl::handler& hd) {                                         \
        hd.parallel_for(                                                      \
            sycl::nd_range<1>((size_t)M * blocks * RNS_WGS, RNS_WGS),         \
            Resadd_norm_gemv_q4k_silu_kernel<KT>{                             \
                h, res, nr, nw, q4_w, q4_sc, q4_mn, y,                        \
                eps, M, I, blocks});                                          \
    });                                                                       \
    return true;

    switch (K) {
        case 2048: LAUNCH_RNS(2048)
        case 2560: LAUNCH_RNS(2560)
        case 3072: LAUNCH_RNS(3072)
        case 4096: LAUNCH_RNS(4096)
        case 5120: LAUNCH_RNS(5120)
        case 6144: LAUNCH_RNS(6144)
        default: return false;
    }
#undef LAUNCH_RNS
}
