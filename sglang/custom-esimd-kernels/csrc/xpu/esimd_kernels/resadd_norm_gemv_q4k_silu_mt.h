/* resadd_norm_gemv_q4k_silu_mt.h — M-tiled variant of
 * resadd_norm_gemv_q4k_silu.h.
 *
 * Same math, same layouts, same output. See resadd_norm_gemv_kq_mt.h for the
 * full rationale; in short, the original launches M * blocks work-groups so
 * each of the M tokens re-reads the whole gate_up weight matrix, and at the
 * M = 4 of a speculative-decoding verify step that bandwidth cost is larger
 * than the dispatch saving. Here work-group b owns output columns
 * [b*ROWS, (b+1)*ROWS) for ALL M tokens, dequantises each weight tile once,
 * and dots it against all M normed activations held in shared local memory.
 *
 * As in the M = 1 kernel, one work-item owns output column j and walks both
 * gate row j and up row I + j, so the shared activation tile is read once per
 * K-tile per token rather than twice.
 *
 * Included into esimd_kernel.sycl (utils.h provides fp16 + esimd namespace).
 */
#pragma once

static constexpr int RNSM_VL   = 256;
static constexpr int RNSM_WGS  = 32;   // work-items per work-group
static constexpr int RNSM_ROWS = 32;   // output columns per work-group
static constexpr int RNSM_SLM_MAX = 64 * 1024;

struct RnsmArgs {
    const fp16*    h;      // [M, KT]
    const fp16*    res;    // [M, KT]
    fp16*          nr;     // [M, KT] new residual (must not alias res)
    const fp16*    nw;     // [KT]

    const uint8_t* q4_w;   // [2I, KT/2] nibble-interleaved
    const fp16*    q4_sc;  // [2I, KT/32]
    const fp16*    q4_mn;  // [2I, KT/32]
    fp16*          y;      // [M, I]

    float eps;
    int I;
    int blocks;
};

template <int KT, int MT>
struct Resadd_norm_gemv_q4k_silu_mt_kernel : RnsmArgs {
    static constexpr int SLM_X   = MT * KT * (int)sizeof(fp16);
    static constexpr int SLM_RED = RNSM_WGS * (int)sizeof(float);

    // One q4_K row dotted against all MT shared-local-memory activations.
    // `out` receives one partial-sum vector per token.
    inline void row_dot_mt(int row, simd<float, MT>& out) const {
        constexpr int VL      = RNSM_VL;
        constexpr int VL_H    = VL / 2;
        constexpr int VL_G32  = VL / 32;
        constexpr int K_ITERS = KT / VL;

        simd<float, MT * 8> acc(0.0f);
        int ai = 0;
        for (int iter = 0; iter < K_ITERS; iter++) {
            const int k = iter * VL;
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

            #pragma unroll
            for (int t = 0; t < MT; t++) {
                simd<float, VL> act = slm_block_load<fp16, VL>(
                    t * KT * (int)sizeof(fp16) + k * (int)sizeof(fp16));
                acc[t * 8 + ai] += esimd_detail::sum<float, float, VL>(weight_f * act);
            }
            ai = (ai + 1) & 7;
        }

        #pragma unroll
        for (int t = 0; t < MT; t++) {
            simd<float, 8> a = acc.template select<8, 1>(t * 8);
            out[t] = esimd_detail::sum<float, float, 8>(a);
        }
    }

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        slm_init<SLM_X + SLM_RED>();

        const int blk = (int)item.get_group(0);
        const int lid = (int)item.get_local_id(0);

        constexpr int VL    = RNSM_VL;
        constexpr int NTILE = KT / VL;

        // ---- phase 1: residual-add + RMSNorm of every token into SLM -------
        #pragma unroll
        for (int m = 0; m < MT; m++) {
            const fp16* hr   = h + (size_t)m * KT;
            const fp16* rr   = res + (size_t)m * KT;
            const int sb = m * KT * (int)sizeof(fp16);

            float part = 0.0f;
            for (int t = lid; t < NTILE; t += RNSM_WGS) {
                const int k = t * VL;
                // fp16 add first: the reference rounds the residual to fp16
                // BEFORE taking the fp32 variance.
                simd<fp16, VL> vh = block_load<fp16, VL>(hr + k) +
                                    block_load<fp16, VL>(rr + k);
                simd<float, VL> v = vh;
                part += esimd_detail::sum<float, float, VL>(v * v);
                slm_block_store<fp16, VL>(sb + k * (int)sizeof(fp16), vh);
                if (blk == 0)
                    block_store<fp16, VL>(nr + (size_t)m * KT + k, vh);
            }

            slm_block_store<float, 1>(SLM_X + lid * (int)sizeof(float),
                                      simd<float, 1>(part));
            barrier();
            simd<float, RNSM_WGS> parts = slm_block_load<float, RNSM_WGS>(SLM_X);
            const float rstd = 1.0f / sycl::sqrt(
                esimd_detail::sum<float, float, RNSM_WGS>(parts) / (float)KT + eps);

            for (int t = lid; t < NTILE; t += RNSM_WGS) {
                const int k = t * VL;
                simd<float, VL> v =
                    slm_block_load<fp16, VL>(sb + k * (int)sizeof(fp16));
                slm_block_store<fp16, VL>(sb + k * (int)sizeof(fp16), convert<fp16>(
                    v * rstd * simd<float, VL>(block_load<fp16, VL>(nw + k))));
            }
            // Also fences the reduction slot before the next token rewrites it.
            barrier();
        }

        // ---- phase 2: one output column per work-item, all MT tokens -------
        const int j = blk * RNSM_ROWS + lid;   // RNSM_ROWS == RNSM_WGS
        if (j >= I) return;

        simd<float, MT> gs, us;
        row_dot_mt(j, gs);
        row_dot_mt(I + j, us);

        #pragma unroll
        for (int m = 0; m < MT; m++) {
            // g and u are rounded to fp16 before the activation, matching the
            // unfused path where silu_and_mul consumes the fp16 GEMV output.
            const float gf = (float)(fp16)gs[m];
            const float uf = (float)(fp16)us[m];
            y[(size_t)m * I + j] = (fp16)(gf / (1.0f + sycl::exp(-gf)) * uf);
        }
    }
};

// See resadd_norm_gemv_kq_mt.h: being a function template lets `if constexpr`
// discard oversized shared-local-memory requests before they reach slm_init<>.
template <int KT, int MT>
inline bool rnsm_launch(const RnsmArgs& a, int blocks, sycl::queue& q) {
    if constexpr (MT * KT * (int)sizeof(fp16) +
                      RNSM_WGS * (int)sizeof(float) > RNSM_SLM_MAX) {
        (void)a; (void)blocks; (void)q;
        return false;
    } else {
        q.submit([&](sycl::handler& hd) {
            hd.parallel_for(
                sycl::nd_range<1>((size_t)blocks * RNSM_WGS, RNSM_WGS),
                Resadd_norm_gemv_q4k_silu_mt_kernel<KT, MT>{a});
        });
        return true;
    }
}

// Returns false when the (K, M) pair is unsupported so the caller can fall
// back to the per-token kernel or to the unfused path.
inline bool resadd_norm_gemv_q4k_silu_mt_host(
    const fp16* h, const fp16* res, fp16* nr, const fp16* nw,
    const uint8_t* q4_w, const fp16* q4_sc, const fp16* q4_mn, fp16* y,
    float eps, int M, int K, int I, sycl::queue& q) {
    if (K % RNSM_VL != 0 || I <= 0) return false;
    if (M < 2) return false;                       // M == 1 uses the plain kernel
    if ((long)M * K * (long)sizeof(fp16) + RNSM_WGS * (long)sizeof(float) >
        RNSM_SLM_MAX)
        return false;

    const int blocks = (I + RNSM_ROWS - 1) / RNSM_ROWS;
    const RnsmArgs a{h, res, nr, nw, q4_w, q4_sc, q4_mn, y, eps, I, blocks};

#define RNSM_BY_M(KT)                                                         \
    switch (M) {                                                              \
        case 2: return rnsm_launch<KT, 2>(a, blocks, q);                      \
        case 3: return rnsm_launch<KT, 3>(a, blocks, q);                      \
        case 4: return rnsm_launch<KT, 4>(a, blocks, q);                      \
        default: return false;                                                \
    }

    switch (K) {
        case 2048: RNSM_BY_M(2048)
        case 4096: RNSM_BY_M(4096)
        case 5120: RNSM_BY_M(5120)
        case 6144: RNSM_BY_M(6144)
        default: return false;
    }
#undef RNSM_BY_M
}
