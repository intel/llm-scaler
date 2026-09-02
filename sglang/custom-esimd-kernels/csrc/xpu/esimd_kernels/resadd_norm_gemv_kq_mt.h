/* resadd_norm_gemv_kq_mt.h — M-tiled variant of resadd_norm_gemv_kq.h.
 *
 * Same math, same layouts, same outputs. The only difference is which
 * work-group owns which token.
 *
 * Why a separate kernel
 * ---------------------
 * resadd_norm_gemv_kq.h launches M * blocks work-groups: work-group (t, b)
 * owns token t and output-row block b. Every one of the M tokens therefore
 * re-reads the whole weight block, so weight traffic scales with M. At M = 1
 * (plain decode) that is free, which is why the original kernel is written
 * that way. At M = 4 (speculative-decoding TARGET_VERIFY with
 * num_draft_tokens = 4) it means reading the q4_K/q6_K weights four times, and
 * measurement showed that cost exceeds the dispatch savings: forcing the
 * GEMV-shaped fusions on at M = 4 regressed end-to-end TPOT by 10%.
 *
 * This kernel launches `blocks` work-groups instead. Work-group b owns output
 * row block b for ALL M tokens: it loads each weight tile exactly once and
 * dots it against all M normed activations, which are staged together in SLM.
 * Weight traffic is then independent of M, so the fusion's dispatch savings
 * are kept without paying for them in bandwidth. This is accumulator tiling:
 * the per-token partial sums live in registers (MT * 8 floats per work-item),
 * not in extra memory passes.
 *
 * Cost of the tiling
 * ------------------
 *   - SLM grows from KT to MT * KT fp16, so the host must reject shapes where
 *     that exceeds the per-work-group budget (see RNQM_SLM_MAX).
 *   - Fewer work-groups are launched (blocks instead of M * blocks), so the
 *     grid must still be wide enough to fill the machine. That holds for the
 *     projection shapes this is used on, where `blocks` is in the hundreds.
 *
 * MT is a compile-time template parameter and the host requires M == MT, so
 * the per-token loops fully unroll and every SLM offset is a constant.
 *
 * Included into esimd_kernel.sycl (utils.h provides fp16 + esimd namespace).
 */
#pragma once

// Must match the tile size the host-side q6_K qh pre-shuffle was built for,
// and must match RNQ_VL in resadd_norm_gemv_kq.h.
static constexpr int RNQM_VL   = 512;
static constexpr int RNQM_WGS  = 32;   // work-items per work-group
static constexpr int RNQM_ROWS = 32;   // output rows per work-group

// Per-work-group shared-local-memory budget we are willing to spend. The
// normed activations for all MT tokens have to fit, so this caps MT * KT.
static constexpr int RNQM_SLM_MAX = 64 * 1024;

// Plain-old-data argument pack, so the launcher below can be a function
// template (and therefore use `if constexpr` to discard shapes whose shared
// local memory request would not compile) without repeating 25 arguments.
struct RnqmArgs {
    const fp16*    h;        // [MT, KT]
    const fp16*    res;      // [MT, KT]
    fp16*          nr;       // [MT, KT] new residual (must not alias res)
    const fp16*    nw;       // [KT]
    fp16*          xn;       // [MT, KT] normed activation, may be null

    const uint8_t* q4_w;     // [N4, KT/2]
    const fp16*    q4_sc;    // [N4, KT/32]
    const fp16*    q4_mn;    // [N4, KT/32]
    fp16*          o4;
    int            ld4, of4;

    const uint8_t* q6_ql;    // [N6, KT/2]
    const uint8_t* q6_qh;    // [N6, KT/4] pre-shuffled per RNQM_VL tile
    const fp16*    q6_sc;    // [N6, KT/16]
    fp16*          o6;
    int            ld6, of6;

    const fp16*    wf;       // [NF, KT]
    fp16*          of;
    int            ldf, off;

    float eps;
    int N4, N6, NF;
    int nb4, nb6, blocks;
};

template <int KT, int MT>
struct Resadd_norm_gemv_kq_mt_kernel : RnqmArgs {
    static constexpr int SLM_X   = MT * KT * (int)sizeof(fp16);
    static constexpr int SLM_RED = RNQM_WGS * (int)sizeof(float);

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        slm_init<SLM_X + SLM_RED>();

        const int blk = (int)item.get_group(0);
        const int lid = (int)item.get_local_id(0);

        constexpr int VL    = RNQM_VL;
        constexpr int NTILE = KT / VL;

        // ---- phase 1: residual-add + RMSNorm of every token into SLM -------
        // One token at a time so the cross-work-item reduction slot is reused.
        #pragma unroll
        for (int m = 0; m < MT; m++) {
            const fp16* hr   = h + (size_t)m * KT;
            const fp16* rr   = res + (size_t)m * KT;
            const int sb = m * KT * (int)sizeof(fp16);

            float part = 0.0f;
            for (int t = lid; t < NTILE; t += RNQM_WGS) {
                const int k = t * VL;
                // fp16 add first: the reference rounds the residual to fp16
                // BEFORE taking the fp32 variance, so a fp32 accumulate would
                // not match.
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
            simd<float, RNQM_WGS> parts = slm_block_load<float, RNQM_WGS>(SLM_X);
            const float rstd = 1.0f / sycl::sqrt(
                esimd_detail::sum<float, float, RNQM_WGS>(parts) / (float)KT + eps);

            for (int t = lid; t < NTILE; t += RNQM_WGS) {
                const int k = t * VL;
                simd<float, VL> v =
                    slm_block_load<fp16, VL>(sb + k * (int)sizeof(fp16));
                simd<fp16, VL> xv = convert<fp16>(
                    v * rstd * simd<float, VL>(block_load<fp16, VL>(nw + k)));
                slm_block_store<fp16, VL>(sb + k * (int)sizeof(fp16), xv);
                if (blk == 0 && xn)
                    block_store<fp16, VL>(xn + (size_t)m * KT + k, xv);
            }
            // Also fences the reduction slot before the next token rewrites it.
            barrier();
        }

        // ---- phase 2: one output row per work-item, all MT tokens at once ---
        int kind, rbase, nrow;
        if (blk < nb4)            { kind = 0; rbase = blk * RNQM_ROWS;              nrow = N4; }
        else if (blk < nb4 + nb6) { kind = 1; rbase = (blk - nb4) * RNQM_ROWS;      nrow = N6; }
        else                      { kind = 2; rbase = (blk - nb4 - nb6) * RNQM_ROWS; nrow = NF; }

        const int row = rbase + lid;   // RNQM_ROWS == RNQM_WGS: one row per lane
        if (row >= nrow) return;

        constexpr int VL_HALF = VL / 2;
        constexpr int VL_QTR  = VL / 4;
        constexpr int VL_G32  = VL / 32;
        constexpr int VL_G16  = VL / 16;
        constexpr int K_ITERS = KT / VL;

        // MT independent accumulator chains, each 8-way rotated to keep the
        // fused-multiply-add pipeline busy.
        simd<float, MT * 8> acc(0.0f);
        int ai = 0;

        for (int iter = 0; iter < K_ITERS; iter++) {
            const int k = iter * VL;
            simd<float, VL> weight_f;

            // The weight tile is dequantised ONCE here and reused for every
            // token below. This is the entire point of the M-tiled variant.
            if (kind == 0) {
                // ---- q4_K: nibble * scale - min, per-32 group ----
                simd<uint8_t, VL_HALF> w_data = block_load<uint8_t, VL_HALF>(
                    q4_w + (size_t)row * (KT / 2) + k / 2);
                simd<float, VL_G32> sc_f =
                    block_load<fp16, VL_G32>(q4_sc + (size_t)row * (KT / 32) + k / 32);
                simd<float, VL_G32> mn_f =
                    block_load<fp16, VL_G32>(q4_mn + (size_t)row * (KT / 32) + k / 32);
                #pragma unroll
                for (int c = 0; c < VL_HALF / 64; c++) {
                    auto p = w_data.template select<64, 1>(c * 64);
                    simd<float, 64> lo = p & 0x0F;
                    simd<float, 64> hi = (p >> 4) & 0x0F;
                    weight_f.template select<64, 2>(c * 128) = lo;
                    weight_f.template select<64, 2>(c * 128 + 1) = hi;
                }
                #pragma unroll
                for (int sb2 = 0; sb2 < VL_G32; sb2++) {
                    const float s = sc_f[sb2], m2 = mn_f[sb2];
                    weight_f.template select<32, 1>(sb2 * 32) =
                        weight_f.template select<32, 1>(sb2 * 32) * s - m2;
                }
            } else if (kind == 1) {
                // ---- q6_K: (ql | qh<<4) - 32, scaled per-16 group ----
                simd<uint8_t, VL_HALF> ql_data = block_load<uint8_t, VL_HALF>(
                    q6_ql + (size_t)row * (KT / 2) + k / 2);
                simd<uint8_t, VL_QTR> qh_data = block_load<uint8_t, VL_QTR>(
                    q6_qh + (size_t)row * (KT / 4) + k / 4);
                simd<float, VL_G16> sc_f =
                    block_load<fp16, VL_G16>(q6_sc + (size_t)row * (KT / 16) + k / 16);
                #pragma unroll
                for (int c = 0; c < VL_HALF / 64; c++) {
                    auto p = ql_data.template select<64, 1>(c * 64);
                    simd<float, 64> lo = p & 0x0F;
                    simd<float, 64> hi = (p >> 4) & 0x0F;
                    weight_f.template select<64, 2>(c * 128) = lo;
                    weight_f.template select<64, 2>(c * 128 + 1) = hi;
                }
                // qh is pre-shuffled so field p covers elements [p*VL_QTR, ...)
                #pragma unroll
                for (int p = 0; p < 4; p++) {
                    simd<float, VL_QTR> ef = (qh_data >> (2 * p)) & 3;
                    weight_f.template select<VL_QTR, 1>(p * VL_QTR) += ef * 16.0f;
                }
                #pragma unroll
                for (int sb2 = 0; sb2 < VL_G16; sb2++) {
                    const float s = sc_f[sb2];
                    weight_f.template select<16, 1>(sb2 * 16) =
                        (weight_f.template select<16, 1>(sb2 * 16) - 32.0f) * s;
                }
            } else {
                weight_f = block_load<fp16, VL>(wf + (size_t)row * KT + k);
            }

            #pragma unroll
            for (int m = 0; m < MT; m++) {
                simd<float, VL> act = slm_block_load<fp16, VL>(
                    m * KT * (int)sizeof(fp16) + k * (int)sizeof(fp16));
                acc[m * 8 + ai] += esimd_detail::sum<float, float, VL>(weight_f * act);
            }
            ai = (ai + 1) & 7;
        }

        #pragma unroll
        for (int m = 0; m < MT; m++) {
            simd<float, 8> a = acc.template select<8, 1>(m * 8);
            const fp16 val = (fp16)esimd_detail::sum<float, float, 8>(a);
            if (kind == 0)      o4[(size_t)m * ld4 + of4 + row] = val;
            else if (kind == 1) o6[(size_t)m * ld6 + of6 + row] = val;
            else                of[(size_t)m * ldf + off + row] = val;
        }
    }
};

// Launch helper. Being a function template lets `if constexpr` discard, without
// compiling it, any (KT, MT) pair whose shared-local-memory request exceeds the
// budget -- slm_init<> would otherwise be a hard compile error.
template <int KT, int MT>
inline bool rnqm_launch(const RnqmArgs& a, int blocks, sycl::queue& q) {
    if constexpr (MT * KT * (int)sizeof(fp16) +
                      RNQM_WGS * (int)sizeof(float) > RNQM_SLM_MAX) {
        (void)a; (void)blocks; (void)q;
        return false;
    } else {
        q.submit([&](sycl::handler& hd) {
            hd.parallel_for(
                sycl::nd_range<1>((size_t)blocks * RNQM_WGS, RNQM_WGS),
                Resadd_norm_gemv_kq_mt_kernel<KT, MT>{a});
        });
        return true;
    }
}

// Returns false when the (K, M) pair is unsupported so the caller can fall
// back to the per-token kernel or to the unfused path.
//
// Only the (hidden size, token count) pairs a speculative-decoding verify step
// actually produces are instantiated; every extra pair is another full kernel
// in the binary and another chunk of compile time. Anything else returns false
// and takes the existing path.
inline bool resadd_norm_gemv_kq_mt_host(
    const fp16* h, const fp16* res, fp16* nr, const fp16* nw, fp16* xn,
    const uint8_t* q4_w, const fp16* q4_sc, const fp16* q4_mn,
    fp16* o4, int ld4, int of4,
    const uint8_t* q6_ql, const uint8_t* q6_qh, const fp16* q6_sc,
    fp16* o6, int ld6, int of6,
    const fp16* wf, fp16* of, int ldf, int off,
    float eps, int M, int K, int N4, int N6, int NF, sycl::queue& q) {
    if (K % RNQM_VL != 0) return false;
    if (M < 2) return false;                       // M == 1 uses the plain kernel
    if ((long)M * K * (long)sizeof(fp16) + RNQM_WGS * (long)sizeof(float) >
        RNQM_SLM_MAX)
        return false;

    const int nb4 = (N4 + RNQM_ROWS - 1) / RNQM_ROWS;
    const int nb6 = (N6 + RNQM_ROWS - 1) / RNQM_ROWS;
    const int nbf = (NF + RNQM_ROWS - 1) / RNQM_ROWS;
    const int blocks = (nb4 + nb6 + nbf) > 0 ? (nb4 + nb6 + nbf) : 1;

    const RnqmArgs a{h, res, nr, nw, xn,
                     q4_w, q4_sc, q4_mn, o4, ld4, of4,
                     q6_ql, q6_qh, q6_sc, o6, ld6, of6,
                     wf, of, ldf, off,
                     eps, N4, N6, NF, nb4, nb6, blocks};

#define RNQM_BY_M(KT)                                                         \
    switch (M) {                                                              \
        case 2: return rnqm_launch<KT, 2>(a, blocks, q);                      \
        case 3: return rnqm_launch<KT, 3>(a, blocks, q);                      \
        case 4: return rnqm_launch<KT, 4>(a, blocks, q);                      \
        default: return false;                                                \
    }

    switch (K) {
        case 2048: RNQM_BY_M(2048)
        case 4096: RNQM_BY_M(4096)
        case 5120: RNQM_BY_M(5120)
        case 6144: RNQM_BY_M(6144)
        default: return false;
    }
#undef RNQM_BY_M
}
