/* resadd_norm_gemv_kq.h — fused (residual-add + GemmaRMSNorm + k-quant GEMVs)
 * for the GGUF decode path on Intel XPU (ESIMD).
 *
 *   nr[t,:]  = fp16( h[t,:] + res[t,:] )                  (new residual)
 *   v        = float(nr[t,:])
 *   xn[t,:]  = fp16( v * rsqrt(mean(v^2) + eps) * nw )    (nw = 1 + gemma w)
 *   o4[t,:]  = xn[t,:] @ dequant_q4_K(q4_w, q4_sc, q4_mn)^T   (N4 rows)
 *   o6[t,:]  = xn[t,:] @ dequant_q6_K(q6_ql, q6_qh, q6_sc)^T  (N6 rows)
 *   of[t,:]  = xn[t,:] @ wf^T                                 (NF fp16 rows)
 *
 * Any of the three matrices may be absent (row count 0), so one op covers all
 * three fusion sites of a Q4_K_M Qwen3.5/3.6 layer:
 *
 *   input_layernorm  + in_proj_qkvz (q6_K run | q4_K run) + in_proj_ba (fp16)
 *   input_layernorm  + qkv          (q4_K run | q6_K run)
 *   post_attn_norm   + gate_up      (q4_K)
 *
 * Why this exists
 * ---------------
 * GGUF decode at M=1 is launch-bound (~37us of host dispatch per kernel call,
 * measured by correlating launches/token against e2e step time), so the win is
 * in removing op CALLS. Each site above costs 2-4 dispatches (a standalone
 * gemma_fused_add_rmsnorm, one GEMV per quant-kind run, and for GDN layers an
 * extra fp16 mm for in_proj_ba); this collapses every site to one.
 *
 * Why the normed vector goes through SLM
 * --------------------------------------
 * The obvious port of Moe_norm_q8_kernel (every work-group re-derives the norm
 * from its own global load of h/res) costs one extra K-element read of the
 * activation per work-group. That is acceptable when the GEMV is a small
 * router (E ~ 128 rows) but not here: with 4 rows per work-group the activation
 * re-read is ~2/3 of the q6_K weight traffic, which would roughly double DRAM
 * traffic on the single largest GEMV of the step. Instead one work-group owns
 * ROWS rows, computes the norm cooperatively across its work-items, and parks
 * the normed activation in SLM (K fp16 = 10 KB at hidden=5120, well inside the
 * 64 KB budget). The activation re-read then amortises over ROWS rows instead
 * of 4, dropping it to <10% of the weight traffic.
 *
 * Layouts are the ones the standalone GEMVs already consume, so no extra
 * repack is needed:
 *   q4_K: q4_w [N4, K/2] u8 nibble-interleaved, q4_sc/q4_mn [N4, K/32] fp16,
 *         asymmetric  w = sc*nibble - mn
 *   q6_K: q6_ql [N6, K/2] u8 nibble-interleaved, q6_qh [N6, K/4] u8 2-bit
 *         PRE-SHUFFLED PER VL=512 TILE (so the K loop below must also step 512),
 *         q6_sc [N6, K/16] fp16, symmetric  w = sc*(v6 - 32)
 *
 * Included into esimd_kernel.sycl (utils.h provides fp16 + esimd namespace).
 */
#pragma once

// Must match the tile size the host-side q6_K qh pre-shuffle was built for.
static constexpr int RNQ_VL   = 512;
static constexpr int RNQ_WGS  = 32;   // work-items per work-group
static constexpr int RNQ_ROWS = 32;   // output rows per work-group

// KT is the exact hidden size, needed at compile time for slm_init.
template <int KT>
struct Resadd_norm_gemv_kq_kernel {
    const fp16*    h;        // [M, KT]
    const fp16*    res;      // [M, KT]
    fp16*          nr;       // [M, KT] new residual (must not alias res)
    const fp16*    nw;       // [KT]
    fp16*          xn;       // [M, KT] normed activation, may be null

    const uint8_t* q4_w;     // [N4, KT/2]
    const fp16*    q4_sc;    // [N4, KT/32]
    const fp16*    q4_mn;    // [N4, KT/32]
    fp16*          o4;       // base of the q4_K output
    int            ld4, of4; // row stride / column offset within it

    const uint8_t* q6_ql;    // [N6, KT/2]
    const uint8_t* q6_qh;    // [N6, KT/4]
    const fp16*    q6_sc;    // [N6, KT/16]
    fp16*          o6;       // base of the q6_K output (may be the same
                             // buffer as o4 with a different column offset)
    int            ld6, of6;

    const fp16*    wf;       // [NF, KT]
    fp16*          of;       // base of the fp16 output
    int            ldf, off;

    float eps;
    int M, N4, N6, NF;
    int nb4, nb6, blocks;    // block ranges: [0,nb4) q4 | [nb4,nb4+nb6) q6 | rest fp16

    static constexpr int SLM_X    = KT * (int)sizeof(fp16);
    static constexpr int SLM_RED  = RNQ_WGS * (int)sizeof(float);

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        slm_init<SLM_X + SLM_RED>();

        const int gid   = (int)item.get_group(0);
        const int lid   = (int)item.get_local_id(0);
        const int token = gid / blocks;
        const int blk   = gid % blocks;

        // ---- phase 1: cooperative residual-add + RMSNorm into SLM ----------
        // Grid-strided over VL-sized tiles so KT only has to be a multiple of
        // RNQ_VL, not of RNQ_VL * RNQ_WGS.
        constexpr int NTILE = KT / RNQ_VL;
        const fp16* hr = h + (size_t)token * KT;
        const fp16* rr = res + (size_t)token * KT;

        float part = 0.0f;
        for (int t = lid; t < NTILE; t += RNQ_WGS) {
            const int k = t * RNQ_VL;
            // fp16 add first: the reference rounds the residual to fp16 BEFORE
            // taking the fp32 variance, so a fp32 accumulate would not match.
            simd<fp16, RNQ_VL> vh = block_load<fp16, RNQ_VL>(hr + k) +
                                    block_load<fp16, RNQ_VL>(rr + k);
            simd<float, RNQ_VL> v = vh;
            part += esimd_detail::sum<float, float, RNQ_VL>(v * v);
            // Stash the pre-norm sum in SLM; phase 1b rescales it in place so
            // h/res are read exactly once per work-group.
            slm_block_store<fp16, RNQ_VL>(k * sizeof(fp16), vh);
            if (blk == 0) block_store<fp16, RNQ_VL>(nr + (size_t)token * KT + k, vh);
        }

        slm_block_store<float, 1>(SLM_X + lid * sizeof(float), simd<float, 1>(part));
        barrier();
        simd<float, RNQ_WGS> parts = slm_block_load<float, RNQ_WGS>(SLM_X);
        const float rstd = 1.0f / sycl::sqrt(
            esimd_detail::sum<float, float, RNQ_WGS>(parts) / (float)KT + eps);

        for (int t = lid; t < NTILE; t += RNQ_WGS) {
            const int k = t * RNQ_VL;
            simd<float, RNQ_VL> v = slm_block_load<fp16, RNQ_VL>(k * sizeof(fp16));
            simd<fp16, RNQ_VL> xv = convert<fp16>(
                v * rstd * simd<float, RNQ_VL>(block_load<fp16, RNQ_VL>(nw + k)));
            slm_block_store<fp16, RNQ_VL>(k * sizeof(fp16), xv);
            if (blk == 0 && xn)
                block_store<fp16, RNQ_VL>(xn + (size_t)token * KT + k, xv);
        }
        barrier();

        // ---- phase 2: one row per work-item, K-tiled over the SLM copy -----
        int kind, rbase, nrow;
        if (blk < nb4)            { kind = 0; rbase = blk * RNQ_ROWS;              nrow = N4; }
        else if (blk < nb4 + nb6) { kind = 1; rbase = (blk - nb4) * RNQ_ROWS;      nrow = N6; }
        else                      { kind = 2; rbase = (blk - nb4 - nb6) * RNQ_ROWS; nrow = NF; }

        const int row = rbase + lid;   // RNQ_ROWS == RNQ_WGS: one row per lane
        if (row >= nrow) return;

        constexpr int VL      = RNQ_VL;
        constexpr int VL_HALF = VL / 2;
        constexpr int VL_QTR  = VL / 4;
        constexpr int VL_G32  = VL / 32;
        constexpr int VL_G16  = VL / 16;
        constexpr int K_ITERS = KT / VL;

        simd<float, 8> acc(0.0f);
        int ai = 0;

        for (int iter = 0; iter < K_ITERS; iter++) {
            const int k = iter * VL;
            simd<float, VL> act = slm_block_load<fp16, VL>(k * sizeof(fp16));
            simd<float, VL> weight_f;

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
                for (int sb = 0; sb < VL_G32; sb++) {
                    const float s = sc_f[sb], m = mn_f[sb];
                    weight_f.template select<32, 1>(sb * 32) =
                        weight_f.template select<32, 1>(sb * 32) * s - m;
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
                for (int sb = 0; sb < VL_G16; sb++) {
                    const float s = sc_f[sb];
                    weight_f.template select<16, 1>(sb * 16) =
                        (weight_f.template select<16, 1>(sb * 16) - 32.0f) * s;
                }
            } else {
                weight_f = block_load<fp16, VL>(wf + (size_t)row * KT + k);
            }

            acc[ai] += esimd_detail::sum<float, float, VL>(weight_f * act);
            ai = (ai + 1) & 7;
        }

        const fp16 val = (fp16)esimd_detail::sum<float, float, 8>(acc);
        if (kind == 0)      o4[(size_t)token * ld4 + of4 + row] = val;
        else if (kind == 1) o6[(size_t)token * ld6 + of6 + row] = val;
        else                of[(size_t)token * ldf + off + row] = val;
    }
};

// Returns false when the shape is unsupported so the caller can fall back to
// the separate norm + per-kind GEMV path.
inline bool resadd_norm_gemv_kq_host(
    const fp16* h, const fp16* res, fp16* nr, const fp16* nw, fp16* xn,
    const uint8_t* q4_w, const fp16* q4_sc, const fp16* q4_mn,
    fp16* o4, int ld4, int of4,
    const uint8_t* q6_ql, const uint8_t* q6_qh, const fp16* q6_sc,
    fp16* o6, int ld6, int of6,
    const fp16* wf, fp16* of, int ldf, int off,
    float eps, int M, int K, int N4, int N6, int NF, sycl::queue& q) {
    if (K % RNQ_VL != 0) return false;

    const int nb4 = (N4 + RNQ_ROWS - 1) / RNQ_ROWS;
    const int nb6 = (N6 + RNQ_ROWS - 1) / RNQ_ROWS;
    const int nbf = (NF + RNQ_ROWS - 1) / RNQ_ROWS;
    const int blocks = (nb4 + nb6 + nbf) > 0 ? (nb4 + nb6 + nbf) : 1;

#define LAUNCH_RNQ(KT)                                                        \
    q.submit([&](sycl::handler& hd) {                                         \
        hd.parallel_for(                                                      \
            sycl::nd_range<1>((size_t)M * blocks * RNQ_WGS, RNQ_WGS),         \
            Resadd_norm_gemv_kq_kernel<KT>{                                   \
                h, res, nr, nw, xn,                                           \
                q4_w, q4_sc, q4_mn, o4, ld4, of4,                             \
                q6_ql, q6_qh, q6_sc, o6, ld6, of6,                            \
                wf, of, ldf, off,                                             \
                eps, M, N4, N6, NF, nb4, nb6, blocks});                       \
    });                                                                       \
    return true;

    switch (K) {
        case 2048: LAUNCH_RNQ(2048)
        case 2560: LAUNCH_RNQ(2560)
        case 3072: LAUNCH_RNQ(3072)
        case 4096: LAUNCH_RNQ(4096)
        case 5120: LAUNCH_RNQ(5120)
        case 6144: LAUNCH_RNQ(6144)
        default: return false;
    }
#undef LAUNCH_RNQ
}
