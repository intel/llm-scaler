/* norm_gemv_q5k.h — fused (RMSNormGated + q5_K GEMV) for the GGUF GDN
 * out_proj on Intel XPU (ESIMD).
 *
 *   for each value head hh:
 *     xh          = float(x[t, hh, :])                      (V elements)
 *     normed[hh]  = xh * rsqrt(mean(xh^2) + eps) * nw * silu(float(z[t,hh,:]))
 *   y[t, :]       = fp16( normed @ dequant_q5_K(ql, qh, sc, mn)^T )
 *
 * This is the q5_K counterpart of `esimd_norm_gemv_q8_0` (norm_gemv_fused.h
 * covers fp8, and the q8_0 variant covers the Q8_0 GGUF builds). Qwen3.6-27B
 * Q4_K_M stores ssm_out as q5_K, so neither existing fusion matches and the
 * decode step pays a standalone `gdn_rms_norm_gated` dispatch plus the reshape
 * and cast glue around it on every one of the 48 GDN layers.
 *
 * Why SLM rather than the per-work-group recompute used by the fp8 kernel
 * -----------------------------------------------------------------------
 * The fp8 kernel gives each work-group one output row and lets it re-derive
 * the whole gated norm, which is fine because the norm inputs (~4 KB) stay
 * resident in L3. That does not carry over here: q5_K's high-bit plane is
 * PRE-SHUFFLED per VL=512 K-tile, so the GEMV must consume K in 512-element
 * tiles, which straddle four V=128 heads. Deriving the norm per tile would
 * re-do each head's reduction four times. Instead one work-group computes the
 * gated norm once (each work-item owning whole heads, so the V-wide reduction
 * stays inside a single work-item and needs no barrier), parks it in SLM
 * (K fp16 = 6 KB at K=3072), and then runs ROWS q5_K rows against it.
 *
 * q5_K layout is the one q5_k_GEMV.h already consumes, so no repack:
 *   ql [N, K/2] u8 nibble-interleaved, qh [N, K/8] u8 1-bit PRE-SHUFFLED per
 *   VL=512 tile, sc/mn [N, K/32] fp16, asymmetric  w = sc*v5 - mn
 *
 * Included into esimd_kernel.sycl (utils.h provides fp16 + esimd namespace).
 */
#pragma once

// Must match the tile size the host-side q5_K qh pre-shuffle was built for.
static constexpr int NGQ5_VL   = 512;
static constexpr int NGQ5_WGS  = 32;   // work-items per work-group
static constexpr int NGQ5_ROWS = 32;   // output rows per work-group
static constexpr int NGQ5_V    = 128;  // head_v_dim

// KT is the exact per-rank GDN output width (HV * V), needed for slm_init.
template <int KT>
struct Norm_gemv_q5k_kernel {
    const fp16*    x;      // [M, KT] core_attn_out
    const fp16*    z;      // [M, KT] gate
    const fp16*    nw;     // [V]     per-head norm weight, shared across heads

    const uint8_t* ql;     // [N, KT/2]
    const uint8_t* qh;     // [N, KT/8] pre-shuffled
    const fp16*    sc;     // [N, KT/32]
    const fp16*    mn;     // [N, KT/32]
    fp16*          y;      // [M, N]

    float eps;
    int M, N, blocks;

    static constexpr int HV    = KT / NGQ5_V;
    static constexpr int SLM_X = KT * (int)sizeof(fp16);

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        slm_init<SLM_X>();

        const int gid   = (int)item.get_group(0);
        const int lid   = (int)item.get_local_id(0);
        const int token = gid / blocks;
        const int blk   = gid % blocks;

        // ---- phase 1: gated RMSNorm, one whole head per work-item ----------
        simd<float, NGQ5_V> nwv = block_load<fp16, NGQ5_V>(nw);
        const fp16* xr = x + (size_t)token * KT;
        const fp16* zr = z + (size_t)token * KT;

        for (int hh = lid; hh < HV; hh += NGQ5_WGS) {
            const int off = hh * NGQ5_V;
            simd<float, NGQ5_V> xf = block_load<fp16, NGQ5_V>(xr + off);
            simd<float, NGQ5_V> zf = block_load<fp16, NGQ5_V>(zr + off);

            const float mean_sq =
                esimd_detail::sum<float, float, NGQ5_V>(xf * xf) / (float)NGQ5_V;
            const float inv_rms = 1.0f / sycl::sqrt(mean_sq + eps);

            // silu(z) = z * sigmoid(z), evaluated in fp32 as RMSNormGated does.
            simd<float, NGQ5_V> silu_z =
                zf / (1.0f + sycl::ext::intel::esimd::exp(-zf));
            slm_block_store<fp16, NGQ5_V>(
                off * sizeof(fp16), convert<fp16>(xf * inv_rms * nwv * silu_z));
        }
        barrier();

        // ---- phase 2: one q5_K row per work-item, K-tiled over SLM ---------
        const int row = blk * NGQ5_ROWS + lid;   // NGQ5_ROWS == NGQ5_WGS
        if (row >= N) return;

        constexpr int VL     = NGQ5_VL;
        constexpr int VL_H   = VL / 2;    // ql bytes per tile
        constexpr int VL_8   = VL / 8;    // qh bytes per tile
        constexpr int VL_G32 = VL / 32;   // sc/mn entries per tile
        constexpr int K_ITERS = KT / VL;

        simd<float, 8> acc(0.0f);
        int ai = 0;

        for (int iter = 0; iter < K_ITERS; iter++) {
            const int k = iter * VL;
            simd<float, VL> act = slm_block_load<fp16, VL>(k * sizeof(fp16));

            simd<uint8_t, VL_H> ql_data = block_load<uint8_t, VL_H>(
                ql + (size_t)row * (KT / 2) + k / 2);
            simd<uint8_t, VL_8> qh_data = block_load<uint8_t, VL_8>(
                qh + (size_t)row * (KT / 8) + k / 8);
            simd<float, VL_G32> sc_f =
                block_load<fp16, VL_G32>(sc + (size_t)row * (KT / 32) + k / 32);
            simd<float, VL_G32> mn_f =
                block_load<fp16, VL_G32>(mn + (size_t)row * (KT / 32) + k / 32);

            simd<float, VL> weight_f;
            #pragma unroll
            for (int c = 0; c < VL_H / 64; c++) {
                auto p = ql_data.template select<64, 1>(c * 64);
                simd<float, 64> lo = p & 0x0F;
                simd<float, 64> hi = (p >> 4) & 0x0F;
                weight_f.template select<64, 2>(c * 128) = lo;
                weight_f.template select<64, 2>(c * 128 + 1) = hi;
            }
            // qh is pre-shuffled so bit b of byte t covers element b*VL_8 + t.
            #pragma unroll
            for (int bit = 0; bit < 8; bit++) {
                simd<float, VL_8> ef = (qh_data >> bit) & 1;
                weight_f.template select<VL_8, 1>(bit * VL_8) += ef * 16.0f;
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

        y[(size_t)token * N + row] =
            (fp16)esimd_detail::sum<float, float, 8>(acc);
    }
};

// Returns false when the shape is unsupported so the caller can fall back to
// the separate gdn_rms_norm_gated + q5_K GEMV path.
inline bool norm_gemv_q5k_host(
    const fp16* x, const fp16* z, const fp16* nw,
    const uint8_t* ql, const uint8_t* qh, const fp16* sc, const fp16* mn,
    fp16* y, float eps, int M, int K, int N, int V, sycl::queue& q) {
    if (V != NGQ5_V || K % NGQ5_VL != 0 || K % NGQ5_V != 0) return false;

    const int blocks = (N + NGQ5_ROWS - 1) / NGQ5_ROWS;

#define LAUNCH_NGQ5(KT)                                                       \
    q.submit([&](sycl::handler& hd) {                                         \
        hd.parallel_for(                                                      \
            sycl::nd_range<1>((size_t)M * blocks * NGQ5_WGS, NGQ5_WGS),       \
            Norm_gemv_q5k_kernel<KT>{                                         \
                x, z, nw, ql, qh, sc, mn, y, eps, M, N, blocks});             \
    });                                                                       \
    return true;

    switch (K) {
        case 1024: LAUNCH_NGQ5(1024)
        case 1536: LAUNCH_NGQ5(1536)
        case 2048: LAUNCH_NGQ5(2048)
        case 3072: LAUNCH_NGQ5(3072)
        case 4096: LAUNCH_NGQ5(4096)
        case 6144: LAUNCH_NGQ5(6144)
        default: return false;
    }
#undef LAUNCH_NGQ5
}
