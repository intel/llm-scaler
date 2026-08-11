/* gdn_norm_gated.h — standalone RMSNormGated head for the GGUF GDN out_proj.
 *
 *   y[h,:] = fp16( x[h,:] * rsqrt(mean(x[h,:]^2) + eps) * nw[:] * silu(z[h,:]) )
 *
 * i.e. RMSNormGated(norm_before_gate=True, activation="swish"), per value head.
 *
 * `norm_gemv_fused.h` already fuses this into an fp8 GEMV, but the GGUF build's
 * out_proj is an ESIMD q8_0 GEMV, so that kernel never applies. Rather than
 * duplicate the q8_0 dequant inside a second fused kernel, this one only emits
 * the normed activation; the caller enqueues the existing, already-validated
 * q8_0 GEMV right behind it on the same in-order queue. Two kernels, but still
 * ONE torch op dispatch — and at bs=1 decode the dispatch is what costs
 * (~13-25us) while a kernel of this size costs ~2us.
 *
 * The RMS reduction is per head (V dims), so one work-item owns a whole head
 * and no cross-lane/SLM reduction is needed at all.
 *
 * Math copied from NormGEMV_fp8_pert_kernel so both paths stay bit-comparable.
 */
#pragma once

template <int V>
struct Gdn_norm_gated_kernel {
    const fp16* x;   // [HV, V]
    const fp16* z;   // [HV, V]
    const fp16* nw;  // [V]
    fp16*       y;   // [HV, V]  out
    float eps;

    void operator()(sycl::item<1> item) const SYCL_ESIMD_KERNEL {
        const size_t off = (size_t)item.get_id(0) * V;

        simd<float, V> x_f = block_load<fp16, V>(x + off);
        float mean_sq = reduce<float>(x_f * x_f, std::plus<>()) * (1.0f / (float)V);
        float inv_rms =
            sycl::ext::intel::esimd::rsqrt(simd<float, 8>(mean_sq + eps))[0];

        simd<float, V> z_f = block_load<fp16, V>(z + off);
        simd<float, V> silu_z = z_f / (1.0f + sycl::ext::intel::esimd::exp(-z_f));

        simd<float, V> normed =
            x_f * inv_rms * simd<float, V>(block_load<fp16, V>(nw)) * silu_z;
        block_store<fp16, V>(y + off, convert<fp16>(normed));
    }
};

// Returns false for an unsupported head width so the caller can fall back.
inline bool gdn_norm_gated_host(
    const fp16* x, const fp16* z, const fp16* nw, fp16* y,
    float eps, int HV, int V, sycl::queue& q) {
#define LAUNCH_GDN_NG(VV)                                                      \
    q.submit([&](sycl::handler& hd) {                                          \
        hd.parallel_for(sycl::range<1>((size_t)HV),                            \
                        Gdn_norm_gated_kernel<VV>{x, z, nw, y, eps});          \
    });                                                                        \
    return true;

    switch (V) {
        case 64:  LAUNCH_GDN_NG(64)
        case 128: LAUNCH_GDN_NG(128)
        case 256: LAUNCH_GDN_NG(256)
        default:  return false;
    }
#undef LAUNCH_GDN_NG
}

/* ---------------------------------------------------------------------------
 * Gdn_norm_gated_q8_kernel — RMSNormGated fused with the q8_0 out_proj GEMV.
 *
 * One work-group = HVT lanes; lane `l` owns value head `l` (V elements), so the
 * gated norm needs no cross-lane communication and the normalised head stays in
 * registers. The GEMV then contracts over the flattened [HVT*V] vector with each
 * lane supplying the partial dot of its own head, combined through SLM — the
 * same K-split shape q8_0_GEMV.h uses, with K_SPLIT == HVT and KP == V.
 *
 * Saves one enqueue per GDN layer; at M=1 decode we are launch-bound, so the
 * redundant per-WG norm recompute (register/L2 local) is cheaper than the
 * second kernel launch.
 *
 * Batched decode: the WG grid is M*blocks groups, `token = gid / blocks`
 * selecting the row of x/z/y/out and `blk = gid % blocks` the band of output
 * rows, exactly as Moe_norm_q8_kernel does. The weight offset within K stays
 * `lid*V` (shared by all tokens) while the activation offset carries the extra
 * `token*K`, so the weights are read once per band and reused across tokens.
 * ------------------------------------------------------------------------- */
template <int V, int HVT>
struct Gdn_norm_gated_q8_kernel {
    static constexpr int ROWS = HVT < 8 ? HVT : 8;
    const fp16*   x;    // [M*HV, V]
    const fp16*   z;    // [M*HV, V]
    const fp16*   nw;   // [V]
    fp16*         y;    // [M*HV, V]  may be null
    const int8_t* qs;   // [N, HVT*V]
    const fp16*   sc;   // [N, HVT*V/32]
    fp16*         out;  // [M, N]
    float eps;
    int N, K, M, blocks;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        slm_init<HVT*(ROWS + 1) * sizeof(float)>();
        const int gid   = (int)item.get_group(0);
        const int lid   = (int)item.get_local_id(0);
        const int token = gid / blocks;
        const int blk   = gid % blocks;
        // Offset of this lane's head inside the contraction dim K (shared by
        // every token), and the same offset inside this token's activation row.
        const size_t koff = (size_t)lid * V;
        const size_t xoff = (size_t)token * K + koff;

        simd<float, V> x_f = block_load<fp16, V>(x + xoff);
        float mean_sq = reduce<float>(x_f * x_f, std::plus<>()) * (1.0f / (float)V);
        float inv_rms =
            sycl::ext::intel::esimd::rsqrt(simd<float, 8>(mean_sq + eps))[0];

        simd<float, V> z_f = block_load<fp16, V>(z + xoff);
        simd<float, V> silu_z = z_f / (1.0f + sycl::ext::intel::esimd::exp(-z_f));
        simd<fp16, V> yv = convert<fp16>(
            x_f * inv_rms * simd<float, V>(block_load<fp16, V>(nw)) * silu_z);
        if (blk == 0 && y) block_store<fp16, V>(y + xoff, yv);

        simd<float, V> xf = simd<float, V>(yv);
        const int rbase = blk * ROWS;
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const int row = rbase + r;
            float p = 0.0f;
            if (row < N) {
                const int8_t* wrow = qs + (size_t)row * K + koff;
                const fp16*   srow = sc + (size_t)row * (K / 32) + (koff / 32);
                simd<float, 32> acc = 0.0f;
                for (int k = 0; k < V; k += 32) {
                    simd<float, 32> wf =
                        convert<float>(block_load<int8_t, 32>(wrow + k));
                    float s = static_cast<float>(srow[k >> 5]);
                    acc += xf.template select<32, 1>(k) * (wf * s);
                }
                p = reduce<float>(acc, std::plus<>());
            }
            slm_block_store<float, 1>((HVT * (1 + r) + lid) * sizeof(float),
                                      simd<float, 1>(p));
        }
        barrier();
        if (lid < ROWS) {
            const int row = rbase + lid;
            if (row < N) {
                simd<float, HVT> d =
                    slm_block_load<float, HVT>(HVT * (1 + lid) * sizeof(float));
                out[(size_t)token * N + row] = fp16(reduce<float>(d, std::plus<>()));
            }
        }
    }
};

inline bool gdn_norm_gated_q8_host(
    const fp16* x, const fp16* z, const fp16* nw, fp16* y,
    const int8_t* qs, const fp16* sc, fp16* out,
    float eps, int HV, int V, int N, int M, sycl::queue& q) {
    const int rows   = HV < 8 ? HV : 8;
    const int K      = HV * V;
    const int blocks = (N + rows - 1) / rows;
    if (M < 1) return false;

#define LAUNCH_GDN_NQ(VV, HH)                                                  \
    q.submit([&](sycl::handler& hd) {                                          \
        hd.parallel_for(sycl::nd_range<1>((size_t)M * blocks * HH, HH),        \
                        Gdn_norm_gated_q8_kernel<VV, HH>{                \
                            x, z, nw, y, qs, sc, out, eps, N, K, M, blocks});  \
    });                                                                        \
    return true;
#define DISPATCH_HV(VV)                                                        \
    switch (HV) {                                                              \
        case 2:  LAUNCH_GDN_NQ(VV, 2)                                          \
        case 4:  LAUNCH_GDN_NQ(VV, 4)                                          \
        case 8:  LAUNCH_GDN_NQ(VV, 8)                                          \
        case 16: LAUNCH_GDN_NQ(VV, 16)                                         \
        case 32: LAUNCH_GDN_NQ(VV, 32)                                         \
        default: return false;                                                 \
    }

    if (V % 32 != 0) return false;
    switch (V) {
        case 64:  DISPATCH_HV(64)
        case 128: DISPATCH_HV(128)
        case 256: DISPATCH_HV(256)
        default:  return false;
    }
#undef DISPATCH_HV
#undef LAUNCH_GDN_NQ
}
