/* moe_norm_router.h — fused (residual-add + GemmaRMSNorm + router GEMV) head
 * for the GGUF decode MoE op (Intel XPU, ESIMD).
 *
 *   res_out[t,:]= fp16( h[t,:] + res[t,:] )
 *   v           = float(res_out[t,:])
 *   xn[t,:]     = fp16( v * rsqrt(mean(v^2) + eps) * nw )      (nw = 1 + gemma w)
 *   logits[t,e] = sum_k xn[t,k] * rw[e,k]
 *
 * Motivation: at M=1 decode the model is HOST-bound (~13-25us of torch dispatch
 * per op call vs ~4us of actual enqueue), so the win is in collapsing op CALLS,
 * not kernel time. Folding `gemma_fused_add_rmsnorm` + the fp16 router `linear`
 * into the MoE op removes two python/dispatcher round-trips per layer.
 *
 * One work-group per (token, block of ROWS expert rows); K_SPLIT threads split
 * the hidden reduction and combine through SLM. Every WG recomputes the norm
 * from values it has already loaded into registers (no extra DRAM traffic), and
 * only block 0 of a token stores xn/res back, so there is no cross-WG race.
 *
 * Included into esimd_kernel.sycl alongside the other moe_* headers.
 */
#pragma once

template <int K_SPLIT, int KP, int ROWS>
struct Moe_norm_router_kernel {
    const fp16* h;       // [M, hidden]
    const fp16* res;     // [M, hidden]  in
    fp16*       res_out; // [M, hidden]  out (MUST NOT alias `res`: block 0
                         // stores while the other blocks are still loading)
    const fp16* nw;      // [hidden]     (1 + gemma weight)
    const fp16* rw;      // [E, hidden]
    fp16*       xn;      // [M, hidden]  out
    fp16*       logits;  // [M, E]       out
    float       eps;
    int M, hidden, E, blocks;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        slm_init<K_SPLIT*(ROWS + 1) * sizeof(float)>();
        const int gid   = (int)item.get_group(0);
        const int lid   = (int)item.get_local_id(0);
        const int token = gid / blocks;
        const int blk   = gid % blocks;
        const int kbeg  = lid * KP;

        const fp16* hr = h + (size_t)token * hidden;
        const fp16* rr = res + (size_t)token * hidden;

        // fp16 add first: the reference rounds the residual to fp16 BEFORE the
        // fp32 variance, so accumulating in fp32 here would not match.
        simd<fp16, KP> vh = block_load<fp16, KP>(hr + kbeg) +
                            block_load<fp16, KP>(rr + kbeg);
        simd<float, KP> v = simd<float, KP>(vh);

        slm_block_store<float, 1>(lid * sizeof(float),
                                  simd<float, 1>(reduce<float>(v * v, std::plus<>())));
        barrier();
        simd<float, K_SPLIT> parts = slm_block_load<float, K_SPLIT>(0);
        float rstd = 1.0f / sycl::sqrt(reduce<float>(parts, std::plus<>()) / (float)hidden + eps);

        simd<fp16, KP> xv = convert<fp16>(
            v * rstd * simd<float, KP>(block_load<fp16, KP>(nw + kbeg)));
        if (blk == 0) {
            block_store<fp16, KP>(res_out + (size_t)token * hidden + kbeg, vh);
            block_store<fp16, KP>(xn + (size_t)token * hidden + kbeg, xv);
        }
        simd<float, KP> xf = simd<float, KP>(xv);

        const int rbase = blk * ROWS;
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const int row = rbase + r;
            float p = 0.0f;
            if (row < E)
                p = reduce<float>(
                    xf * simd<float, KP>(
                             block_load<fp16, KP>(rw + (size_t)row * hidden + kbeg)),
                    std::plus<>());
            slm_block_store<float, 1>((K_SPLIT * (1 + r) + lid) * sizeof(float),
                                      simd<float, 1>(p));
        }
        barrier();
        if (lid < ROWS) {
            const int row = rbase + lid;
            if (row < E) {
                simd<float, K_SPLIT> d = slm_block_load<float, K_SPLIT>(
                    K_SPLIT * (1 + lid) * sizeof(float));
                logits[(size_t)token * E + row] =
                    fp16(reduce<float>(d, std::plus<>()));
            }
        }
    }
};

// Returns false when the shape is unsupported, so the caller can fall back to
// the separate norm + router path.
inline bool moe_norm_router_host(
    const fp16* h, const fp16* res, fp16* res_out, const fp16* nw, const fp16* rw,
    fp16* xn, fp16* logits, float eps,
    int M, int hidden, int E, sycl::queue& q) {
    constexpr int KS   = 8;
    constexpr int ROWS = 4;
    // E == 0 means "norm only, no GEMV": still launch one block per token so
    // xn / res are written (the `row < E` guards make the GEMV half a no-op).
    const int blocks = E > 0 ? (E + ROWS - 1) / ROWS : 1;

#define LAUNCH_MOE_NR(KP)                                                      \
    q.submit([&](sycl::handler& hd) {                                          \
        hd.parallel_for(                                                       \
            sycl::nd_range<1>((size_t)M * blocks * KS, KS),                    \
            Moe_norm_router_kernel<KS, KP, ROWS>{h, res, res_out, nw, rw, xn,  \
                                                 logits, eps, M, hidden, E,    \
                                                 blocks});                     \
    });                                                                        \
    return true;

    switch (hidden / KS) {
        case 128: LAUNCH_MOE_NR(128)
        case 256: LAUNCH_MOE_NR(256)
        case 512: LAUNCH_MOE_NR(512)
        default:  return false;
    }
#undef LAUNCH_MOE_NR
}

/* ---------------------------------------------------------------------------
 * Moe_norm_q8_kernel — same head as above, but the WG grid is split into two
 * ranges so ONE kernel launch produces both GEMVs:
 *
 *   blk <  nb_r : rows of the fp16 matrix `rw`   -> logits[M, E]
 *   blk >= nb_r : rows of the q8_0 matrix qs/sc  -> out0[M, N0]
 *
 * At M=1 decode we are launch-bound (~4-8us of host per enqueue vs ~2-20us of
 * GPU), so collapsing the (norm-kernel + gemv-kernel) pair into a single
 * enqueue is worth the redundant norm recompute, which is register/L2 local.
 * ------------------------------------------------------------------------- */
template <int K_SPLIT, int KP, int ROWS>
struct Moe_norm_q8_kernel {
    const fp16*   h;
    const fp16*   res;
    fp16*         res_out;   // may be null
    const fp16*   nw;
    const fp16*   rw;        // [E, hidden] fp16   (may be null when E == 0)
    fp16*         logits;    // [M, E]
    const int8_t* qs;        // [N0, hidden] int8  (may be null when N0 == 0)
    const fp16*   sc;        // [N0, hidden/32]
    fp16*         out0;      // [M, N0]
    fp16*         xn;        // [M, hidden] may be null
    float         eps;
    int M, hidden, E, N0, nb_r, blocks;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        slm_init<K_SPLIT*(ROWS + 1) * sizeof(float)>();
        const int gid   = (int)item.get_group(0);
        const int lid   = (int)item.get_local_id(0);
        const int token = gid / blocks;
        const int blk   = gid % blocks;
        const int kbeg  = lid * KP;

        simd<fp16, KP> vh =
            block_load<fp16, KP>(h + (size_t)token * hidden + kbeg) +
            block_load<fp16, KP>(res + (size_t)token * hidden + kbeg);
        simd<float, KP> v = simd<float, KP>(vh);

        slm_block_store<float, 1>(lid * sizeof(float),
                                  simd<float, 1>(reduce<float>(v * v, std::plus<>())));
        barrier();
        simd<float, K_SPLIT> parts = slm_block_load<float, K_SPLIT>(0);
        float rstd = 1.0f / sycl::sqrt(reduce<float>(parts, std::plus<>()) / (float)hidden + eps);

        simd<fp16, KP> xv = convert<fp16>(
            v * rstd * simd<float, KP>(block_load<fp16, KP>(nw + kbeg)));
        if (blk == 0) {
            if (res_out) block_store<fp16, KP>(res_out + (size_t)token * hidden + kbeg, vh);
            if (xn)      block_store<fp16, KP>(xn + (size_t)token * hidden + kbeg, xv);
        }
        simd<float, KP> xf = simd<float, KP>(xv);

        const bool q8_range = (blk >= nb_r);
        const int  rbase    = (q8_range ? (blk - nb_r) : blk) * ROWS;
        const int  nrow     = q8_range ? N0 : E;

#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const int row = rbase + r;
            float p = 0.0f;
            if (row < nrow) {
                if (q8_range) {
                    const int8_t* wrow = qs + (size_t)row * hidden + kbeg;
                    const fp16*   srow = sc + (size_t)row * (hidden / 32) + (kbeg / 32);
                    simd<float, 32> acc = 0.0f;
                    for (int k = 0; k < KP; k += 32) {
                        simd<float, 32> wf =
                            convert<float>(block_load<int8_t, 32>(wrow + k));
                        float s = static_cast<float>(srow[k >> 5]);
                        acc += xf.template select<32, 1>(k) * (wf * s);
                    }
                    p = reduce<float>(acc, std::plus<>());
                } else {
                    p = reduce<float>(
                        xf * simd<float, KP>(
                                 block_load<fp16, KP>(rw + (size_t)row * hidden + kbeg)),
                        std::plus<>());
                }
            }
            slm_block_store<float, 1>((K_SPLIT * (1 + r) + lid) * sizeof(float),
                                      simd<float, 1>(p));
        }
        barrier();
        if (lid < ROWS) {
            const int row = rbase + lid;
            if (row < nrow) {
                simd<float, K_SPLIT> d = slm_block_load<float, K_SPLIT>(
                    K_SPLIT * (1 + lid) * sizeof(float));
                fp16 val = fp16(reduce<float>(d, std::plus<>()));
                if (q8_range) out0[(size_t)token * N0 + row] = val;
                else          logits[(size_t)token * E + row] = val;
            }
        }
    }
};

inline bool moe_norm_q8_host(
    const fp16* h, const fp16* res, fp16* res_out, const fp16* nw,
    const fp16* rw, fp16* logits,
    const int8_t* qs, const fp16* sc, fp16* out0,
    fp16* xn, float eps, int M, int hidden, int E, int N0, sycl::queue& q) {
    constexpr int KS   = 8;
    constexpr int ROWS = 4;
    if (hidden % (KS * 32) != 0) return false;
    const int nb_r   = E  > 0 ? (E  + ROWS - 1) / ROWS : 0;
    const int nb_q   = N0 > 0 ? (N0 + ROWS - 1) / ROWS : 0;
    const int blocks = (nb_r + nb_q) > 0 ? (nb_r + nb_q) : 1;

#define LAUNCH_MOE_NQ(KP)                                                      \
    q.submit([&](sycl::handler& hd) {                                          \
        hd.parallel_for(sycl::nd_range<1>((size_t)M * blocks * KS, KS),        \
                        Moe_norm_q8_kernel<KS, KP, ROWS>{                      \
                            h, res, res_out, nw, rw, logits, qs, sc, out0, xn, \
                            eps, M, hidden, E, N0, nb_r, blocks});             \
    });                                                                        \
    return true;

    switch (hidden / KS) {
        case 128: LAUNCH_MOE_NQ(128)
        case 256: LAUNCH_MOE_NQ(256)
        case 512: LAUNCH_MOE_NQ(512)
        default:  return false;
    }
#undef LAUNCH_MOE_NQ
}
