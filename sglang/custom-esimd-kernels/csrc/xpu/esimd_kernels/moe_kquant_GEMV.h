/* moe_kquant_GEMV.h — fused GGUF k-quant MoE for Intel XPU (ESIMD), decode.
 *
 * Replaces the per-expert Python GEMV loop in GGUFMoEXPUMethod.apply (which on
 * decode launched top_k*3 kernels/token + host syncs = launch-bound). One
 * kernel launch per stage handles ALL routed (token,expert) pairs, inlining the
 * k-quant dequant — structure ported from pathB moe_int4.sycl
 * moe_up_routed_int4_kernel (GGML N-major layout), but with Q4_K/Q5_K/Q6_K
 * group-32 dequant instead of g128 sym_int4.
 *
 * Expert weights are stacked GGML-style [E, N, K-packed]:
 *   Q4_K (gate/up): ql [E, N, K/2] u8 interleaved + scale,min [E, N, K/32] fp16
 *   Q5_K (down):    u5 stored as the packed (ql nibble + pre-shuffled qh) rep —
 *                   BUT for MoE we use the simpler uint8/elem path: down weights
 *                   repacked to u8 [E, N, K] + scale,min [E, N, K/32].
 *   Q6_K (down):    u6 u8 [E, N, K] + scale [E, N, K/16] (symmetric, v6-32).
 * (down uint8/elem keeps the kernel simple; experts are the bulk but uint8 is
 *  only +overhead on the down tensors, acceptable per §10d analysis.)
 *
 * Dequant (matches q4_k_GEMV.h / q5_k / q6_k):
 *   Q4_K: w = scale[k/32]*nibble - min[k/32]   (interleaved: lo->even, hi->odd)
 *   Q5_K(u8): w = scale[k/32]*u8 - min[k/32]
 *   Q6_K(u8): w = scale[k/16]*(u8 - 32)
 *
 * Included into esimd_kernel.sycl (utils.h: fp16 + esimd namespace + detail).
 */
#pragma once

namespace esimd_detail2 = sycl::ext::intel::esimd::detail;

static constexpr int MOE_Q4K_GROUP = 32;
static constexpr int MOE_Q4K_HALF = 16;

// Gated activation applied to the up/gate stage. SiLU covers Qwen; GELU_TANH
// ("gelu_pytorch_tanh") covers gemma-4, whose MoE block is otherwise the same
// Q4_K gate/up shape.
static constexpr int MOE_ACT_SILU = 0;
static constexpr int MOE_ACT_GELU_TANH = 1;

template <int ACT>
inline float moe_act(float g) {
    if constexpr (ACT == MOE_ACT_GELU_TANH) {
        // ESIMD has no sycl::tanh, so use the exact algebraic rewrite
        // 0.5*g*(1+tanh(z)) == g / (1 + exp(-2z)), z = c*(g + 0.044715 g^3).
        const float c = 0.7978845608028654f;  // sqrt(2/pi)
        const float z = c * (g + 0.044715f * g * g * g);
        return g / (1.0f + sycl::exp(-2.0f * z));
    } else {
        return g / (1.0f + sycl::exp(-g));
    }
}

// ── Up/gate stage: Q4_K gate + Q4_K up -> silu(gate)*up -> intermediate ──────
// grid (n_routed = n_tokens*top_k, intermediate_size). Each WI: one n_col.
// VL elements per iteration: the original stepped 32 at a time, issuing 16-byte
// nibble loads (below LSC granularity) and re-reading a scalar scale/min per
// block. Wider tiles cut the load count by VL/32 and keep one flat accumulator.
template <int VL, int ACT = MOE_ACT_SILU>
struct Moe_up_q4k_kernel {
    const fp16*    x;          // [n_tokens, hidden]
    const uint8_t* gate_ql;    // [E, inter, hidden/2]
    const fp16*    gate_sc;    // [E, inter, hidden/32]
    const fp16*    gate_mn;    // [E, inter, hidden/32]
    const uint8_t* up_ql;      // [E, inter, hidden/2]
    const fp16*    up_sc;
    const fp16*    up_mn;
    const int*     sel_experts; // [n_routed]
    fp16*          inter;       // [n_routed, intermediate_size]
    int n_tokens, hidden, intermediate, top_k;

    void operator()(sycl::id<2> idx) const SYCL_ESIMD_KERNEL {
        run((int)idx[0], (int)idx[1]);
    }

    // Split out so Moe_up_fused_kernel can reuse the exact same body without
    // duplicating the q4_k dequant.
    void run(int route, int n_col) const {
        const int token = route / top_k;
        const int eid = sel_experts[route];

        constexpr int VH  = VL / 2;                 // packed bytes per tile
        constexpr int NSC = VL / MOE_Q4K_GROUP;     // scale/min entries per tile
        const int Kh = hidden / 2;
        const int Kg = hidden / MOE_Q4K_GROUP;
        const fp16* x_row = x + (size_t)token * hidden;
        const uint8_t* gq = gate_ql + ((size_t)eid * intermediate + n_col) * Kh;
        const fp16*    gs = gate_sc + ((size_t)eid * intermediate + n_col) * Kg;
        const fp16*    gm = gate_mn + ((size_t)eid * intermediate + n_col) * Kg;
        const uint8_t* uq = up_ql + ((size_t)eid * intermediate + n_col) * Kh;
        const fp16*    us = up_sc + ((size_t)eid * intermediate + n_col) * Kg;
        const fp16*    um = up_mn + ((size_t)eid * intermediate + n_col) * Kg;

        simd<float, VL> ag = 0.0f, au = 0.0f;
        for (int k = 0; k < hidden; k += VL) {
            const int gi = k / MOE_Q4K_GROUP;
            simd<fp16, VL>  iv = block_load<fp16, VL>(x_row + k);
            simd<float, VL> xf = simd<float, VL>(iv);

            simd<uint8_t, VH>  graw = block_load<uint8_t, VH>(gq + k / 2);
            simd<uint8_t, VH>  uraw = block_load<uint8_t, VH>(uq + k / 2);
            simd<fp16, NSC>    gsv  = block_load<fp16, NSC>(gs + gi);
            simd<fp16, NSC>    gmv  = block_load<fp16, NSC>(gm + gi);
            simd<fp16, NSC>    usv  = block_load<fp16, NSC>(us + gi);
            simd<fp16, NSC>    umv  = block_load<fp16, NSC>(um + gi);

            simd<uint16_t, VH> g16 = convert<uint16_t>(graw);
            simd<uint16_t, VH> u16 = convert<uint16_t>(uraw);
            simd<float, VL> gw, uw;
            gw.template select<VH, 2>(0) = convert<float>(g16 & 0x000F);
            gw.template select<VH, 2>(1) = convert<float>((g16 >> 4) & 0x000F);
            uw.template select<VH, 2>(0) = convert<float>(u16 & 0x000F);
            uw.template select<VH, 2>(1) = convert<float>((u16 >> 4) & 0x000F);

            #pragma unroll
            for (int c = 0; c < NSC; c++) {
                fp16 gsc = gsv[c], gmn = gmv[c], usc = usv[c], umn = umv[c];
                gw.template select<MOE_Q4K_GROUP, 1>(c * MOE_Q4K_GROUP) =
                    gw.template select<MOE_Q4K_GROUP, 1>(c * MOE_Q4K_GROUP)
                        * (float)gsc - (float)gmn;
                uw.template select<MOE_Q4K_GROUP, 1>(c * MOE_Q4K_GROUP) =
                    uw.template select<MOE_Q4K_GROUP, 1>(c * MOE_Q4K_GROUP)
                        * (float)usc - (float)umn;
            }
            ag += xf * gw;
            au += xf * uw;
        }
        float g = reduce<float>(ag, std::plus<>());
        float u = reduce<float>(au, std::plus<>());
        inter[(size_t)route * intermediate + n_col] = fp16(moe_act<ACT>(g) * u);
    }
};

inline void moe_up_q4k_host(
    const fp16* x, const uint8_t* gq, const fp16* gs, const fp16* gm,
    const uint8_t* uq, const fp16* us, const fp16* um,
    const int* sel, fp16* inter,
    int n_tokens, int hidden, int intermediate, int top_k, sycl::queue& q,
    int act = MOE_ACT_SILU) {
#define LAUNCH_MOE_UP_Q4K_A(V, A)                                            \
    q.submit([&](sycl::handler& h) {                                         \
        h.parallel_for(sycl::range<2>((size_t)n_tokens * top_k, intermediate),\
            Moe_up_q4k_kernel<V, A>{x, gq, gs, gm, uq, us, um, sel, inter,   \
                                    n_tokens, hidden, intermediate, top_k}); \
    });
#define LAUNCH_MOE_UP_Q4K(V)                                                 \
    if (act == MOE_ACT_GELU_TANH) { LAUNCH_MOE_UP_Q4K_A(V, MOE_ACT_GELU_TANH) } \
    else                          { LAUNCH_MOE_UP_Q4K_A(V, MOE_ACT_SILU) }

    if      (hidden % 128 == 0) { LAUNCH_MOE_UP_Q4K(128) }
    else if (hidden % 64  == 0) { LAUNCH_MOE_UP_Q4K(64)  }
    else                        { LAUNCH_MOE_UP_Q4K(32)  }
#undef LAUNCH_MOE_UP_Q4K
#undef LAUNCH_MOE_UP_Q4K_A
}

// ── Down stage: PACKED (zero extra memory, mirrors q5_k/q6_k_GEMV.h) ─────────
// grid (n_routed, hidden). Each WI: dot(inter, dequant(down[eid,h_col,:])) *
// topk_w -> per-route partial [n_routed, hidden] (host sums over top_k).
// Both down kernels read PLAIN qh and loop per 32-elem superblock at fixed simd
// width 32, so intermediate (=K) may be any multiple of 32 (TP1 512 / TP2 256 /
// TP4 128).

// Q5_K down: ql [E,N,K/2] nibble + qh [E,N,K/8] PLAIN 1-bit (byte j bit b = elem
// 8j+b) + scale,min [E,N,K/32] fp16. v5 = nibble|(qh<<4); w = v5*scale - min.
// VL elements per iteration, ROWS output cols per work-item. The original
// stepped one 32-elem superblock at a time (16B nibble + 4B qh loads, plus a
// full horizontal sum folded into a serial `dot` chain every block) and only
// reached ~180GB/s. Wider tiles remove the tiny loads and the serial reduction;
// ROWS>1 additionally amortises the `inter` row load, which every work-item
// otherwise re-reads in full (N=2048 work-items x 512B = 1MB of cache traffic
// per route for a 4KB tensor).
// Q5_1 (legacy, gemma-4 down) shares this kernel verbatim: the host repacks it
// to the SAME layout (interleaved nibble + plain 1-bit qh, one scale/min pair
// per 32 elements), and the only arithmetic difference is the sign of the
// offset term -- q5_K is `v*scale - min`, q5_1 is `v*d + m`. ADDMIN selects it.
template <int VL, int ROWS, bool ADDMIN = false>
struct Moe_down_q5k_kernel {
    const fp16*    inter;      // [n_routed, K]
    const uint8_t* ql;         // [E, N, K/2]
    const uint8_t* qh;         // [E, N, K/8] PLAIN (byte j bit b = elem 8j+b)
    const fp16*    sc;         // [E, N, K/32]
    const fp16*    mn;         // [E, N, K/32]
    const int*     sel_experts;
    const fp16*    topk_w;     // [n_routed]
    fp16*          out;        // [n_routed, N] partial
    int n_tokens, N, K, top_k;

    // Unscaled dot products for the ROWS output cols starting at `n0` of one
    // (token, expert-slot) route. Split out of operator() so the fused
    // down+finalize kernel can reuse the dequant math verbatim.
    inline simd<float, ROWS> dots_for(int route, int n0) const {
        const int eid = sel_experts[route];
        const int Kh = K / 2, Kq = K / 8, Kg = K / 32;
        const fp16* i_row = inter + (size_t)route * K;
        const size_t rbase = (size_t)eid * N + n0;

        constexpr int VH  = VL / 2;    // nibble bytes per tile
        constexpr int VW  = VL / 32;   // qh dwords / scale entries per tile
        const simd<uint32_t, 32> lane(0u, 1u);   // bit index within a qh dword

        simd<float, ROWS> dots = 0.0f;
        for (int k = 0; k < K; k += VL) {
            const int gi = k / 32;
            simd<fp16, VL>  iv = block_load<fp16, VL>(i_row + k);
            simd<float, VL> xf = simd<float, VL>(iv);

            #pragma unroll
            for (int r = 0; r < ROWS; r++) {
                const uint8_t* qlr = ql + (rbase + r) * Kh;
                const uint8_t* qhr = qh + (rbase + r) * Kq;
                const fp16*    scr = sc + (rbase + r) * Kg;
                const fp16*    mnr = mn + (rbase + r) * Kg;

                simd<uint8_t, VH>  qd  = block_load<uint8_t, VH>(qlr + k / 2);
                simd<uint32_t, VW> qhw = block_load<uint32_t, VW>(
                    reinterpret_cast<const uint32_t*>(qhr + k / 8));
                simd<fp16, VW>     scv = block_load<fp16, VW>(scr + gi);
                simd<fp16, VW>     mnv = block_load<fp16, VW>(mnr + gi);

                simd<float, VL> wf;
                wf.template select<VH, 2>(0) = convert<float>(qd & 0x0F);
                wf.template select<VH, 2>(1) = convert<float>((qd >> 4) & 0x0F);

                #pragma unroll
                for (int c = 0; c < VW; c++) {
                    uint32_t qh32 = qhw[c];
                    simd<uint32_t, 32> hb = (simd<uint32_t, 32>(qh32) >> lane) & 1u;
                    fp16 s = scv[c], m = mnv[c];
                    auto blk = wf.template select<32, 1>(c * 32);
                    if constexpr (ADDMIN)
                        blk = (blk + simd<float, 32>(hb) * 16.0f) * (float)s + (float)m;
                    else
                        blk = (blk + simd<float, 32>(hb) * 16.0f) * (float)s - (float)m;
                }
                dots[r] += reduce<float>(xf * wf, std::plus<>());
            }
        }
        return dots;
    }

    void operator()(sycl::id<2> idx) const SYCL_ESIMD_KERNEL {
        const int route = (int)idx[0];
        const int n0 = (int)idx[1] * ROWS;     // first hidden output col
        simd<float, ROWS> dots = dots_for(route, n0);
        const float tw = (float)topk_w[route];
        #pragma unroll
        for (int r = 0; r < ROWS; r++)
            out[(size_t)route * N + n0 + r] = fp16(dots[r] * tw);
    }
};

// Q6_K down: ql [E,N,K/2] nibble + qh [E,N,K/4] PLAIN 2-bit (byte j field p =
// elem 4j+p) + scale [E,N,K/16] fp16. v6 = nibble|(qh<<4); w = scale*(v6-32).
// Per-32-elem block (= 2 scale sub-blocks of 16); fixed simd width 32, any K%32==0.
struct Moe_down_q6k_kernel {
    const fp16*    inter;
    const uint8_t* ql;         // [E, N, K/2]
    const uint8_t* qh;         // [E, N, K/4] PLAIN 2-bit (byte j field p = elem 4j+p)
    const fp16*    sc;         // [E, N, K/16]
    const int*     sel_experts;
    const fp16*    topk_w;
    fp16*          out;
    int n_tokens, N, K, top_k;

    void operator()(sycl::id<2> idx) const SYCL_ESIMD_KERNEL {
        const int route = (int)idx[0];
        const int n = (int)idx[1];
        const int eid = sel_experts[route];
        const int Kh = K / 2, Kq = K / 4, Kg = K / 16;
        const fp16*    i_row = inter + (size_t)route * K;
        const uint8_t* qlr = ql + ((size_t)eid * N + n) * Kh;
        const uint8_t* qhr = qh + ((size_t)eid * N + n) * Kq;
        const fp16*    scr = sc + ((size_t)eid * N + n) * Kg;

        // Each iter = 32 elems = 2 Q6_K scale sub-blocks of 16.
        const simd<uint32_t, 16> lane2(0u, 2u);  // 0,2,..,30 = 2-bit field offsets
        float dot = 0.0f;
        for (int blk = 0; blk < K / 32; blk++) {
            const int e0 = blk * 32;
            simd<fp16, 32>    iv = block_load<fp16, 32>(i_row + e0);
            simd<uint8_t, 16> qd = block_load<uint8_t, 16>(qlr + e0 / 2);
            simd<uint32_t, 2> qhw =
                block_load<uint32_t, 2>(
                    reinterpret_cast<const uint32_t*>(qhr + e0 / 4));
            const uint32_t w0 = qhw[0], w1 = qhw[1];
            simd<float, 32> wf;
            wf.template select<16, 2>(0) = qd & 0x0F;         // low  nibble -> 2j
            wf.template select<16, 2>(1) = (qd >> 4) & 0x0F;  // high nibble -> 2j+1
            // 2-bit high: elem e (0..15) high = bits [2e,2e+1] of w0; 16..31 of w1.
            simd<uint32_t, 16> h0 = (simd<uint32_t, 16>(w0) >> lane2) & 3u;
            simd<uint32_t, 16> h1 = (simd<uint32_t, 16>(w1) >> lane2) & 3u;
            wf.template select<16, 1>(0)  += simd<float, 16>(h0) * 16.0f;
            wf.template select<16, 1>(16) += simd<float, 16>(h1) * 16.0f;
            const float s0 = (float)scr[blk * 2], s1 = (float)scr[blk * 2 + 1];
            wf.template select<16, 1>(0)  = (wf.template select<16, 1>(0)  - 32.0f) * s0;
            wf.template select<16, 1>(16) = (wf.template select<16, 1>(16) - 32.0f) * s1;
            simd<float, 32> prod = simd<float, 32>(iv) * wf;
            dot += esimd_detail2::sum<float, float, 32>(prod);
        }
        out[(size_t)route * N + n] = fp16(dot * (float)topk_w[route]);
    }
};

// Q8_0 routed down: qs [E,N,K] int8 + scale [E,N,K/32] fp16, symmetric
// (w = scale * qs). gemma-4's Q4_K_M mix leaves a single layer's down tensor at
// Q8_0 while the other 29 are Q5_1; without this it alone would fall back to
// the per-route python path and dominate the step.
template <int VL, int ROWS>
struct Moe_down_q8_kernel {
    const fp16*   inter;       // [n_routed, K]
    const int8_t* qs;          // [E, N, K]
    const fp16*   sc;          // [E, N, K/32]
    const int*    sel_experts;
    const fp16*   topk_w;      // [n_routed]
    fp16*         out;         // [n_routed, N] partial
    int n_tokens, N, K, top_k;

    void operator()(sycl::id<2> idx) const SYCL_ESIMD_KERNEL {
        const int route = (int)idx[0];
        const int n0 = (int)idx[1] * ROWS;
        const int eid = sel_experts[route];
        const int Kg = K / 32;
        constexpr int VW = VL / 32;               // scale entries per tile
        const fp16* i_row = inter + (size_t)route * K;
        const size_t rbase = (size_t)eid * N + n0;

        simd<float, ROWS> dots = 0.0f;
        for (int k = 0; k < K; k += VL) {
            const int gi = k / 32;
            simd<fp16, VL>  iv = block_load<fp16, VL>(i_row + k);
            simd<float, VL> xf = simd<float, VL>(iv);
            #pragma unroll
            for (int r = 0; r < ROWS; r++) {
                simd<int8_t, VL> qr =
                    block_load<int8_t, VL>(qs + (rbase + r) * K + k);
                simd<fp16, VW>   sv = block_load<fp16, VW>(sc + (rbase + r) * Kg + gi);
                simd<float, VL>  wf = convert<float>(qr);
                #pragma unroll
                for (int c = 0; c < VW; c++) {
                    fp16 s = sv[c];
                    wf.template select<32, 1>(c * 32) =
                        wf.template select<32, 1>(c * 32) * (float)s;
                }
                dots[r] += reduce<float>(xf * wf, std::plus<>());
            }
        }
        const float tw = (float)topk_w[route];
        #pragma unroll
        for (int r = 0; r < ROWS; r++)
            out[(size_t)route * N + n0 + r] = fp16(dots[r] * tw);
    }
};

inline void moe_down_q8_host(
    const fp16* inter, const int8_t* qs, const fp16* sc, const int* sel,
    const fp16* topk_w, fp16* out_partial,
    int n_tokens, int N, int K, int top_k, sycl::queue& q) {
#define LAUNCH_MOE_DOWN_Q8(V, R)                                              \
    q.submit([&](sycl::handler& h) {                                          \
        h.parallel_for(sycl::range<2>((size_t)n_tokens * top_k, N / (R)),     \
            Moe_down_q8_kernel<V, R>{inter, qs, sc, sel, topk_w,              \
                                     out_partial, n_tokens, N, K, top_k});    \
    });

    const int R = (N % 4 == 0) ? 4 : 1;
    if (K % 256 == 0) {
        if (R == 4) { LAUNCH_MOE_DOWN_Q8(256, 4) } else { LAUNCH_MOE_DOWN_Q8(256, 1) }
    } else if (K % 128 == 0) {
        if (R == 4) { LAUNCH_MOE_DOWN_Q8(128, 4) } else { LAUNCH_MOE_DOWN_Q8(128, 1) }
    } else if (K % 64 == 0) {
        if (R == 4) { LAUNCH_MOE_DOWN_Q8(64, 4) } else { LAUNCH_MOE_DOWN_Q8(64, 1) }
    } else {
        if (R == 4) { LAUNCH_MOE_DOWN_Q8(32, 4) } else { LAUNCH_MOE_DOWN_Q8(32, 1) }
    }
#undef LAUNCH_MOE_DOWN_Q8
}

inline void moe_down_q5k_host(
    const fp16* inter, const uint8_t* ql, const uint8_t* qh, const fp16* sc,
    const fp16* mn, const int* sel, const fp16* topk_w, fp16* out_partial,
    int n_tokens, int N, int K, int top_k, sycl::queue& q,
    bool add_min = false) {
#define LAUNCH_MOE_DOWN_Q5K_S(V, R, A)                                        \
    q.submit([&](sycl::handler& h) {                                          \
        h.parallel_for(sycl::range<2>((size_t)n_tokens * top_k, N / (R)),     \
            Moe_down_q5k_kernel<V, R, A>{inter, ql, qh, sc, mn, sel, topk_w,  \
                                         out_partial, n_tokens, N, K, top_k});\
    });
#define LAUNCH_MOE_DOWN_Q5K(V, R)                                             \
    if (add_min) { LAUNCH_MOE_DOWN_Q5K_S(V, R, true) }                        \
    else         { LAUNCH_MOE_DOWN_Q5K_S(V, R, false) }

    const int R = (N % 4 == 0) ? 4 : 1;
    if (K % 256 == 0) {
        if (R == 4) { LAUNCH_MOE_DOWN_Q5K(256, 4) } else { LAUNCH_MOE_DOWN_Q5K(256, 1) }
    } else if (K % 128 == 0) {
        if (R == 4) { LAUNCH_MOE_DOWN_Q5K(128, 4) } else { LAUNCH_MOE_DOWN_Q5K(128, 1) }
    } else if (K % 64 == 0) {
        if (R == 4) { LAUNCH_MOE_DOWN_Q5K(64, 4) } else { LAUNCH_MOE_DOWN_Q5K(64, 1) }
    } else {
        if (R == 4) { LAUNCH_MOE_DOWN_Q5K(32, 4) } else { LAUNCH_MOE_DOWN_Q5K(32, 1) }
    }
#undef LAUNCH_MOE_DOWN_Q5K
#undef LAUNCH_MOE_DOWN_Q5K_S
}

inline void moe_down_q6k_host(
    const fp16* inter, const uint8_t* ql, const uint8_t* qh, const fp16* sc,
    const int* sel, const fp16* topk_w, fp16* out_partial,
    int n_tokens, int N, int K, int top_k, sycl::queue& q) {
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<2>((size_t)n_tokens * top_k, N),
            Moe_down_q6k_kernel{inter, ql, qh, sc, sel, topk_w, out_partial,
                                n_tokens, N, K, top_k});
    });
}
