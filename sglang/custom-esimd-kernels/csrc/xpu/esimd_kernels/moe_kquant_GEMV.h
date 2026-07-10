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

// ── Up/gate stage: Q4_K gate + Q4_K up -> silu(gate)*up -> intermediate ──────
// grid (n_routed = n_tokens*top_k, intermediate_size). Each WI: one n_col.
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
        const int route = (int)idx[0];
        const int n_col = (int)idx[1];
        const int token = route / top_k;
        const int eid = sel_experts[route];

        const int Kh = hidden / 2;           // packed bytes per row
        const int Kg = hidden / MOE_Q4K_GROUP;
        const fp16* x_row = x + (size_t)token * hidden;
        const uint8_t* gq = gate_ql + ((size_t)eid * intermediate + n_col) * Kh;
        const fp16*    gs = gate_sc + ((size_t)eid * intermediate + n_col) * Kg;
        const fp16*    gm = gate_mn + ((size_t)eid * intermediate + n_col) * Kg;
        const uint8_t* uq = up_ql + ((size_t)eid * intermediate + n_col) * Kh;
        const fp16*    us = up_sc + ((size_t)eid * intermediate + n_col) * Kg;
        const fp16*    um = up_mn + ((size_t)eid * intermediate + n_col) * Kg;

        simd<float, MOE_Q4K_HALF> ag_e = 0.0f, ag_o = 0.0f, au_e = 0.0f, au_o = 0.0f;
        int gi = 0;
        for (int k = 0; k < hidden; k += MOE_Q4K_GROUP) {
            simd<fp16, MOE_Q4K_GROUP> iv = block_load<fp16, MOE_Q4K_GROUP>(x_row + k);
            simd<float, MOE_Q4K_HALF> ie = iv.template select<MOE_Q4K_HALF, 2>(0);
            simd<float, MOE_Q4K_HALF> io = iv.template select<MOE_Q4K_HALF, 2>(1);

            simd<uint8_t, MOE_Q4K_HALF> graw = block_load<uint8_t, MOE_Q4K_HALF>(gq + k / 2);
            simd<uint16_t, MOE_Q4K_HALF> g16 = convert<uint16_t>(graw);
            float gsc = (float)gs[gi], gmn = (float)gm[gi];
            ag_e += ie * (convert<float>(g16 & 0x000F) * gsc - gmn);
            ag_o += io * (convert<float>((g16 >> 4) & 0x000F) * gsc - gmn);

            simd<uint8_t, MOE_Q4K_HALF> uraw = block_load<uint8_t, MOE_Q4K_HALF>(uq + k / 2);
            simd<uint16_t, MOE_Q4K_HALF> u16 = convert<uint16_t>(uraw);
            float usc = (float)us[gi], umn = (float)um[gi];
            au_e += ie * (convert<float>(u16 & 0x000F) * usc - umn);
            au_o += io * (convert<float>((u16 >> 4) & 0x000F) * usc - umn);
            gi++;
        }
        float g = reduce<float>(ag_e, std::plus<>()) + reduce<float>(ag_o, std::plus<>());
        float u = reduce<float>(au_e, std::plus<>()) + reduce<float>(au_o, std::plus<>());
        float silu = g / (1.0f + sycl::exp(-g));
        inter[(size_t)route * intermediate + n_col] = fp16(silu * u);
    }
};

inline void moe_up_q4k_host(
    const fp16* x, const uint8_t* gq, const fp16* gs, const fp16* gm,
    const uint8_t* uq, const fp16* us, const fp16* um,
    const int* sel, fp16* inter,
    int n_tokens, int hidden, int intermediate, int top_k, sycl::queue& q) {
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<2>((size_t)n_tokens * top_k, intermediate),
            Moe_up_q4k_kernel{x, gq, gs, gm, uq, us, um, sel, inter,
                              n_tokens, hidden, intermediate, top_k});
    });
}

// ── Down stage: PACKED (zero extra memory, mirrors q5_k/q6_k_GEMV.h) ─────────
// grid (n_routed, hidden). Each WI: dot(inter, dequant(down[eid,h_col,:])) *
// topk_w -> per-route partial [n_routed, hidden] (host sums over top_k).
// intermediate (=K of the down weight) must be a multiple of VL=512 (the
// pre-shuffle tile) — true for the 35B (inter=512).
static constexpr int MOE_DOWN_VL = 512;

// Q5_K down: ql [E,N,K/2] nibble + qh [E,N,K/8] pre-shuffled 1-bit +
// scale,min [E,N,K/32] fp16. v5 = nibble|(qh<<4); w = scale*v5 - min.
struct Moe_down_q5k_kernel {
    const fp16*    inter;      // [n_routed, K]
    const uint8_t* ql;         // [E, N, K/2]
    const uint8_t* qh;         // [E, N, K/8] pre-shuffled
    const fp16*    sc;         // [E, N, K/32]
    const fp16*    mn;         // [E, N, K/32]
    const int*     sel_experts;
    const fp16*    topk_w;     // [n_routed]
    fp16*          out;        // [n_routed, N] partial
    int n_tokens, N, K, top_k;

    void operator()(sycl::id<2> idx) const SYCL_ESIMD_KERNEL {
        const int route = (int)idx[0];
        const int n = (int)idx[1];            // hidden output col
        const int eid = sel_experts[route];
        const int Kh = K / 2, Kq = K / 8, Kg = K / 32;
        const fp16*    i_row = inter + (size_t)route * K;
        const uint8_t* qlr = ql + ((size_t)eid * N + n) * Kh;
        const uint8_t* qhr = qh + ((size_t)eid * N + n) * Kq;
        const fp16*    scr = sc + ((size_t)eid * N + n) * Kg;
        const fp16*    mnr = mn + ((size_t)eid * N + n) * Kg;

        simd<float, MOE_DOWN_VL> acc = 0.0f;  // reused per tile, sum at end
        float dot = 0.0f;
        constexpr int VL = MOE_DOWN_VL, VL2 = VL / 2, VL8 = VL / 8, VLG = VL / 32;
        for (int t = 0; t < K / VL; t++) {
            const int kb = t * VL;
            simd<fp16, VL> iv = block_load<fp16, VL>(i_row + kb);
            simd<uint8_t, VL2> qd = block_load<uint8_t, VL2>(qlr + kb / 2);
            simd<float, VL> wf;
            #pragma unroll
            for (int c = 0; c < VL2 / 64; c++) {
                auto p = qd.template select<64, 1>(c * 64);
                wf.template select<64, 2>(c * 128) = p & 0x0F;
                wf.template select<64, 2>(c * 128 + 1) = (p >> 4) & 0x0F;
            }
            simd<uint8_t, VL8> qhd = block_load<uint8_t, VL8>(qhr + kb / 8);
            #pragma unroll
            for (int b = 0; b < 8; b++) {
                simd<float, VL8> ef = (qhd >> b) & 1;
                wf.template select<VL8, 1>(b * VL8) += ef * 16.0f;
            }
            #pragma unroll
            for (int sb = 0; sb < VLG; sb++) {
                float s = (float)scr[t * VLG + sb], m = (float)mnr[t * VLG + sb];
                wf.template select<32, 1>(sb * 32) =
                    wf.template select<32, 1>(sb * 32) * s - m;
            }
            acc = simd<float, VL>(iv) * wf;
            dot += esimd_detail2::sum<float, float, VL>(acc);
        }
        out[(size_t)route * N + n] = fp16(dot * (float)topk_w[route]);
    }
};

// Q6_K down: ql [E,N,K/2] nibble + qh [E,N,K/4] pre-shuffled 2-bit +
// scale [E,N,K/16] fp16. v6 = nibble|(qh<<4); w = scale*(v6-32). symmetric.
struct Moe_down_q6k_kernel {
    const fp16*    inter;
    const uint8_t* ql;         // [E, N, K/2]
    const uint8_t* qh;         // [E, N, K/4] pre-shuffled 2-bit
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

        float dot = 0.0f;
        constexpr int VL = MOE_DOWN_VL, VL2 = VL / 2, VLQ = VL / 4, VLG = VL / 16;
        for (int t = 0; t < K / VL; t++) {
            const int kb = t * VL;
            simd<fp16, VL> iv = block_load<fp16, VL>(i_row + kb);
            simd<uint8_t, VL2> qd = block_load<uint8_t, VL2>(qlr + kb / 2);
            simd<float, VL> wf;
            #pragma unroll
            for (int c = 0; c < VL2 / 64; c++) {
                auto p = qd.template select<64, 1>(c * 64);
                wf.template select<64, 2>(c * 128) = p & 0x0F;
                wf.template select<64, 2>(c * 128 + 1) = (p >> 4) & 0x0F;
            }
            simd<uint8_t, VLQ> qhd = block_load<uint8_t, VLQ>(qhr + kb / 4);
            #pragma unroll
            for (int p = 0; p < 4; p++) {
                simd<float, VLQ> ef = (qhd >> (2 * p)) & 3;
                wf.template select<VLQ, 1>(p * VLQ) += ef * 16.0f;
            }
            #pragma unroll
            for (int sb = 0; sb < VLG; sb++) {
                float s = (float)scr[t * VLG + sb];
                wf.template select<16, 1>(sb * 16) =
                    (wf.template select<16, 1>(sb * 16) - 32.0f) * s;
            }
            simd<float, VL> prod = simd<float, VL>(iv) * wf;
            dot += esimd_detail2::sum<float, float, VL>(prod);
        }
        out[(size_t)route * N + n] = fp16(dot * (float)topk_w[route]);
    }
};

inline void moe_down_q5k_host(
    const fp16* inter, const uint8_t* ql, const uint8_t* qh, const fp16* sc,
    const fp16* mn, const int* sel, const fp16* topk_w, fp16* out_partial,
    int n_tokens, int N, int K, int top_k, sycl::queue& q) {
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<2>((size_t)n_tokens * top_k, N),
            Moe_down_q5k_kernel{inter, ql, qh, sc, mn, sel, topk_w, out_partial,
                                n_tokens, N, K, top_k});
    });
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
