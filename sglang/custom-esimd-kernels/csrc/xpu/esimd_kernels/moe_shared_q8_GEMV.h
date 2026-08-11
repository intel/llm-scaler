/* moe_shared_q8_GEMV.h — fused GGUF Q8_0 SHARED-expert MLP for Intel XPU
 * (ESIMD), decode. Collapses the ~6 python dispatches of the Qwen2MoE shared
 * expert (gate_up GEMV + SiLU_and_mul + down GEMV + shared_expert_gate GEMV +
 * sigmoid + mul) into ONE host op = ONE python dispatch:
 *
 *   g[t]        = sigmoid( sum_k x[t,k] * wg[k] )                 (gate scalar)
 *   inter[t,c]  = silu( <x[t], W_gate[c]> ) * <x[t], W_up[c]>     (Q8_0 gate/up)
 *   out[t,n]    = g[t] * sum_c inter[t,c] * W_down[n,c]           (Q8_0 down)
 *
 * Q8_0 rep (matches gguf.py _xpu_repack_q8_0): symmetric, group-32,
 *   qs   [N, K]     int8   (per-element quant)
 *   scale[N, K/32]  fp16   (per-group);  w[n,k] = scale[n,k/32] * qs[n,k].
 * gate_up is the MERGED rep: rows [0, inter_s) = gate, [inter_s, 2*inter_s) = up
 * (MergedColumnParallelLinear shard order gate,up). down K = inter_s.
 * K (=hidden for gate/up, =inter_s for down) is a multiple of 32 (Q8_0 block).
 *
 * Included into esimd_kernel.sycl after moe_kquant_GEMV.h (utils.h namespaces).
 */
#pragma once

static constexpr int MOE_Q8_GROUP = 32;

// ── Shared gate scalar: g[t] = sigmoid( dot(x[t], wg) ) ──────────────────────
struct Moe_shared_gate_kernel {
    const fp16* x;    // [M, hidden]
    const fp16* wg;   // [hidden]  (shared_expert_gate weight row 0)
    fp16*       g;    // [M]
    int hidden;

    void operator()(sycl::id<1> idx) const SYCL_ESIMD_KERNEL {
        const int token = (int)idx[0];
        const fp16* xr = x + (size_t)token * hidden;
        simd<float, MOE_Q8_GROUP> acc = 0.0f;
        for (int k = 0; k < hidden; k += MOE_Q8_GROUP) {
            simd<fp16, MOE_Q8_GROUP> xv = block_load<fp16, MOE_Q8_GROUP>(xr + k);
            simd<fp16, MOE_Q8_GROUP> wv = block_load<fp16, MOE_Q8_GROUP>(wg + k);
            acc += simd<float, MOE_Q8_GROUP>(xv) * simd<float, MOE_Q8_GROUP>(wv);
        }
        float d = reduce<float>(acc, std::plus<>());
        g[token] = fp16(1.0f / (1.0f + sycl::exp(-d)));
    }
};

inline void moe_shared_gate_host(
    const fp16* x, const fp16* wg, fp16* g, int M, int hidden, sycl::queue& q) {
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>((size_t)M),
            Moe_shared_gate_kernel{x, wg, g, hidden});
    });
}

// ── Up/gate stage: Q8_0 gate + Q8_0 up -> silu(gate)*up -> intermediate ───────
// One work-group per (token, col); K_SPLIT threads split the `hidden` reduction
// and combine through SLM (same structure as q8_0_gemv, which sustains ~470GB/s
// while the old flat range<2>(M, inter_s) launch only reached ~120GB/s because
// inter_s=256 work-items cannot fill the machine).
//
// The shared-expert gate scalar rides along as one extra column (col ==
// inter_s), which removes a whole kernel launch whose single work-item spent
// ~7.7us on a serial 64-iteration dependent load chain.
template <int K_SPLIT>
struct Moe_shared_up_q8_kernel {
    const fp16*   x;       // [M, hidden]
    const int8_t* gu_qs;   // [2*inter_s, hidden] (rows: gate then up)
    const fp16*   gu_sc;   // [2*inter_s, hidden/32]
    const fp16*   wg;      // [hidden]  shared_expert_gate row (may be null)
    fp16*         inter;   // [M, inter_s]
    fp16*         g;       // [M]       shared gate scalar (may be null)
    int M, hidden, inter_s, n_cols;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        slm_init<2 * K_SPLIT * sizeof(float)>();
        run((int)item.get_group(0), (int)item.get_local_id(0));
    }

    // Body without slm_init, so Moe_up_fused_kernel can call it after doing its
    // own (single, uniform) slm_init. Contains barriers -> must be entered by
    // every lane of the work-group, which holds because the caller branches on
    // the group id only.
    void run(int wgid, int lid) const {
        const int token = wgid / n_cols;
        const int col   = wgid % n_cols;
        const int Kg    = hidden / MOE_Q8_GROUP;
        const int kp    = hidden / K_SPLIT;
        const int kbeg  = lid * kp;
        const int kend  = kbeg + kp;
        const fp16* xr  = x + (size_t)token * hidden;

        if (col == inter_s) {  // shared-expert gate scalar
            simd<float, MOE_Q8_GROUP> acc = 0.0f;
            for (int k = kbeg; k < kend; k += MOE_Q8_GROUP) {
                simd<fp16, MOE_Q8_GROUP> xv = block_load<fp16, MOE_Q8_GROUP>(xr + k);
                simd<fp16, MOE_Q8_GROUP> wv = block_load<fp16, MOE_Q8_GROUP>(wg + k);
                acc += simd<float, MOE_Q8_GROUP>(xv) * simd<float, MOE_Q8_GROUP>(wv);
            }
            float part = reduce<float>(acc, std::plus<>());
            slm_block_store<float, 1>(lid * sizeof(float), simd<float, 1>(part));
            barrier();
            if (lid == 0) {
                simd<float, K_SPLIT> p = slm_block_load<float, K_SPLIT>(0);
                float d = reduce<float>(p, std::plus<>());
                g[token] = fp16(1.0f / (1.0f + sycl::exp(-d)));
            }
            return;
        }

        const int grow = col;
        const int urow = inter_s + col;
        const int8_t* gq = gu_qs + (size_t)grow * hidden;
        const fp16*   gs = gu_sc + (size_t)grow * Kg;
        const int8_t* uq = gu_qs + (size_t)urow * hidden;
        const fp16*   us = gu_sc + (size_t)urow * Kg;

        simd<float, MOE_Q8_GROUP> ag = 0.0f, au = 0.0f;
        int gi = kbeg / MOE_Q8_GROUP;
        for (int k = kbeg; k < kend; k += MOE_Q8_GROUP) {
            simd<fp16, MOE_Q8_GROUP> xv = block_load<fp16, MOE_Q8_GROUP>(xr + k);
            simd<float, MOE_Q8_GROUP> xf = simd<float, MOE_Q8_GROUP>(xv);
            simd<int8_t, MOE_Q8_GROUP> gr = block_load<int8_t, MOE_Q8_GROUP>(gq + k);
            simd<int8_t, MOE_Q8_GROUP> ur = block_load<int8_t, MOE_Q8_GROUP>(uq + k);
            float gsc = (float)gs[gi], usc = (float)us[gi];
            ag += xf * (convert<float>(gr) * gsc);
            au += xf * (convert<float>(ur) * usc);
            gi++;
        }
        float pg = reduce<float>(ag, std::plus<>());
        float pu = reduce<float>(au, std::plus<>());

        if constexpr (K_SPLIT == 1) {
            float silu = pg / (1.0f + sycl::exp(-pg));
            inter[(size_t)token * inter_s + col] = fp16(silu * pu);
        } else {
            slm_block_store<float, 1>(lid * sizeof(float), simd<float, 1>(pg));
            slm_block_store<float, 1>((K_SPLIT + lid) * sizeof(float),
                                      simd<float, 1>(pu));
            barrier();
            if (lid == 0) {
                simd<float, K_SPLIT> vg = slm_block_load<float, K_SPLIT>(0);
                simd<float, K_SPLIT> vu =
                    slm_block_load<float, K_SPLIT>(K_SPLIT * sizeof(float));
                float gate = reduce<float>(vg, std::plus<>());
                float up   = reduce<float>(vu, std::plus<>());
                float silu = gate / (1.0f + sycl::exp(-gate));
                inter[(size_t)token * inter_s + col] = fp16(silu * up);
            }
        }
    }
};

// wg/g may be null: then no gate column is launched (standalone use).
inline void moe_shared_up_q8_host(
    const fp16* x, const int8_t* gu_qs, const fp16* gu_sc,
    const fp16* wg, fp16* inter, fp16* g,
    int M, int hidden, int inter_s, sycl::queue& q) {
    const int n_cols = inter_s + (g != nullptr ? 1 : 0);
    int ks = 8;
    while (ks > 1 && (hidden / ks) % MOE_Q8_GROUP != 0) ks /= 2;

#define LAUNCH_MOE_SH_UP(S)                                                  \
    q.submit([&](sycl::handler& h) {                                         \
        h.parallel_for(                                                      \
            sycl::nd_range<1>((size_t)M * n_cols * (S), (S)),                \
            Moe_shared_up_q8_kernel<S>{x, gu_qs, gu_sc, wg, inter, g,        \
                                       M, hidden, inter_s, n_cols});         \
    });

    if      (ks == 8) { LAUNCH_MOE_SH_UP(8) }
    else if (ks == 4) { LAUNCH_MOE_SH_UP(4) }
    else if (ks == 2) { LAUNCH_MOE_SH_UP(2) }
    else              { LAUNCH_MOE_SH_UP(1) }
#undef LAUNCH_MOE_SH_UP
}

// ── Down stage: Q8_0 down -> * gate scalar -> shared output ───────────────────
// grid (M, hidden). Each WI: one hidden output col. K = inter_s.
struct Moe_shared_down_q8_kernel {
    const fp16*   inter;   // [M, inter_s]
    const int8_t* d_qs;    // [hidden, inter_s]
    const fp16*   d_sc;    // [hidden, inter_s/32]
    const fp16*   g;       // [M]
    fp16*         out;     // [M, hidden]
    int M, hidden, inter_s;

    void operator()(sycl::id<2> idx) const SYCL_ESIMD_KERNEL {
        const int token = (int)idx[0];
        const int n     = (int)idx[1];
        const int Kg    = inter_s / MOE_Q8_GROUP;
        const fp16*   ir = inter + (size_t)token * inter_s;
        const int8_t* dq = d_qs + (size_t)n * inter_s;
        const fp16*   ds = d_sc + (size_t)n * Kg;

        simd<float, MOE_Q8_GROUP> acc = 0.0f;
        int gi = 0;
        for (int k = 0; k < inter_s; k += MOE_Q8_GROUP) {
            simd<fp16, MOE_Q8_GROUP> iv = block_load<fp16, MOE_Q8_GROUP>(ir + k);
            simd<int8_t, MOE_Q8_GROUP> dr = block_load<int8_t, MOE_Q8_GROUP>(dq + k);
            float dsc = (float)ds[gi];
            acc += simd<float, MOE_Q8_GROUP>(iv) * (convert<float>(dr) * dsc);
            gi++;
        }
        float dot = reduce<float>(acc, std::plus<>());
        out[(size_t)token * hidden + n] = fp16(dot * (float)g[token]);
    }
};

inline void moe_shared_down_q8_host(
    const fp16* inter, const int8_t* d_qs, const fp16* d_sc, const fp16* g,
    fp16* out, int M, int hidden, int inter_s, sycl::queue& q) {
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<2>((size_t)M, hidden),
            Moe_shared_down_q8_kernel{inter, d_qs, d_sc, g, out,
                                      M, hidden, inter_s});
    });
}

// ═══════════════ FULL fusion: topk + routed + shared + finalize ═══════════════
// Mirrors the fp8 moe_forward_full: ONE host op = ONE python dispatch. Router
// GEMV stays in python (logits input); everything else is chained on the queue.

// ── Softmax over all experts + top-k heap select + renorm (per token) ─────────
// Ported from moe_batch/moe_topk.h (validated). n_experts padded to 512.
static constexpr int MOE_TOPK_PAD = 512;

// Fast top-k: the original kernel ran a 32-slot min-heap with one serial
// iteration per expert (n_experts=256 -> 256 dependent hmin/pack_mask/merge
// chains), costing ~20us for a single work-item. Instead we pack each score
// into an order-preserving uint32 key (monotonic fp16 bits << 16 | inverted
// lane id) and extract the top-k with `top_k` hmax reductions, i.e. 8 steps
// instead of 256.
//
// Weights are mathematically identical to "full softmax -> select -> renorm":
//   norm=true  -> softmax restricted to the selected logits
//   norm=false -> full softmax probability of the selected logits
// Both use the global max (== the top-1 logit) as the shift.
template <int PAD>
struct Moe_topk_gguf_kernel {
    const fp16* logits;   // [M, n_experts]
    int*        sel;      // [M*top_k]  expert ids
    fp16*       tw;       // [M*top_k]  renormalized weights
    int n_experts, top_k;
    bool norm;

    void operator()(sycl::id<1> idx) const SYCL_ESIMD_KERNEL {
        const int nid = (int)idx[0];
        const fp16* row_ptr = logits + (size_t)nid * n_experts;

        simd<fp16, PAD> scores(fp16(-65504.f));
        // Load whole 32-blocks, then the tail one element at a time. Reading a
        // full block past the end of the row would both run off the logits
        // allocation on the last token and clobber the -65504 padding, letting a
        // garbage lane win the hmax below and yield a selected expert id >=
        // n_experts -- which then indexes the expert weights out of range. The
        // tail loop is dead code whenever n_experts % 32 == 0, which is the case
        // for every shipped config (Qwen3.5-35B-A3B has 256 routed experts), so
        // the fast path is unchanged.
        int i = 0;
        for (; i + 32 <= n_experts; i += 32)
            scores.template select<32, 1>(i) = block_load<fp16, 32>(row_ptr + i);
        for (; i < n_experts; ++i)
            scores[i] = row_ptr[i];

        // fp16 bits -> monotonically ordered uint16 key.
        simd<uint16_t, PAD> bits = scores.template bit_cast_view<uint16_t>();
        simd<uint16_t, PAD> key  = bits | (uint16_t)0x8000;
        key.merge(~bits, (bits >> 15) != 0);

        // Low 16 bits break ties towards the smaller expert id.
        simd<uint32_t, PAD> lane(0, 1);
        simd<uint32_t, PAD> keys =
            (simd<uint32_t, PAD>(key) << 16) | ((uint32_t)(PAD - 1) - lane);

        int   top_i[32];
        float top_l[32];
        for (int k = 0; k < top_k; ++k) {
            uint32_t mx = hmax<uint32_t>(keys);
            int i = (int)((uint32_t)(PAD - 1) - (mx & 0xFFFFu));
            top_i[k] = i;
            fp16 sv = scores[i];
            top_l[k] = (float)sv;
            keys.merge(simd<uint32_t, PAD>(0), keys == mx);
        }

        const float mx_l = top_l[0];  // global max logit
        float denom;
        if (norm) {
            denom = 0.f;
            for (int k = 0; k < top_k; ++k) denom += sycl::exp(top_l[k] - mx_l);
        } else {
            simd<float, PAD> e = exp(convert<float>(scores) - mx_l);
            denom = reduce<float>(e, std::plus<>());
        }
        const float inv = 1.0f / denom;

        int*  idx_base = sel + (size_t)nid * top_k;
        fp16* w_base   = tw  + (size_t)nid * top_k;
        for (int k = 0; k < top_k; ++k) {
            idx_base[k] = top_i[k];
            w_base[k]   = fp16(sycl::exp(top_l[k] - mx_l) * inv);
        }
    }
};

inline void moe_topk_gguf_host(
    const fp16* logits, int* sel, fp16* tw,
    int M, int n_experts, int top_k, bool norm, sycl::queue& q) {
    q.submit([&](sycl::handler& h) {
        if (n_experts <= 256)
            h.parallel_for(sycl::range<1>((size_t)M),
                Moe_topk_gguf_kernel<256>{logits, sel, tw, n_experts, top_k, norm});
        else
            h.parallel_for(sycl::range<1>((size_t)M),
                Moe_topk_gguf_kernel<512>{logits, sel, tw, n_experts, top_k, norm});
    });
}

// ── Finalize: routed top_k combine + shared Q8_0 down + gate*sigmoid ──────────
// grid (M, hidden). final[t,n] = sum_{k<top_k} out_partial[t*top_k+k, n]
//                                + g[t] * sum_c inter_sh[t,c] * W_down_sh[n,c]
// VL = elements consumed per iteration; the old version stepped 32 at a time
// (8 tiny dependent loads for inter_s=256), which capped it near 140GB/s.
template <int VL>
struct Moe_finalize_gguf_kernel {
    const fp16*   out_partial; // [M*top_k, hidden]  (routed, topk_w already applied)
    const fp16*   inter_sh;    // [M, inter_s]
    const int8_t* d_qs;        // [hidden, inter_s]  shared down Q8_0
    const fp16*   d_sc;        // [hidden, inter_s/32]
    const fp16*   g;           // [M]  shared gate scalar
    fp16*         out;         // [M, hidden]
    int M, hidden, inter_s, top_k;

    void operator()(sycl::id<2> idx) const SYCL_ESIMD_KERNEL {
        const int token = (int)idx[0];
        const int n     = (int)idx[1];
        float acc = 0.0f;
        for (int k = 0; k < top_k; k++)
            acc += (float)out_partial[((size_t)token * top_k + k) * hidden + n];

        constexpr int NSC = VL / MOE_Q8_GROUP;
        const int Kg = inter_s / MOE_Q8_GROUP;
        const fp16*   ir = inter_sh + (size_t)token * inter_s;
        const int8_t* dq = d_qs + (size_t)n * inter_s;
        const fp16*   ds = d_sc + (size_t)n * Kg;

        simd<float, VL> sacc = 0.0f;
        for (int kk = 0; kk < inter_s; kk += VL) {
            simd<fp16, VL>    iv = block_load<fp16, VL>(ir + kk);
            simd<int8_t, VL>  dr = block_load<int8_t, VL>(dq + kk);
            simd<fp16, NSC>   sh = block_load<fp16, NSC>(ds + kk / MOE_Q8_GROUP);
            simd<float, VL>   wf = convert<float>(dr);
            #pragma unroll
            for (int sb = 0; sb < NSC; sb++) {
                fp16 sv = sh[sb];
                wf.template select<MOE_Q8_GROUP, 1>(sb * MOE_Q8_GROUP) =
                    wf.template select<MOE_Q8_GROUP, 1>(sb * MOE_Q8_GROUP) * (float)sv;
            }
            sacc += simd<float, VL>(iv) * wf;
        }
        float sdot = reduce<float>(sacc, std::plus<>());
        out[(size_t)token * hidden + n] = fp16(acc + (float)g[token] * sdot);
    }
};

inline void moe_finalize_gguf_host(
    const fp16* out_partial, const fp16* inter_sh, const int8_t* d_qs,
    const fp16* d_sc, const fp16* g, fp16* out,
    int M, int hidden, int inter_s, int top_k, sycl::queue& q) {
#define LAUNCH_MOE_FIN(V)                                                     \
    q.submit([&](sycl::handler& h) {                                          \
        h.parallel_for(sycl::range<2>((size_t)M, hidden),                     \
            Moe_finalize_gguf_kernel<V>{out_partial, inter_sh, d_qs, d_sc, g, \
                                        out, M, hidden, inter_s, top_k});     \
    });

    if      (inter_s % 256 == 0) { LAUNCH_MOE_FIN(256) }
    else if (inter_s % 128 == 0) { LAUNCH_MOE_FIN(128) }
    else if (inter_s % 64  == 0) { LAUNCH_MOE_FIN(64)  }
    else                         { LAUNCH_MOE_FIN(32)  }
#undef LAUNCH_MOE_FIN
}
