/* moe_up_fused.h — single-launch routed-up (Q4_K) + shared-up (Q8_0) stage.
 *
 * The two stages are independent: both only read the layer input `x`, and they
 * write disjoint buffers (inter_r / inter_sh+g). At M=1 decode the MoE op is
 * launch-bound (~30-48us of host per enqueue against ~5-12us of GPU per
 * kernel), so folding them into one enqueue removes 40 launches per step
 * without changing the work done.
 *
 * The grid is a concatenation of the two original grids:
 *
 *   group <  split : routed q4_k, work-item (group*K_SPLIT + lid) -> (route,col)
 *   group >= split : shared q8_0, work-group (group - split) exactly as before
 *
 * Every lane of a group takes the same side (the branch is on the group id
 * only), so the barriers inside the shared body stay uniform.
 *
 * Both bodies are reused verbatim through Moe_*_kernel::run(), so this file
 * carries no copy of the dequant math.
 */
#pragma once
#include <cstdlib>

template <int VL, int K_SPLIT>
struct Moe_up_fused_kernel {
    Moe_up_q4k_kernel<VL>        routed;
    Moe_shared_up_q8_kernel<K_SPLIT> shared;
    int split;      // first group index belonging to the shared stage
    int n_routed_wi;  // M * top_k * intermediate

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        slm_init<2 * K_SPLIT * sizeof(float)>();
        const int grp = (int)item.get_group(0);
        const int lid = (int)item.get_local_id(0);
        if (grp < split) {
            const int wi = grp * K_SPLIT + lid;
            if (wi < n_routed_wi)
                routed.run(wi / routed.intermediate, wi % routed.intermediate);
        } else {
            shared.run(grp - split, lid);
        }
    }
};

// Returns false when the shape/tuning combination is outside the fused fast
// path; the caller then falls back to the two separate launches.
inline bool moe_up_fused_host(
    const fp16* x,
    const uint8_t* gq, const fp16* gs, const fp16* gm,
    const uint8_t* uq, const fp16* us, const fp16* um,
    const int* sel, fp16* inter_r,
    const int8_t* gu_qs, const fp16* gu_sc, const fp16* wg,
    fp16* inter_sh, fp16* g,
    int M, int hidden, int intermediate, int top_k, int inter_s,
    sycl::queue& q) {
    // Escape hatch for A/B testing the fused path against the original pair.
    static const bool disabled = getenv("SGL_ESIMD_NO_MOE_UP_FUSE") != nullptr;
    if (disabled) return false;
    if (hidden % 128 != 0) return false;               // routed wants VL=128
    if ((hidden / 8) % MOE_Q8_GROUP != 0) return false;  // shared wants K_SPLIT=8
    constexpr int VL = 128, KS = 8;

    const int n_cols      = inter_s + (g != nullptr ? 1 : 0);
    const int n_routed_wi = M * top_k * intermediate;
    const int split       = (n_routed_wi + KS - 1) / KS;
    const int groups      = split + M * n_cols;

    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>((size_t)groups * KS, KS),
            Moe_up_fused_kernel<VL, KS>{
                Moe_up_q4k_kernel<VL>{x, gq, gs, gm, uq, us, um, sel, inter_r,
                                      M, hidden, intermediate, top_k},
                Moe_shared_up_q8_kernel<KS>{x, gu_qs, gu_sc, wg, inter_sh, g,
                                            M, hidden, inter_s, n_cols},
                split, n_routed_wi});
    });
    return true;
}
