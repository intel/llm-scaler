/* moe_down_fin_fused.h — single-launch routed-down (Q5_K) + finalize stage.
 *
 * The unfused pair is
 *
 *   down:     out_part[t*top_k+k, n] = topk_w[t*top_k+k]
 *                                    * sum_c inter_r[t*top_k+k, c] * Wd[eid,n,c]
 *   finalize: out[t, n] = sum_k out_part[t*top_k+k, n]
 *                       + g[t] * sum_c inter_sh[t, c] * Wsh[n, c]
 *
 * The only reason `out_part` exists is that `down` splits the top_k reduction
 * across work-items. Giving one work-item the whole (token, col-tile) instead
 * lets it accumulate over k in registers, so the finalize launch AND the
 * [M*top_k, hidden] scratch round-trip both disappear.
 *
 * Decode/verify is host-bound here (~48us of host per enqueue against ~10-30us
 * of GPU per kernel), so trading work-item count for one fewer launch is the
 * right side of the deal even though the grid shrinks by top_k.
 *
 * The dequant math is not duplicated: both halves call the existing
 * Moe_down_q5k_kernel::dots_for() / Moe_finalize_gguf_kernel::shared_dot().
 */
#pragma once
#include <cstdlib>
#include <cstring>

// VL  = routed-down K tile (Moe_down_q5k_kernel)
// ROWS= output cols per work-item
// VLS = shared-down inter_s tile (Moe_finalize_gguf_kernel)
template <int VL, int ROWS, int VLS>
struct Moe_down_fin_q5k_kernel {
    Moe_down_q5k_kernel<VL, ROWS>  down;
    Moe_finalize_gguf_kernel<VLS>  fin;
    int top_k;

    void operator()(sycl::id<2> idx) const SYCL_ESIMD_KERNEL {
        const int token = (int)idx[0];
        const int n0    = (int)idx[1] * ROWS;
        const int N     = down.N;

        simd<float, ROWS> acc = 0.0f;
        for (int k = 0; k < top_k; k++) {
            const int route = token * top_k + k;
            acc += down.dots_for(route, n0) * (float)down.topk_w[route];
        }

        const float gv = (float)fin.g[token];
        #pragma unroll
        for (int r = 0; r < ROWS; r++)
            acc[r] += gv * fin.shared_dot(token, n0 + r);

        #pragma unroll
        for (int r = 0; r < ROWS; r++)
            fin.out[(size_t)token * (size_t)N + n0 + r] = fp16(acc[r]);
    }
};

// Returns false when the shape/tuning combination is outside the fused fast
// path; the caller then falls back to the two separate launches.
//
// ROWS trades work-item count for register reuse. Fusing already divides the
// grid by top_k (one work-item now owns the whole top_k reduction), so at small
// M a large ROWS starves the machine: M=5, hidden=2048, ROWS=4 leaves 2560
// work-items where the unfused down had 20480. ROWS is therefore picked from M
// to keep the grid near MOE_DF_WI_TARGET work-items.
static constexpr int MOE_DF_WI_TARGET = 8192;

inline bool moe_down_fin_q5k_host(
    const fp16* inter, const uint8_t* ql, const uint8_t* qh, const fp16* sc,
    const fp16* mn, const int* sel, const fp16* topk_w,
    const fp16* inter_sh, const int8_t* d_qs, const fp16* d_sc, const fp16* g,
    fp16* out,
    int M, int hidden, int intermediate, int top_k, int inter_s,
    sycl::queue& q) {
    // Escape hatch for A/B testing the fused path against the original pair.
    static const bool disabled = getenv("SGL_ESIMD_NO_MOE_DOWN_FIN_FUSE") != nullptr;
    if (disabled) return false;
    const int K = intermediate;

    // Override for tuning; 0/unset = pick from M.
    static const int rows_env = []() {
        const char* s = getenv("SGL_ESIMD_MOE_DF_ROWS");
        return s ? atoi(s) : 0;
    }();
    int rows = rows_env;
    if (rows <= 0) {
        rows = 4;
        while (rows > 1 && (size_t)M * (hidden / rows) < MOE_DF_WI_TARGET)
            rows >>= 1;
    }
    if (rows != 1 && rows != 2 && rows != 4) return false;
    if (hidden % rows != 0) return false;

    // Only the (VL, VLS) pairs the GGUF decode path actually hits are
    // instantiated; anything else keeps the unfused pair.
#define LAUNCH_MOE_DF(V, VS, R)                                               \
    q.submit([&](sycl::handler& h) {                                          \
        h.parallel_for(sycl::range<2>((size_t)M, hidden / (R)),               \
            Moe_down_fin_q5k_kernel<V, R, VS>{                                \
                Moe_down_q5k_kernel<V, R>{inter, ql, qh, sc, mn, sel,         \
                                          topk_w, nullptr, M, hidden, K,      \
                                          top_k},                             \
                Moe_finalize_gguf_kernel<VS>{nullptr, inter_sh, d_qs, d_sc,   \
                                             g, out, M, hidden, inter_s,      \
                                             top_k},                          \
                top_k});                                                      \
    });                                                                       \
    return true;

#define LAUNCH_MOE_DF_ROWS(V, VS)                                             \
    if (rows == 4) { LAUNCH_MOE_DF(V, VS, 4) }                                \
    else if (rows == 2) { LAUNCH_MOE_DF(V, VS, 2) }                           \
    else { LAUNCH_MOE_DF(V, VS, 1) }

    if (K % 256 == 0) {
        if      (inter_s % 256 == 0) { LAUNCH_MOE_DF_ROWS(256, 256) }
        else if (inter_s % 128 == 0) { LAUNCH_MOE_DF_ROWS(256, 128) }
    } else if (K % 128 == 0) {
        if      (inter_s % 256 == 0) { LAUNCH_MOE_DF_ROWS(128, 256) }
        else if (inter_s % 128 == 0) { LAUNCH_MOE_DF_ROWS(128, 128) }
    }
#undef LAUNCH_MOE_DF_ROWS
#undef LAUNCH_MOE_DF
    return false;
}
