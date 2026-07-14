/* q8_0_GEMV.h — GGUF q8_0 GEMV for Intel XPU (ESIMD), decode M=1.
 *
 * GGML block_q8_0 = { half d; int8 qs[32] }, group=32, SYMMETRIC (no min).
 * Real dequant (dequantize.cuh:71): w = d * qs, qs signed int8. We match this
 * exactly — NOT the skill's synthetic uint8+min q8_0 asset.
 *
 * Consumes the split-buffer repack (q8_0_repack_ref.py, bit-exact vs gguf-lib):
 *   input   [1, K]      fp16
 *   weight  [N, K]      int8   (signed quants, contiguous per row)
 *   scale   [N, K/32]   fp16   (per-block d)
 *   output  [1, N]      fp16
 *
 * dequant: w[n,k] = scale[n, k/32] * (float)qs[n,k]
 *
 * Structure mirrors q4_0_GEMV.h (K_SPLIT threads/WG + SLM reduce) rather than
 * the skill's ROWS=4 BMG layout — q8_0 in Qwen3.5 is the N=32 ssm_alpha/beta
 * tensors, so grid=N gives only 32 WGs; K_SPLIT keeps the 12 PTL cores busy.
 *
 * Included into esimd_kernel.sycl (utils.h provides fp16 + esimd namespace).
 */
#pragma once

static constexpr int Q8_0_GROUP = 32;  // elements per q8_0 block

// VL fixed at 32 (one q8_0 block per iter -> one scale load/iter).
// K_SPLIT distributes the K reduction across threads; kp = K/K_SPLIT must stay
// a multiple of 32 so no block is split across threads.
inline void select_ks_q8_0(uint32_t N, uint32_t K, int& ks) {
    ks = 1;
    if      (N <= 128 && K >= 2048) ks = 8;
    else if (N <= 512 && K >= 2048) ks = 4;
    int kp = K / ks;
    while ((kp % Q8_0_GROUP != 0) && ks > 1) {
        ks /= 2;
        kp = K / ks;
    }
}

template <int K_SPLIT>
struct Q8_0_gemv_kernel {
    const fp16*   input;   // [1, K]
    const int8_t* weight;  // [N, K]
    const fp16*   scale;   // [N, K/32]
    fp16*         output;  // [1, N]
    int N, K;
    int n_groups;          // K / 32

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        if constexpr (K_SPLIT > 1) {
            slm_init<K_SPLIT * sizeof(float)>();
        }
        int n   = item.get_group(0);
        int lid = item.get_local_id(0);
        if (n >= N) return;

        int kp = K / K_SPLIT;
        int kstart = lid * kp;

        simd<float, Q8_0_GROUP> acc = 0.0f;

        const int8_t* w_row = weight + (size_t)n * K;
        const fp16*   s_row = scale  + (size_t)n * n_groups;
        int group_idx = kstart / Q8_0_GROUP;

        for (int k = kstart; k < kstart + kp; k += Q8_0_GROUP) {
            // input: 32 fp16
            simd<fp16, Q8_0_GROUP> iv = block_load<fp16, Q8_0_GROUP>(input + k);
            // weight: 32 signed int8 -> float
            simd<int8_t, Q8_0_GROUP> raw = block_load<int8_t, Q8_0_GROUP>(w_row + k);
            simd<float, Q8_0_GROUP> wf = convert<float>(raw);
            // one fp16 scale per 32-block
            float s = static_cast<float>(s_row[group_idx]);
            group_idx += 1;
            acc += simd<float, Q8_0_GROUP>(iv) * (wf * s);
        }

        float my_sum = reduce<float>(acc, std::plus<>());

        if constexpr (K_SPLIT == 1) {
            output[n] = fp16(my_sum);
        } else {
            slm_block_store<float, 1>(lid * sizeof(float), simd<float, 1>(my_sum));
            barrier();
            if (lid == 0) {
                simd<float, K_SPLIT> parts = slm_block_load<float, K_SPLIT>(0);
                output[n] = fp16(reduce<float>(parts, std::plus<>()));
            }
        }
    }
};

inline void q8_0_gemv_host(
    const fp16* input, const int8_t* weight, const fp16* scale, fp16* output,
    uint32_t N, uint32_t K, sycl::queue& q) {
    int n_groups = K / Q8_0_GROUP;
    int ks;
    select_ks_q8_0(N, K, ks);
    int global = N * ks;
    int local = ks;

#define LAUNCH_Q8_0(S)                                                  \
    q.submit([&](sycl::handler& h) {                                    \
        h.parallel_for(sycl::nd_range<1>(global, local),                \
            Q8_0_gemv_kernel<S>{input, weight, scale, output,           \
                                (int)N, (int)K, n_groups});             \
    });

    if      (ks == 1) { LAUNCH_Q8_0(1) }
    else if (ks == 2) { LAUNCH_Q8_0(2) }
    else if (ks == 4) { LAUNCH_Q8_0(4) }
    else if (ks == 8) { LAUNCH_Q8_0(8) }
    else              { LAUNCH_Q8_0(1) }
#undef LAUNCH_Q8_0
}

// ===================================================================
// Small-M (M in 2..16) Q8_0 dense GEMV — weights-read-once across M rows.
//
// For the MTP target-verify forward the dense attn projections (qkv/o_proj,
// Q8_0) run at M = draft_token_num (2..4). The M==1 GEMV would relaunch per
// row, and the M>1 path was routing to oneDNN jit:gemm:any which CACHE-MISSes
// and JIT-recompiles for EACH new (M,K,N) shape (a major XPU-time hog).
// This kernel mirrors q6_k_GEMV.h's M-tiled design: one weight row per
// work-item, dequant the int8 tile ONCE per K-block, reuse across all M
// activation rows. grid = N/ROWS WGs (N is large for dense proj, plenty).
//
//   input  [M, K]    fp16
//   weight [N, K]    int8   (signed q8_0 quants, contiguous per row)
//   scale  [N, K/32] fp16   (per-32-block d)
//   output [M, N]    fp16
// dequant: w[n,k] = scale[n,k/32] * (float)qs[n,k]  (same as M=1 kernel)

static constexpr int Q8_0_M_VL   = 256;  // elems/iter (8 q8_0 blocks); K%256==0 for 2048/4096
static constexpr int Q8_0_M_ROWS = 4;    // weight rows per work-group

template <int M>
struct Q8_0_gemv_M_kernel {
    const fp16*   input;   // [M, K]
    const int8_t* weight;  // [N, K]
    const fp16*   scale;   // [N, K/32]
    fp16*         output;  // [M, N]
    int N, K;

    void operator()(sycl::nd_item<1> ndi) const SYCL_ESIMD_KERNEL {
        const int row = (int)ndi.get_group(0) * Q8_0_M_ROWS + (int)ndi.get_local_id(0);
        if (row >= N) return;
        constexpr int VL    = Q8_0_M_VL;
        constexpr int VL_GS = VL / Q8_0_GROUP;   // scales per tile (8)
        const int K_ITERS  = K / VL;
        const int W_STRIDE = K;                  // int8 per row
        const int SC_STRIDE = K / Q8_0_GROUP;

        simd<float, 8> acc[M];
        #pragma unroll
        for (int m = 0; m < M; m++) acc[m] = 0.0f;
        int ai = 0;

        for (int iter = 0; iter < K_ITERS; iter++) {
            const int k = iter * VL;
            // --- load + dequant the weight tile ONCE ---
            simd<int8_t, VL> raw = block_load<int8_t, VL>(
                weight + (size_t)row * W_STRIDE + k);
            simd<fp16, VL_GS> sc_h = block_load<fp16, VL_GS>(
                scale + (size_t)row * SC_STRIDE + k / Q8_0_GROUP);
            simd<float, VL_GS> sc_f = sc_h;
            simd<float, VL> weight_f = convert<float>(raw);
            #pragma unroll
            for (int sb = 0; sb < VL_GS; sb++) {
                weight_f.template select<Q8_0_GROUP, 1>(sb * Q8_0_GROUP) =
                    weight_f.template select<Q8_0_GROUP, 1>(sb * Q8_0_GROUP) * sc_f[sb];
            }
            // --- reuse weight_f across all M activation rows ---
            #pragma unroll
            for (int m = 0; m < M; m++) {
                simd<fp16, VL> act = block_load<fp16, VL>(input + (size_t)m * K + k);
                simd<float, VL> prod = weight_f * simd<float, VL>(act);
                acc[m][ai] += reduce<float>(prod, std::plus<>());
            }
            ai = (ai + 1) & 7;
        }
        #pragma unroll
        for (int m = 0; m < M; m++)
            output[(size_t)m * N + row] = (fp16)reduce<float>(acc[m], std::plus<>());
    }
};

template <int M>
inline void q8_0_gemv_M_launch(
    const fp16* input, const int8_t* weight, const fp16* scale, fp16* output,
    uint32_t N, uint32_t K, sycl::queue& q) {
    const int NWG = ((int)N + Q8_0_M_ROWS - 1) / Q8_0_M_ROWS;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>((size_t)NWG * Q8_0_M_ROWS, Q8_0_M_ROWS),
            Q8_0_gemv_M_kernel<M>{input, weight, scale, output, (int)N, (int)K});
    });
}

// Dispatch arbitrary M onto fixed-M kernels by tiling in {4,2,1} chunks; each
// tile reads the weights once for its M rows. M==1 -> the K_SPLIT GEMV. The MTP
// verify M = draft_token_num (typically <=4), so {4,2} covers the hot case; >4
// just re-tiles (weights re-read per tile, still far cheaper than oneDNN JIT).
// NO <8> tile: the AOT gen compiler (-device ptl-u) fails to lower the <8>
// instantiation (large live simd<float,VL> set); {4,2} link fine and cover
// draft_token_num. Requires K % Q8_0_M_VL (256) == 0 (dense proj K=2048/4096);
// caller falls back to oneDNN otherwise.
inline void q8_0_gemv_M_host(
    const fp16* input, const int8_t* weight, const fp16* scale, fp16* output,
    uint32_t M, uint32_t N, uint32_t K, sycl::queue& q) {
    if (M == 1) { q8_0_gemv_host(input, weight, scale, output, N, K, q); return; }
    uint32_t m0 = 0;
    while (m0 < M) {
        uint32_t r = M - m0;
        const fp16* in = input + (size_t)m0 * K;
        fp16* out = output + (size_t)m0 * N;
        if      (r >= 4) { q8_0_gemv_M_launch<4>(in, weight, scale, out, N, K, q); m0 += 4; }
        else if (r >= 2) { q8_0_gemv_M_launch<2>(in, weight, scale, out, N, K, q); m0 += 2; }
        else             { q8_0_gemv_M_launch<1>(in, weight, scale, out, N, K, q); m0 += 1; }
    }
}
