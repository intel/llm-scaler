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

// The m loop must stay fully unrolled (vacc[m] needs a compile-time index), so
// every M row adds a live simd<float,VL> accumulator plus its act/prod
// temporaries. M=4/VL=256 fits; M=8/VL=256 is what the AOT gen compiler refuses
// to lower.
template <int M>
struct Q8_0_gemv_M_kernel {
    const fp16*   input;   // [M, K]
    const int8_t* weight;  // [N, K]
    const fp16*   scale;   // [N, K/32]
    fp16*         output;  // [M, N]
    int N, K;
    int MT;                // number of M-tiles folded into the grid

    void operator()(sycl::nd_item<1> ndi) const SYCL_ESIMD_KERNEL {
        // Groups are laid out mt-fastest so the MT groups that share a weight
        // row block are adjacent: tile mt+1 then finds that block still in L2
        // instead of re-pulling it from memory.
        const int g   = (int)ndi.get_group(0);
        const int mt  = g % MT;
        const int wgN = g / MT;
        const int row = wgN * Q8_0_M_ROWS + (int)ndi.get_local_id(0);
        if (row >= N) return;
        const fp16* input  = this->input  + (size_t)mt * M * K;
        fp16*       output = this->output + (size_t)mt * M * N;
        constexpr int VL    = Q8_0_M_VL;
        constexpr int VL_GS = VL / Q8_0_GROUP;   // scales per tile (8)
        const int K_ITERS  = K / VL;
        const int W_STRIDE = K;                  // int8 per row
        const int SC_STRIDE = K / Q8_0_GROUP;

        // Accumulate lane-wise and fold ONCE at the end. A per-iteration
        // reduce<>() would cost M * K_ITERS cross-lane reductions (128 per
        // work-item at M=4,K=2048) whose shuffles do not produce useful FLOPs,
        // and this kernel is ALU bound, not bandwidth bound.
        simd<float, VL> vacc[M];
        #pragma unroll
        for (int m = 0; m < M; m++) vacc[m] = 0.0f;

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
                vacc[m] += weight_f * simd<float, VL>(act);
            }
        }
        #pragma unroll
        for (int m = 0; m < M; m++)
            output[(size_t)m * N + row] = (fp16)reduce<float>(vacc[m], std::plus<>());
    }
};

// Launches MT M-tiles of M rows each in a SINGLE enqueue by folding the tile
// index into the grid. Each work-group still owns Q8_0_M_ROWS weight rows and
// M activation rows, so its register footprint is independent of MT.
template <int M>
inline void q8_0_gemv_M_launch(
    const fp16* input, const int8_t* weight, const fp16* scale, fp16* output,
    uint32_t N, uint32_t K, sycl::queue& q, int MT = 1) {
    const int NWG = ((int)N + Q8_0_M_ROWS - 1) / Q8_0_M_ROWS;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>((size_t)NWG * MT * Q8_0_M_ROWS, Q8_0_M_ROWS),
            Q8_0_gemv_M_kernel<M>{input, weight, scale, output,
                                  (int)N, (int)K, MT});
    });
}

// ===================================================================
// Small-M Q8_0 dense GEMV on XMX (DPAS).
//
// The XVE path above dequantizes the same weight tile once per 4-row M-tile, so
// at M=16 the dequant ALU is paid 4x while XMX sits idle. DPAS folds all M
// tokens into one systolic op, so the dequant is paid ONCE per weight tile
// regardless of M, and the MACs move off the saturated XVE pipe.
//
//   C[_M=MC weight rows, _N=NTOK tokens] = A[_M,_K] * B[_K,_N]
//   A = wsub, row-major   [_M,_K]      -> wsub[r*KK + k]
//   B = xsub, VNNI K-major[_K,_N]      -> xsub[k*NTOK + t]
//   acc row-major                      -> acc[r*NTOK + t]
// Note the ESIMD call order is dpas(C, B, A).
//
// MC=8 is the finest weight-row tile DPAS allows, so N=2048 yields only 256
// work-items — far short of the 2048 HW threads. K_SPLIT work-items per group
// each cover K/K_SPLIT and reduce through SLM to restore occupancy.
namespace xmx_ns = sycl::ext::intel::esimd::xmx;
namespace xesimd = sycl::ext::intel::experimental::esimd;

static constexpr int Q8_0_DPAS_MC   = 8;   // DPAS RepeatCount: weight rows/tile
static constexpr int Q8_0_DPAS_KK   = 16;  // DPAS _K for fp16 (SystolicDepth 8 x 2 ops/chan)
static constexpr int Q8_0_DPAS_NTOK = 16;  // DPAS ExecutionSize on BMG: token slots

// NTOK=16 covers the whole MTP verify batch (M<=16) in one tile, so the weight
// dequant is amortized over every token with no M-tiling at all.
template <int K_SPLIT>
struct Q8_0_gemv_dpas_kernel {
    const fp16*   input;   // [M, K]
    const int8_t* weight;  // [N, K]
    const fp16*   scale;   // [N, K/32]
    fp16*         output;  // [M, N]
    int N, K, M;

    void operator()(sycl::nd_item<1> ndi) const SYCL_ESIMD_KERNEL {
        constexpr int MC   = Q8_0_DPAS_MC;
        constexpr int KK   = Q8_0_DPAS_KK;
        constexpr int NTOK = Q8_0_DPAS_NTOK;
        constexpr int BS   = Q8_0_GROUP;      // 32 = 2 DPAS K-steps
        constexpr int NSUB = BS / KK;         // 2
        constexpr int BN   = KK * NTOK;       // 256: B operand elems per K-step
        constexpr int ACC  = MC * NTOK;       // 128 floats
        slm_init<K_SPLIT * ACC * sizeof(float)>();

        const int row0 = (int)ndi.get_group(0) * MC;
        const int lid  = (int)ndi.get_local_id(0);

        const int nblocks = K / BS;
        const int bpart   = nblocks / K_SPLIT;   // 32-blocks handled by this lid
        const int blk0    = lid * bpart;

        const fp16* s_base = scale + (size_t)row0 * nblocks;
        const simd<uint32_t, MC> scl_off =
            simd<uint32_t, MC>(0u, 1u) * (uint32_t)(nblocks * sizeof(fp16));
        // Token lanes past M are clamped onto the last valid row so the gather
        // stays in bounds; their accumulator columns are simply never stored.
        const simd<uint32_t, NTOK> in_off =
            min(simd<uint32_t, NTOK>(0u, 1u),
                simd<uint32_t, NTOK>((uint32_t)(M - 1))) *
            (uint32_t)(K * sizeof(fp16));

        simd<float, ACC> acc = 0.0f;

        for (int blk = blk0; blk < blk0 + bpart; blk++) {
            // --- dequant MC weight rows x BS values, ONCE for all M tokens ---
            // A operand is row-major [_M=MC, _K=KK]: wsub[r*KK + k].
            const simd<fp16, MC> sc = gather<fp16, MC>(s_base + blk, scl_off);
            simd<fp16, MC * BS> wfull;
            #pragma unroll
            for (int r = 0; r < MC; r++) {
                simd<int8_t, BS> raw = block_load<int8_t, BS>(
                    weight + (size_t)(row0 + r) * K + (size_t)blk * BS);
                wfull.template select<BS, 1>(r * BS) =
                    convert<fp16>(convert<float>(raw) * (float)sc[r]);
            }
            #pragma unroll
            for (int sub = 0; sub < NSUB; sub++) {
                // One gather yields the DPAS VNNI B operand directly: with
                // NElts=8 dwords across NTOK lanes the result is SoA, so dword j
                // of token t lands at j*NTOK + t — exactly the K-pair packing
                // DPAS expects. Assembling it with strided register moves
                // instead cost more than the DPAS itself saved.
                const uint32_t koff =
                    (uint32_t)((blk * BS + sub * KK) * sizeof(fp16));
                simd<uint32_t, BN / 2> xraw =
                    xesimd::lsc_gather<uint32_t, KK / 2,
                        xesimd::lsc_data_size::u32,
                        xesimd::cache_hint::cached,
                        xesimd::cache_hint::cached,
                        NTOK, uint32_t>(
                        reinterpret_cast<const uint32_t*>(input),
                        in_off + koff);
                simd<fp16, BN> xsub = xraw.template bit_cast_view<fp16>();

                simd<fp16, MC * KK> wsub;
                #pragma unroll
                for (int r = 0; r < MC; r++)
                    wsub.template select<KK, 1>(r * KK) =
                        wfull.template select<KK, 1>(r * BS + sub * KK);
                acc = xmx_ns::dpas<8, 8, float, float, fp16, fp16>(acc, xsub, wsub);
            }
        }

        // --- cross-lane K reduction through SLM ---
        if constexpr (K_SPLIT > 1) {
            slm_block_store<float, ACC>(lid * ACC * sizeof(float), acc);
            barrier();
            if (lid != 0) return;
            #pragma unroll
            for (int p = 1; p < K_SPLIT; p++)
                acc += slm_block_load<float, ACC>(p * ACC * sizeof(float));
        }

        #pragma unroll
        for (int t = 0; t < NTOK; t++) {
            if (t < M) {
                // acc[r*NTOK + t]: the MC rows for one token sit at stride NTOK
                simd<float, MC> col = acc.template select<MC, NTOK>(t);
                block_store<fp16, MC>(output + (size_t)t * N + row0,
                                      convert<fp16>(col));
            }
        }
    }
};

template <int K_SPLIT>
inline void q8_0_gemv_dpas_launch(
    const fp16* input, const int8_t* weight, const fp16* scale, fp16* output,
    uint32_t M, uint32_t N, uint32_t K, sycl::queue& q) {
    const int NWG = (int)N / Q8_0_DPAS_MC;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>((size_t)NWG * K_SPLIT, K_SPLIT),
            Q8_0_gemv_dpas_kernel<K_SPLIT>{input, weight, scale, output,
                                           (int)N, (int)K, (int)M});
    });
}

// DPAS runs at a near-constant cost from M=4 to M=16 because it is bandwidth
// bound, while the XVE path degrades with M; measured on BMG the crossover is
// at M=5 (at M=4 the XVE path is already at ~75% of the bandwidth roofline and
// DPAS would waste 12 of its 16 token lanes). It also needs N>=1024 for enough
// weight-row tiles, and loses on K>=4096 where the per-step setup stops
// amortizing. M>NTOK would need M-tiling, which the MTP verify batch never hits.
inline bool q8_0_gemv_dpas_ok(uint32_t M, uint32_t N, uint32_t K) {
    static const bool off = getenv("SGL_ESIMD_NO_Q8_DPAS") != nullptr;
    if (off) return false;
    if (M < 5 || M > (uint32_t)Q8_0_DPAS_NTOK) return false;
    if (N < 1024 || N % Q8_0_DPAS_MC != 0) return false;
    if (K > 2048) return false;
    return (K % (Q8_0_GROUP * 8)) == 0;
}

inline void q8_0_gemv_dpas_host(
    const fp16* input, const int8_t* weight, const fp16* scale, fp16* output,
    uint32_t M, uint32_t N, uint32_t K, sycl::queue& q) {
    q8_0_gemv_dpas_launch<8>(input, weight, scale, output, M, N, K, q);
}

// Dispatch arbitrary M onto fixed-M kernels. All floor(M/4) full tiles go out in
// ONE enqueue (the tile index lives in the grid, see q8_0_gemv_M_launch); the
// 0..3 leftover rows keep the {2,1} tiling. M==1 -> the K_SPLIT GEMV.
//
// A wider (8-row) tile was tried to halve the weight re-reads across tiles and
// measured slightly SLOWER: the mt-fastest group order already lets L2 serve
// the repeats, and fitting 8 rows in registers needs VL=128, which costs more
// than the traffic it saves.
//
// Requires K % Q8_0_M_VL (256) == 0 (dense proj K=2048/4096); caller falls back
// to oneDNN otherwise.
inline void q8_0_gemv_M_host(
    const fp16* input, const int8_t* weight, const fp16* scale, fp16* output,
    uint32_t M, uint32_t N, uint32_t K, sycl::queue& q) {
    if (M == 1) { q8_0_gemv_host(input, weight, scale, output, N, K, q); return; }
    if (q8_0_gemv_dpas_ok(M, N, K)) {
        q8_0_gemv_dpas_host(input, weight, scale, output, M, N, K, q);
        return;
    }
    uint32_t m0 = 0;
    if (M >= 4) {
        const uint32_t nt = M / 4;
        q8_0_gemv_M_launch<4>(input, weight, scale, output, N, K, q, (int)nt);
        m0 = nt * 4;
    }
    while (m0 < M) {
        uint32_t r = M - m0;
        const fp16* in = input + (size_t)m0 * K;
        fp16* out = output + (size_t)m0 * N;
        if      (r >= 4) { q8_0_gemv_M_launch<4>(in, weight, scale, out, N, K, q); m0 += 4; }
        else if (r >= 2) { q8_0_gemv_M_launch<2>(in, weight, scale, out, N, K, q); m0 += 2; }
        else             { q8_0_gemv_M_launch<1>(in, weight, scale, out, N, K, q); m0 += 1; }
    }
}
