/* q4_k_GEMV.h — GGUF q4_K GEMV for Intel XPU (ESIMD), decode M=1.
 *
 * GGML block_q4_K (ggml-common.h:87, QK_K=256): {half2 dm; u8 scales[12];
 * u8 qs[128]}, 8 sub-blocks of 32, ASYMMETRIC with 6-bit sub-scale + 6-bit
 * sub-min. The 6-bit unpack (get_scale_min_k4) and dall/dmin multiply are done
 * on the HOST (q4_k_repack_ref.py, skill Stage 1) so the GPU sees per-32-block
 * fp16 scale + fp16 min — identical granularity to q4_0's group-32.
 *
 * Consumes the interleaved repack (same nibble layout as q4_0_GEMV):
 *   input   [1, K]      fp16
 *   weight  [N, K/2]    uint8  (byte j: low nibble -> elem 2j, high -> elem 2j+1)
 *   scale   [N, K/32]   fp16   (= dall * sc6, pre-computed)
 *   min     [N, K/32]   fp16   (= dmin * mn6, pre-computed)
 *   output  [1, N]      fp16
 *
 * dequant: w[k] = scale[k/32] * nibble[k] - min[k/32]   (nibble in 0..15)
 *
 * Structure: the default M=1 path is Q4_K_gemv_wide_kernel, a VL=512 K-tile
 * loop (grid = ceil(N/ROWS) x ROWS), matching q5_K/q6_K. The older
 * Q4_K_gemv_kernel (q4_0_GEMV.h-style K_SPLIT + SLM reduce) is kept as the
 * fallback for shards whose K is not a multiple of 256. The only algorithmic
 * difference from q4_0 is: q4_0 does (nibble-8)*scale (symmetric), q4_K does
 * nibble*scale - min (asymmetric). Same interleaved deinterleave.
 *
 * Included into esimd_kernel.sycl (utils.h provides fp16 + esimd namespace).
 */
#pragma once

// Alias also (re)declared in q5_k_GEMV.h; repeating a namespace alias with the
// same target is legal and keeps this header include-order independent.
namespace esimd_detail = sycl::ext::intel::esimd::detail;

static constexpr int Q4_K_GROUP = 32;  // q4_K sub-block size
static constexpr int Q4_K_HALF = 16;   // qs bytes per 32-block (= group/2)

static constexpr int Q4_K_VL   = 512;  // K-tile (wide path, M=1 and M-tiled)
static constexpr int Q4_K_ROWS = 4;    // rows per work-group (wide path)

inline void select_ks_q4_k(uint32_t N, uint32_t K, int& ks) {
    ks = 1;
    if      (N <= 128 && K >= 2048) ks = 8;
    else if (N <= 512 && K >= 2048) ks = 4;
    int kp = K / ks;
    while ((kp % Q4_K_GROUP != 0) && ks > 1) {
        ks /= 2;
        kp = K / ks;
    }
}

template <int K_SPLIT>
struct Q4_K_gemv_kernel {
    const fp16*    input;   // [1, K]
    const uint8_t* weight;  // [N, K/2]
    const fp16*    scale;   // [N, K/32]
    const fp16*    minv;    // [N, K/32]
    fp16*          output;  // [1, N]
    int N, K;
    int n_groups;           // K / 32

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        if constexpr (K_SPLIT > 1) {
            slm_init<K_SPLIT * sizeof(float)>();
        }
        int n   = item.get_group(0);
        int lid = item.get_local_id(0);
        if (n >= N) return;

        int kp = K / K_SPLIT;
        int kstart = lid * kp;

        // Even/odd K-position accumulators (interleaved layout, like q4_0).
        simd<float, Q4_K_HALF> acc_even = 0.0f;
        simd<float, Q4_K_HALF> acc_odd  = 0.0f;

        const uint8_t* w_row = weight + (size_t)n * (K / 2);
        const fp16*    s_row = scale  + (size_t)n * n_groups;
        const fp16*    m_row = minv   + (size_t)n * n_groups;
        int group_idx = kstart / Q4_K_GROUP;

        for (int k = kstart; k < kstart + kp; k += Q4_K_GROUP) {
            // input: 32 fp16 -> deinterleave into even[16] + odd[16]
            simd<fp16, Q4_K_GROUP> iv = block_load<fp16, Q4_K_GROUP>(input + k);
            simd<float, Q4_K_HALF> in_even = iv.template select<Q4_K_HALF, 2>(0);
            simd<float, Q4_K_HALF> in_odd  = iv.template select<Q4_K_HALF, 2>(1);

            // weight: 16 packed bytes -> low nibble = even K, high = odd K
            simd<uint8_t, Q4_K_HALF> raw =
                block_load<uint8_t, Q4_K_HALF>(w_row + k / 2);
            simd<uint16_t, Q4_K_HALF> u16 = convert<uint16_t>(raw);
            simd<float, Q4_K_HALF> nib_even = convert<float>(u16 & 0x000F);
            simd<float, Q4_K_HALF> nib_odd  = convert<float>((u16 >> 4) & 0x000F);

            // per-32-block scale + min (asymmetric): w = scale*nibble - min
            float s = static_cast<float>(s_row[group_idx]);
            float m = static_cast<float>(m_row[group_idx]);
            group_idx += 1;
            simd<float, Q4_K_HALF> w_even = nib_even * s - m;
            simd<float, Q4_K_HALF> w_odd  = nib_odd  * s - m;

            acc_even += in_even * w_even;
            acc_odd  += in_odd  * w_odd;
        }

        float my_sum = reduce<float>(acc_even, std::plus<>())
                     + reduce<float>(acc_odd,  std::plus<>());

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

// ---- Wide-tile q4_K GEMV (M=1 decode, default path) ----
//
// The K_SPLIT kernel above steps 32 elements at a time, which issues 16-byte
// nibble loads (below LSC granularity) and re-reads a scalar scale/min per
// block. This is the same shape that was measured at ~180 GB/s in
// moe_kquant_GEMV.h before it was widened. Since q4_K carries ~2/3 of the
// weight bytes in a Q4_K_M file, the M=1 decode path was the dominant term in
// the step time while q5_K/q6_K had already moved to the VL=512 structure.
//
// This kernel is the q4_K instance of that same structure: one VL=512 tile per
// iteration, so the weight load becomes block_load<uint8_t, 256> and the
// scale/min are loaded as VL_GS-wide vectors instead of scalars. Grid is
// ceil(N/ROWS) x ROWS with one row per work-item, matching q5_K/q6_K.
//
// The dequant/unpack body is identical to Q4_K_gemv_M_kernel's (already
// validated) with M fixed to 1.
template <int VLP>
struct Q4_K_gemv_wide_kernel {
    const fp16*    input;   // [1, K]
    const uint8_t* weight;  // [N, K/2]
    const fp16*    scale;   // [N, K/32]
    const fp16*    minv;    // [N, K/32]
    fp16*          output;  // [1, N]
    int N, K;

    void operator()(sycl::nd_item<1> ndi) const SYCL_ESIMD_KERNEL {
        const int row = (int)ndi.get_group(0) * Q4_K_ROWS + (int)ndi.get_local_id(0);
        if (row >= N) return;

        constexpr int VL = VLP;
        constexpr int VL_HALF = VL / 2;         // packed bytes per tile
        constexpr int VL_GS = VL / Q4_K_GROUP;  // scale/min entries per tile
        const int K_ITERS = K / VL;
        const int W_STRIDE = K / 2;
        const int SC_STRIDE = K / Q4_K_GROUP;

        const uint8_t* w_row = weight + (size_t)row * W_STRIDE;
        const fp16*    s_row = scale  + (size_t)row * SC_STRIDE;
        const fp16*    m_row = minv   + (size_t)row * SC_STRIDE;

        // 8 rotating accumulators break the serial dependency between the
        // per-tile horizontal sums (same trick as q5_K/q6_K).
        simd<float, 8> acc(0.0f);
        int ai = 0;

        for (int iter = 0; iter < K_ITERS; iter++) {
            const int k = iter * VL;
            simd<fp16, VL> act = block_load<fp16, VL>(input + k);

            simd<uint8_t, VL_HALF> w_data =
                block_load<uint8_t, VL_HALF>(w_row + k / 2);
            simd<fp16, VL_GS> sc_h =
                block_load<fp16, VL_GS>(s_row + k / Q4_K_GROUP);
            simd<fp16, VL_GS> mn_h =
                block_load<fp16, VL_GS>(m_row + k / Q4_K_GROUP);
            simd<float, VL_GS> sc_f = sc_h, mn_f = mn_h;

            // nibble unpack: byte j low -> elem 2j, high -> elem 2j+1
            simd<float, VL> weight_f;
            #pragma unroll
            for (int c = 0; c < VL_HALF / 64; c++) {
                auto p = w_data.template select<64, 1>(c * 64);
                simd<float, 64> lo = p & 0x0F;
                simd<float, 64> hi = (p >> 4) & 0x0F;
                weight_f.template select<64, 2>(c * 128) = lo;
                weight_f.template select<64, 2>(c * 128 + 1) = hi;
            }
            // asymmetric dequant w = scale*nibble - min (per 32-block)
            #pragma unroll
            for (int sb = 0; sb < VL_GS; sb++) {
                float s = sc_f[sb], m = mn_f[sb];
                weight_f.template select<Q4_K_GROUP, 1>(sb * Q4_K_GROUP) =
                    weight_f.template select<Q4_K_GROUP, 1>(sb * Q4_K_GROUP) * s - m;
            }

            simd<float, VL> prod = weight_f * simd<float, VL>(act);
            acc[ai] += esimd_detail::sum<float, float, VL>(prod);
            ai = (ai + 1) & 7;
        }
        output[row] = (fp16)esimd_detail::sum<float, float, 8>(acc);
    }
};

inline void q4_k_gemv_host(
    const fp16* input, const uint8_t* weight, const fp16* scale,
    const fp16* minv, fp16* output, uint32_t N, uint32_t K, sycl::queue& q) {
    // Wide path requires K % VL == 0. Every GGUF k-quant tensor has
    // K % 256 == 0 (256 = super-block), and TP row-splits keep that property,
    // so the 512/256 pair covers every shard seen in practice. The K_SPLIT
    // kernel below stays as the fallback for anything else.
    //
    // Kill switch: the wide path wins the isolated microbenchmark by ~2.6x but
    // was measured to LOSE ~1.8 ms of e2e decode on a launch-bound step, where
    // the extra work-groups it spawns compete with host dispatch. Set
    // SGL_XPU_Q4K_WIDE=0 to fall back to the K_SPLIT kernel.
    static const bool wide_enabled = [] {
        const char* e = std::getenv("SGL_XPU_Q4K_WIDE");
        return !(e && e[0] == '0');
    }();
    const bool wide512 = wide_enabled && (K % Q4_K_VL) == 0;
    const bool wide256 = wide_enabled && (K % (Q4_K_VL / 2)) == 0;
    if (wide512 || wide256) {
        const int NWG = ((int)N + Q4_K_ROWS - 1) / Q4_K_ROWS;
        q.submit([&](sycl::handler& h) {
            sycl::nd_range<1> r((size_t)NWG * Q4_K_ROWS, Q4_K_ROWS);
            if (wide512) {
                h.parallel_for(r, Q4_K_gemv_wide_kernel<Q4_K_VL>{
                    input, weight, scale, minv, output, (int)N, (int)K});
            } else {
                h.parallel_for(r, Q4_K_gemv_wide_kernel<Q4_K_VL / 2>{
                    input, weight, scale, minv, output, (int)N, (int)K});
            }
        });
        return;
    }

    int n_groups = K / Q4_K_GROUP;
    int ks;
    select_ks_q4_k(N, K, ks);
    int global = N * ks;
    int local = ks;

#define LAUNCH_Q4_K(S)                                                  \
    q.submit([&](sycl::handler& h) {                                    \
        h.parallel_for(sycl::nd_range<1>(global, local),                \
            Q4_K_gemv_kernel<S>{input, weight, scale, minv, output,     \
                                (int)N, (int)K, n_groups});             \
    });

    if      (ks == 1) { LAUNCH_Q4_K(1) }
    else if (ks == 2) { LAUNCH_Q4_K(2) }
    else if (ks == 4) { LAUNCH_Q4_K(4) }
    else if (ks == 8) { LAUNCH_Q4_K(8) }
    else              { LAUNCH_Q4_K(1) }
#undef LAUNCH_Q4_K
}

// ---- M-tiled q4_K GEMV (small M: MTP verify, or plain decode at batch>1) ----
//
// The generic M>1 path dequantizes q4_K into a fp16 table (4x the bytes) on
// EVERY call and then runs a dense GEMM; the dequant cost is independent of M,
// so at M=2 it already dominates. This kernel loads+unpacks+dequants each
// K-tile ONCE and multiplies it against all M activation rows, keeping the
// weights resident in their 4.5-bit form. input [M,K], output [M,N] row-major.
//
// Uses the same VL=512 tile structure as q5_K/q6_K rather than the M=1
// K_SPLIT structure: with M rows to amortize, per-row work-group splitting is
// no longer needed and a tile loop keeps the weight registers live.

// VLP is the K-tile length. 512 is the fast default; 256 covers shards whose K
// is not a multiple of 512 (gemma-4 hidden_size 5376). Every K-quant tensor has
// K % 256 == 0 because that is the GGUF super-block size, so the two
// instantiations together span every possible shard.
template <int M, int VLP>
struct Q4_K_gemv_M_kernel {
    const fp16*    input;   // [M, K]
    const uint8_t* weight;  // [N, K/2]
    const fp16*    scale;   // [N, K/32]
    const fp16*    minv;    // [N, K/32]
    fp16*          output;  // [M, N]
    int N, K;
    // Row stride of `output`, so a caller can write a column slice of a
    // wider buffer (the GGUF mixed-kind group path) without a torch.cat.
    int ldo;

    void operator()(sycl::nd_item<1> ndi) const SYCL_ESIMD_KERNEL {
        const int row = (int)ndi.get_group(0) * Q4_K_ROWS + (int)ndi.get_local_id(0);
        if (row >= N) return;

        constexpr int VL = VLP;
        constexpr int VL_HALF = VL / 2;         // 256 packed bytes/tile
        constexpr int VL_GS = VL / Q4_K_GROUP;  // 16 scale/min per tile
        const int K_ITERS = K / VL;
        const int W_STRIDE = K / 2;
        const int SC_STRIDE = K / Q4_K_GROUP;

        // Accumulate lane-wise and fold ONCE at the end. A per-iteration
        // sum<VL>() costs M * K_ITERS cross-lane reduction trees whose shuffles
        // produce no useful FLOPs; at M=16 that tree is about half of the
        // per-row instruction count and this kernel is ALU bound.
        // Folding the VL-wide product into an AW-wide accumulator as VL/AW
        // strided FMAs issues the same number of instructions as one VL-wide
        // FMA, so AW can stay small: AW*4*M is 4 KB at M=16, against the 32 KB
        // a full VL-wide accumulator would need (the GRF is 8 KB).
        constexpr int AW = 64;
        simd<float, AW> vacc[M];
        #pragma unroll
        for (int m = 0; m < M; m++) vacc[m] = 0.0f;

        for (int iter = 0; iter < K_ITERS; iter++) {
            const int k = iter * VL;
            // --- load + unpack + dequant the weight tile ONCE ---
            simd<uint8_t, VL_HALF> w_data = block_load<uint8_t, VL_HALF>(
                weight + (size_t)row * W_STRIDE + k / 2);
            simd<fp16, VL_GS> sc_h = block_load<fp16, VL_GS>(
                scale + (size_t)row * SC_STRIDE + k / Q4_K_GROUP);
            simd<fp16, VL_GS> mn_h = block_load<fp16, VL_GS>(
                minv + (size_t)row * SC_STRIDE + k / Q4_K_GROUP);
            simd<float, VL_GS> sc_f = sc_h, mn_f = mn_h;

            // nibble unpack: byte j low -> elem 2j, high -> elem 2j+1
            simd<float, VL> weight_f;
            #pragma unroll
            for (int c = 0; c < VL_HALF / 64; c++) {
                auto p = w_data.template select<64, 1>(c * 64);
                simd<float, 64> lo = p & 0x0F;
                simd<float, 64> hi = (p >> 4) & 0x0F;
                weight_f.template select<64, 2>(c * 128) = lo;
                weight_f.template select<64, 2>(c * 128 + 1) = hi;
            }
            // asymmetric dequant w = scale*nibble - min (per 32-block)
            #pragma unroll
            for (int sb = 0; sb < VL_GS; sb++) {
                float s = sc_f[sb], m = mn_f[sb];
                weight_f.template select<Q4_K_GROUP, 1>(sb * Q4_K_GROUP) =
                    weight_f.template select<Q4_K_GROUP, 1>(sb * Q4_K_GROUP) * s - m;
            }
            // --- reuse weight_f across all M activation rows ---
            #pragma unroll
            for (int m = 0; m < M; m++) {
                simd<fp16, VL> act = block_load<fp16, VL>(input + (size_t)m * K + k);
                #pragma unroll
                for (int c = 0; c < VL / AW; c++)
                    vacc[m] += weight_f.template select<AW, 1>(c * AW) *
                               simd<float, AW>(act.template select<AW, 1>(c * AW));
            }
        }
        #pragma unroll
        for (int m = 0; m < M; m++)
            output[(size_t)m * ldo + row] = (fp16)esimd_detail::sum<float, float, AW>(vacc[m]);
    }
};

template <int M>
inline void q4_k_gemv_M_launch(
    const fp16* input, const uint8_t* weight, const fp16* scale,
    const fp16* minv, fp16* output, uint32_t N, uint32_t K, uint32_t ldo, sycl::queue& q) {
    const int NWG = ((int)N + Q4_K_ROWS - 1) / Q4_K_ROWS;
    const bool wide = (K % Q4_K_VL) == 0;
    q.submit([&](sycl::handler& h) {
        sycl::nd_range<1> r((size_t)NWG * Q4_K_ROWS, Q4_K_ROWS);
        if (wide) {
            h.parallel_for(
                r, Q4_K_gemv_M_kernel<M, Q4_K_VL>{input, weight, scale, minv, output,
                                                  (int)N, (int)K, (int)ldo});
        } else {
            h.parallel_for(
                r, Q4_K_gemv_M_kernel<M, Q4_K_VL / 2>{input, weight, scale, minv, output,
                                                      (int)N, (int)K, (int)ldo});
        }
    });
}

// Dispatch arbitrary M onto fixed-M kernels by tiling in chunks of
// {16,8,4,2,1}. Each tile streams the whole weight matrix once, so the
// widest tile that fits M decides how many times the weights are read:
// M=16 (MTP verify at concurrency 4 x 4 draft tokens) costs one pass with
// the 16-tile but two with the 8-tile. M only adds one AW-wide accumulator
// per row, while the weight/product registers are M-independent.
// q4_k_gemv_M_launch picks the VL=512 or VL=256 tile instantiation from K, so
// every K is served by the tiled kernel (no per-row M=1 fan-out).
inline void q4_k_gemv_M_host(
    const fp16* input, const uint8_t* weight, const fp16* scale,
    const fp16* minv, fp16* output, uint32_t M, uint32_t N, uint32_t K, uint32_t ldo,
    sycl::queue& q) {
    uint32_t m0 = 0;
    while (m0 < M) {
        uint32_t r = M - m0;
        const fp16* in = input + (size_t)m0 * K;
        fp16* out = output + (size_t)m0 * ldo;
        if      (r >= 16) { q4_k_gemv_M_launch<16>(in, weight, scale, minv, out, N, K, ldo, q); m0 += 16; }
        else if (r >= 8) { q4_k_gemv_M_launch<8>(in, weight, scale, minv, out, N, K, ldo, q); m0 += 8; }
        else if (r >= 4) { q4_k_gemv_M_launch<4>(in, weight, scale, minv, out, N, K, ldo, q); m0 += 4; }
        else if (r >= 2) { q4_k_gemv_M_launch<2>(in, weight, scale, minv, out, N, K, ldo, q); m0 += 2; }
        else { q4_k_gemv_host(in, weight, scale, minv, out, N, K, q); m0 += 1; }
    }
}
