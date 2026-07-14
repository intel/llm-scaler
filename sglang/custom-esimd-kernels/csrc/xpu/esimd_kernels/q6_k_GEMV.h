/* q6_k_GEMV.h — GGUF q6_K GEMV for Intel XPU (ESIMD), decode M=1.
 *
 * PACKED layout (temp gemv-gguf-quant style, ZERO extra memory vs GGUF):
 *   input   [1, K]      fp16
 *   ql      [N, K/2]    uint8  — 4 low bits, nibble (byte j: low->elem 2j,
 *                                high->elem 2j+1, sequential element order)
 *   qh      [N, K/4]    uint8  — upper 2 bits, 2-bit packed AND PRE-SHUFFLED
 *                                on host (per VL=512-elem tile: field p of
 *                                shuffled byte t -> element p*128+t, so the GPU
 *                                high-bit add is stride-1, not stride-4)
 *   scale   [N, K/16]   fp16   — = d * sc_int8  (per 16-block, may be negative)
 *   output  [1, N]      fp16
 *
 * value v6 = ql_nibble | (qh_2bit << 4) (0..63); SYMMETRIC dequant
 * w = scale * (v6 - 32). NO min (real GGML q6_K is symmetric, group=16).
 *
 * Processes K in VL=512-element tiles. grid = ceil(N/ROWS) x ROWS. N,K runtime.
 * Requires K % VL == 0 (Qwen3.5 q6_K: ffn_down K=9216, token_embd K=2560).
 * Validated: cc_workspace/tools/q5q6_packed_repack_ref.py (vs gguf-lib).
 *
 * Included into esimd_kernel.sycl (utils.h: fp16 + esimd namespace + detail).
 */
#pragma once

static constexpr int Q6_K_VL   = 512;
static constexpr int Q6_K_ROWS = 4;
static constexpr int Q6_K_GS   = 16;    // q6_K scale group (per-16, not 32)

struct Q6_K_gemv_kernel {
    const fp16*    input;   // [1, K]
    const uint8_t* ql;      // [N, K/2]
    const uint8_t* qh;      // [N, K/4] pre-shuffled
    const fp16*    scale;   // [N, K/16]
    fp16*          output;  // [1, N]
    int N, K;

    void operator()(sycl::nd_item<1> ndi) const SYCL_ESIMD_KERNEL {
        const int row = (int)ndi.get_group(0) * Q6_K_ROWS + (int)ndi.get_local_id(0);
        if (row >= N) return;

        constexpr int VL = Q6_K_VL;
        constexpr int VL_HALF = VL / 2;     // 256 ql bytes/tile
        constexpr int VL_QTR = VL / 4;      // 128 qh bytes/tile
        constexpr int VL_GS = VL / Q6_K_GS; // 32 scale per tile (group-16)
        const int K_ITERS = K / VL;
        const int QL_STRIDE = K / 2;
        const int QH_STRIDE = K / 4;
        const int SC_STRIDE = K / Q6_K_GS;

        simd<float, 8> acc(0.0f);
        int ai = 0;

        for (int iter = 0; iter < K_ITERS; iter++) {
            const int k = iter * VL;
            simd<fp16, VL> act = block_load<fp16, VL>(input + k);
            simd<float, VL> act_f = act;

            simd<uint8_t, VL_HALF> ql_data = block_load<uint8_t, VL_HALF>(
                ql + (size_t)row * QL_STRIDE + k / 2);
            simd<uint8_t, VL_QTR> qh_data = block_load<uint8_t, VL_QTR>(
                qh + (size_t)row * QH_STRIDE + k / 4);
            simd<fp16, VL_GS> sc_h = block_load<fp16, VL_GS>(
                scale + (size_t)row * SC_STRIDE + k / Q6_K_GS);
            simd<float, VL_GS> sc_f = sc_h;

            // ql nibble unpack (stride-2 interleave)
            simd<float, VL> weight_f;
            #pragma unroll
            for (int c = 0; c < VL_HALF / 64; c++) {
                auto p = ql_data.template select<64, 1>(c * 64);
                simd<float, 64> lo = p & 0x0F;
                simd<float, 64> hi = (p >> 4) & 0x0F;
                weight_f.template select<64, 2>(c * 128) = lo;
                weight_f.template select<64, 2>(c * 128 + 1) = hi;
            }

            // upper 2 bits — stride-1 contiguous (pre-shuffled)
            #pragma unroll
            for (int p = 0; p < 4; p++) {
                simd<uint8_t, VL_QTR> ext = (qh_data >> (2 * p)) & 3;
                simd<float, VL_QTR> ef = ext;
                weight_f.template select<VL_QTR, 1>(p * VL_QTR) += ef * 16.0f;
            }

            // SYMMETRIC dequant w = scale * (v6 - 32)  (per 16-block)
            #pragma unroll
            for (int sb = 0; sb < VL_GS; sb++) {
                float s = sc_f[sb];
                weight_f.template select<16, 1>(sb * 16) =
                    (weight_f.template select<16, 1>(sb * 16) - 32.0f) * s;
            }

            simd<float, VL> prod = weight_f * act_f;
            acc[ai] += esimd_detail::sum<float, float, VL>(prod);
            ai = (ai + 1) & 7;
        }
        output[row] = (fp16)esimd_detail::sum<float, float, 8>(acc);
    }
};

inline void q6_k_gemv_host(
    const fp16* input, const uint8_t* ql, const uint8_t* qh,
    const fp16* scale, fp16* output, uint32_t N, uint32_t K, sycl::queue& q) {
    const int NWG = ((int)N + Q6_K_ROWS - 1) / Q6_K_ROWS;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>((size_t)NWG * Q6_K_ROWS, Q6_K_ROWS),
            Q6_K_gemv_kernel{input, ql, qh, scale, output, (int)N, (int)K});
    });
}

// M-tiled q6_K GEMV (small M, e.g. MTP verify M<=16). The M>1 dense path
// dequantizes Q6_K -> a 2.5x-bigger fp16 table + generic GEMM (reads 1017MB at
// N=248320). This reads the 413MB Q6_K ONCE: per K-tile, load+unpack+dequant
// weight_f a single time, then multiply against ALL M activation rows (M
// accumulators). Much less weight BW than the fp16-table GEMM. input [M,K] row-
// major, output [M,N] row-major.
template <int M>
struct Q6_K_gemv_M_kernel {
    const fp16*    input;   // [M, K]
    const uint8_t* ql;      // [N, K/2]
    const uint8_t* qh;      // [N, K/4] pre-shuffled
    const fp16*    scale;   // [N, K/16]
    fp16*          output;  // [M, N]
    int N, K;

    void operator()(sycl::nd_item<1> ndi) const SYCL_ESIMD_KERNEL {
        const int row = (int)ndi.get_group(0) * Q6_K_ROWS + (int)ndi.get_local_id(0);
        if (row >= N) return;
        constexpr int VL = Q6_K_VL;
        constexpr int VL_HALF = VL / 2;
        constexpr int VL_QTR = VL / 4;
        constexpr int VL_GS = VL / Q6_K_GS;
        const int K_ITERS = K / VL;
        const int QL_STRIDE = K / 2;
        const int QH_STRIDE = K / 4;
        const int SC_STRIDE = K / Q6_K_GS;

        simd<float, 8> acc[M];
        #pragma unroll
        for (int m = 0; m < M; m++) acc[m] = 0.0f;
        int ai = 0;

        for (int iter = 0; iter < K_ITERS; iter++) {
            const int k = iter * VL;
            // --- load + unpack + dequant the weight tile ONCE ---
            simd<uint8_t, VL_HALF> ql_data = block_load<uint8_t, VL_HALF>(
                ql + (size_t)row * QL_STRIDE + k / 2);
            simd<uint8_t, VL_QTR> qh_data = block_load<uint8_t, VL_QTR>(
                qh + (size_t)row * QH_STRIDE + k / 4);
            simd<fp16, VL_GS> sc_h = block_load<fp16, VL_GS>(
                scale + (size_t)row * SC_STRIDE + k / Q6_K_GS);
            simd<float, VL_GS> sc_f = sc_h;
            simd<float, VL> weight_f;
            #pragma unroll
            for (int c = 0; c < VL_HALF / 64; c++) {
                auto p = ql_data.template select<64, 1>(c * 64);
                simd<float, 64> lo = p & 0x0F;
                simd<float, 64> hi = (p >> 4) & 0x0F;
                weight_f.template select<64, 2>(c * 128) = lo;
                weight_f.template select<64, 2>(c * 128 + 1) = hi;
            }
            #pragma unroll
            for (int p = 0; p < 4; p++) {
                simd<uint8_t, VL_QTR> ext = (qh_data >> (2 * p)) & 3;
                simd<float, VL_QTR> ef = ext;
                weight_f.template select<VL_QTR, 1>(p * VL_QTR) += ef * 16.0f;
            }
            #pragma unroll
            for (int sb = 0; sb < VL_GS; sb++) {
                float s = sc_f[sb];
                weight_f.template select<16, 1>(sb * 16) =
                    (weight_f.template select<16, 1>(sb * 16) - 32.0f) * s;
            }
            // --- reuse weight_f across all M activation rows ---
            #pragma unroll
            for (int m = 0; m < M; m++) {
                simd<fp16, VL> act = block_load<fp16, VL>(input + (size_t)m * K + k);
                simd<float, VL> prod = weight_f * simd<float, VL>(act);
                acc[m][ai] += esimd_detail::sum<float, float, VL>(prod);
            }
            ai = (ai + 1) & 7;
        }
        #pragma unroll
        for (int m = 0; m < M; m++)
            output[(size_t)m * N + row] = (fp16)esimd_detail::sum<float, float, 8>(acc[m]);
    }
};

template <int M>
inline void q6_k_gemv_M_launch(
    const fp16* input, const uint8_t* ql, const uint8_t* qh,
    const fp16* scale, fp16* output, uint32_t N, uint32_t K, sycl::queue& q) {
    const int NWG = ((int)N + Q6_K_ROWS - 1) / Q6_K_ROWS;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>((size_t)NWG * Q6_K_ROWS, Q6_K_ROWS),
            Q6_K_gemv_M_kernel<M>{input, ql, qh, scale, output, (int)N, (int)K});
    });
}

// Dispatch arbitrary M (2..16) onto fixed-M kernels by tiling in chunks; the
// common MTP verify M is exactly draft_token_num (<=8). Round up to {2,4,8,16}.
inline void q6_k_gemv_M_host(
    const fp16* input, const uint8_t* ql, const uint8_t* qh,
    const fp16* scale, fp16* output, uint32_t M, uint32_t N, uint32_t K,
    sycl::queue& q) {
    if (M == 1) { q6_k_gemv_host(input, ql, qh, scale, output, N, K, q); return; }
    // process in fixed tiles; each tile reads the weights once for its rows.
    uint32_t m0 = 0;
    while (m0 < M) {
        uint32_t r = M - m0;
        const fp16* in = input + (size_t)m0 * K;
        fp16* out = output + (size_t)m0 * N;
        if      (r >= 8) { q6_k_gemv_M_launch<8>(in, ql, qh, scale, out, N, K, q); m0 += 8; }
        else if (r >= 4) { q6_k_gemv_M_launch<4>(in, ql, qh, scale, out, N, K, q); m0 += 4; }
        else if (r >= 2) { q6_k_gemv_M_launch<2>(in, ql, qh, scale, out, N, K, q); m0 += 2; }
        else { q6_k_gemv_host(in, ql, qh, scale, out, N, K, q); m0 += 1; }
    }
}
