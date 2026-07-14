/* q5_k_GEMV.h — GGUF q5_K GEMV for Intel XPU (ESIMD), decode M=1.
 *
 * PACKED layout (temp gemv-gguf-quant style, ZERO extra memory vs GGUF):
 *   input   [1, K]      fp16
 *   ql      [N, K/2]    uint8  — 4 low bits, nibble (byte j: low->elem 2j,
 *                                high->elem 2j+1, sequential element order)
 *   qh      [N, K/8]    uint8  — 5th bit, 1-bit packed AND PRE-SHUFFLED on host
 *                                (per VL=512-elem tile: bit b of shuffled byte t
 *                                 -> element b*64+t, so the GPU high-bit add is
 *                                 a stride-1 contiguous write, not stride-8)
 *   scale   [N, K/32]   fp16   — = dall * sc6  (per 32-block)
 *   min     [N, K/32]   fp16   — = dmin * mn6
 *   output  [1, N]      fp16
 *
 * value v5 = ql_nibble | (qh_bit << 4) (0..31); dequant w = scale*v5 - min.
 *
 * Processes K in VL=512-element tiles (pre-shuffle is per-tile, so no K-split
 * inside a tile). grid = ceil(N/ROWS) work-groups x ROWS rows. N, K runtime.
 * Requires K % VL == 0 (true for all Qwen3.5 q5_K: K=2560,4096). Validated
 * layout: cc_workspace/tools/q5q6_packed_repack_ref.py (vs gguf-lib).
 *
 * Included into esimd_kernel.sycl (utils.h: fp16 + esimd namespace + detail).
 */
#pragma once

namespace esimd_detail = sycl::ext::intel::esimd::detail;

static constexpr int Q5_K_VL   = 512;   // K-tile (matches host pre-shuffle chunk)
static constexpr int Q5_K_ROWS = 4;     // rows per work-group
static constexpr int Q5_K_GS   = 32;    // scale/min group

struct Q5_K_gemv_kernel {
    const fp16*    input;   // [1, K]
    const uint8_t* ql;      // [N, K/2]
    const uint8_t* qh;      // [N, K/8] pre-shuffled
    const fp16*    scale;   // [N, K/32]
    const fp16*    minv;    // [N, K/32]
    fp16*          output;  // [1, N]
    int N, K;

    void operator()(sycl::nd_item<1> ndi) const SYCL_ESIMD_KERNEL {
        const int row = (int)ndi.get_group(0) * Q5_K_ROWS + (int)ndi.get_local_id(0);
        if (row >= N) return;

        constexpr int VL = Q5_K_VL;
        constexpr int VL_HALF = VL / 2;     // 256 ql bytes/tile
        constexpr int VL_8TH = VL / 8;      // 64 qh bytes/tile
        constexpr int VL_GS = VL / Q5_K_GS; // 16 scale/min per tile
        const int K_ITERS = K / VL;
        const int QL_STRIDE = K / 2;
        const int QH_STRIDE = K / 8;
        const int SC_STRIDE = K / Q5_K_GS;

        simd<float, 8> acc(0.0f);
        int ai = 0;

        for (int iter = 0; iter < K_ITERS; iter++) {
            const int k = iter * VL;
            simd<fp16, VL> act = block_load<fp16, VL>(input + k);
            simd<float, VL> act_f = act;

            simd<uint8_t, VL_HALF> ql_data = block_load<uint8_t, VL_HALF>(
                ql + (size_t)row * QL_STRIDE + k / 2);
            simd<uint8_t, VL_8TH> qh_data = block_load<uint8_t, VL_8TH>(
                qh + (size_t)row * QH_STRIDE + k / 8);
            simd<fp16, VL_GS> sc_h = block_load<fp16, VL_GS>(
                scale + (size_t)row * SC_STRIDE + k / Q5_K_GS);
            simd<fp16, VL_GS> mn_h = block_load<fp16, VL_GS>(
                minv + (size_t)row * SC_STRIDE + k / Q5_K_GS);
            simd<float, VL_GS> sc_f = sc_h, mn_f = mn_h;

            // ql nibble unpack (stride-2 interleave: lo->even, hi->odd)
            simd<float, VL> weight_f;
            #pragma unroll
            for (int c = 0; c < VL_HALF / 64; c++) {
                auto p = ql_data.template select<64, 1>(c * 64);
                simd<float, 64> lo = p & 0x0F;
                simd<float, 64> hi = (p >> 4) & 0x0F;
                weight_f.template select<64, 2>(c * 128) = lo;
                weight_f.template select<64, 2>(c * 128 + 1) = hi;
            }

            // 5th bit — stride-1 contiguous (pre-shuffled)
            #pragma unroll
            for (int bit = 0; bit < 8; bit++) {
                simd<uint8_t, VL_8TH> ext = (qh_data >> bit) & 1;
                simd<float, VL_8TH> ef = ext;
                weight_f.template select<VL_8TH, 1>(bit * VL_8TH) += ef * 16.0f;
            }

            // dequant w = scale*v5 - min  (per 32-block)
            #pragma unroll
            for (int sb = 0; sb < VL_GS; sb++) {
                float s = sc_f[sb], m = mn_f[sb];
                weight_f.template select<32, 1>(sb * 32) =
                    weight_f.template select<32, 1>(sb * 32) * s - m;
            }

            simd<float, VL> prod = weight_f * act_f;
            acc[ai] += esimd_detail::sum<float, float, VL>(prod);
            ai = (ai + 1) & 7;
        }
        output[row] = (fp16)esimd_detail::sum<float, float, 8>(acc);
    }
};

inline void q5_k_gemv_host(
    const fp16* input, const uint8_t* ql, const uint8_t* qh,
    const fp16* scale, const fp16* minv, fp16* output,
    uint32_t N, uint32_t K, sycl::queue& q) {
    const int NWG = ((int)N + Q5_K_ROWS - 1) / Q5_K_ROWS;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>((size_t)NWG * Q5_K_ROWS, Q5_K_ROWS),
            Q5_K_gemv_kernel{input, ql, qh, scale, minv, output, (int)N, (int)K});
    });
}
