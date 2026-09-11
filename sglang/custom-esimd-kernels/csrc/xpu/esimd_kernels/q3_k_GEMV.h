/* q3_k_GEMV.h — canonical GGUF Q3_K GEMV for Intel XPU.
 *
 * SGLang normalizes block_q3_K to:
 *   input    [M,K]    fp16
 *   ql       [N,K/4]  uint8, four adjacent low-2-bit values per byte
 *   qh       [N,K/8]  uint8, bit=1 means subtract four
 *   scale    [N,K/16] fp16, final d*signed_scale6
 *   output   [M,N]    fp16
 *
 * weight[k] = scale[k/16] * (low2[k] - 4*subtract[k]).
 */
#pragma once

namespace q3k_esimd_detail = sycl::ext::intel::esimd::detail;

static constexpr int Q3K_GROUP = 16;
static constexpr int Q3K_VL = 512;
static constexpr int Q3K_ROWS = 4;

template <int VL>
SYCL_ESIMD_FUNCTION inline simd<float, VL> q3k_dequant_tile(
    simd<uint8_t, VL / 4> ql_data,
    simd<uint8_t, VL / 8> qh_data,
    simd<fp16, VL / Q3K_GROUP> scale_h) {
    static_assert(VL % 256 == 0);
    simd<float, VL> weight_f;

    #pragma unroll
    for (int field = 0; field < 4; field++) {
        simd<uint8_t, VL / 4> low =
            (ql_data >> (2 * field)) & uint8_t(3);
        weight_f.template select<VL / 4, 4>(field) =
            convert<float>(low);
    }
    #pragma unroll
    for (int bit = 0; bit < 8; bit++) {
        simd<uint8_t, VL / 8> subtract =
            (qh_data >> bit) & uint8_t(1);
        weight_f.template select<VL / 8, 8>(bit) -=
            convert<float>(subtract) * 4.0f;
    }

    simd<float, VL / Q3K_GROUP> scale_f = scale_h;
    #pragma unroll
    for (int group = 0; group < VL / Q3K_GROUP; group++) {
        weight_f.template select<Q3K_GROUP, 1>(group * Q3K_GROUP) =
            weight_f.template select<Q3K_GROUP, 1>(group * Q3K_GROUP)
            * scale_f[group];
    }
    return weight_f;
}

template <int VL>
struct Q3K_gemv_kernel {
    const fp16* input;
    const uint8_t* ql;
    const uint8_t* qh;
    const fp16* scale;
    fp16* output;
    int N, K;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        const int row = (int)item.get_group(0) * Q3K_ROWS
                      + (int)item.get_local_id(0);
        if (row >= N) return;

        const int ql_stride = K / 4;
        const int qh_stride = K / 8;
        const int scale_stride = K / Q3K_GROUP;
        simd<float, 8> acc(0.0f);
        int ai = 0;
        for (int k = 0; k < K; k += VL) {
            simd<fp16, VL> act = block_load<fp16, VL>(input + k);
            simd<uint8_t, VL / 4> ql_data = block_load<uint8_t, VL / 4>(
                ql + (size_t)row * ql_stride + k / 4);
            simd<uint8_t, VL / 8> qh_data = block_load<uint8_t, VL / 8>(
                qh + (size_t)row * qh_stride + k / 8);
            simd<fp16, VL / Q3K_GROUP> scale_h =
                block_load<fp16, VL / Q3K_GROUP>(
                    scale + (size_t)row * scale_stride + k / Q3K_GROUP);
            simd<float, VL> weight_f =
                q3k_dequant_tile<VL>(ql_data, qh_data, scale_h);
            simd<float, VL> product = weight_f * simd<float, VL>(act);
            acc[ai] += q3k_esimd_detail::sum<float, float, VL>(product);
            ai = (ai + 1) & 7;
        }
        output[row] = fp16(
            q3k_esimd_detail::sum<float, float, 8>(acc));
    }
};

inline void q3k_gemv_host(
    const fp16* input, const uint8_t* ql, const uint8_t* qh,
    const fp16* scale, fp16* output, uint32_t N, uint32_t K,
    sycl::queue& q) {
    const int nwg = ((int)N + Q3K_ROWS - 1) / Q3K_ROWS;
    const bool wide = (K % Q3K_VL) == 0;
    q.submit([&](sycl::handler& h) {
        sycl::nd_range<1> range((size_t)nwg * Q3K_ROWS, Q3K_ROWS);
        if (wide) {
            h.parallel_for(range, Q3K_gemv_kernel<Q3K_VL>{
                input, ql, qh, scale, output, (int)N, (int)K});
        } else {
            h.parallel_for(range, Q3K_gemv_kernel<Q3K_VL / 2>{
                input, ql, qh, scale, output, (int)N, (int)K});
        }
    });
}

template <int M, int VL>
struct Q3K_gemv_M_kernel {
    const fp16* input;
    const uint8_t* ql;
    const uint8_t* qh;
    const fp16* scale;
    fp16* output;
    int N, K, ldo;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        const int row = (int)item.get_group(0) * Q3K_ROWS
                      + (int)item.get_local_id(0);
        if (row >= N) return;

        constexpr int AW = 64;
        const int ql_stride = K / 4;
        const int qh_stride = K / 8;
        const int scale_stride = K / Q3K_GROUP;
        simd<float, AW> acc[M];
        #pragma unroll
        for (int m = 0; m < M; m++) acc[m] = 0.0f;

        for (int k = 0; k < K; k += VL) {
            simd<uint8_t, VL / 4> ql_data = block_load<uint8_t, VL / 4>(
                ql + (size_t)row * ql_stride + k / 4);
            simd<uint8_t, VL / 8> qh_data = block_load<uint8_t, VL / 8>(
                qh + (size_t)row * qh_stride + k / 8);
            simd<fp16, VL / Q3K_GROUP> scale_h =
                block_load<fp16, VL / Q3K_GROUP>(
                    scale + (size_t)row * scale_stride + k / Q3K_GROUP);
            simd<float, VL> weight_f =
                q3k_dequant_tile<VL>(ql_data, qh_data, scale_h);
            #pragma unroll
            for (int m = 0; m < M; m++) {
                simd<fp16, VL> act = block_load<fp16, VL>(
                    input + (size_t)m * K + k);
                #pragma unroll
                for (int c = 0; c < VL / AW; c++) {
                    acc[m] += weight_f.template select<AW, 1>(c * AW)
                            * simd<float, AW>(
                                act.template select<AW, 1>(c * AW));
                }
            }
        }
        #pragma unroll
        for (int m = 0; m < M; m++) {
            output[(size_t)m * ldo + row] = fp16(
                q3k_esimd_detail::sum<float, float, AW>(acc[m]));
        }
    }
};

template <int M>
inline void q3k_gemv_M_launch(
    const fp16* input, const uint8_t* ql, const uint8_t* qh,
    const fp16* scale, fp16* output, uint32_t N, uint32_t K,
    uint32_t ldo, sycl::queue& q) {
    const int nwg = ((int)N + Q3K_ROWS - 1) / Q3K_ROWS;
    const bool wide = (K % Q3K_VL) == 0;
    q.submit([&](sycl::handler& h) {
        sycl::nd_range<1> range((size_t)nwg * Q3K_ROWS, Q3K_ROWS);
        if (wide) {
            h.parallel_for(range, Q3K_gemv_M_kernel<M, Q3K_VL>{
                input, ql, qh, scale, output,
                (int)N, (int)K, (int)ldo});
        } else {
            h.parallel_for(range, Q3K_gemv_M_kernel<M, Q3K_VL / 2>{
                input, ql, qh, scale, output,
                (int)N, (int)K, (int)ldo});
        }
    });
}

inline void q3k_gemv_M_host(
    const fp16* input, const uint8_t* ql, const uint8_t* qh,
    const fp16* scale, fp16* output, uint32_t M, uint32_t N, uint32_t K,
    uint32_t ldo, sycl::queue& q) {
    uint32_t m0 = 0;
    while (m0 < M) {
        const uint32_t remaining = M - m0;
        const fp16* in = input + (size_t)m0 * K;
        fp16* out = output + (size_t)m0 * ldo;
        if (remaining >= 16) {
            q3k_gemv_M_launch<16>(in, ql, qh, scale, out, N, K, ldo, q);
            m0 += 16;
        } else if (remaining >= 8) {
            q3k_gemv_M_launch<8>(in, ql, qh, scale, out, N, K, ldo, q);
            m0 += 8;
        } else if (remaining >= 4) {
            q3k_gemv_M_launch<4>(in, ql, qh, scale, out, N, K, ldo, q);
            m0 += 4;
        } else if (remaining >= 2) {
            q3k_gemv_M_launch<2>(in, ql, qh, scale, out, N, K, ldo, q);
            m0 += 2;
        } else {
            q3k_gemv_host(in, ql, qh, scale, out, N, K, q);
            m0 += 1;
        }
    }
}
