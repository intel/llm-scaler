/* iq4_GEMV.h — shared canonical IQ4_NL/IQ4_XS GEMV for Intel XPU.
 *
 * The GGUF raw block layouts are normalized by SGLang before this kernel:
 *   input   [M, K]      fp16
 *   weight  [N, K/2]    uint8  (low nibble -> elem 2j, high -> elem 2j+1)
 *   scale   [N, K/32]   fp16   (final per-32-element scale)
 *   output  [M, N]      fp16
 *
 * Both formats then use the same dequantization:
 *   w[k] = scale[k/32] * IQ4_LUT[index[k]]
 *
 * IQ4_LUT = {-127,-104,-83,-65,-49,-35,-22,-10,
 *              1,  13, 25, 38, 53, 69, 89,113}
 */
#pragma once

namespace iq4_esimd_detail = sycl::ext::intel::esimd::detail;

static constexpr int IQ4_GROUP = 32;
static constexpr int IQ4_VL = 512;
static constexpr int IQ4_ROWS = 4;

template <int VL>
SYCL_ESIMD_FUNCTION inline simd<float, VL> iq4_dequant_tile(
    simd<uint8_t, VL / 2> packed, simd<fp16, VL / IQ4_GROUP> scale_h) {
    static_assert(VL % IQ4_GROUP == 0);
    simd<int16_t, 16> lut;
    lut[0] = -127; lut[1] = -104; lut[2] = -83; lut[3] = -65;
    lut[4] = -49;  lut[5] = -35;  lut[6] = -22; lut[7] = -10;
    lut[8] = 1;    lut[9] = 13;    lut[10] = 25; lut[11] = 38;
    lut[12] = 53;  lut[13] = 69;   lut[14] = 89; lut[15] = 113;

    simd<float, VL> weight_f;
    #pragma unroll
    for (int c = 0; c < VL / 32; c++) {
        auto bytes = packed.template select<16, 1>(c * 16);
        simd<uint16_t, 16> lo = convert<uint16_t>(bytes & 0x0F);
        simd<uint16_t, 16> hi = convert<uint16_t>((bytes >> 4) & 0x0F);
        weight_f.template select<16, 2>(c * 32) =
            convert<float>(lut.template iselect<16>(lo));
        weight_f.template select<16, 2>(c * 32 + 1) =
            convert<float>(lut.template iselect<16>(hi));
    }

    simd<float, VL / IQ4_GROUP> scale_f = scale_h;
    #pragma unroll
    for (int group = 0; group < VL / IQ4_GROUP; group++) {
        weight_f.template select<IQ4_GROUP, 1>(group * IQ4_GROUP) =
            weight_f.template select<IQ4_GROUP, 1>(group * IQ4_GROUP)
            * scale_f[group];
    }
    return weight_f;
}

inline void select_ks_iq4(uint32_t N, uint32_t K, int& ks) {
    ks = 1;
    if (N <= 128 && K >= 2048) ks = 8;
    else if (N <= 512 && K >= 2048) ks = 4;
    int kp = K / ks;
    while ((kp % IQ4_GROUP != 0) && ks > 1) {
        ks /= 2;
        kp = K / ks;
    }
}

template <int K_SPLIT>
struct IQ4_gemv_split_kernel {
    const fp16* input;
    const uint8_t* weight;
    const fp16* scale;
    fp16* output;
    int N, K, n_groups;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        if constexpr (K_SPLIT > 1)
            slm_init<K_SPLIT * sizeof(float)>();
        const int row = item.get_group(0);
        const int lid = item.get_local_id(0);
        if (row >= N) return;

        const int kp = K / K_SPLIT;
        const int kstart = lid * kp;
        const uint8_t* w_row = weight + (size_t)row * (K / 2);
        const fp16* s_row = scale + (size_t)row * n_groups;
        simd<float, 8> acc(0.0f);
        int ai = 0;
        for (int k = kstart; k < kstart + kp; k += IQ4_GROUP) {
            simd<fp16, IQ4_GROUP> act =
                block_load<fp16, IQ4_GROUP>(input + k);
            simd<uint8_t, IQ4_GROUP / 2> packed =
                block_load<uint8_t, IQ4_GROUP / 2>(w_row + k / 2);
            simd<fp16, 1> scale_h(s_row[k / IQ4_GROUP]);
            simd<float, IQ4_GROUP> weight_f =
                iq4_dequant_tile<IQ4_GROUP>(packed, scale_h);
            simd<float, IQ4_GROUP> product =
                weight_f * simd<float, IQ4_GROUP>(act);
            acc[ai] += iq4_esimd_detail::sum<float, float, IQ4_GROUP>(product);
            ai = (ai + 1) & 7;
        }
        const float sum = iq4_esimd_detail::sum<float, float, 8>(acc);
        if constexpr (K_SPLIT == 1) {
            output[row] = fp16(sum);
        } else {
            slm_block_store<float, 1>(lid * sizeof(float), simd<float, 1>(sum));
            barrier();
            if (lid == 0) {
                simd<float, K_SPLIT> parts =
                    slm_block_load<float, K_SPLIT>(0);
                output[row] = fp16(
                    iq4_esimd_detail::sum<float, float, K_SPLIT>(parts));
            }
        }
    }
};

template <int VLP>
struct IQ4_gemv_wide_kernel {
    const fp16* input;
    const uint8_t* weight;
    const fp16* scale;
    fp16* output;
    int N, K;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        const int row = (int)item.get_group(0) * IQ4_ROWS
                      + (int)item.get_local_id(0);
        if (row >= N) return;

        constexpr int VL = VLP;
        constexpr int VL_HALF = VL / 2;
        constexpr int VL_GROUPS = VL / IQ4_GROUP;
        const int w_stride = K / 2;
        const int s_stride = K / IQ4_GROUP;
        const uint8_t* w_row = weight + (size_t)row * w_stride;
        const fp16* s_row = scale + (size_t)row * s_stride;
        simd<float, 8> acc(0.0f);
        int ai = 0;

        for (int k = 0; k < K; k += VL) {
            simd<fp16, VL> act = block_load<fp16, VL>(input + k);
            simd<uint8_t, VL_HALF> packed =
                block_load<uint8_t, VL_HALF>(w_row + k / 2);
            simd<fp16, VL_GROUPS> scale_h =
                block_load<fp16, VL_GROUPS>(s_row + k / IQ4_GROUP);
            simd<float, VL> weight_f = iq4_dequant_tile<VL>(packed, scale_h);
            simd<float, VL> product = weight_f * simd<float, VL>(act);
            acc[ai] += iq4_esimd_detail::sum<float, float, VL>(product);
            ai = (ai + 1) & 7;
        }
        output[row] = fp16(iq4_esimd_detail::sum<float, float, 8>(acc));
    }
};

inline void iq4_gemv_host(
    const fp16* input, const uint8_t* weight, const fp16* scale,
    fp16* output, uint32_t N, uint32_t K, sycl::queue& q) {
    const bool wide512 = (K % IQ4_VL) == 0;
    const bool wide256 = (K % (IQ4_VL / 2)) == 0;
    if (wide512 || wide256) {
        const int nwg = ((int)N + IQ4_ROWS - 1) / IQ4_ROWS;
        q.submit([&](sycl::handler& h) {
            sycl::nd_range<1> range((size_t)nwg * IQ4_ROWS, IQ4_ROWS);
            if (wide512) {
                h.parallel_for(range, IQ4_gemv_wide_kernel<IQ4_VL>{
                    input, weight, scale, output, (int)N, (int)K});
            } else {
                h.parallel_for(range, IQ4_gemv_wide_kernel<IQ4_VL / 2>{
                    input, weight, scale, output, (int)N, (int)K});
            }
        });
        return;
    }

    const int n_groups = K / IQ4_GROUP;
    int ks;
    select_ks_iq4(N, K, ks);
    const int global = N * ks;
    const int local = ks;
#define LAUNCH_IQ4(S)                                                     \
    q.submit([&](sycl::handler& h) {                                      \
        h.parallel_for(sycl::nd_range<1>(global, local),                  \
            IQ4_gemv_split_kernel<S>{input, weight, scale, output,        \
                                      (int)N, (int)K, n_groups});          \
    });
    if (ks == 1) { LAUNCH_IQ4(1) }
    else if (ks == 2) { LAUNCH_IQ4(2) }
    else if (ks == 4) { LAUNCH_IQ4(4) }
    else if (ks == 8) { LAUNCH_IQ4(8) }
    else { LAUNCH_IQ4(1) }
#undef LAUNCH_IQ4
}

template <int M, int VLP>
struct IQ4_gemv_M_kernel {
    const fp16* input;
    const uint8_t* weight;
    const fp16* scale;
    fp16* output;
    int N, K, ldo;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        const int row = (int)item.get_group(0) * IQ4_ROWS
                      + (int)item.get_local_id(0);
        if (row >= N) return;

        constexpr int VL = VLP;
        constexpr int VL_HALF = VL / 2;
        constexpr int VL_GROUPS = VL / IQ4_GROUP;
        constexpr int AW = 64;
        const int w_stride = K / 2;
        const int s_stride = K / IQ4_GROUP;
        simd<float, AW> acc[M];
        #pragma unroll
        for (int m = 0; m < M; m++) acc[m] = 0.0f;

        for (int k = 0; k < K; k += VL) {
            simd<uint8_t, VL_HALF> packed = block_load<uint8_t, VL_HALF>(
                weight + (size_t)row * w_stride + k / 2);
            simd<fp16, VL_GROUPS> scale_h = block_load<fp16, VL_GROUPS>(
                scale + (size_t)row * s_stride + k / IQ4_GROUP);
            simd<float, VL> weight_f = iq4_dequant_tile<VL>(packed, scale_h);
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
                iq4_esimd_detail::sum<float, float, AW>(acc[m]));
        }
    }
};

template <int M>
inline void iq4_gemv_M_launch(
    const fp16* input, const uint8_t* weight, const fp16* scale,
    fp16* output, uint32_t N, uint32_t K, uint32_t ldo, sycl::queue& q) {
    const int nwg = ((int)N + IQ4_ROWS - 1) / IQ4_ROWS;
    const bool wide512 = (K % IQ4_VL) == 0;
    q.submit([&](sycl::handler& h) {
        sycl::nd_range<1> range((size_t)nwg * IQ4_ROWS, IQ4_ROWS);
        if (wide512) {
            h.parallel_for(range, IQ4_gemv_M_kernel<M, IQ4_VL>{
                input, weight, scale, output, (int)N, (int)K, (int)ldo});
        } else {
            h.parallel_for(range, IQ4_gemv_M_kernel<M, IQ4_VL / 2>{
                input, weight, scale, output, (int)N, (int)K, (int)ldo});
        }
    });
}

inline void iq4_gemv_M_host(
    const fp16* input, const uint8_t* weight, const fp16* scale,
    fp16* output, uint32_t M, uint32_t N, uint32_t K, uint32_t ldo,
    sycl::queue& q) {
    if (K % (IQ4_VL / 2) != 0) {
        for (uint32_t m = 0; m < M; m++) {
            iq4_gemv_host(input + (size_t)m * K, weight, scale,
                           output + (size_t)m * ldo, N, K, q);
        }
        return;
    }
    uint32_t m0 = 0;
    while (m0 < M) {
        const uint32_t remaining = M - m0;
        const fp16* in = input + (size_t)m0 * K;
        fp16* out = output + (size_t)m0 * ldo;
        if (remaining >= 16) {
            iq4_gemv_M_launch<16>(in, weight, scale, out, N, K, ldo, q);
            m0 += 16;
        } else if (remaining >= 8) {
            iq4_gemv_M_launch<8>(in, weight, scale, out, N, K, ldo, q);
            m0 += 8;
        } else if (remaining >= 4) {
            iq4_gemv_M_launch<4>(in, weight, scale, out, N, K, ldo, q);
            m0 += 4;
        } else if (remaining >= 2) {
            iq4_gemv_M_launch<2>(in, weight, scale, out, N, K, ldo, q);
            m0 += 2;
        } else {
            iq4_gemv_host(in, weight, scale, out, N, K, q);
            m0 += 1;
        }
    }
}
