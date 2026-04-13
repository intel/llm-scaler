#pragma once
#include "utils.h"

// FP8 GEMV with FP32 accumulation, element-wise acc + deferred scale.
// Optimized for decode (M=1) on BMG. Achieves ~97% BW utilization on large shapes.
//
// Supports both E4M3 and E5M2 via fp8_mode (0=E4M3, 1=E5M2).
// Input: [1, K] fp16, Weight: [N, K] fp8, Output: [1, N] fp16
//
// E4M3 dequant: sign<<15 | (exp+8)<<10 | mant<<7  (bias shift 7->15 = +8)
// E5M2 dequant: sign<<15 | exp<<10 | mant<<8       (same bias, no shift)
// Subnormal (exp==0): flush to signed zero

// FP8 -> FP16 dequant: returns fp16 simd from raw uint8 simd
template<int VL>
SYCL_ESIMD_FUNCTION inline simd<float, VL> fp8_dequant(
    simd<uint8_t, VL> raw, int fp8_mode) {
    simd<uint16_t, VL> u16 = convert<uint16_t>(raw);
    simd<uint16_t, VL> fp8_sign = (u16 >> 7) & 1;
    simd<uint16_t, VL> fp16_bits;

    if (fp8_mode == 0) {
        // E4M3: 4-bit exp (bias 7), 3-bit mant
        simd<uint16_t, VL> fp8_exp  = (u16 >> 3) & 0xF;
        simd<uint16_t, VL> fp8_mant = u16 & 0x7;
        fp16_bits = (fp8_sign << 15) | ((fp8_exp + 8) << 10) | (fp8_mant << 7);
        fp16_bits.merge(fp8_sign << 15, fp8_exp == 0);
    } else {
        // E5M2: 5-bit exp (bias 15), 2-bit mant
        simd<uint16_t, VL> fp8_exp  = (u16 >> 2) & 0x1F;
        simd<uint16_t, VL> fp8_mant = u16 & 0x3;
        fp16_bits = (fp8_sign << 15) | (fp8_exp << 10) | (fp8_mant << 8);
        fp16_bits.merge(fp8_sign << 15, fp8_exp == 0);
    }

    simd<fp16, VL> wh = fp16_bits.template bit_cast_view<fp16>().read();
    return simd<float, VL>(wh);
}

// ============================================================================
// Per-N scale (pern): scale is fp16[N]
// ============================================================================

template<int VL, int K_SPLIT>
struct GEMV_fp8_pern_kernel {
    const fp16*    input;
    const uint8_t* weight;
    const fp16*    scale;
    fp16*          output;
    int N, K;
    int fp8_mode; // 0=E4M3, 1=E5M2

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        if constexpr (K_SPLIT > 1) {
            slm_init<K_SPLIT * sizeof(float)>();
        }

        int n   = item.get_group(0);
        int lid = item.get_local_id(0);
        if (n >= N) return;

        int kp = K / K_SPLIT;
        int ks = lid * kp;

        simd<float, VL> acc = 0.0f;

        for (int k = ks; k < ks + kp; k += VL) {
            simd<fp16, VL> iv = block_load<fp16, VL>(input + k);
            simd<float, VL> input_f = iv;

            simd<uint8_t, VL> raw = block_load<uint8_t, VL>(weight + (size_t)n * K + k);
            simd<float, VL> wf = fp8_dequant<VL>(raw, fp8_mode);

            acc += input_f * wf;
        }

        float my_sum = reduce<float>(acc, std::plus<>()) * static_cast<float>(scale[n]);

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

// VL/K_SPLIT auto-selection helper (shared by all dispatchers)
inline void select_vl_ks(uint32_t N, uint32_t K, int& vl, int& ks) {
    vl = 512; ks = 1;

    if (K < 512) {
        vl = 128; ks = 1;
    } else if (K == 512) {
        vl = 256; ks = 1;
    }

    if (N <= 128 && K >= 2048) {
        vl = 128; ks = 8;
    } else if (N <= 512 && K >= 2048) {
        vl = 128; ks = 4;
    }

    int kpt = K / ks;
    while (vl > kpt || kpt % vl != 0) {
        if (vl > 128) {
            vl /= 2;
        } else if (ks > 1) {
            ks /= 2;
            kpt = K / ks;
        } else {
            break;
        }
    }
}

inline void GEMV_fp8_pern_host(
    uint8_t* input_data,
    uint8_t* weight_data,
    uint8_t* scale_data,
    uint8_t* output_data,
    uint32_t N,
    uint32_t K,
    int fp8_mode,
    sycl::queue& q) {

    auto* p_in  = reinterpret_cast<const fp16*>(input_data);
    auto* p_w   = reinterpret_cast<const uint8_t*>(weight_data);
    auto* p_sc  = reinterpret_cast<const fp16*>(scale_data);
    auto* p_out = reinterpret_cast<fp16*>(output_data);

    int vl, ks;
    select_vl_ks(N, K, vl, ks);

    int global = N * ks;
    int local  = ks;

    #define LAUNCH(V, S) \
        q.submit([&](sycl::handler& h) { \
            h.parallel_for(sycl::nd_range<1>(global, local), \
                GEMV_fp8_pern_kernel<V, S>{p_in, p_w, p_sc, p_out, (int)N, (int)K, fp8_mode}); \
        });

    if (vl == 512 && ks == 1) { LAUNCH(512, 1) }
    else if (vl == 512 && ks == 2) { LAUNCH(512, 2) }
    else if (vl == 256 && ks == 1) { LAUNCH(256, 1) }
    else if (vl == 256 && ks == 2) { LAUNCH(256, 2) }
    else if (vl == 256 && ks == 4) { LAUNCH(256, 4) }
    else if (vl == 128 && ks == 1) { LAUNCH(128, 1) }
    else if (vl == 128 && ks == 2) { LAUNCH(128, 2) }
    else if (vl == 128 && ks == 4) { LAUNCH(128, 4) }
    else if (vl == 128 && ks == 8) { LAUNCH(128, 8) }
    else { LAUNCH(128, 1) }

    #undef LAUNCH
}

// ============================================================================
// Fused per-N scale
// ============================================================================

template<int VL, int K_SPLIT, int GEMV_COUNT>
struct GEMV_fp8_pern_fused_kernel {
    const fp16*    input;
    const uint8_t* weights[GEMV_COUNT];
    const fp16*    scales[GEMV_COUNT];
    fp16*          outputs[GEMV_COUNT];
    int            N[GEMV_COUNT];
    int            N_cumsum[GEMV_COUNT];
    int            K;
    int            fp8_mode;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        if constexpr (K_SPLIT > 1) {
            slm_init<K_SPLIT * sizeof(float)>();
        }

        int gid = item.get_group(0);
        int lid = item.get_local_id(0);

        int mat_idx = 0;
        int local_n = gid;
        #pragma unroll
        for (int i = 0; i < GEMV_COUNT; i++) {
            if (gid < N_cumsum[i]) {
                mat_idx = i;
                local_n = (i == 0) ? gid : gid - N_cumsum[i - 1];
                break;
            }
        }

        const uint8_t* w_ptr = weights[mat_idx];
        const fp16*    s_ptr = scales[mat_idx];
        fp16*          o_ptr = outputs[mat_idx];
        int            n     = local_n;

        int kp = K / K_SPLIT;
        int ks = lid * kp;

        simd<float, VL> acc = 0.0f;

        for (int k = ks; k < ks + kp; k += VL) {
            simd<fp16, VL> iv = block_load<fp16, VL>(input + k);
            simd<float, VL> input_f = iv;

            simd<uint8_t, VL> raw = block_load<uint8_t, VL>(w_ptr + (size_t)n * K + k);
            simd<float, VL> wf = fp8_dequant<VL>(raw, fp8_mode);

            acc += input_f * wf;
        }

        float my_sum = reduce<float>(acc, std::plus<>()) * static_cast<float>(s_ptr[n]);

        if constexpr (K_SPLIT == 1) {
            o_ptr[n] = fp16(my_sum);
        } else {
            slm_block_store<float, 1>(lid * sizeof(float), simd<float, 1>(my_sum));
            barrier();
            if (lid == 0) {
                simd<float, K_SPLIT> parts = slm_block_load<float, K_SPLIT>(0);
                o_ptr[n] = fp16(reduce<float>(parts, std::plus<>()));
            }
        }
    }
};

template<int GEMV_COUNT>
inline void GEMV_fp8_pern_fused_host(
    uint8_t* input_data,
    uint8_t* weight_ptrs[GEMV_COUNT],
    uint8_t* scale_ptrs[GEMV_COUNT],
    uint8_t* output_ptrs[GEMV_COUNT],
    uint32_t Ns[GEMV_COUNT],
    uint32_t K,
    int fp8_mode,
    sycl::queue& q) {

    auto* p_in = reinterpret_cast<const fp16*>(input_data);

    uint32_t total_N = 0;
    for (int i = 0; i < GEMV_COUNT; i++) total_N += Ns[i];

    int vl, ks;
    select_vl_ks(total_N, K, vl, ks);

    int global = total_N * ks;
    int local  = ks;

    #define LAUNCH_FUSED(V, S) \
        q.submit([&](sycl::handler& h) { \
            GEMV_fp8_pern_fused_kernel<V, S, GEMV_COUNT> kern; \
            kern.input = p_in; \
            kern.K = (int)K; \
            kern.fp8_mode = fp8_mode; \
            uint32_t cum = 0; \
            for (int i = 0; i < GEMV_COUNT; i++) { \
                kern.weights[i] = reinterpret_cast<const uint8_t*>(weight_ptrs[i]); \
                kern.scales[i]  = reinterpret_cast<const fp16*>(scale_ptrs[i]); \
                kern.outputs[i] = reinterpret_cast<fp16*>(output_ptrs[i]); \
                kern.N[i] = (int)Ns[i]; \
                cum += Ns[i]; \
                kern.N_cumsum[i] = (int)cum; \
            } \
            h.parallel_for(sycl::nd_range<1>(global, local), kern); \
        });

    if (vl == 512 && ks == 1) { LAUNCH_FUSED(512, 1) }
    else if (vl == 512 && ks == 2) { LAUNCH_FUSED(512, 2) }
    else if (vl == 256 && ks == 1) { LAUNCH_FUSED(256, 1) }
    else if (vl == 256 && ks == 2) { LAUNCH_FUSED(256, 2) }
    else if (vl == 256 && ks == 4) { LAUNCH_FUSED(256, 4) }
    else if (vl == 128 && ks == 1) { LAUNCH_FUSED(128, 1) }
    else if (vl == 128 && ks == 2) { LAUNCH_FUSED(128, 2) }
    else if (vl == 128 && ks == 4) { LAUNCH_FUSED(128, 4) }
    else if (vl == 128 && ks == 8) { LAUNCH_FUSED(128, 8) }
    else { LAUNCH_FUSED(128, 1) }

    #undef LAUNCH_FUSED
}

// ============================================================================
// Per-tensor scale (pert): scale is a single float per matrix
// ============================================================================

template<int VL, int K_SPLIT>
struct GEMV_fp8_pert_kernel {
    const fp16*    input;
    const uint8_t* weight;
    const float*   scale_ptr;  // device pointer — no host sync
    fp16*          output;
    int N, K;
    int fp8_mode;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        if constexpr (K_SPLIT > 1) {
            slm_init<K_SPLIT * sizeof(float)>();
        }

        int n   = item.get_group(0);
        int lid = item.get_local_id(0);
        if (n >= N) return;

        int kp = K / K_SPLIT;
        int ks = lid * kp;

        simd<float, VL> acc = 0.0f;

        for (int k = ks; k < ks + kp; k += VL) {
            simd<fp16, VL> iv = block_load<fp16, VL>(input + k);
            simd<float, VL> input_f = iv;

            simd<uint8_t, VL> raw = block_load<uint8_t, VL>(weight + (size_t)n * K + k);
            simd<float, VL> wf = fp8_dequant<VL>(raw, fp8_mode);

            acc += input_f * wf;
        }

        float my_sum = reduce<float>(acc, std::plus<>()) * *scale_ptr;

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

inline void GEMV_fp8_pert_host(
    uint8_t* input_data,
    uint8_t* weight_data,
    uint8_t* scale_data,  // device pointer to float scalar — no host sync
    uint8_t* output_data,
    uint32_t N,
    uint32_t K,
    int fp8_mode,
    sycl::queue& q) {

    auto* p_in  = reinterpret_cast<const fp16*>(input_data);
    auto* p_w   = reinterpret_cast<const uint8_t*>(weight_data);
    auto* p_sc  = reinterpret_cast<const float*>(scale_data);
    auto* p_out = reinterpret_cast<fp16*>(output_data);

    int vl, ks;
    select_vl_ks(N, K, vl, ks);

    int global = N * ks;
    int local  = ks;

    #define LAUNCH_PERT(V, S) \
        q.submit([&](sycl::handler& h) { \
            h.parallel_for(sycl::nd_range<1>(global, local), \
                GEMV_fp8_pert_kernel<V, S>{p_in, p_w, p_sc, p_out, (int)N, (int)K, fp8_mode}); \
        });

    if (vl == 512 && ks == 1) { LAUNCH_PERT(512, 1) }
    else if (vl == 512 && ks == 2) { LAUNCH_PERT(512, 2) }
    else if (vl == 256 && ks == 1) { LAUNCH_PERT(256, 1) }
    else if (vl == 256 && ks == 2) { LAUNCH_PERT(256, 2) }
    else if (vl == 256 && ks == 4) { LAUNCH_PERT(256, 4) }
    else if (vl == 128 && ks == 1) { LAUNCH_PERT(128, 1) }
    else if (vl == 128 && ks == 2) { LAUNCH_PERT(128, 2) }
    else if (vl == 128 && ks == 4) { LAUNCH_PERT(128, 4) }
    else if (vl == 128 && ks == 8) { LAUNCH_PERT(128, 8) }
    else { LAUNCH_PERT(128, 1) }

    #undef LAUNCH_PERT
}

// ============================================================================
// Fused per-tensor scale
// ============================================================================

template<int VL, int K_SPLIT, int GEMV_COUNT>
struct GEMV_fp8_pert_fused_kernel {
    const fp16*    input;
    const uint8_t* weights[GEMV_COUNT];
    const float*   scale_ptrs[GEMV_COUNT];  // device pointers — no host sync
    fp16*          outputs[GEMV_COUNT];
    int            N[GEMV_COUNT];
    int            N_cumsum[GEMV_COUNT];
    int            K;
    int            fp8_mode;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        if constexpr (K_SPLIT > 1) {
            slm_init<K_SPLIT * sizeof(float)>();
        }

        int gid = item.get_group(0);
        int lid = item.get_local_id(0);

        int mat_idx = 0;
        int local_n = gid;
        #pragma unroll
        for (int i = 0; i < GEMV_COUNT; i++) {
            if (gid < N_cumsum[i]) {
                mat_idx = i;
                local_n = (i == 0) ? gid : gid - N_cumsum[i - 1];
                break;
            }
        }

        const uint8_t* w_ptr = weights[mat_idx];
        float          s_val = *scale_ptrs[mat_idx];
        fp16*          o_ptr = outputs[mat_idx];
        int            n     = local_n;

        int kp = K / K_SPLIT;
        int ks = lid * kp;

        simd<float, VL> acc = 0.0f;

        for (int k = ks; k < ks + kp; k += VL) {
            simd<fp16, VL> iv = block_load<fp16, VL>(input + k);
            simd<float, VL> input_f = iv;

            simd<uint8_t, VL> raw = block_load<uint8_t, VL>(w_ptr + (size_t)n * K + k);
            simd<float, VL> wf = fp8_dequant<VL>(raw, fp8_mode);

            acc += input_f * wf;
        }

        float my_sum = reduce<float>(acc, std::plus<>()) * s_val;

        if constexpr (K_SPLIT == 1) {
            o_ptr[n] = fp16(my_sum);
        } else {
            slm_block_store<float, 1>(lid * sizeof(float), simd<float, 1>(my_sum));
            barrier();
            if (lid == 0) {
                simd<float, K_SPLIT> parts = slm_block_load<float, K_SPLIT>(0);
                o_ptr[n] = fp16(reduce<float>(parts, std::plus<>()));
            }
        }
    }
};

template<int GEMV_COUNT>
inline void GEMV_fp8_pert_fused_host(
    uint8_t* input_data,
    uint8_t* weight_ptrs[GEMV_COUNT],
    uint8_t* scale_ptrs[GEMV_COUNT],  // device pointers to float scalars — no host sync
    uint8_t* output_ptrs[GEMV_COUNT],
    uint32_t Ns[GEMV_COUNT],
    uint32_t K,
    int fp8_mode,
    sycl::queue& q) {

    auto* p_in = reinterpret_cast<const fp16*>(input_data);

    uint32_t total_N = 0;
    for (int i = 0; i < GEMV_COUNT; i++) total_N += Ns[i];

    int vl, ks;
    select_vl_ks(total_N, K, vl, ks);

    int global = total_N * ks;
    int local  = ks;

    #define LAUNCH_PERT_FUSED(V, S) \
        q.submit([&](sycl::handler& h) { \
            GEMV_fp8_pert_fused_kernel<V, S, GEMV_COUNT> kern; \
            kern.input = p_in; \
            kern.K = (int)K; \
            kern.fp8_mode = fp8_mode; \
            uint32_t cum = 0; \
            for (int i = 0; i < GEMV_COUNT; i++) { \
                kern.weights[i] = reinterpret_cast<const uint8_t*>(weight_ptrs[i]); \
                kern.scale_ptrs[i] = reinterpret_cast<const float*>(scale_ptrs[i]); \
                kern.outputs[i] = reinterpret_cast<fp16*>(output_ptrs[i]); \
                kern.N[i] = (int)Ns[i]; \
                cum += Ns[i]; \
                kern.N_cumsum[i] = (int)cum; \
            } \
            h.parallel_for(sycl::nd_range<1>(global, local), kern); \
        });

    if (vl == 512 && ks == 1) { LAUNCH_PERT_FUSED(512, 1) }
    else if (vl == 512 && ks == 2) { LAUNCH_PERT_FUSED(512, 2) }
    else if (vl == 256 && ks == 1) { LAUNCH_PERT_FUSED(256, 1) }
    else if (vl == 256 && ks == 2) { LAUNCH_PERT_FUSED(256, 2) }
    else if (vl == 256 && ks == 4) { LAUNCH_PERT_FUSED(256, 4) }
    else if (vl == 128 && ks == 1) { LAUNCH_PERT_FUSED(128, 1) }
    else if (vl == 128 && ks == 2) { LAUNCH_PERT_FUSED(128, 2) }
    else if (vl == 128 && ks == 4) { LAUNCH_PERT_FUSED(128, 4) }
    else if (vl == 128 && ks == 8) { LAUNCH_PERT_FUSED(128, 8) }
    else { LAUNCH_PERT_FUSED(128, 1) }

    #undef LAUNCH_PERT_FUSED
}
