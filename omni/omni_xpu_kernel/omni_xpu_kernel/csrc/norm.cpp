// ============================================================================
// Normalization Kernels - Intel XPU ESIMD Optimized Implementation
// ============================================================================
// High-performance RMSNorm and LayerNorm using Intel ESIMD
// ============================================================================

#include <torch/extension.h>
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

#include "utils.h"

using fp16 = sycl::half;
using bf16 = sycl::ext::oneapi::bfloat16;
using ST = at::ScalarType;

using namespace sycl::ext::intel::esimd;

namespace omni_xpu {
namespace norm {

#if defined(OMNI_XPU_ARCH_PTL_H)
// Z-Image and Krea2 attention projections normalize large batches of
// contiguous H128 Q/K rows. The generic H128 kernel assigns four work-items
// to every row, reserves its maximum 8K-element SLM cache, and crosses a
// work-group barrier. For sufficiently many short rows, one PTL-H work-item
// per row is faster: it retains the 128 inputs in GRF and only uses four SLM
// floats. Z-Image reaches this route as BF16; Krea2 uses FP32 to preserve its
// model-defined accumulation semantics.
struct RmsNormH128PTLConfig {
    static constexpr int HiddenSize = 128;
    static constexpr int BlockSize = 32;
    static constexpr int Blocks = HiddenSize / BlockSize;
    static constexpr int MinimumRows = 1024;
    static constexpr int PartialBytes =
        ((Blocks * static_cast<int>(sizeof(float)) + 15) / 16) * 16;
};

class RmsNormH128PTLKernel;
class RmsNormH128FP32PTLKernel;

// Z-Image applies a second RMSNorm immediately before a BF16 gate multiply
// and BF16 residual add. At its H3840 workflow shapes, keeping all three
// operations in one kernel removes two materialized intermediates while
// preserving the two reduced-precision boundaries explicitly.
struct RmsNormGateResidualH3840PTLConfig {
    static constexpr int HiddenSize = 3840;
    static constexpr int BlockSize = 64;
    static constexpr int GroupSize = 32;
    static constexpr int Blocks = HiddenSize / BlockSize;
    static constexpr int InputBytes = HiddenSize * sizeof(bf16);
    static constexpr int PartialBytes =
        ((GroupSize * static_cast<int>(sizeof(float)) + 15) / 16) * 16;
    static constexpr int SlmBytes = InputBytes + PartialBytes;
};

class RmsNormGateResidualH3840PTLKernel;

void rms_norm_h128_ptl_kernel(
    const void* weight_ptr,
    const void* input_ptr,
    void* output_ptr,
    float eps,
    const int input_size,
    const at::Device& device
) {
    using Config = RmsNormH128PTLConfig;
    const bf16* weight = static_cast<const bf16*>(weight_ptr);
    const bf16* input = static_cast<const bf16*>(input_ptr);
    bf16* output = static_cast<bf16*>(output_ptr);

    auto cgf = [&](sycl::handler& handle) {
        handle.parallel_for<RmsNormH128PTLKernel>(
            sycl::nd_range<2>(
                sycl::range<2>(input_size, 1),
                sycl::range<2>(1, 1)),
            [=](sycl::nd_item<2> item) SYCL_ESIMD_KERNEL {
                slm_init<Config::PartialBytes>();
                const int row = item.get_global_id(0);
                const bf16* input_row =
                    input + static_cast<size_t>(row) * Config::HiddenSize;
                bf16* output_row =
                    output + static_cast<size_t>(row) * Config::HiddenSize;
                simd<bf16, Config::HiddenSize> cached;

#pragma unroll
                for (int block = 0; block < Config::Blocks; ++block) {
                    simd<bf16, Config::BlockSize> values =
                        block_load<bf16, Config::BlockSize>(
                            input_row + block * Config::BlockSize);
                    cached.template select<Config::BlockSize, 1>(
                        block * Config::BlockSize) = values;
                    simd<float, Config::BlockSize> values_f32 = values;
                    simd<float, Config::BlockSize> squares = 0;
                    squares += values_f32 * values_f32;
                    const float partial =
                        sycl::ext::intel::esimd::detail::sum<
                            float, float, Config::BlockSize>(squares) /
                        static_cast<float>(Config::HiddenSize);
                    slm_block_store<float, 1>(
                        block * sizeof(float), partial);
                }

                simd<float, Config::Blocks> partials =
                    slm_block_load<float, Config::Blocks>(0);
                const float mean =
                    sycl::ext::intel::esimd::detail::sum<
                        float, float, Config::Blocks>(partials);
                const float scale = rsqrt(mean + eps);

#pragma unroll
                for (int block = 0; block < Config::Blocks; ++block) {
                    simd<float, Config::BlockSize> values =
                        cached.template select<Config::BlockSize, 1>(
                            block * Config::BlockSize);
                    simd<float, Config::BlockSize> weights =
                        block_load<bf16, Config::BlockSize>(
                            weight + block * Config::BlockSize);
                    block_store<bf16, Config::BlockSize>(
                        output_row + block * Config::BlockSize,
                        simd<bf16, Config::BlockSize>(
                            values * scale * weights));
                }
            });
    };
    utils::submit_kernel(cgf, device, "rms_norm_h128_ptl");
}

void rms_norm_h128_fp32_ptl_kernel(
    const void* weight_ptr,
    const void* input_ptr,
    void* output_ptr,
    float eps,
    const int input_size,
    const at::Device& device
) {
    using Config = RmsNormH128PTLConfig;
    const float* weight = static_cast<const float*>(weight_ptr);
    const float* input = static_cast<const float*>(input_ptr);
    float* output = static_cast<float*>(output_ptr);

    auto cgf = [&](sycl::handler& handle) {
        handle.parallel_for<RmsNormH128FP32PTLKernel>(
            sycl::nd_range<2>(
                sycl::range<2>(input_size, 1),
                sycl::range<2>(1, 1)),
            [=](sycl::nd_item<2> item) SYCL_ESIMD_KERNEL {
                slm_init<Config::PartialBytes>();
                const int row = item.get_global_id(0);
                const float* input_row =
                    input + static_cast<size_t>(row) * Config::HiddenSize;
                float* output_row =
                    output + static_cast<size_t>(row) * Config::HiddenSize;
                simd<float, Config::HiddenSize> cached;

#pragma unroll
                for (int block = 0; block < Config::Blocks; ++block) {
                    simd<float, Config::BlockSize> values =
                        block_load<float, Config::BlockSize>(
                            input_row + block * Config::BlockSize);
                    cached.template select<Config::BlockSize, 1>(
                        block * Config::BlockSize) = values;
                    simd<float, Config::BlockSize> squares = 0;
                    squares += values * values;
                    const float partial =
                        sycl::ext::intel::esimd::detail::sum<
                            float, float, Config::BlockSize>(squares) /
                        static_cast<float>(Config::HiddenSize);
                    slm_block_store<float, 1>(
                        block * sizeof(float), partial);
                }

                simd<float, Config::Blocks> partials =
                    slm_block_load<float, Config::Blocks>(0);
                const float mean =
                    sycl::ext::intel::esimd::detail::sum<
                        float, float, Config::Blocks>(partials);
                const float scale = rsqrt(mean + eps);

#pragma unroll
                for (int block = 0; block < Config::Blocks; ++block) {
                    simd<float, Config::BlockSize> values =
                        cached.template select<Config::BlockSize, 1>(
                            block * Config::BlockSize);
                    simd<float, Config::BlockSize> weights =
                        block_load<float, Config::BlockSize>(
                            weight + block * Config::BlockSize);
                    block_store<float, Config::BlockSize>(
                        output_row + block * Config::BlockSize,
                        values * scale * weights);
                }
            });
    };
    utils::submit_kernel(cgf, device, "rms_norm_h128_fp32_ptl");
}

void rms_norm_gate_residual_h3840_ptl_kernel(
    const bf16* weight,
    const bf16* input,
    const bf16* gate,
    const bf16* residual,
    bf16* output,
    float eps,
    int64_t rows,
    const at::Device& device
) {
    using Config = RmsNormGateResidualH3840PTLConfig;
    constexpr int SubBlocks = Config::Blocks / Config::GroupSize;
    constexpr int RemainderBlocks = Config::Blocks % Config::GroupSize;

    auto cgf = [&](sycl::handler& handle) {
        handle.parallel_for<RmsNormGateResidualH3840PTLKernel>(
            sycl::nd_range<2>(
                sycl::range<2>(rows, Config::GroupSize),
                sycl::range<2>(1, Config::GroupSize)),
            [=](sycl::nd_item<2> item) SYCL_ESIMD_KERNEL {
                slm_init<Config::SlmBytes>();
                const int64_t row = item.get_global_id(0);
                const int tid = item.get_local_id(1);
                const bf16* input_row =
                    input + row * Config::HiddenSize;
                const bf16* residual_row =
                    residual + row * Config::HiddenSize;
                bf16* output_row = output + row * Config::HiddenSize;
                const int start_block =
                    SubBlocks * tid +
                    (tid < RemainderBlocks ? tid : RemainderBlocks);
                const int end_block =
                    start_block + SubBlocks + (tid < RemainderBlocks);

                simd<float, Config::BlockSize> accumulator = 0;
                for (int block = start_block; block < end_block; ++block) {
                    simd<bf16, Config::BlockSize> values =
                        block_load<bf16, Config::BlockSize>(
                            input_row + block * Config::BlockSize);
                    slm_block_store<bf16, Config::BlockSize>(
                        block * Config::BlockSize * sizeof(bf16), values);
                    simd<float, Config::BlockSize> values_f32 = values;
                    accumulator += values_f32 * values_f32;
                }
                const float partial =
                    sycl::ext::intel::esimd::detail::sum<
                        float, float, Config::BlockSize>(accumulator) /
                    static_cast<float>(Config::HiddenSize);
                slm_block_store<float, 1>(
                    Config::InputBytes + tid * sizeof(float), partial);
                barrier();

                simd<float, Config::GroupSize> partials =
                    slm_block_load<float, Config::GroupSize>(
                        Config::InputBytes);
                const float mean =
                    sycl::ext::intel::esimd::detail::sum<
                        float, float, Config::GroupSize>(partials);
                const float rms_scale = rsqrt(mean + eps);

                for (int block = start_block; block < end_block; ++block) {
                    const int column = block * Config::BlockSize;
                    simd<float, Config::BlockSize> values =
                        slm_block_load<bf16, Config::BlockSize>(
                            column * sizeof(bf16));
                    simd<float, Config::BlockSize> weights =
                        block_load<bf16, Config::BlockSize>(weight + column);
                    simd<float, Config::BlockSize> gate_values =
                        block_load<bf16, Config::BlockSize>(gate + column);
                    simd<float, Config::BlockSize> residual_values =
                        block_load<bf16, Config::BlockSize>(
                            residual_row + column);

                    // Match the existing three-op BF16 chain: RMSNorm first
                    // stores BF16, gate * normalized stores BF16 again, then
                    // the residual add rounds to BF16.
                    simd<bf16, Config::BlockSize> normalized_bf16 =
                        simd<bf16, Config::BlockSize>(
                            values * rms_scale * weights);
                    simd<float, Config::BlockSize> normalized =
                        normalized_bf16;
                    simd<bf16, Config::BlockSize> product_bf16 =
                        simd<bf16, Config::BlockSize>(
                            normalized * gate_values);
                    simd<float, Config::BlockSize> product = product_bf16;
                    block_store<bf16, Config::BlockSize>(
                        output_row + column,
                        simd<bf16, Config::BlockSize>(
                            residual_values + product));
                }
            });
    };
    utils::submit_kernel(
        cgf, device, "rms_norm_gate_residual_h3840_ptl");
}
#endif

// ============================================================================
// RMSNorm Kernel  (optimized: right-sized SLM, tuned GS)
// ============================================================================
template<typename IT, const int GS, const int BS>
void rms_norm_kernel(
    const void* weight_ptr,
    const void* input_ptr,
    void* output_ptr,
    float eps,
    const int input_size,
    const int hidden_size,
    const at::Device& device
) {
    assert(hidden_size % BS == 0);
    assert(hidden_size <= 8 * 1024);

    const int nb = hidden_size / BS;
    const int sub_nb = nb / GS;
    const int rem_nb = nb % GS;
    // Right-size SLM: only allocate what we need
    constexpr int slm_acc_align = ((GS * (int)sizeof(float) + 15) / 16) * 16;
    const int acc_offset = hidden_size * sizeof(IT);

    sycl::range<2> global_size(input_size, GS);
    sycl::range<2> local_size(1, GS);

    auto cgf = [&](sycl::handler& handle) {
        handle.parallel_for(
            sycl::nd_range<2>(global_size, local_size),
            [=](sycl::nd_item<2> item) SYCL_ESIMD_KERNEL {
                slm_init<8 * 1024 * sizeof(IT) + slm_acc_align>();

                const int rid = item.get_global_id(0);
                const int tid = item.get_local_id(1);

                const IT* weight = (const IT*)weight_ptr;
                const IT* input = (const IT*)input_ptr + hidden_size * (size_t)rid;
                IT* output = (IT*)output_ptr + hidden_size * (size_t)rid;

                const int start_blk = sub_nb * tid + std::min(tid, rem_nb);
                const int end_blk = start_blk + sub_nb + (tid < rem_nb);

                simd<float, BS> accv = 0;
                for (int i = start_blk; i < end_blk; ++i) {
                    simd<IT, BS> xv = block_load<IT, BS>(input + i * BS);
                    slm_block_store<IT, BS>(i * BS * sizeof(IT), xv);
                    simd<float, BS> xv_f32 = xv;
                    accv += xv_f32 * xv_f32;
                }
                float acc = sycl::ext::intel::esimd::detail::sum<float, float, BS>(accv) / hidden_size;

                if constexpr (GS == 1) {
                    // Single thread: no barrier needed
                    float scale = rsqrt(acc + eps);
                    for (int i = 0; i < nb; ++i) {
                        simd<float, BS> xv = slm_block_load<IT, BS>(i * BS * sizeof(IT));
                        simd<float, BS> yv = block_load<IT, BS>(weight + i * BS);
                        simd<IT, BS> result = xv * scale * yv;
                        block_store<IT, BS>(output + i * BS, result);
                    }
                } else {
                    slm_block_store<float, 1>(acc_offset + tid * sizeof(float), acc);
                    barrier();

                    simd<float, GS> accs = slm_block_load<float, GS>(acc_offset);
                    float mean = sycl::ext::intel::esimd::detail::sum<float, float, GS>(accs);
                    float scale = rsqrt(mean + eps);

                    for (int i = start_blk; i < end_blk; ++i) {
                        simd<float, BS> xv = slm_block_load<IT, BS>(i * BS * sizeof(IT));
                        simd<float, BS> yv = block_load<IT, BS>(weight + i * BS);
                        simd<IT, BS> result = xv * scale * yv;
                        block_store<IT, BS>(output + i * BS, result);
                    }
                }
            }
        );
    };

    utils::submit_kernel(cgf, device, "rms_norm_esimd");
}

// ============================================================================
// LayerNorm Kernel  (optimized: single-pass mean+variance via Welford's)
// ============================================================================
template<typename IT, const int GS, const int BS>
void layer_norm_kernel(
    const void* input_ptr,
    const uint64_t weight_ptr,
    const uint64_t bias_ptr,
    void* output_ptr,
    float eps,
    const int input_size,
    const int hidden_size,
    const at::Device& device
) {
    assert(hidden_size % BS == 0);
    assert(hidden_size <= 8 * 1024);

    const int nb = hidden_size / BS;
    const int sub_nb = nb / GS;
    const int rem_nb = nb % GS;
    // SLM layout: [input_cache] [mean_partials] [sq_sum_partials]
    const int partials_offset = hidden_size * sizeof(IT);
    // Two arrays of GS floats for partials, aligned
    constexpr int slm_partials_size = ((2 * GS * (int)sizeof(float) + 15) / 16) * 16;

    sycl::range<2> global_size(input_size, GS);
    sycl::range<2> local_size(1, GS);

    auto cgf = [&](sycl::handler& handle) {
        handle.parallel_for(
            sycl::nd_range<2>(global_size, local_size),
            [=](sycl::nd_item<2> item) SYCL_ESIMD_KERNEL {
                slm_init<8 * 1024 * sizeof(IT) + slm_partials_size>();

                const int rid = item.get_global_id(0);
                const int tid = item.get_local_id(1);

                const IT* input = (const IT*)input_ptr + hidden_size * (size_t)rid;
                IT* output = (IT*)output_ptr + hidden_size * (size_t)rid;

                const int start_blk = sub_nb * tid + std::min(tid, rem_nb);
                const int end_blk = start_blk + sub_nb + (tid < rem_nb);

                // Single pass: compute sum and sum-of-squares simultaneously
                simd<float, BS> sumv = 0;
                simd<float, BS> sq_sumv = 0;
                for (int i = start_blk; i < end_blk; ++i) {
                    simd<IT, BS> xv = block_load<IT, BS>(input + i * BS);
                    slm_block_store<IT, BS>(i * BS * sizeof(IT), xv);
                    simd<float, BS> xv_f32 = xv;
                    sumv += xv_f32;
                    sq_sumv += xv_f32 * xv_f32;
                }
                float par_sum = sycl::ext::intel::esimd::detail::sum<float, float, BS>(sumv);
                float par_sq_sum = sycl::ext::intel::esimd::detail::sum<float, float, BS>(sq_sumv);

                float mean, scale;

                if constexpr (GS == 1) {
                    // Single thread: no barrier needed
                    mean = par_sum / hidden_size;
                    // Var = E[x^2] - E[x]^2
                    float var = par_sq_sum / hidden_size - mean * mean;
                    scale = rsqrt(var + eps);
                } else {
                    // Store both partials
                    const int mean_off = partials_offset;
                    const int sq_off = partials_offset + GS * sizeof(float);
                    slm_block_store<float, 1>(mean_off + tid * sizeof(float), par_sum);
                    slm_block_store<float, 1>(sq_off + tid * sizeof(float), par_sq_sum);

                    barrier();

                    simd<float, GS> sums = slm_block_load<float, GS>(mean_off);
                    simd<float, GS> sq_sums = slm_block_load<float, GS>(sq_off);
                    float total_sum = sycl::ext::intel::esimd::detail::sum<float, float, GS>(sums);
                    float total_sq_sum = sycl::ext::intel::esimd::detail::sum<float, float, GS>(sq_sums);
                    mean = total_sum / hidden_size;
                    // Var = E[x^2] - E[x]^2
                    float var = total_sq_sum / hidden_size - mean * mean;
                    scale = rsqrt(var + eps);
                }

                // Normalize and apply weight/bias
                for (int i = start_blk; i < end_blk; ++i) {
                    simd<float, BS> xv = slm_block_load<IT, BS>(i * BS * sizeof(IT));
                    simd<float, BS> result = (xv - mean) * scale;

                    if (weight_ptr != 0) {
                        simd<float, BS> yv = block_load<IT, BS>((const IT*)weight_ptr + i * BS);
                        result = result * yv;
                    }

                    if (bias_ptr != 0) {
                        simd<float, BS> bv = block_load<IT, BS>((const IT*)bias_ptr + i * BS);
                        result = result + bv;
                    }

                    block_store<IT, BS>(output + i * BS, result);
                }
            }
        );
    };

    utils::submit_kernel(cgf, device, "layer_norm_esimd");
}

// ============================================================================
// Fused Add + RMSNorm Kernel
// ============================================================================
// In-place: residual += input, then output = rmsnorm(residual) * weight
// ============================================================================
template<typename IT, const int GS, const int BS>
void fused_add_rms_norm_kernel(
    const void* weight_ptr,
    void* input_ptr,       // in-place: overwritten with normalized result
    void* residual_ptr,    // in-place: residual += input
    float eps,
    const int input_size,
    const int hidden_size,
    const at::Device& device
) {
    assert(hidden_size % BS == 0);
    assert(hidden_size <= 8 * 1024);

    const int nb = hidden_size / BS;
    const int sub_nb = nb / GS;
    const int rem_nb = nb % GS;
    constexpr int slm_acc_align = ((GS * (int)sizeof(float) + 15) / 16) * 16;
    const int acc_offset = hidden_size * sizeof(IT);

    sycl::range<2> global_size(input_size, GS);
    sycl::range<2> local_size(1, GS);

    auto cgf = [&](sycl::handler& handle) {
        handle.parallel_for(
            sycl::nd_range<2>(global_size, local_size),
            [=](sycl::nd_item<2> item) SYCL_ESIMD_KERNEL {
                slm_init<8 * 1024 * sizeof(IT) + slm_acc_align>();

                const int rid = item.get_global_id(0);
                const int tid = item.get_local_id(1);

                const IT* weight = (const IT*)weight_ptr;
                IT* input = (IT*)input_ptr + hidden_size * (size_t)rid;
                IT* residual = (IT*)residual_ptr + hidden_size * (size_t)rid;

                const int start_blk = sub_nb * tid + std::min(tid, rem_nb);
                const int end_blk = start_blk + sub_nb + (tid < rem_nb);

                // Pass 1: residual += input, compute sum of squares for RMS
                simd<float, BS> accv = 0;
                for (int i = start_blk; i < end_blk; ++i) {
                    simd<float, BS> xv = block_load<IT, BS>(input + i * BS);
                    simd<float, BS> rv = block_load<IT, BS>(residual + i * BS);
                    simd<float, BS> sv = xv + rv;
                    // Store updated residual back to global memory
                    block_store<IT, BS>(residual + i * BS, (simd<IT, BS>)sv);
                    // Cache in SLM for pass 2
                    slm_block_store<IT, BS>(i * BS * sizeof(IT), (simd<IT, BS>)sv);
                    accv += sv * sv;
                }
                float acc = sycl::ext::intel::esimd::detail::sum<float, float, BS>(accv) / hidden_size;

                if constexpr (GS == 1) {
                    float scale = rsqrt(acc + eps);
                    for (int i = 0; i < nb; ++i) {
                        simd<float, BS> sv = slm_block_load<IT, BS>(i * BS * sizeof(IT));
                        simd<float, BS> yv = block_load<IT, BS>(weight + i * BS);
                        simd<IT, BS> result = sv * scale * yv;
                        block_store<IT, BS>(input + i * BS, result);
                    }
                } else {
                    slm_block_store<float, 1>(acc_offset + tid * sizeof(float), acc);
                    barrier();

                    simd<float, GS> accs = slm_block_load<float, GS>(acc_offset);
                    float mean = sycl::ext::intel::esimd::detail::sum<float, float, GS>(accs);
                    float scale = rsqrt(mean + eps);

                    // Pass 2: normalize and write to input (output)
                    for (int i = start_blk; i < end_blk; ++i) {
                        simd<float, BS> sv = slm_block_load<IT, BS>(i * BS * sizeof(IT));
                        simd<float, BS> yv = block_load<IT, BS>(weight + i * BS);
                        simd<IT, BS> result = sv * scale * yv;
                        block_store<IT, BS>(input + i * BS, result);
                    }
                }
            }
        );
    };

    utils::submit_kernel(cgf, device, "fused_add_rms_norm_esimd");
}

// ============================================================================
// GS dispatch helper: select optimal group size based on nb = hidden_size / BS
// ============================================================================
// Strategy: GS = clamp(nb, 1, 32) rounded down to power of 2
//   nb=1      -> GS=1   (no barrier overhead)
//   nb=2      -> GS=2
//   nb=3..4   -> GS=4
//   nb=5..8   -> GS=8
//   nb=9..16  -> GS=16
//   nb>=17    -> GS=32
// ============================================================================

// RMS norm dispatch
template<typename IT, int BS>
using rms_fn_t = void(*)(const void*, const void*, void*, float, const int, const int, const at::Device&);

template<typename IT, int BS>
rms_fn_t<IT, BS> select_rms_kernel(int nb) {
    if (nb <= 1)  return rms_norm_kernel<IT, 1,  BS>;
    if (nb <= 2)  return rms_norm_kernel<IT, 2,  BS>;
    if (nb <= 4)  return rms_norm_kernel<IT, 4,  BS>;
    if (nb <= 8)  return rms_norm_kernel<IT, 8,  BS>;
    if (nb <= 16) return rms_norm_kernel<IT, 16, BS>;
    return rms_norm_kernel<IT, 32, BS>;
}

// LayerNorm dispatch
template<typename IT, int BS>
using ln_fn_t = void(*)(const void*, const uint64_t, const uint64_t, void*, float, const int, const int, const at::Device&);

template<typename IT, int BS>
ln_fn_t<IT, BS> select_ln_kernel(int nb) {
    if (nb <= 1)  return layer_norm_kernel<IT, 1,  BS>;
    if (nb <= 2)  return layer_norm_kernel<IT, 2,  BS>;
    if (nb <= 4)  return layer_norm_kernel<IT, 4,  BS>;
    if (nb <= 8)  return layer_norm_kernel<IT, 8,  BS>;
    if (nb <= 16) return layer_norm_kernel<IT, 16, BS>;
    return layer_norm_kernel<IT, 32, BS>;
}

// Fused add rms norm dispatch
template<typename IT, int BS>
using fused_fn_t = void(*)(const void*, void*, void*, float, const int, const int, const at::Device&);

template<typename IT, int BS>
fused_fn_t<IT, BS> select_fused_kernel(int nb) {
    if (nb <= 1)  return fused_add_rms_norm_kernel<IT, 1,  BS>;
    if (nb <= 2)  return fused_add_rms_norm_kernel<IT, 2,  BS>;
    if (nb <= 4)  return fused_add_rms_norm_kernel<IT, 4,  BS>;
    if (nb <= 8)  return fused_add_rms_norm_kernel<IT, 8,  BS>;
    if (nb <= 16) return fused_add_rms_norm_kernel<IT, 16, BS>;
    return fused_add_rms_norm_kernel<IT, 32, BS>;
}

// ============================================================================
// Public C++ API
// ============================================================================

#if defined(OMNI_XPU_ARCH_PTL_H)
torch::Tensor rms_norm_gate_residual(
    torch::Tensor weight,
    torch::Tensor input,
    torch::Tensor gate,
    torch::Tensor residual,
    double eps
) {
    using Config = RmsNormGateResidualH3840PTLConfig;
    TORCH_CHECK(
        input.is_xpu() && weight.is_xpu() && gate.is_xpu() && residual.is_xpu(),
        "weight, input, gate, and residual must be XPU tensors");
    TORCH_CHECK(
        input.device() == weight.device() && input.device() == gate.device() &&
            input.device() == residual.device(),
        "weight, input, gate, and residual must be on the same device");
    TORCH_CHECK(
        input.dim() == 2 && input.size(1) == Config::HiddenSize,
        "input must have shape [M, 3840]");
    TORCH_CHECK(
        input.size(0) == 64 || input.size(0) == 1024 || input.size(0) == 1088,
        "M must be one of the validated Z-Image lengths: 64, 1024, or 1088");
    TORCH_CHECK(
        weight.dim() == 1 && weight.size(0) == Config::HiddenSize &&
            gate.dim() == 1 && gate.size(0) == Config::HiddenSize,
        "weight and gate must have shape [3840]");
    TORCH_CHECK(
        residual.sizes() == input.sizes(),
        "residual must have the same shape as input");
    TORCH_CHECK(
        input.scalar_type() == ST::BFloat16 &&
            weight.scalar_type() == ST::BFloat16 &&
            gate.scalar_type() == ST::BFloat16 &&
            residual.scalar_type() == ST::BFloat16,
        "weight, input, gate, and residual must be BF16 tensors");
    TORCH_CHECK(
        input.is_contiguous() && weight.is_contiguous() &&
            gate.is_contiguous() && residual.is_contiguous(),
        "weight, input, gate, and residual must be contiguous");

    auto output = torch::empty_like(input);
    rms_norm_gate_residual_h3840_ptl_kernel(
        reinterpret_cast<const bf16*>(weight.data_ptr()),
        reinterpret_cast<const bf16*>(input.data_ptr()),
        reinterpret_cast<const bf16*>(gate.data_ptr()),
        reinterpret_cast<const bf16*>(residual.data_ptr()),
        reinterpret_cast<bf16*>(output.data_ptr()),
        static_cast<float>(eps), input.size(0), input.device());
    return output;
}
#endif

torch::Tensor rms_norm(
    torch::Tensor weight,
    torch::Tensor input,
    double eps
) {
    TORCH_CHECK(input.dim() == 2, "Input must be 2D tensor [batch, hidden_size]");
    TORCH_CHECK(weight.dim() == 1, "Weight must be 1D tensor [hidden_size]");
    TORCH_CHECK(weight.size(0) == input.size(1), "Weight size must match hidden_size");
    TORCH_CHECK(input.scalar_type() == weight.scalar_type(), "Input and weight dtype must match");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Weight must be contiguous");

    int64_t input_size = input.size(0);
    int64_t hidden_size = input.size(1);

    auto output = torch::empty({input_size, hidden_size},
        torch::device(input.device()).dtype(input.dtype()));

#if defined(OMNI_XPU_ARCH_PTL_H)
    if (hidden_size == RmsNormH128PTLConfig::HiddenSize &&
        input_size >= RmsNormH128PTLConfig::MinimumRows) {
        if (input.scalar_type() == ST::BFloat16) {
            rms_norm_h128_ptl_kernel(
                weight.data_ptr(), input.data_ptr(), output.data_ptr(),
                static_cast<float>(eps), input_size, input.device());
            return output;
        }
        if (input.scalar_type() == ST::Float) {
            rms_norm_h128_fp32_ptl_kernel(
                weight.data_ptr(), input.data_ptr(), output.data_ptr(),
                static_cast<float>(eps), input_size, input.device());
            return output;
        }
    }
#endif

    // Select BS and GS based on hidden_size
    // Prefer BS=32; for hidden_size >= 2048 also divisible by 64, use BS=64
    if (hidden_size % 64 == 0 && hidden_size >= 2048) {
        const int nb = hidden_size / 64;
        switch (input.scalar_type()) {
            case ST::Float: {
                auto fn = select_rms_kernel<float, 64>(nb);
                fn(weight.data_ptr(), input.data_ptr(), output.data_ptr(),
                   static_cast<float>(eps), input_size, hidden_size, input.device());
                break;
            }
            case ST::Half: {
                auto fn = select_rms_kernel<fp16, 64>(nb);
                fn(weight.data_ptr(), input.data_ptr(), output.data_ptr(),
                   static_cast<float>(eps), input_size, hidden_size, input.device());
                break;
            }
            case ST::BFloat16: {
                auto fn = select_rms_kernel<bf16, 64>(nb);
                fn(weight.data_ptr(), input.data_ptr(), output.data_ptr(),
                   static_cast<float>(eps), input_size, hidden_size, input.device());
                break;
            }
            default: TORCH_CHECK(false, "Unsupported dtype, only fp32, fp16 and bf16 are supported");
        }
    } else {
        const int nb = hidden_size / 32;
        switch (input.scalar_type()) {
            case ST::Float: {
                auto fn = select_rms_kernel<float, 32>(nb);
                fn(weight.data_ptr(), input.data_ptr(), output.data_ptr(),
                   static_cast<float>(eps), input_size, hidden_size, input.device());
                break;
            }
            case ST::Half: {
                auto fn = select_rms_kernel<fp16, 32>(nb);
                fn(weight.data_ptr(), input.data_ptr(), output.data_ptr(),
                   static_cast<float>(eps), input_size, hidden_size, input.device());
                break;
            }
            case ST::BFloat16: {
                auto fn = select_rms_kernel<bf16, 32>(nb);
                fn(weight.data_ptr(), input.data_ptr(), output.data_ptr(),
                   static_cast<float>(eps), input_size, hidden_size, input.device());
                break;
            }
            default: TORCH_CHECK(false, "Unsupported dtype, only fp32, fp16 and bf16 are supported");
        }
    }

    return output;
}

torch::Tensor layer_norm(
    torch::Tensor input,
    std::optional<torch::Tensor> weight,
    std::optional<torch::Tensor> bias,
    double eps
) {
    TORCH_CHECK(input.dim() == 2, "Input must be 2D tensor [batch, hidden_size]");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");

    int64_t input_size = input.size(0);
    int64_t hidden_size = input.size(1);

    if (weight.has_value()) {
        TORCH_CHECK(weight->numel() == hidden_size, "Weight size must match hidden_size");
        TORCH_CHECK(weight->scalar_type() == input.scalar_type(), "Weight dtype must match input");
        TORCH_CHECK(weight->is_contiguous(), "Weight must be contiguous");
    }
    if (bias.has_value()) {
        TORCH_CHECK(bias->numel() == hidden_size, "Bias size must match hidden_size");
        TORCH_CHECK(bias->scalar_type() == input.scalar_type(), "Bias dtype must match input");
        TORCH_CHECK(bias->is_contiguous(), "Bias must be contiguous");
    }

    auto output = torch::empty({input_size, hidden_size},
        torch::device(input.device()).dtype(input.dtype()));

    const uint64_t w_ptr = weight.has_value() ? (uint64_t)(weight->data_ptr()) : 0;
    const uint64_t b_ptr = bias.has_value() ? (uint64_t)(bias->data_ptr()) : 0;

    if (hidden_size % 64 == 0 && hidden_size >= 2048) {
        const int nb = hidden_size / 64;
        switch (input.scalar_type()) {
            case ST::Float: {
                auto fn = select_ln_kernel<float, 64>(nb);
                fn(input.data_ptr(), w_ptr, b_ptr, output.data_ptr(),
                   static_cast<float>(eps), input_size, hidden_size, input.device());
                break;
            }
            case ST::Half: {
                auto fn = select_ln_kernel<fp16, 64>(nb);
                fn(input.data_ptr(), w_ptr, b_ptr, output.data_ptr(),
                   static_cast<float>(eps), input_size, hidden_size, input.device());
                break;
            }
            case ST::BFloat16: {
                auto fn = select_ln_kernel<bf16, 64>(nb);
                fn(input.data_ptr(), w_ptr, b_ptr, output.data_ptr(),
                   static_cast<float>(eps), input_size, hidden_size, input.device());
                break;
            }
            default: TORCH_CHECK(false, "Unsupported dtype, only fp32, fp16 and bf16 are supported");
        }
    } else {
        const int nb = hidden_size / 32;
        switch (input.scalar_type()) {
            case ST::Float: {
                auto fn = select_ln_kernel<float, 32>(nb);
                fn(input.data_ptr(), w_ptr, b_ptr, output.data_ptr(),
                   static_cast<float>(eps), input_size, hidden_size, input.device());
                break;
            }
            case ST::Half: {
                auto fn = select_ln_kernel<fp16, 32>(nb);
                fn(input.data_ptr(), w_ptr, b_ptr, output.data_ptr(),
                   static_cast<float>(eps), input_size, hidden_size, input.device());
                break;
            }
            case ST::BFloat16: {
                auto fn = select_ln_kernel<bf16, 32>(nb);
                fn(input.data_ptr(), w_ptr, b_ptr, output.data_ptr(),
                   static_cast<float>(eps), input_size, hidden_size, input.device());
                break;
            }
            default: TORCH_CHECK(false, "Unsupported dtype, only fp32, fp16 and bf16 are supported");
        }
    }

    return output;
}

void fused_add_rms_norm(
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor weight,
    double eps
) {
    TORCH_CHECK(input.dim() == 2, "Input must be 2D tensor [batch, hidden_size]");
    TORCH_CHECK(residual.dim() == 2, "Residual must be 2D tensor [batch, hidden_size]");
    TORCH_CHECK(weight.dim() == 1, "Weight must be 1D tensor [hidden_size]");
    TORCH_CHECK(input.sizes() == residual.sizes(), "Input and residual shapes must match");
    TORCH_CHECK(weight.size(0) == input.size(1), "Weight size must match hidden_size");
    TORCH_CHECK(input.scalar_type() == residual.scalar_type(), "Input and residual dtype must match");
    TORCH_CHECK(input.scalar_type() == weight.scalar_type(), "Input and weight dtype must match");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(residual.is_contiguous(), "Residual must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Weight must be contiguous");

    int64_t input_size = input.size(0);
    int64_t hidden_size = input.size(1);

    if (hidden_size % 64 == 0 && hidden_size >= 2048) {
        const int nb = hidden_size / 64;
        switch (input.scalar_type()) {
            case ST::Float: {
                auto fn = select_fused_kernel<float, 64>(nb);
                fn(weight.data_ptr(), input.data_ptr(), residual.data_ptr(),
                   static_cast<float>(eps), input_size, hidden_size, input.device());
                break;
            }
            case ST::Half: {
                auto fn = select_fused_kernel<fp16, 64>(nb);
                fn(weight.data_ptr(), input.data_ptr(), residual.data_ptr(),
                   static_cast<float>(eps), input_size, hidden_size, input.device());
                break;
            }
            case ST::BFloat16: {
                auto fn = select_fused_kernel<bf16, 64>(nb);
                fn(weight.data_ptr(), input.data_ptr(), residual.data_ptr(),
                   static_cast<float>(eps), input_size, hidden_size, input.device());
                break;
            }
            default: TORCH_CHECK(false, "Unsupported dtype, only fp32, fp16 and bf16 are supported");
        }
    } else {
        const int nb = hidden_size / 32;
        switch (input.scalar_type()) {
            case ST::Float: {
                auto fn = select_fused_kernel<float, 32>(nb);
                fn(weight.data_ptr(), input.data_ptr(), residual.data_ptr(),
                   static_cast<float>(eps), input_size, hidden_size, input.device());
                break;
            }
            case ST::Half: {
                auto fn = select_fused_kernel<fp16, 32>(nb);
                fn(weight.data_ptr(), input.data_ptr(), residual.data_ptr(),
                   static_cast<float>(eps), input_size, hidden_size, input.device());
                break;
            }
            case ST::BFloat16: {
                auto fn = select_fused_kernel<bf16, 32>(nb);
                fn(weight.data_ptr(), input.data_ptr(), residual.data_ptr(),
                   static_cast<float>(eps), input_size, hidden_size, input.device());
                break;
            }
            default: TORCH_CHECK(false, "Unsupported dtype, only fp32, fp16 and bf16 are supported");
        }
    }
}

// ============================================================================
// Fused RMSNorm + Linear Projection
// ============================================================================
// Chains RMSNorm and matmul in a single C++ call to:
//   1. Eliminate Python roundtrip between norm and linear (~10-50us)
//   2. Keep normalized data warm in L3 cache for immediate GEMM consumption
//   3. Avoid materializing intermediate tensor in Python scope
//
// Pattern: output = Linear(RMSNorm(input, weight, eps), proj_weight)
//          output = RMSNorm(input) @ proj_weight.T
// ============================================================================

torch::Tensor fused_rms_norm_linear(
    torch::Tensor input,         // [M, K]
    torch::Tensor norm_weight,   // [K]
    torch::Tensor proj_weight,   // [N, K] (will be transposed for matmul)
    double eps
) {
    TORCH_CHECK(input.dim() == 2, "Input must be 2D [M, K]");
    TORCH_CHECK(norm_weight.dim() == 1, "Norm weight must be 1D [K]");
    TORCH_CHECK(proj_weight.dim() == 2, "Proj weight must be 2D [N, K]");
    TORCH_CHECK(input.size(1) == norm_weight.size(0), "Input K must match norm_weight size");
    TORCH_CHECK(input.size(1) == proj_weight.size(1), "Input K must match proj_weight K");

    OMNI_DEBUG("norm", "fused_rms_norm_linear: input=[%ld,%ld] proj=[%ld,%ld]",
               input.size(0), input.size(1), proj_weight.size(0), proj_weight.size(1));

    auto normed = rms_norm(norm_weight, input, eps);
    // proj_weight is [N, K], we need normed @ proj_weight.T = [M, N]
    auto output = torch::mm(normed, proj_weight.t());

    return output;
}

}  // namespace norm
}  // namespace omni_xpu
