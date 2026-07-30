// ============================================================================
// Fused INT8 Quantization Kernels
// ============================================================================
// High-performance quantization kernels that fuse:
//   absmax reduction + scale computation + divide + round + clamp + cast
// into minimal kernel launches, eliminating Python-level multi-op overhead.
//
// Input:  [M, K] bf16/f16
// Output: [M, K] int8 + [M, 1] float32 scales
//
// Design: One plain-SYCL sub-group cooperatively processes each row. Each lane
// handles contiguous vector chunks so both passes use wide coalesced accesses.
// ============================================================================

#include <torch/extension.h>
#include <sycl/sycl.hpp>
#if defined(OMNI_XPU_ARCH_BMG)
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#endif
#include <type_traits>

#if defined(OMNI_XPU_ARCH_BMG)
#include "bmg_kernel_policy.h"
#include "device_utils.h"
#endif
#include "utils.h"

using fp16 = sycl::half;
using bf16 = sycl::ext::oneapi::bfloat16;
#if defined(OMNI_XPU_ARCH_BMG)
namespace esimd = sycl::ext::intel::esimd;
namespace xmx = sycl::ext::intel::esimd::xmx;
#endif

namespace omni_xpu {
namespace int8_ops {

#ifndef OMNI_SILU_MUL_ELEMENTS_PER_WI
#if defined(OMNI_XPU_ARCH_PTL_H)
// PTL-H's math pipeline benefits from exposing each exp as an independent
// work-item instead of serializing several transcendental operations.
#define OMNI_SILU_MUL_ELEMENTS_PER_WI 1
#elif defined(OMNI_XPU_ARCH_BMG)
// BMG workflow shapes are also math-pipeline limited: exposing each exp as an
// independent work-item is faster than serializing eight transcendental
// operations while preserving byte-exact FP16/BF16 output.
#define OMNI_SILU_MUL_ELEMENTS_PER_WI 1
#else
#error "Define OMNI_XPU_ARCH_PTL_H or OMNI_XPU_ARCH_BMG"
#endif
#endif

// ============================================================================
// Kernel: Fused per-row INT8 quantization
// ============================================================================
// Each sub-group processes one full row of K elements:
//   1. Compute absmax across the row (vectorized reduce)
//   2. Compute scale = absmax / 127
//   3. Quantize: round(x / scale).clamp(-128, 127) → int8
// ============================================================================

template <typename InputT>
void quantize_int8_rowwise_kernel(
    const InputT* __restrict__ input,  // [M, K]
    int8_t* __restrict__ output,       // [M, K]
    float* __restrict__ scales,        // [M]
    int64_t M,
    int64_t K,
    const at::Device& device
) {
    // Subgroup-cooperative rowwise INT8 quantization (plain SYCL).
    //
    // One sub-group (SG lanes) processes one row of K elements. Large aligned
    // rows use contiguous VEC-element chunks per lane; other rows use scalar
    // interleaving. Both layouts give fully coalesced HBM access. Row absmax is
    // reduced via a sub-group collective (no SLM, no barrier). Pass 2 re-reads
    // the row (served from L2 for typical K) and writes int8.
    //
    // This replaces the previous ESIMD SLM kernel, which suffered a large IGC
    // JIT register-spill penalty inside the multi-kernel _C module (measured
    // ~38 GB/s vs ~140 GB/s for the original scalar sub-group design at
    // M=4128, K=3840).
    //
    // HBM traffic: K*2 read (pass1) + K*2 read (pass2, L2-cached) + K*1 write.
    constexpr int SG = 32;            // sub-group size (lanes per row)
    constexpr int ROWS_PER_WG = 8;    // rows (sub-groups) per work-group
    constexpr int WG = SG * ROWS_PER_WG;
    constexpr int VEC = 8;            // contiguous elements handled by each lane
    constexpr int MIN_VECTOR_K = SG * VEC * 2;

    const int64_t n_wg = (M + ROWS_PER_WG - 1) / ROWS_PER_WG;
    sycl::range<1> global_size(static_cast<size_t>(n_wg) * WG);
    sycl::range<1> local_size(WG);

    auto cgf = [&](sycl::handler& handle) {
        handle.parallel_for(
            sycl::nd_range<1>(global_size, local_size),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG)]] {
                auto sg = item.get_sub_group();
                const int lane = static_cast<int>(sg.get_local_linear_id());
                const int64_t row =
                    static_cast<int64_t>(item.get_group(0)) * ROWS_PER_WG +
                    sg.get_group_linear_id();
                if (row >= M) return;

                const InputT* __restrict__ row_ptr = input + row * K;
                int8_t* __restrict__ out_ptr = output + row * K;

                // Process aligned rows with wider per-lane transactions. One
                // sub-group iteration covers SG*VEC contiguous elements, which
                // reduces loop/address overhead without increasing GRF pressure
                // enough to trigger the spills seen in the old ESIMD kernel.
                float local_max = 0.0f;
                const bool use_vector = K >= MIN_VECTOR_K && K % VEC == 0;
                if (use_vector) {
                    using InputVec = sycl::vec<InputT, VEC>;
                    const int64_t lane_start = static_cast<int64_t>(lane) * VEC;
                    for (int64_t k = lane_start; k < K; k += SG * VEC) {
                        const InputVec values =
                            *reinterpret_cast<const InputVec*>(row_ptr + k);
                        #pragma unroll
                        for (int i = 0; i < VEC; ++i) {
                            const float v = static_cast<float>(values[i]);
                            local_max = sycl::fmax(local_max, sycl::fabs(v));
                        }
                    }
                } else {
                    for (int64_t k = lane; k < K; k += SG) {
                        const float v = static_cast<float>(row_ptr[k]);
                        local_max = sycl::fmax(local_max, sycl::fabs(v));
                    }
                }
                float row_max =
                    sycl::reduce_over_group(sg, local_max, sycl::maximum<float>());

                float scale = row_max / 127.0f;
                if (scale < 1e-30f) scale = 1e-30f;
                float inv_scale = 1.0f / scale;
                if (lane == 0) scales[row] = scale;

                // Pass 2: coalesced quantize + write (round-to-nearest-even).
                if (use_vector) {
                    using InputVec = sycl::vec<InputT, VEC>;
                    using OutputVec = sycl::vec<int8_t, VEC>;
                    const int64_t lane_start = static_cast<int64_t>(lane) * VEC;
                    for (int64_t k = lane_start; k < K; k += SG * VEC) {
                        const InputVec values =
                            *reinterpret_cast<const InputVec*>(row_ptr + k);
                        OutputVec quantized;
                        #pragma unroll
                        for (int i = 0; i < VEC; ++i) {
                            float r = sycl::rint(
                                static_cast<float>(values[i]) * inv_scale);
                            r = sycl::fmax(-128.0f, sycl::fmin(127.0f, r));
                            quantized[i] =
                                static_cast<int8_t>(static_cast<int32_t>(r));
                        }
                        *reinterpret_cast<OutputVec*>(out_ptr + k) = quantized;
                    }
                } else {
                    for (int64_t k = lane; k < K; k += SG) {
                        float r = sycl::rint(
                            static_cast<float>(row_ptr[k]) * inv_scale);
                        r = sycl::fmax(-128.0f, sycl::fmin(127.0f, r));
                        out_ptr[k] =
                            static_cast<int8_t>(static_cast<int32_t>(r));
                    }
                }
            }
        );
    };
    utils::submit_kernel(cgf, device, "quantize_int8_rowwise_sg2pass");
}

#if defined(OMNI_XPU_ARCH_PTL_H) || defined(OMNI_XPU_ARCH_BMG)
#if defined(OMNI_XPU_ARCH_PTL_H)
// Large PTL BF16 ConvRot activations no longer remain cache-resident between
// the generic kernel's two global-memory passes. One work-group therefore owns
// one row, stages it in SLM, and cooperatively performs reduction and
// quantization. Keep every launch geometry tied to its measured workflow shape.
struct RowwiseQuantizeLargePTLConfig {
    static constexpr int Columns = 10240;
    static constexpr int MinimumRows = 1024;
    static constexpr int SubgroupSize = 32;
    static constexpr int SubgroupsPerRow = 24;
    static constexpr int WorkgroupSize = SubgroupSize * SubgroupsPerRow;
    static constexpr int VectorWidth = 16;
};

// Krea2 Turbo INT8 at 1024x1024 produces BF16 [4192, 6144] activations. Eight
// subgroups loop over three 2048-column stripes while retaining the complete
// 12 KiB row in SLM, eliminating the second global read.
struct RowwiseQuantizeKrea2PTLConfig {
    static constexpr int Columns = 6144;
    static constexpr int Rows = 4192;
    static constexpr int SubgroupSize = 32;
    static constexpr int SubgroupsPerRow = 8;
    static constexpr int WorkgroupSize = SubgroupSize * SubgroupsPerRow;
    static constexpr int VectorWidth = 8;
};

// The Krea2 FFN down projection quantizes a much wider BF16 activation before
// its K=16384 -> N=6144 INT8 matmul. Sixteen subgroups stage the complete
// 32 KiB row in SLM and preserve the generic max/scale/rounding result while
// eliminating its second global-memory read.
struct RowwiseQuantizeKrea2FFNDownPTLConfig {
    static constexpr int Columns = 16384;
    static constexpr int Rows = 4192;
    static constexpr int SubgroupSize = 32;
    static constexpr int SubgroupsPerRow = 16;
    static constexpr int WorkgroupSize = SubgroupSize * SubgroupsPerRow;
    static constexpr int VectorWidth = 8;
};

// Boogu Image Turbo at 1024x1024 uses FP16 activations with either the image
// stream (4096 rows) or the joined image/instruction stream (4205 rows).  Its
// hidden projections quantize K=3360, while FFN down projections quantize
// K=13568.  Both shapes benefit from retaining one complete row in SLM; twenty
// SG32 subgroups with VEC16 were selected by a core-matched, process-isolated
// sweep.  Keep the row guard exact so unrelated FP16 models retain the generic
// path.
struct RowwiseQuantizeBooguHiddenPTLConfig {
    static constexpr int Columns = 3360;
    static constexpr int ImageRows = 4096;
    static constexpr int JointRows = 4205;
    static constexpr int SubgroupSize = 32;
    static constexpr int SubgroupsPerRow = 20;
    static constexpr int WorkgroupSize = SubgroupSize * SubgroupsPerRow;
    static constexpr int VectorWidth = 16;
};

struct RowwiseQuantizeBooguFFNDownPTLConfig {
    static constexpr int Columns = 13568;
    static constexpr int ImageRows = 4096;
    static constexpr int JointRows = 4205;
    static constexpr int SubgroupSize = 32;
    static constexpr int SubgroupsPerRow = 20;
    static constexpr int WorkgroupSize = SubgroupSize * SubgroupsPerRow;
    static constexpr int VectorWidth = 16;
};
#endif

#if defined(OMNI_XPU_ARCH_BMG)
// BMG BF16 routes use exact workflow shapes. Z-Image produces M=4096 and
// M=4128 at K=3840 and K=10240; Krea2 produces M=4192 at K=6144 and K=16384.
// Process-isolated continuous benchmarks selected VEC8/SG16 for K=3840,
// VEC8/SG20 for K=10240, VEC8/SG12 for K=6144, and VEC8/SG32 for K=16384.
// K=6144 SG12 and SG20 were effectively tied in the continuous benchmark;
// SG12 is the configuration validated end to end. All selected configurations
// preserve byte-exact Q/scale results.
struct RowwiseQuantizeZImageHiddenBMGConfig {
    static constexpr int Columns = 3840;
    static constexpr int ImageRows = 4096;
    static constexpr int JointRows = 4128;
    static constexpr int SubgroupSize = 32;
    static constexpr int SubgroupsPerRow = 16;
    static constexpr int WorkgroupSize = SubgroupSize * SubgroupsPerRow;
    static constexpr int VectorWidth = 8;
};

struct RowwiseQuantizeZImageBMGConfig {
    static constexpr int Columns = 10240;
    static constexpr int ImageRows = 4096;
    static constexpr int JointRows = 4128;
    static constexpr int SubgroupSize = 32;
    static constexpr int SubgroupsPerRow = 20;
    static constexpr int WorkgroupSize = SubgroupSize * SubgroupsPerRow;
    static constexpr int VectorWidth = 8;
};

struct RowwiseQuantizeKrea2BMGConfig {
    static constexpr int Columns = 6144;
    static constexpr int Rows = 4192;
    static constexpr int SubgroupSize = 32;
    static constexpr int SubgroupsPerRow = 12;
    static constexpr int WorkgroupSize = SubgroupSize * SubgroupsPerRow;
    static constexpr int VectorWidth = 8;
};

struct RowwiseQuantizeKrea2FFNDownBMGConfig {
    static constexpr int Columns = 16384;
    static constexpr int Rows = 4192;
    static constexpr int SubgroupSize = 32;
    static constexpr int SubgroupsPerRow = 32;
    static constexpr int WorkgroupSize = SubgroupSize * SubgroupsPerRow;
    static constexpr int VectorWidth = 8;
};

// BMG was tuned independently for the same Boogu Image Turbo 1024x1024 FP16
// shapes. VEC16/SG20, which is the PTL-H winner, regresses BMG's K=3360 route.
// A process-isolated BMG sweep selected VEC8/SG16 while preserving byte-exact
// quantized output and scales for both deterministic and random inputs.
struct RowwiseQuantizeBooguHiddenBMGConfig {
    static constexpr int Columns = 3360;
    static constexpr int ImageRows = 4096;
    static constexpr int JointRows = 4205;
    static constexpr int SubgroupSize = 32;
    static constexpr int SubgroupsPerRow = 16;
    static constexpr int WorkgroupSize = SubgroupSize * SubgroupsPerRow;
    static constexpr int VectorWidth = 8;
};

struct RowwiseQuantizeBooguFFNDownBMGConfig {
    static constexpr int Columns = 13568;
    static constexpr int ImageRows = 4096;
    static constexpr int JointRows = 4205;
    static constexpr int SubgroupSize = 32;
    static constexpr int SubgroupsPerRow = 16;
    static constexpr int WorkgroupSize = SubgroupSize * SubgroupsPerRow;
    static constexpr int VectorWidth = 8;
};
#endif

// Retain the historical PTL kernel identity so existing PTL traces remain
// comparable. BMG uses the same SLM algorithm with independently tuned config
// types and a BMG-specific submission label.
template <
    typename InputT,
    int Columns,
    int VectorWidth,
    int SubgroupsPerRow>
class QuantizeInt8RowwiseLargePTLKernel;

template <typename InputT, typename Config>
void quantize_int8_rowwise_large_ptl_kernel(
    const InputT* __restrict__ input,
    int8_t* __restrict__ output,
    float* __restrict__ scales,
    int64_t rows,
    const at::Device& device,
    const char* submission_name = "quantize_int8_rowwise_large_ptl") {
    using InputVector = sycl::vec<InputT, Config::VectorWidth>;
    using OutputVector = sycl::vec<int8_t, Config::VectorWidth>;
    static_assert(Config::WorkgroupSize <= 1024);

    const sycl::range<1> local(Config::WorkgroupSize);
    const sycl::range<1> global(
        static_cast<size_t>(rows) * Config::WorkgroupSize);
    auto cgf = [&](sycl::handler& handle) {
        sycl::local_accessor<InputT, 1> row_cache(
            sycl::range<1>(Config::Columns), handle);
        sycl::local_accessor<float, 1> subgroup_maxima(
            sycl::range<1>(Config::SubgroupsPerRow), handle);
        handle.parallel_for<QuantizeInt8RowwiseLargePTLKernel<
            InputT,
            Config::Columns,
            Config::VectorWidth,
            Config::SubgroupsPerRow>>(
            sycl::nd_range<1>(global, local),
            [=](sycl::nd_item<1> item)
                [[sycl::reqd_sub_group_size(Config::SubgroupSize)]] {
                auto subgroup = item.get_sub_group();
                const int lane =
                    static_cast<int>(subgroup.get_local_linear_id());
                const int subgroup_index =
                    static_cast<int>(subgroup.get_group_linear_id());
                const int workgroup_lane =
                    subgroup_index * Config::SubgroupSize + lane;
                const int64_t row = static_cast<int64_t>(item.get_group(0));
                const InputT* __restrict__ row_input =
                    input + row * Config::Columns;
                int8_t* __restrict__ row_output =
                    output + row * Config::Columns;
                InputT* local_ptr = row_cache
                    .template get_multi_ptr<sycl::access::decorated::no>()
                    .get();

                float local_max = 0.0f;
                constexpr int VectorStride =
                    Config::WorkgroupSize * Config::VectorWidth;
                for (int first = workgroup_lane * Config::VectorWidth;
                     first < Config::Columns;
                     first += VectorStride) {
                    const InputVector values =
                        *reinterpret_cast<const InputVector*>(
                            row_input + first);
#pragma unroll
                    for (int element = 0; element < Config::VectorWidth;
                         ++element) {
                        local_ptr[first + element] = values[element];
                        local_max = sycl::fmax(
                            local_max,
                            sycl::fabs(static_cast<float>(values[element])));
                    }
                }

                const float subgroup_max = sycl::reduce_over_group(
                    subgroup, local_max, sycl::maximum<float>());
                if (lane == 0) {
                    subgroup_maxima[subgroup_index] = subgroup_max;
                }
                sycl::group_barrier(item.get_group());

                if (subgroup_index == 0) {
                    const float row_max = sycl::reduce_over_group(
                        subgroup,
                        lane < Config::SubgroupsPerRow
                            ? subgroup_maxima[lane]
                            : 0.0f,
                        sycl::maximum<float>());
                    if (lane == 0) {
                        float scale = row_max / 127.0f;
                        if (scale < 1e-30f) scale = 1e-30f;
                        subgroup_maxima[0] = scale;
                        scales[row] = scale;
                    }
                }
                sycl::group_barrier(item.get_group());

                for (int first = workgroup_lane * Config::VectorWidth;
                     first < Config::Columns;
                     first += VectorStride) {
                    InputVector values;
#pragma unroll
                    for (int element = 0; element < Config::VectorWidth;
                         ++element) {
                        values[element] = local_ptr[first + element];
                    }
                    const float inverse_scale = 1.0f / subgroup_maxima[0];
                    OutputVector quantized;
#pragma unroll
                    for (int element = 0; element < Config::VectorWidth;
                         ++element) {
                        float rounded = sycl::rint(
                            static_cast<float>(values[element]) *
                            inverse_scale);
                        rounded = sycl::fmax(
                            -128.0f, sycl::fmin(127.0f, rounded));
                        quantized[element] = static_cast<int8_t>(
                            static_cast<int32_t>(rounded));
                    }
                    *reinterpret_cast<OutputVector*>(row_output + first) =
                        quantized;
                }
            });
    };
    utils::submit_kernel(cgf, device, submission_name);
}
#endif

template <typename InputT>
inline float silu_mul_rounded(InputT x1, InputT x2) {
    const float a = static_cast<float>(x1);
    const float b = static_cast<float>(x2);

    // Stable sigmoid formulation avoids exp overflow for large negative input.
    const float exp_value = sycl::exp(a >= 0.0f ? -a : a);
    const float sigmoid = a >= 0.0f
        ? 1.0f / (1.0f + exp_value)
        : exp_value / (1.0f + exp_value);

    // Match the existing PyTorch inference boundary as closely as possible:
    // F.silu(x1) is stored in the input dtype before the multiply, and the
    // product is stored in that dtype before rowwise quantization.
    const InputT silu_value = static_cast<InputT>(a * sigmoid);
    const InputT product = static_cast<InputT>(
        static_cast<float>(silu_value) * b);
    float value = static_cast<float>(product);

    // Lumina's clamp_fp16 applies nan_to_num to the floating SwiGLU result.
    if constexpr (std::is_same_v<InputT, fp16>) {
        if (sycl::isnan(value)) {
            value = 0.0f;
        } else {
            value = sycl::fmax(-65504.0f, sycl::fmin(65504.0f, value));
        }
    }
    return value;
}

template <typename InputT>
void fused_silu_mul_kernel(
    const InputT* __restrict__ input1,
    const InputT* __restrict__ input2,
    InputT* __restrict__ output,
    int64_t numel,
    const at::Device& device
) {
    constexpr int WG = 256;
    constexpr int VEC = OMNI_SILU_MUL_ELEMENTS_PER_WI;
    const int64_t work_items = (numel + VEC - 1) / VEC;
    const int64_t global_items = ((work_items + WG - 1) / WG) * WG;

    auto cgf = [&](sycl::handler& handle) {
        handle.parallel_for(
            sycl::nd_range<1>(
                sycl::range<1>(static_cast<size_t>(global_items)),
                sycl::range<1>(WG)),
            [=](sycl::nd_item<1> item) {
                const int64_t base =
                    static_cast<int64_t>(item.get_global_linear_id()) * VEC;
                #pragma unroll
                for (int i = 0; i < VEC; ++i) {
                    const int64_t index = base + i;
                    if (index < numel) {
                        output[index] = static_cast<InputT>(
                            silu_mul_rounded(input1[index], input2[index]));
                    }
                }
            });
    };
    utils::submit_kernel(cgf, device, "fused_silu_mul");
}

// One subgroup owns one row. The first pass recomputes the rounded SwiGLU
// value for absmax; the second pass recomputes it for quantization. This avoids
// the full floating [M,K] intermediate without introducing per-row scratch
// storage that would exceed registers/SLM at K=10240.
template <typename InputT>
void fused_silu_mul_quantize_rowwise_kernel(
    const InputT* __restrict__ input1,
    const InputT* __restrict__ input2,
    int8_t* __restrict__ output,
    float* __restrict__ scales,
    int64_t M,
    int64_t K,
    const at::Device& device
) {
    constexpr int SG = 32;
    constexpr int ROWS_PER_WG = 8;
    constexpr int WG = SG * ROWS_PER_WG;
    constexpr int VEC = 8;
    constexpr int MIN_VECTOR_K = SG * VEC * 2;

    const int64_t n_wg = (M + ROWS_PER_WG - 1) / ROWS_PER_WG;
    sycl::range<1> global_size(static_cast<size_t>(n_wg) * WG);
    sycl::range<1> local_size(WG);

    auto cgf = [&](sycl::handler& handle) {
        handle.parallel_for(
            sycl::nd_range<1>(global_size, local_size),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG)]] {
                auto sg = item.get_sub_group();
                const int lane = static_cast<int>(sg.get_local_linear_id());
                const int64_t row =
                    static_cast<int64_t>(item.get_group(0)) * ROWS_PER_WG +
                    sg.get_group_linear_id();
                if (row >= M) return;

                const InputT* __restrict__ row1 = input1 + row * K;
                const InputT* __restrict__ row2 = input2 + row * K;
                int8_t* __restrict__ out_ptr = output + row * K;

                float local_max = 0.0f;
                const bool use_vector = K >= MIN_VECTOR_K && K % VEC == 0;
                if (use_vector) {
                    using InputVec = sycl::vec<InputT, VEC>;
                    const int64_t lane_start = static_cast<int64_t>(lane) * VEC;
                    for (int64_t k = lane_start; k < K; k += SG * VEC) {
                        const InputVec values1 =
                            *reinterpret_cast<const InputVec*>(row1 + k);
                        const InputVec values2 =
                            *reinterpret_cast<const InputVec*>(row2 + k);
                        #pragma unroll
                        for (int i = 0; i < VEC; ++i) {
                            const float value =
                                silu_mul_rounded(values1[i], values2[i]);
                            local_max = sycl::fmax(
                                local_max, sycl::fabs(value));
                        }
                    }
                } else {
                    for (int64_t k = lane; k < K; k += SG) {
                        const float value =
                            silu_mul_rounded(row1[k], row2[k]);
                        local_max = sycl::fmax(
                            local_max, sycl::fabs(value));
                    }
                }

                const float row_max = sycl::reduce_over_group(
                    sg, local_max, sycl::maximum<float>());
                float scale = row_max / 127.0f;
                if (scale < 1e-30f) scale = 1e-30f;
                const float inv_scale = 1.0f / scale;
                if (lane == 0) scales[row] = scale;

                if (use_vector) {
                    using InputVec = sycl::vec<InputT, VEC>;
                    using OutputVec = sycl::vec<int8_t, VEC>;
                    const int64_t lane_start = static_cast<int64_t>(lane) * VEC;
                    for (int64_t k = lane_start; k < K; k += SG * VEC) {
                        const InputVec values1 =
                            *reinterpret_cast<const InputVec*>(row1 + k);
                        const InputVec values2 =
                            *reinterpret_cast<const InputVec*>(row2 + k);
                        OutputVec quantized;
                        #pragma unroll
                        for (int i = 0; i < VEC; ++i) {
                            float rounded = sycl::rint(
                                silu_mul_rounded(values1[i], values2[i]) *
                                inv_scale);
                            rounded = sycl::fmax(
                                -128.0f, sycl::fmin(127.0f, rounded));
                            quantized[i] = static_cast<int8_t>(
                                static_cast<int32_t>(rounded));
                        }
                        *reinterpret_cast<OutputVec*>(out_ptr + k) = quantized;
                    }
                } else {
                    for (int64_t k = lane; k < K; k += SG) {
                        float rounded = sycl::rint(
                            silu_mul_rounded(row1[k], row2[k]) * inv_scale);
                        rounded = sycl::fmax(
                            -128.0f, sycl::fmin(127.0f, rounded));
                        out_ptr[k] = static_cast<int8_t>(
                            static_cast<int32_t>(rounded));
                    }
                }
            });
    };
    utils::submit_kernel(
        cgf, device, "fused_silu_mul_quantize_rowwise_sg2pass");
}

// ============================================================================
// Public C++ API for fused quantization
// ============================================================================

torch::Tensor fused_silu_mul(
    torch::Tensor x1,
    torch::Tensor x2
) {
    TORCH_CHECK(x1.device().is_xpu(), "x1 must be on XPU device");
    TORCH_CHECK(x2.device().is_xpu(), "x2 must be on XPU device");
    TORCH_CHECK(x1.device() == x2.device(),
        "x1 and x2 must be on the same XPU device");
    TORCH_CHECK(x1.sizes() == x2.sizes(),
        "x1 and x2 must have identical shapes, got ",
        x1.sizes(), " and ", x2.sizes());
    TORCH_CHECK(x1.scalar_type() == x2.scalar_type(),
        "x1 and x2 must have identical dtypes");
    TORCH_CHECK(
        x1.scalar_type() == torch::kBFloat16 ||
        x1.scalar_type() == torch::kHalf,
        "x1 and x2 must be bf16 or f16, got ", x1.scalar_type());

    x1 = x1.contiguous();
    x2 = x2.contiguous();
    auto output = torch::empty_like(x1);
    if (x1.numel() == 0) return output;

    if (x1.scalar_type() == torch::kBFloat16) {
        fused_silu_mul_kernel<bf16>(
            reinterpret_cast<const bf16*>(x1.data_ptr()),
            reinterpret_cast<const bf16*>(x2.data_ptr()),
            reinterpret_cast<bf16*>(output.data_ptr()),
            x1.numel(), x1.device());
    } else {
        fused_silu_mul_kernel<fp16>(
            reinterpret_cast<const fp16*>(x1.data_ptr()),
            reinterpret_cast<const fp16*>(x2.data_ptr()),
            reinterpret_cast<fp16*>(output.data_ptr()),
            x1.numel(), x1.device());
    }
    return output;
}

std::tuple<torch::Tensor, torch::Tensor> quantize_int8_rowwise_fused(
    torch::Tensor x
) {
    TORCH_CHECK(x.device().is_xpu(), "x must be on XPU device");
    TORCH_CHECK(x.dim() >= 1, "x must be at least 1D");
    TORCH_CHECK(
        x.scalar_type() == torch::kBFloat16 ||
            x.scalar_type() == torch::kHalf ||
            x.scalar_type() == torch::kFloat,
        "x must be bf16, f16, or f32, got ", x.scalar_type()
    );

    x = x.contiguous();
    TORCH_CHECK(x.size(-1) > 0, "x last dimension must be non-empty");
    const int64_t M = x.numel() / x.size(-1);
    const int64_t K = x.size(-1);

    torch::Tensor output = torch::empty_like(x, torch::TensorOptions().dtype(torch::kInt8));
    torch::Tensor scales = torch::empty({M}, torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));

    if (x.scalar_type() == torch::kBFloat16) {
#if defined(OMNI_XPU_ARCH_PTL_H)
        if (M >= RowwiseQuantizeLargePTLConfig::MinimumRows &&
            K == RowwiseQuantizeLargePTLConfig::Columns) {
            quantize_int8_rowwise_large_ptl_kernel<
                bf16, RowwiseQuantizeLargePTLConfig>(
                reinterpret_cast<const bf16*>(x.data_ptr()),
                reinterpret_cast<int8_t*>(output.data_ptr()),
                scales.data_ptr<float>(), M, x.device());
        } else if (M == RowwiseQuantizeKrea2PTLConfig::Rows &&
                   K == RowwiseQuantizeKrea2PTLConfig::Columns) {
            quantize_int8_rowwise_large_ptl_kernel<
                bf16, RowwiseQuantizeKrea2PTLConfig>(
                reinterpret_cast<const bf16*>(x.data_ptr()),
                reinterpret_cast<int8_t*>(output.data_ptr()),
                scales.data_ptr<float>(), M, x.device());
        } else if (M == RowwiseQuantizeKrea2FFNDownPTLConfig::Rows &&
                   K == RowwiseQuantizeKrea2FFNDownPTLConfig::Columns) {
            quantize_int8_rowwise_large_ptl_kernel<
                bf16, RowwiseQuantizeKrea2FFNDownPTLConfig>(
                reinterpret_cast<const bf16*>(x.data_ptr()),
                reinterpret_cast<int8_t*>(output.data_ptr()),
                scales.data_ptr<float>(), M, x.device());
        } else {
#elif defined(OMNI_XPU_ARCH_BMG)
        if ((M == RowwiseQuantizeZImageHiddenBMGConfig::ImageRows ||
             M == RowwiseQuantizeZImageHiddenBMGConfig::JointRows) &&
            K == RowwiseQuantizeZImageHiddenBMGConfig::Columns) {
            quantize_int8_rowwise_large_ptl_kernel<
                bf16, RowwiseQuantizeZImageHiddenBMGConfig>(
                reinterpret_cast<const bf16*>(x.data_ptr()),
                reinterpret_cast<int8_t*>(output.data_ptr()),
                scales.data_ptr<float>(), M, x.device(),
                "quantize_int8_rowwise_large_bmg");
        } else if ((M == RowwiseQuantizeZImageBMGConfig::ImageRows ||
             M == RowwiseQuantizeZImageBMGConfig::JointRows) &&
            K == RowwiseQuantizeZImageBMGConfig::Columns) {
            quantize_int8_rowwise_large_ptl_kernel<
                bf16, RowwiseQuantizeZImageBMGConfig>(
                reinterpret_cast<const bf16*>(x.data_ptr()),
                reinterpret_cast<int8_t*>(output.data_ptr()),
                scales.data_ptr<float>(), M, x.device(),
                "quantize_int8_rowwise_large_bmg");
        } else if (M == RowwiseQuantizeKrea2BMGConfig::Rows &&
                   K == RowwiseQuantizeKrea2BMGConfig::Columns) {
            quantize_int8_rowwise_large_ptl_kernel<
                bf16, RowwiseQuantizeKrea2BMGConfig>(
                reinterpret_cast<const bf16*>(x.data_ptr()),
                reinterpret_cast<int8_t*>(output.data_ptr()),
                scales.data_ptr<float>(), M, x.device(),
                "quantize_int8_rowwise_large_bmg");
        } else if (M == RowwiseQuantizeKrea2FFNDownBMGConfig::Rows &&
                   K == RowwiseQuantizeKrea2FFNDownBMGConfig::Columns) {
            quantize_int8_rowwise_large_ptl_kernel<
                bf16, RowwiseQuantizeKrea2FFNDownBMGConfig>(
                reinterpret_cast<const bf16*>(x.data_ptr()),
                reinterpret_cast<int8_t*>(output.data_ptr()),
                scales.data_ptr<float>(), M, x.device(),
                "quantize_int8_rowwise_large_bmg");
        } else {
#endif
        quantize_int8_rowwise_kernel<bf16>(
            reinterpret_cast<const bf16*>(x.data_ptr()),
            reinterpret_cast<int8_t*>(output.data_ptr()),
            scales.data_ptr<float>(),
            M, K, x.device()
        );
#if defined(OMNI_XPU_ARCH_PTL_H) || defined(OMNI_XPU_ARCH_BMG)
        }
#endif
    } else if (x.scalar_type() == torch::kHalf) {
#if defined(OMNI_XPU_ARCH_PTL_H)
        if ((M == RowwiseQuantizeBooguHiddenPTLConfig::ImageRows ||
             M == RowwiseQuantizeBooguHiddenPTLConfig::JointRows) &&
            K == RowwiseQuantizeBooguHiddenPTLConfig::Columns) {
            quantize_int8_rowwise_large_ptl_kernel<
                fp16, RowwiseQuantizeBooguHiddenPTLConfig>(
                reinterpret_cast<const fp16*>(x.data_ptr()),
                reinterpret_cast<int8_t*>(output.data_ptr()),
                scales.data_ptr<float>(), M, x.device());
        } else if ((M == RowwiseQuantizeBooguFFNDownPTLConfig::ImageRows ||
                    M == RowwiseQuantizeBooguFFNDownPTLConfig::JointRows) &&
                   K == RowwiseQuantizeBooguFFNDownPTLConfig::Columns) {
            quantize_int8_rowwise_large_ptl_kernel<
                fp16, RowwiseQuantizeBooguFFNDownPTLConfig>(
                reinterpret_cast<const fp16*>(x.data_ptr()),
                reinterpret_cast<int8_t*>(output.data_ptr()),
                scales.data_ptr<float>(), M, x.device());
        } else {
#elif defined(OMNI_XPU_ARCH_BMG)
        if ((M == RowwiseQuantizeBooguHiddenBMGConfig::ImageRows ||
             M == RowwiseQuantizeBooguHiddenBMGConfig::JointRows) &&
            K == RowwiseQuantizeBooguHiddenBMGConfig::Columns) {
            quantize_int8_rowwise_large_ptl_kernel<
                fp16, RowwiseQuantizeBooguHiddenBMGConfig>(
                reinterpret_cast<const fp16*>(x.data_ptr()),
                reinterpret_cast<int8_t*>(output.data_ptr()),
                scales.data_ptr<float>(), M, x.device(),
                "quantize_int8_rowwise_large_bmg");
        } else if (
            (M == RowwiseQuantizeBooguFFNDownBMGConfig::ImageRows ||
             M == RowwiseQuantizeBooguFFNDownBMGConfig::JointRows) &&
            K == RowwiseQuantizeBooguFFNDownBMGConfig::Columns) {
            quantize_int8_rowwise_large_ptl_kernel<
                fp16, RowwiseQuantizeBooguFFNDownBMGConfig>(
                reinterpret_cast<const fp16*>(x.data_ptr()),
                reinterpret_cast<int8_t*>(output.data_ptr()),
                scales.data_ptr<float>(), M, x.device(),
                "quantize_int8_rowwise_large_bmg");
        } else {
#endif
        quantize_int8_rowwise_kernel<fp16>(
            reinterpret_cast<const fp16*>(x.data_ptr()),
            reinterpret_cast<int8_t*>(output.data_ptr()),
            scales.data_ptr<float>(),
            M, K, x.device()
        );
#if defined(OMNI_XPU_ARCH_PTL_H) || defined(OMNI_XPU_ARCH_BMG)
        }
#endif
    } else {
        quantize_int8_rowwise_kernel<float>(
            x.data_ptr<float>(),
            reinterpret_cast<int8_t*>(output.data_ptr()),
            scales.data_ptr<float>(),
            M, K, x.device()
        );
    }

    // Reshape scales to [..., 1] to match PyTorch convention
    auto scale_shape = x.sizes().vec();
    scale_shape.back() = 1;
    return {output.reshape(x.sizes()), scales.reshape(scale_shape)};
}

std::tuple<torch::Tensor, torch::Tensor> fused_silu_mul_quantize_rowwise(
    torch::Tensor x1,
    torch::Tensor x2
) {
    TORCH_CHECK(x1.device().is_xpu(), "x1 must be on XPU device");
    TORCH_CHECK(x2.device().is_xpu(), "x2 must be on XPU device");
    TORCH_CHECK(x1.device() == x2.device(),
        "x1 and x2 must be on the same XPU device");
    TORCH_CHECK(x1.dim() >= 2 && x2.dim() >= 2,
        "x1 and x2 must be at least 2D");
    TORCH_CHECK(x1.sizes() == x2.sizes(),
        "x1 and x2 must have identical shapes, got ",
        x1.sizes(), " and ", x2.sizes());
    TORCH_CHECK(x1.scalar_type() == x2.scalar_type(),
        "x1 and x2 must have identical dtypes");
    TORCH_CHECK(
        x1.scalar_type() == torch::kBFloat16 ||
        x1.scalar_type() == torch::kHalf,
        "x1 and x2 must be bf16 or f16, got ", x1.scalar_type());

    x1 = x1.contiguous();
    x2 = x2.contiguous();
    TORCH_CHECK(x1.size(-1) > 0,
        "x1 and x2 last dimension must be non-empty");
    const int64_t M = x1.numel() / x1.size(-1);
    const int64_t K = x1.size(-1);

    auto output = torch::empty_like(
        x1, torch::TensorOptions().dtype(torch::kInt8));
    auto scales = torch::empty(
        {M}, torch::TensorOptions().dtype(torch::kFloat32).device(x1.device()));

    if (x1.scalar_type() == torch::kBFloat16) {
        fused_silu_mul_quantize_rowwise_kernel<bf16>(
            reinterpret_cast<const bf16*>(x1.data_ptr()),
            reinterpret_cast<const bf16*>(x2.data_ptr()),
            reinterpret_cast<int8_t*>(output.data_ptr()),
            scales.data_ptr<float>(), M, K, x1.device());
    } else {
        fused_silu_mul_quantize_rowwise_kernel<fp16>(
            reinterpret_cast<const fp16*>(x1.data_ptr()),
            reinterpret_cast<const fp16*>(x2.data_ptr()),
            reinterpret_cast<int8_t*>(output.data_ptr()),
            scales.data_ptr<float>(), M, K, x1.device());
    }

    auto scale_shape = x1.sizes().vec();
    scale_shape.back() = 1;
    return {output.reshape(x1.sizes()), scales.reshape(scale_shape)};
}

#if defined(OMNI_XPU_ARCH_BMG)
namespace {

template<int GroupsPerDPAS, int WorkitemsPerRow>
class QuantizeInt8ConvRotG16BMGKernel;

inline bool convrot_h16_negative(int inner, int output) {
    const bool high_negative = inner / 4 + output / 4 == 3;
    const bool low_negative = inner % 4 + output % 4 == 3;
    return high_negative != low_negative;
}

}  // namespace

template<int GroupsPerDPAS, int WorkitemsPerRow>
static std::tuple<torch::Tensor, torch::Tensor>
quantize_int8_convrot_g16_bmg_policy(
    torch::Tensor input
) {
    TORCH_CHECK(input.device().is_xpu(), "input must be on XPU");
    TORCH_CHECK(input.dim() >= 1, "input must have at least one dimension");
    TORCH_CHECK(input.scalar_type() == torch::kHalf, "input must be FP16");
    TORCH_CHECK(input.size(-1) == 3360, "input must have K=3360");

    input = input.contiguous();
    const int64_t columns = input.size(-1);
    const int64_t rows = input.numel() / columns;
    TORCH_CHECK(
        rows == 109 || rows == 110 || rows == 4096 ||
            rows == 4205 || rows == 4206,
        "BMG G16 ConvRot quantization requires Boogu rows "
        "109, 110, 4096, 4205, or 4206");

    auto output = torch::empty_like(
        input, input.options().dtype(torch::kInt8));
    auto scales = torch::empty(
        {rows}, input.options().dtype(torch::kFloat32));
    const auto* source =
        reinterpret_cast<const fp16*>(input.data_ptr());
    auto* destination = output.data_ptr<int8_t>();
    auto* scale_output = scales.data_ptr<float>();

    constexpr int GroupSize = 16;
    constexpr int MaximaSlots = 32;
    constexpr int CachedGroups = WorkitemsPerRow * GroupsPerDPAS;
    constexpr int RotatedBytes =
        CachedGroups * GroupSize * sizeof(fp16);
    constexpr int MaximaOffset = RotatedBytes;
    constexpr int ScaleOffset =
        MaximaOffset + MaximaSlots * sizeof(float);
    constexpr int SLMBytes = ScaleOffset + 16;
    const int64_t groups_per_row = columns / GroupSize;
    const sycl::range<1> local(WorkitemsPerRow);
    const sycl::range<1> global(
        static_cast<size_t>(rows) * WorkitemsPerRow);

    auto command_group = [&](sycl::handler& handler) {
        handler.parallel_for<
            QuantizeInt8ConvRotG16BMGKernel<
                GroupsPerDPAS,
                WorkitemsPerRow>>(
            sycl::nd_range<1>(global, local),
            [=](sycl::nd_item<1> item) [[intel::sycl_explicit_simd]] {
                esimd::slm_init<SLMBytes>();
                const int workitem =
                    static_cast<int>(item.get_local_linear_id());
                const int64_t row =
                    static_cast<int64_t>(item.get_group(0));
                const int64_t first_group =
                    static_cast<int64_t>(workitem) * GroupsPerDPAS;
                const int64_t remaining_groups =
                    groups_per_row - first_group;
                const int valid_groups = remaining_groups <= 0
                    ? 0
                    : (
                          remaining_groups < GroupsPerDPAS
                              ? static_cast<int>(remaining_groups)
                              : GroupsPerDPAS);

                esimd::simd<fp16, 256> matrix_b;
#pragma unroll
                for (int inner_pair = 0; inner_pair < 8; ++inner_pair) {
#pragma unroll
                    for (int column = 0; column < 16; ++column) {
#pragma unroll
                        for (int pair_element = 0; pair_element < 2;
                             ++pair_element) {
                            const int inner =
                                inner_pair * 2 + pair_element;
                            const int packed =
                                inner_pair * 32 + column * 2 + pair_element;
                            matrix_b[packed] = static_cast<fp16>(
                                convrot_h16_negative(inner, column)
                                    ? -0.25f
                                    : 0.25f);
                        }
                    }
                }

                esimd::simd<fp16, GroupsPerDPAS * GroupSize> matrix_a =
                    0.0f;
                const fp16* row_input = source + row * columns;
                if (valid_groups == GroupsPerDPAS) {
                    matrix_a.copy_from(
                        row_input + first_group * GroupSize);
                } else {
#pragma unroll
                    for (int group = 0; group < GroupsPerDPAS; ++group) {
                        if (group < valid_groups) {
#pragma unroll
                            for (int inner = 0; inner < GroupSize; ++inner) {
                                matrix_a[group * GroupSize + inner] =
                                    row_input[
                                        (first_group + group) * GroupSize +
                                        inner];
                            }
                        }
                    }
                }

                const auto accumulated =
                    xmx::dpas<8, GroupsPerDPAS, float>(
                        matrix_b, matrix_a);
                // Match the public rotate_convrot materialization boundary:
                // oneDNN writes FP16 before rowwise quantization consumes it.
                const esimd::simd<
                    fp16,
                    GroupsPerDPAS * GroupSize> rounded = accumulated;
                const uint32_t cache_offset =
                    static_cast<uint32_t>(
                        workitem *
                        GroupsPerDPAS *
                        GroupSize *
                        sizeof(fp16));
                esimd::slm_block_store<fp16, GroupsPerDPAS * GroupSize>(
                    cache_offset, rounded);
                const esimd::simd<
                    float,
                    GroupsPerDPAS * GroupSize> rounded_fp32 = rounded;
                const float local_max = esimd::hmax<float>(
                    esimd::abs(rounded_fp32));
                esimd::slm_block_store<float, 1>(
                    MaximaOffset + workitem * sizeof(float),
                    esimd::simd<float, 1>(local_max),
                    esimd::overaligned<4>);
                esimd::barrier();

                // Every work-item reads the same maxima vector and executes
                // the same reduction order. Replicating this small reduction
                // avoids publishing inverse_scale through SLM and a second
                // barrier. Only work-item 0 writes the public scale tensor.
                const auto maxima =
                    esimd::slm_block_load<float, MaximaSlots>(
                        MaximaOffset);
                auto valid_maxima = maxima;
#pragma unroll
                for (int slot = WorkitemsPerRow;
                     slot < MaximaSlots;
                     ++slot) {
                    valid_maxima[slot] = 0.0f;
                }
                const float row_max =
                    esimd::hmax<float>(valid_maxima);
                float scale = row_max / 127.0f;
                if (scale < 1e-30f) scale = 1e-30f;
                if (workitem == 0) {
                    scale_output[row] = scale;
                }
                const float inverse_scale = 1.0f / scale;
                const auto cached =
                    esimd::slm_block_load<
                        fp16,
                        GroupsPerDPAS * GroupSize>(cache_offset);
                const esimd::simd<
                    float,
                    GroupsPerDPAS * GroupSize> cached_fp32 = cached;
                auto quantized_fp32 =
                    esimd::rnde<float>(cached_fp32 * inverse_scale);
#pragma unroll
                for (int element = 0;
                     element < GroupsPerDPAS * GroupSize;
                     ++element) {
                    quantized_fp32[element] =
                        quantized_fp32[element] < -128.0f
                            ? -128.0f
                            : quantized_fp32[element];
                    quantized_fp32[element] =
                        quantized_fp32[element] > 127.0f
                            ? 127.0f
                            : quantized_fp32[element];
                }
                esimd::simd<
                    int8_t,
                    GroupsPerDPAS * GroupSize> quantized =
                    quantized_fp32;
                int8_t* row_output = destination + row * columns;
                if (valid_groups == GroupsPerDPAS) {
                    quantized.copy_to(
                        row_output + first_group * GroupSize);
                } else {
#pragma unroll
                    for (int group = 0; group < GroupsPerDPAS; ++group) {
                        if (group < valid_groups) {
                            const esimd::simd<int8_t, GroupSize>
                                group_quantized = quantized
                                    .template select<GroupSize, 1>(
                                        group * GroupSize);
                            group_quantized.copy_to(
                                row_output +
                                (first_group + group) * GroupSize);
                        }
                    }
                }
            });
    };
    utils::submit_kernel(
        command_group,
        input.device(),
        "quantize_int8_convrot_g16_bmg_dpas");

    auto scale_shape = input.sizes().vec();
    scale_shape.back() = 1;
    return {output, scales.reshape(scale_shape)};
}

std::tuple<torch::Tensor, torch::Tensor> quantize_int8_convrot_g16_bmg(
    torch::Tensor input
) {
    TORCH_CHECK(input.device().is_xpu(), "input must be on XPU");
    auto& queue = utils::get_queue(input.device());
    if (device::use_b60_kernel_profile(queue)) {
        return quantize_int8_convrot_g16_bmg_policy<
            device::B60KernelPolicy::convrot_g16_groups_per_dpas,
            device::B60KernelPolicy::convrot_g16_work_items_per_row>(
                input);
    }
    return quantize_int8_convrot_g16_bmg_policy<
        device::B70KernelPolicy::convrot_g16_groups_per_dpas,
        device::B70KernelPolicy::convrot_g16_work_items_per_row>(
            input);
}
#endif

}  // namespace int8_ops
}  // namespace omni_xpu
