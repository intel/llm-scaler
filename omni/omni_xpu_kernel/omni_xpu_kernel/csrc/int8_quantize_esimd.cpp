// ============================================================================
// ESIMD Fused INT8 Quantization Kernels
// ============================================================================
// High-performance single-pass quantization kernels that fuse:
//   absmax reduction + scale computation + divide + round + clamp + cast
// into minimal kernel launches, eliminating Python-level multi-op overhead.
//
// Kernels:
//   1. quantize_int8_rowwise_esimd: Per-row quantization for activations
//      Input:  [M, K] bf16/f16
//      Output: [M, K] int8 + [M, 1] float32 scales
//
//   2. quantize_int8_tensorwise_esimd: Single-scale tensor quantization
//      Input:  [M, K] bf16/f16
//      Output: [M, K] int8 + scalar float32 scale
//
// Design: Each work-item processes one row (K elements). For typical ComfyUI
// shapes (K=4096), this means each WI reads 4096 bf16 values (8KB), computes
// absmax via ESIMD SIMD reduction, then quantizes in vectorized chunks.
// ============================================================================

#include <torch/extension.h>
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

#include "utils.h"

using fp16 = sycl::half;
using bf16 = sycl::ext::oneapi::bfloat16;
using namespace sycl::ext::intel::esimd;

namespace omni_xpu {
namespace int8_ops {

// ============================================================================
// Kernel: Fused per-row INT8 quantization
// ============================================================================
// Each work-item processes one full row of K elements:
//   1. Compute absmax across the row (vectorized reduce)
//   2. Compute scale = absmax / 127
//   3. Quantize: round(x / scale).clamp(-128, 127) → int8
//
// For K ≤ 8192, single WI per row is efficient (memory-bound, good cache line use).
// For larger K, could split into multi-WI per row with sub-group reduction.
// ============================================================================

template <typename InputT>
void quantize_int8_rowwise_esimd_kernel(
    const InputT* __restrict__ input,  // [M, K]
    int8_t* __restrict__ output,       // [M, K]
    float* __restrict__ scales,        // [M]
    int64_t M,
    int64_t K,
    const at::Device& device
) {
    // Subgroup-cooperative rowwise INT8 quantization (plain SYCL).
    //
    // One sub-group (SG lanes) processes one row of K elements. Adjacent lanes
    // read consecutive elements (lane i reads elements i, i+SG, i+2*SG, ...),
    // giving fully coalesced HBM access. Row absmax is reduced via a sub-group
    // collective (no SLM, no barrier). Pass 2 re-reads the row (served from L2
    // for typical K) and writes int8.
    //
    // This replaces the previous ESIMD SLM kernel, which suffered a large IGC
    // JIT register-spill penalty inside the multi-kernel _C module (measured
    // ~38 GB/s vs ~140 GB/s for this design at M=4128, K=3840).
    //
    // HBM traffic: K*2 read (pass1) + K*2 read (pass2, L2-cached) + K*1 write.
    constexpr int SG = 32;            // sub-group size (lanes per row)
    constexpr int ROWS_PER_WG = 8;    // rows (sub-groups) per work-group
    constexpr int WG = SG * ROWS_PER_WG;

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

                // Pass 1: coalesced absmax over the row.
                float local_max = 0.0f;
                for (int64_t k = lane; k < K; k += SG) {
                    float v = static_cast<float>(row_ptr[k]);
                    local_max = sycl::fmax(local_max, sycl::fabs(v));
                }
                float row_max =
                    sycl::reduce_over_group(sg, local_max, sycl::maximum<float>());

                float scale = row_max / 127.0f;
                if (scale < 1e-30f) scale = 1e-30f;
                float inv_scale = 1.0f / scale;
                if (lane == 0) scales[row] = scale;

                // Pass 2: coalesced quantize + write (round-to-nearest-even).
                for (int64_t k = lane; k < K; k += SG) {
                    float v = static_cast<float>(row_ptr[k]) * inv_scale;
                    float r = sycl::rint(v);
                    r = sycl::fmax(-128.0f, sycl::fmin(127.0f, r));
                    out_ptr[k] = static_cast<int8_t>(static_cast<int32_t>(r));
                }
            }
        );
    };
    utils::submit_kernel(cgf, device, "quantize_int8_rowwise_sg2pass");
}

// ============================================================================
// Public C++ API for ESIMD quantization
// ============================================================================

std::tuple<torch::Tensor, torch::Tensor> quantize_int8_rowwise_fused(
    torch::Tensor x
) {
    TORCH_CHECK(x.device().is_xpu(), "x must be on XPU device");
    TORCH_CHECK(x.dim() >= 2, "x must be at least 2D");
    TORCH_CHECK(
        x.scalar_type() == torch::kBFloat16 || x.scalar_type() == torch::kHalf,
        "x must be bf16 or f16, got ", x.scalar_type()
    );

    x = x.contiguous();
    const int64_t M = x.numel() / x.size(-1);
    const int64_t K = x.size(-1);

    torch::Tensor output = torch::empty_like(x, torch::TensorOptions().dtype(torch::kInt8));
    torch::Tensor scales = torch::empty({M}, torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));

    if (x.scalar_type() == torch::kBFloat16) {
        quantize_int8_rowwise_esimd_kernel<bf16>(
            reinterpret_cast<const bf16*>(x.data_ptr()),
            reinterpret_cast<int8_t*>(output.data_ptr()),
            scales.data_ptr<float>(),
            M, K, x.device()
        );
    } else {
        quantize_int8_rowwise_esimd_kernel<fp16>(
            reinterpret_cast<const fp16*>(x.data_ptr()),
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

}  // namespace int8_ops
}  // namespace omni_xpu
