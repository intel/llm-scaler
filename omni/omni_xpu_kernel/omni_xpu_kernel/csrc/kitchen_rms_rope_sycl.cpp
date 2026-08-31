#include <torch/extension.h>
#include <sycl/sycl.hpp>

#include <cstdint>
#include <type_traits>

#include "device_utils.h"
#include "kernel_tuning_overrides.h"
#include "utils.h"

using fp16 = sycl::half;
using bf16 = sycl::ext::oneapi::bfloat16;

namespace omni_xpu {
namespace rotary {
namespace {

constexpr int64_t RMS_ROPE_WG = 64;

#if defined(OMNI_XPU_ARCH_BMG)
static_assert(
    OMNI_H3_RMS_ROPE_FAST_REDUCE == 0 ||
        OMNI_H3_RMS_ROPE_FAST_REDUCE == 1,
    "OMNI_H3_RMS_ROPE_FAST_REDUCE must be zero or one");
static_assert(
    OMNI_H3_RMS_ROPE_SLM_BF16 == 0 ||
        OMNI_H3_RMS_ROPE_SLM_BF16 == 1,
    "OMNI_H3_RMS_ROPE_SLM_BF16 must be zero or one");
constexpr int64_t H3_HEADS = 56;
constexpr int64_t H3_HEAD_DIM = 128;
constexpr int64_t H3_ROT_DIM = 96;
constexpr int64_t H3_ROTARY_PAIRS = H3_ROT_DIM / 2;
constexpr int64_t H3_PACKED_SEQUENCE_STRIDE =
    3 * H3_HEADS * H3_HEAD_DIM;
using H3PairCacheT = std::conditional_t<
    OMNI_H3_RMS_ROPE_SLM_BF16,
    bf16,
    float>;
#endif

struct Tensor4Meta {
    int64_t n0;
    int64_t n1;
    int64_t n2;
    int64_t dim;
    int64_t stride0;
    int64_t stride1;
    int64_t stride2;
    int64_t stride3;
    int64_t out_stride0;
    int64_t out_stride1;
    int64_t out_stride2;
    int64_t out_stride3;
    int64_t rows;
};

struct Freq6Meta {
    int64_t n0;
    int64_t n1;
    int64_t n2;
    int64_t pairs;
    int64_t stride0;
    int64_t stride1;
    int64_t stride2;
    int64_t stride3;
    int64_t stride4;
    int64_t stride5;
};

Tensor4Meta tensor_meta(
    const torch::Tensor& input,
    const torch::Tensor& output) {
    return {
        input.size(0),
        input.size(1),
        input.size(2),
        input.size(3),
        input.stride(0),
        input.stride(1),
        input.stride(2),
        input.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        input.size(0) * input.size(1) * input.size(2),
    };
}

Freq6Meta freq_meta(const torch::Tensor& freqs) {
    return {
        freqs.size(0),
        freqs.size(1),
        freqs.size(2),
        freqs.size(3),
        freqs.stride(0),
        freqs.stride(1),
        freqs.stride(2),
        freqs.stride(3),
        freqs.stride(4),
        freqs.stride(5),
    };
}

bool supported_float(torch::ScalarType dtype) {
    return dtype == torch::kFloat32 || dtype == torch::kFloat16 ||
           dtype == torch::kBFloat16;
}

void check_tensor(const torch::Tensor& x, const char* name) {
    TORCH_CHECK(x.device().is_xpu(), name, " must be on XPU");
    TORCH_CHECK(x.dim() == 4, name, " must be a four-dimensional tensor");
    TORCH_CHECK(supported_float(x.scalar_type()), name, " has an unsupported dtype");
    TORCH_CHECK(x.size(3) > 0 && x.size(3) % 2 == 0,
                name, " last dimension must be positive and even");
}

void check_scale(
    const torch::Tensor& scale,
    const torch::Tensor& x,
    const char* name) {
    TORCH_CHECK(scale.device().is_xpu(), name, " must be on XPU");
    TORCH_CHECK(scale.device() == x.device(), name, " must be on the input XPU");
    TORCH_CHECK(scale.dim() == 1 && scale.numel() == x.size(3),
                name, " must have one element per input feature");
    TORCH_CHECK(supported_float(scale.scalar_type()), name, " has an unsupported dtype");
}

void check_freqs(
    const torch::Tensor& x,
    const torch::Tensor& freqs,
    int64_t rot_dim) {
    TORCH_CHECK(freqs.device().is_xpu(), "freqs_cis must be on XPU");
    TORCH_CHECK(freqs.device() == x.device(),
                "freqs_cis must be on the input XPU");
    TORCH_CHECK(supported_float(freqs.scalar_type()),
                "freqs_cis has an unsupported dtype");
    TORCH_CHECK(freqs.dim() == 6 && freqs.size(4) == 2 && freqs.size(5) == 2,
                "freqs_cis must be six-dimensional and end in a 2x2 transform");
    TORCH_CHECK(rot_dim > 0 && rot_dim <= x.size(3) && rot_dim % 2 == 0,
                "rot_dim must be positive, even, and no larger than the head dimension");
    TORCH_CHECK(freqs.size(0) == 1 || freqs.size(0) == x.size(0),
                "freqs_cis dimension 0 is not broadcastable to the input");
    TORCH_CHECK(freqs.size(1) == 1 || freqs.size(1) == x.size(1),
                "freqs_cis dimension 1 is not broadcastable to the input");
    TORCH_CHECK(freqs.size(2) == 1 || freqs.size(2) >= x.size(2),
                "freqs_cis dimension 2 is shorter than the input");
    TORCH_CHECK(freqs.size(3) == 1 || freqs.size(3) >= rot_dim / 2,
                "freqs_cis does not contain enough rotary pairs");
}

#if defined(OMNI_XPU_ARCH_BMG)
bool is_minimax_h3_rms_rope_contract(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& freqs,
    const torch::Tensor& q_scale,
    const torch::Tensor& k_scale,
    int64_t rot_dim,
    bool split_half,
    bool inplace) {
    if (!split_half || !inplace ||
        q.scalar_type() != torch::kBFloat16 ||
        k.scalar_type() != torch::kBFloat16 ||
        freqs.scalar_type() != torch::kBFloat16 ||
        q_scale.scalar_type() != torch::kBFloat16 ||
        k_scale.scalar_type() != torch::kBFloat16 ||
        q.sizes() != k.sizes() || q.size(0) != 1 || q.size(1) < 1 ||
        q.size(2) != H3_HEADS || q.size(3) != H3_HEAD_DIM ||
        q.stride(0) <= 0 || k.stride(0) <= 0 ||
        q.stride(1) != H3_PACKED_SEQUENCE_STRIDE ||
        k.stride(1) != H3_PACKED_SEQUENCE_STRIDE ||
        q.stride(2) != H3_HEAD_DIM || k.stride(2) != H3_HEAD_DIM ||
        q.stride(3) != 1 || k.stride(3) != 1 ||
        rot_dim != H3_ROT_DIM ||
        !q_scale.is_contiguous() || !k_scale.is_contiguous()) {
        return false;
    }
    return freqs.is_contiguous() && freqs.size(0) == 1 &&
        freqs.size(1) == q.size(1) && freqs.size(2) == 1 &&
        freqs.size(3) == H3_ROTARY_PAIRS && freqs.size(4) == 2 &&
        freqs.size(5) == 2;
}

void launch_minimax_h3_rms_rope(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& freqs,
    const torch::Tensor& q_scale,
    const torch::Tensor& k_scale,
    float epsilon) {
    const auto* q_ptr = reinterpret_cast<const bf16*>(q.data_ptr());
    const auto* k_ptr = reinterpret_cast<const bf16*>(k.data_ptr());
    const auto* f_ptr = reinterpret_cast<const bf16*>(freqs.data_ptr());
    const auto* qs_ptr = reinterpret_cast<const bf16*>(q_scale.data_ptr());
    const auto* ks_ptr = reinterpret_cast<const bf16*>(k_scale.data_ptr());
    auto* qo_ptr = reinterpret_cast<bf16*>(q.data_ptr());
    auto* ko_ptr = reinterpret_cast<bf16*>(k.data_ptr());
    const int64_t rows_per_operand = q.size(1) * H3_HEADS;
    const int64_t total_rows = 2 * rows_per_operand;

    auto cgf = [&](sycl::handler& handler) {
        sycl::local_accessor<float, 1> partial(
            sycl::range<1>(RMS_ROPE_WG), handler);
        sycl::local_accessor<H3PairCacheT, 1> pair_second(
            sycl::range<1>(H3_ROTARY_PAIRS), handler);
        handler.parallel_for(
            sycl::nd_range<1>(
                sycl::range<1>(
                    static_cast<size_t>(total_rows * RMS_ROPE_WG)),
                sycl::range<1>(RMS_ROPE_WG)),
            [=](sycl::nd_item<1> item)
#if OMNI_H3_RMS_ROPE_FAST_REDUCE
                [[sycl::reqd_sub_group_size(16)]]
#endif
            {
                const int64_t local = item.get_local_id(0);
                const int64_t global_row = item.get_group(0);
                const bool is_key = global_row >= rows_per_operand;
                const int64_t row = is_key
                    ? global_row - rows_per_operand
                    : global_row;
                const int64_t token = row / H3_HEADS;
                const int64_t head = row - token * H3_HEADS;
                const int64_t base =
                    token * H3_PACKED_SEQUENCE_STRIDE + head * H3_HEAD_DIM;
                const bf16* source = is_key ? k_ptr : q_ptr;
                bf16* destination = is_key ? ko_ptr : qo_ptr;
                const bf16* scale = is_key ? ks_ptr : qs_ptr;

                // Keep the generic path's exact reduction partition while
                // retaining both input values in registers. Only the 48
                // split-half partners cross work-items through SLM, so the
                // normalization/rotation pass does not reread Q/K globally.
                const float value0 = static_cast<float>(source[base + local]);
                const float value1 = static_cast<float>(
                    source[base + local + RMS_ROPE_WG]);
                float square_sum = value0 * value0;
                square_sum += value1 * value1;
                partial[local] = square_sum;
                if (local >= H3_ROTARY_PAIRS) {
                    pair_second[local - H3_ROTARY_PAIRS] = value0;
                } else if (local < 32) {
                    pair_second[16 + local] = value1;
                }
                item.barrier(sycl::access::fence_space::local_space);

#if OMNI_H3_RMS_ROPE_FAST_REDUCE
                if (local < 32) {
                    partial[local] += partial[local + 32];
                }
                item.barrier(sycl::access::fence_space::local_space);
                if (local < 16) {
                    float reduced = partial[local] + partial[local + 16];
                    const auto subgroup = item.get_sub_group();
                    const uint32_t lane = subgroup.get_local_linear_id();
#pragma unroll
                    for (uint32_t width = 8; width > 0; width /= 2) {
                        const uint32_t remote =
                            lane < width ? lane + width : lane;
                        const float partner = sycl::select_from_group(
                            subgroup, reduced, remote);
                        if (lane < width) {
                            reduced += partner;
                        }
                    }
                    if (lane == 0) {
                        partial[0] = reduced;
                    }
                }
                item.barrier(sycl::access::fence_space::local_space);
#else
                for (int64_t width = RMS_ROPE_WG / 2; width > 0;
                     width /= 2) {
                    if (local < width) {
                        partial[local] += partial[local + width];
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }
#endif
                const float inverse_rms =
                    sycl::rsqrt(partial[0] / H3_HEAD_DIM + epsilon);

                if (local < H3_ROTARY_PAIRS) {
                    const int64_t col0 = local;
                    const int64_t col1 = H3_ROTARY_PAIRS + local;
                    const bf16 normalized0 = static_cast<bf16>(
                        value0 * inverse_rms *
                        static_cast<float>(scale[col0]));
                    const bf16 normalized1 = static_cast<bf16>(
                        static_cast<float>(pair_second[local]) * inverse_rms *
                        static_cast<float>(scale[col1]));
                    const float rotated0 = static_cast<float>(normalized0);
                    const float rotated1 = static_cast<float>(normalized1);
                    const int64_t freq_base =
                        token * H3_ROTARY_PAIRS * 4 + local * 4;
                    const float f00 = static_cast<float>(f_ptr[freq_base]);
                    const float f01 = static_cast<float>(f_ptr[freq_base + 1]);
                    const float f10 = static_cast<float>(f_ptr[freq_base + 2]);
                    const float f11 = static_cast<float>(f_ptr[freq_base + 3]);
                    destination[base + col0] = static_cast<bf16>(
                        f00 * rotated0 + f01 * rotated1);
                    destination[base + col1] = static_cast<bf16>(
                        f10 * rotated0 + f11 * rotated1);
                }

                if (local >= H3_ROT_DIM - RMS_ROPE_WG) {
                    const int64_t col = RMS_ROPE_WG + local;
                    destination[base + col] = static_cast<bf16>(
                        value1 * inverse_rms *
                        static_cast<float>(scale[col]));
                }
            });
    };
    utils::submit_kernel(
        cgf, q.device(), "kitchen_rms_rope_h3_cached_input_bmg");
}

void launch_minimax_h3_rms_rope_b580(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& freqs,
    const torch::Tensor& q_scale,
    const torch::Tensor& k_scale,
    float epsilon) {
    constexpr int64_t HeadsPerGroup = 7;
    constexpr int64_t HeadBlocks = H3_HEADS / HeadsPerGroup;
    static_assert(H3_HEADS % HeadsPerGroup == 0);
    const auto* q_ptr = reinterpret_cast<const bf16*>(q.data_ptr());
    const auto* k_ptr = reinterpret_cast<const bf16*>(k.data_ptr());
    const auto* f_ptr = reinterpret_cast<const bf16*>(freqs.data_ptr());
    const auto* qs_ptr = reinterpret_cast<const bf16*>(q_scale.data_ptr());
    const auto* ks_ptr = reinterpret_cast<const bf16*>(k_scale.data_ptr());
    auto* qo_ptr = reinterpret_cast<bf16*>(q.data_ptr());
    auto* ko_ptr = reinterpret_cast<bf16*>(k.data_ptr());
    const int64_t total_groups = q.size(1) * HeadBlocks;

    auto cgf = [&](sycl::handler& handler) {
        sycl::local_accessor<float, 1> partial(
            sycl::range<1>(RMS_ROPE_WG), handler);
        sycl::local_accessor<float, 1> pair_second(
            sycl::range<1>(H3_ROTARY_PAIRS), handler);
        handler.parallel_for(
            sycl::nd_range<1>(
                sycl::range<1>(
                    static_cast<size_t>(total_groups * RMS_ROPE_WG)),
                sycl::range<1>(RMS_ROPE_WG)),
            [=](sycl::nd_item<1> item) {
                const int64_t local = item.get_local_id(0);
                const int64_t group = item.get_group(0);
                const int64_t token = group / HeadBlocks;
                const int64_t head_begin =
                    (group - token * HeadBlocks) * HeadsPerGroup;

                // Frequencies are identical across all 56 heads and both Q/K
                // operands for one token. Keep one copy per work-item while
                // processing a seven-head block and both operands.
                float f00 = 0.0f;
                float f01 = 0.0f;
                float f10 = 0.0f;
                float f11 = 0.0f;
                if (local < H3_ROTARY_PAIRS) {
                    const int64_t freq_base =
                        token * H3_ROTARY_PAIRS * 4 + local * 4;
                    f00 = static_cast<float>(f_ptr[freq_base]);
                    f01 = static_cast<float>(f_ptr[freq_base + 1]);
                    f10 = static_cast<float>(f_ptr[freq_base + 2]);
                    f11 = static_cast<float>(f_ptr[freq_base + 3]);
                }

                for (int operand = 0; operand < 2; ++operand) {
                    const bool is_key = operand == 1;
                    const bf16* source = is_key ? k_ptr : q_ptr;
                    bf16* destination = is_key ? ko_ptr : qo_ptr;
                    const bf16* scale = is_key ? ks_ptr : qs_ptr;
                    const float scale_low = local < H3_ROTARY_PAIRS
                        ? static_cast<float>(scale[local])
                        : 0.0f;
                    const float scale_second = local < H3_ROTARY_PAIRS
                        ? static_cast<float>(
                              scale[H3_ROTARY_PAIRS + local])
                        : 0.0f;
                    const float scale_tail =
                        local >= H3_ROT_DIM - RMS_ROPE_WG
                        ? static_cast<float>(scale[RMS_ROPE_WG + local])
                        : 0.0f;

#pragma unroll
                    for (int head_offset = 0;
                         head_offset < HeadsPerGroup;
                         ++head_offset) {
                        const int64_t head = head_begin + head_offset;
                        const int64_t base =
                            token * H3_PACKED_SEQUENCE_STRIDE +
                            head * H3_HEAD_DIM;
                        const float value0 =
                            static_cast<float>(source[base + local]);
                        const float value1 = static_cast<float>(
                            source[base + local + RMS_ROPE_WG]);
                        float square_sum = value0 * value0;
                        square_sum += value1 * value1;
                        partial[local] = square_sum;
                        if (local >= H3_ROTARY_PAIRS) {
                            pair_second[local - H3_ROTARY_PAIRS] = value0;
                        } else if (local < 32) {
                            pair_second[16 + local] = value1;
                        }
                        item.barrier(sycl::access::fence_space::local_space);

#pragma unroll
                        for (int64_t width = RMS_ROPE_WG / 2;
                             width > 0;
                             width /= 2) {
                            if (local < width) {
                                partial[local] += partial[local + width];
                            }
                            item.barrier(
                                sycl::access::fence_space::local_space);
                        }
                        const float inverse_rms =
                            sycl::rsqrt(
                                partial[0] / H3_HEAD_DIM + epsilon);

                        if (local < H3_ROTARY_PAIRS) {
                            const int64_t col0 = local;
                            const int64_t col1 = H3_ROTARY_PAIRS + local;
                            const bf16 normalized0 = static_cast<bf16>(
                                value0 * inverse_rms * scale_low);
                            const bf16 normalized1 = static_cast<bf16>(
                                pair_second[local] * inverse_rms *
                                scale_second);
                            const float rotated0 =
                                static_cast<float>(normalized0);
                            const float rotated1 =
                                static_cast<float>(normalized1);
                            destination[base + col0] = static_cast<bf16>(
                                f00 * rotated0 + f01 * rotated1);
                            destination[base + col1] = static_cast<bf16>(
                                f10 * rotated0 + f11 * rotated1);
                        }
                        if (local >= H3_ROT_DIM - RMS_ROPE_WG) {
                            const int64_t col = RMS_ROPE_WG + local;
                            destination[base + col] = static_cast<bf16>(
                                value1 * inverse_rms * scale_tail);
                        }
                        item.barrier(
                            sycl::access::fence_space::local_space);
                    }
                }
            });
    };
    utils::submit_kernel(
        cgf, q.device(), "kitchen_rms_rope_h3_b580_qk_head7");
}

bool use_b580_h3_rms_rope(const torch::Tensor& q) {
    auto& queue = utils::get_queue(q.device());
    const auto selection = device::get_bmg_selection_unwarned(queue);
    return selection.physical_sku == device::BmgSku::b580 &&
        !selection.forced;
}
#endif

template <typename InputT, typename FreqT, bool SplitHalf, bool Pair>
void launch_rms_rope(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& freqs,
    const torch::Tensor& q_scale,
    const torch::Tensor& k_scale,
    torch::Tensor& q_out,
    torch::Tensor& k_out,
    float epsilon,
    int64_t rot_dim) {
    const Tensor4Meta qm = tensor_meta(q, q_out);
    const Tensor4Meta km = Pair ? tensor_meta(k, k_out) : qm;
    const Freq6Meta fm = freq_meta(freqs);

    const auto* q_ptr = reinterpret_cast<const InputT*>(q.data_ptr());
    const auto* k_ptr = Pair
        ? reinterpret_cast<const InputT*>(k.data_ptr())
        : nullptr;
    const auto* f_ptr = reinterpret_cast<const FreqT*>(freqs.data_ptr());
    const auto* qs_ptr = q_scale.data_ptr<float>();
    const auto* ks_ptr = Pair ? k_scale.data_ptr<float>() : nullptr;
    auto* qo_ptr = reinterpret_cast<InputT*>(q_out.data_ptr());
    auto* ko_ptr = Pair
        ? reinterpret_cast<InputT*>(k_out.data_ptr())
        : nullptr;

    const int64_t total_rows = qm.rows + (Pair ? km.rows : 0);
    if (total_rows == 0) return;

    auto cgf = [&](sycl::handler& handler) {
        sycl::local_accessor<float, 1> partial(
            sycl::range<1>(RMS_ROPE_WG), handler);
        handler.parallel_for(
            sycl::nd_range<1>(
                sycl::range<1>(
                    static_cast<size_t>(total_rows * RMS_ROPE_WG)),
                sycl::range<1>(RMS_ROPE_WG)),
            [=](sycl::nd_item<1> item) {
                const int64_t local = item.get_local_id(0);
                const int64_t global_row = item.get_group(0);
                const bool is_key = Pair && global_row >= qm.rows;
                int64_t row = is_key ? global_row - qm.rows : global_row;

                const int64_t n1 = is_key ? km.n1 : qm.n1;
                const int64_t n2 = is_key ? km.n2 : qm.n2;
                const int64_t dim = is_key ? km.dim : qm.dim;
                const int64_t i2 = row % n2;
                row /= n2;
                const int64_t i1 = row % n1;
                const int64_t i0 = row / n1;

                const int64_t in_stride0 = is_key ? km.stride0 : qm.stride0;
                const int64_t in_stride1 = is_key ? km.stride1 : qm.stride1;
                const int64_t in_stride2 = is_key ? km.stride2 : qm.stride2;
                const int64_t in_stride3 = is_key ? km.stride3 : qm.stride3;
                const int64_t out_stride0 = is_key ? km.out_stride0 : qm.out_stride0;
                const int64_t out_stride1 = is_key ? km.out_stride1 : qm.out_stride1;
                const int64_t out_stride2 = is_key ? km.out_stride2 : qm.out_stride2;
                const int64_t out_stride3 = is_key ? km.out_stride3 : qm.out_stride3;
                const int64_t input_base =
                    i0 * in_stride0 + i1 * in_stride1 + i2 * in_stride2;
                const int64_t output_base =
                    i0 * out_stride0 + i1 * out_stride1 + i2 * out_stride2;

                const InputT* source = is_key ? k_ptr : q_ptr;
                InputT* destination = is_key ? ko_ptr : qo_ptr;
                const float* scale = is_key ? ks_ptr : qs_ptr;

                float square_sum = 0.0f;
                for (int64_t col = local; col < dim; col += RMS_ROPE_WG) {
                    const float value = static_cast<float>(
                        source[input_base + col * in_stride3]);
                    square_sum += value * value;
                }
                partial[local] = square_sum;
                item.barrier(sycl::access::fence_space::local_space);
                for (int64_t width = RMS_ROPE_WG / 2; width > 0; width /= 2) {
                    if (local < width) partial[local] += partial[local + width];
                    item.barrier(sycl::access::fence_space::local_space);
                }
                const float inverse_rms = sycl::rsqrt(partial[0] / dim + epsilon);

                const int64_t rotary_pairs = rot_dim / 2;
                for (int64_t pair = local; pair < rotary_pairs;
                     pair += RMS_ROPE_WG) {
                    const int64_t col0 = SplitHalf ? pair : pair * 2;
                    const int64_t col1 = SplitHalf
                        ? rotary_pairs + pair
                        : pair * 2 + 1;
                    const InputT normalized0 = static_cast<InputT>(
                        static_cast<float>(source[input_base + col0 * in_stride3]) *
                        inverse_rms * scale[col0]);
                    const InputT normalized1 = static_cast<InputT>(
                        static_cast<float>(source[input_base + col1 * in_stride3]) *
                        inverse_rms * scale[col1]);
                    const float value0 = static_cast<float>(normalized0);
                    const float value1 = static_cast<float>(normalized1);

                    const int64_t fi0 = fm.n0 == 1 ? 0 : i0;
                    const int64_t fi1 = fm.n1 == 1 ? 0 : i1;
                    const int64_t fi2 = fm.n2 == 1 ? 0 : i2;
                    const int64_t fp = fm.pairs == 1 ? 0 : pair;
                    const int64_t freq_base =
                        fi0 * fm.stride0 + fi1 * fm.stride1 +
                        fi2 * fm.stride2 + fp * fm.stride3;
                    const float f00 = static_cast<float>(f_ptr[freq_base]);
                    const float f01 = static_cast<float>(
                        f_ptr[freq_base + fm.stride5]);
                    const float f10 = static_cast<float>(
                        f_ptr[freq_base + fm.stride4]);
                    const float f11 = static_cast<float>(
                        f_ptr[freq_base + fm.stride4 + fm.stride5]);
                    destination[output_base + col0 * out_stride3] =
                        static_cast<InputT>(f00 * value0 + f01 * value1);
                    destination[output_base + col1 * out_stride3] =
                        static_cast<InputT>(f10 * value0 + f11 * value1);
                }

                for (int64_t col = rot_dim + local; col < dim;
                     col += RMS_ROPE_WG) {
                    destination[output_base + col * out_stride3] =
                        static_cast<InputT>(
                            static_cast<float>(
                                source[input_base + col * in_stride3]) *
                            inverse_rms * scale[col]);
                }
            });
    };
    utils::submit_kernel(
        cgf,
        q.device(),
        Pair ? "kitchen_rms_rope_pair_sycl" : "kitchen_rms_rope_sycl");
}

template <typename InputT, bool SplitHalf, bool Pair>
void dispatch_freq(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& freqs,
    const torch::Tensor& q_scale,
    const torch::Tensor& k_scale,
    torch::Tensor& q_out,
    torch::Tensor& k_out,
    float epsilon,
    int64_t rot_dim) {
    switch (freqs.scalar_type()) {
        case torch::kFloat32:
            launch_rms_rope<InputT, float, SplitHalf, Pair>(
                q, k, freqs, q_scale, k_scale, q_out, k_out, epsilon, rot_dim);
            break;
        case torch::kFloat16:
            launch_rms_rope<InputT, fp16, SplitHalf, Pair>(
                q, k, freqs, q_scale, k_scale, q_out, k_out, epsilon, rot_dim);
            break;
        case torch::kBFloat16:
            launch_rms_rope<InputT, bf16, SplitHalf, Pair>(
                q, k, freqs, q_scale, k_scale, q_out, k_out, epsilon, rot_dim);
            break;
        default:
            TORCH_CHECK(false, "unsupported freqs_cis dtype");
    }
}

template <bool SplitHalf, bool Pair>
void dispatch_input(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& freqs,
    const torch::Tensor& q_scale,
    const torch::Tensor& k_scale,
    torch::Tensor& q_out,
    torch::Tensor& k_out,
    float epsilon,
    int64_t rot_dim) {
    switch (q.scalar_type()) {
        case torch::kFloat32:
            dispatch_freq<float, SplitHalf, Pair>(
                q, k, freqs, q_scale, k_scale, q_out, k_out, epsilon, rot_dim);
            break;
        case torch::kFloat16:
            dispatch_freq<fp16, SplitHalf, Pair>(
                q, k, freqs, q_scale, k_scale, q_out, k_out, epsilon, rot_dim);
            break;
        case torch::kBFloat16:
            dispatch_freq<bf16, SplitHalf, Pair>(
                q, k, freqs, q_scale, k_scale, q_out, k_out, epsilon, rot_dim);
            break;
        default:
            TORCH_CHECK(false, "unsupported input dtype");
    }
}

}  // namespace

torch::Tensor rms_kitchen_rope1(
    torch::Tensor x,
    torch::Tensor freqs_cis,
    torch::Tensor scale,
    double epsilon,
    bool split_half,
    int64_t rot_dim,
    bool inplace) {
    check_tensor(x, "x");
    if (rot_dim == 0) rot_dim = x.size(3);
    check_freqs(x, freqs_cis, rot_dim);
    check_scale(scale, x, "scale");
    scale = scale.to(torch::kFloat32).contiguous();

    auto output = inplace ? x : torch::empty(x.sizes(), x.options());
    auto unused = torch::Tensor();
    if (split_half) {
        dispatch_input<true, false>(
            x, unused, freqs_cis, scale, unused, output, unused,
            static_cast<float>(epsilon), rot_dim);
    } else {
        dispatch_input<false, false>(
            x, unused, freqs_cis, scale, unused, output, unused,
            static_cast<float>(epsilon), rot_dim);
    }
    return output;
}

std::tuple<torch::Tensor, torch::Tensor> rms_kitchen_rope(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor freqs_cis,
    torch::Tensor q_scale,
    torch::Tensor k_scale,
    double epsilon,
    bool split_half,
    int64_t rot_dim,
    bool inplace) {
    check_tensor(q, "q");
    check_tensor(k, "k");
    TORCH_CHECK(q.device() == k.device(), "q and k must be on the same XPU");
    TORCH_CHECK(q.scalar_type() == k.scalar_type(), "q and k dtypes must match");
    TORCH_CHECK(q.size(3) == k.size(3), "q and k head dimensions must match");
    if (rot_dim == 0) rot_dim = q.size(3);
    check_freqs(q, freqs_cis, rot_dim);
    check_freqs(k, freqs_cis, rot_dim);
    check_scale(q_scale, q, "q_scale");
    check_scale(k_scale, k, "k_scale");

    auto q_out = inplace ? q : torch::empty(q.sizes(), q.options());
    auto k_out = inplace ? k : torch::empty(k.sizes(), k.options());
#if defined(OMNI_XPU_ARCH_BMG)
    if (is_minimax_h3_rms_rope_contract(
            q, k, freqs_cis, q_scale, k_scale, rot_dim,
            split_half, inplace)) {
        if (use_b580_h3_rms_rope(q)) {
            launch_minimax_h3_rms_rope_b580(
                q, k, freqs_cis, q_scale, k_scale,
                static_cast<float>(epsilon));
        } else {
            launch_minimax_h3_rms_rope(
                q, k, freqs_cis, q_scale, k_scale,
                static_cast<float>(epsilon));
        }
        return {q_out, k_out};
    }
#endif
    q_scale = q_scale.to(torch::kFloat32).contiguous();
    k_scale = k_scale.to(torch::kFloat32).contiguous();
    if (split_half) {
        dispatch_input<true, true>(
            q, k, freqs_cis, q_scale, k_scale, q_out, k_out,
            static_cast<float>(epsilon), rot_dim);
    } else {
        dispatch_input<false, true>(
            q, k, freqs_cis, q_scale, k_scale, q_out, k_out,
            static_cast<float>(epsilon), rot_dim);
    }
    return {q_out, k_out};
}

}  // namespace rotary
}  // namespace omni_xpu
