#include <cmath>
#include <cstdint>
#include <optional>

#include <torch/extension.h>
#include <sycl/sycl.hpp>

#include "utils.h"

namespace omni_xpu::kitchen {
namespace {

template <typename T>
class DeltaConvStepKernel;

template <typename T>
class GatedDeltaDecodeKernel;

template <typename T>
torch::Tensor launch_delta_conv_step(
    const torch::Tensor& proj,
    torch::Tensor& conv_state,
    const torch::Tensor& conv_w,
    const std::optional<torch::Tensor>& conv_b,
    const std::optional<torch::Tensor>& snapshots) {
    const int64_t batch = proj.size(0);
    const int64_t steps = proj.size(1);
    const int64_t channels = proj.size(2);
    const int64_t kernel_size = conv_state.size(2) + 1;
    auto output = torch::empty({batch, channels, steps}, proj.options());

    const auto* input_ptr = static_cast<const T*>(proj.data_ptr());
    auto* state_ptr = static_cast<T*>(conv_state.data_ptr());
    const auto* weight_ptr = static_cast<const T*>(conv_w.data_ptr());
    const auto* bias_ptr = conv_b.has_value()
        ? static_cast<const T*>(conv_b->data_ptr()) : nullptr;
    auto* snapshot_ptr = snapshots.has_value()
        ? static_cast<T*>(snapshots->data_ptr()) : nullptr;
    auto* output_ptr = static_cast<T*>(output.data_ptr());

    auto cgf = [&](sycl::handler& handler) {
        handler.parallel_for<DeltaConvStepKernel<T>>(
            sycl::range<1>(static_cast<size_t>(batch * channels)),
            [=](sycl::id<1> item) {
                const int64_t index = static_cast<int64_t>(item[0]);
                const int64_t b = index / channels;
                const int64_t c = index % channels;
                T* state = state_ptr + index * (kernel_size - 1);
                const T* weight = weight_ptr + c * kernel_size;
                for (int64_t step = 0; step < steps; ++step) {
                    const T value = input_ptr[(b * steps + step) * channels + c];
                    float sum = bias_ptr ? static_cast<float>(bias_ptr[c]) : 0.0f;
                    for (int64_t tap = 0; tap < kernel_size - 1; ++tap) {
                        sum += static_cast<float>(state[tap]) *
                               static_cast<float>(weight[tap]);
                    }
                    sum += static_cast<float>(value) *
                           static_cast<float>(weight[kernel_size - 1]);
                    const float activated = sum / (1.0f + sycl::exp(-sum));
                    output_ptr[(b * channels + c) * steps + step] =
                        static_cast<T>(activated);
                    for (int64_t tap = 0; tap < kernel_size - 2; ++tap) {
                        state[tap] = state[tap + 1];
                    }
                    state[kernel_size - 2] = value;
                    if (snapshot_ptr && step + 1 < steps) {
                        const int64_t offset =
                            ((step * batch + b) * channels + c) *
                            (kernel_size - 1);
                        for (int64_t tap = 0; tap < kernel_size - 1; ++tap) {
                            snapshot_ptr[offset + tap] = state[tap];
                        }
                    }
                }
            });
    };
    utils::submit_kernel(cgf, proj.device(), "kitchen_deltanet_conv_step");
    return output;
}

template <typename T>
torch::Tensor launch_gated_delta_decode(
    const torch::Tensor& mixed_qkv,
    const torch::Tensor& x,
    const torch::Tensor& w_a,
    const torch::Tensor& w_b,
    const torch::Tensor& dt_bias,
    const torch::Tensor& g_decay,
    torch::Tensor& state,
    int64_t key_dim,
    int64_t key_heads,
    double scale,
    const torch::Tensor& z,
    const torch::Tensor& norm_weight,
    double eps,
    const std::optional<torch::Tensor>& snapshots) {
    const int64_t batch = x.size(0);
    const int64_t steps = x.size(1);
    const int64_t hidden = x.size(2);
    const int64_t heads = state.size(1);
    const int64_t head_dim = state.size(2);
    const int64_t value_dim = state.size(3);
    const int64_t channels = mixed_qkv.size(1);
    const int64_t head_repeat = heads / key_heads;
    auto output = torch::empty({batch, steps, heads, value_dim}, x.options());

    const T* qkv_ptr = static_cast<const T*>(mixed_qkv.data_ptr());
    const T* x_ptr = static_cast<const T*>(x.data_ptr());
    const T* a_ptr = static_cast<const T*>(w_a.data_ptr());
    const T* b_ptr = static_cast<const T*>(w_b.data_ptr());
    const float* dt_ptr = dt_bias.data_ptr<float>();
    const float* decay_ptr = g_decay.data_ptr<float>();
    float* state_ptr = state.data_ptr<float>();
    float* snapshot_ptr = snapshots.has_value()
        ? snapshots->data_ptr<float>() : nullptr;
    const T* z_ptr = static_cast<const T*>(z.data_ptr());
    const T* norm_ptr = static_cast<const T*>(norm_weight.data_ptr());
    T* output_ptr = static_cast<T*>(output.data_ptr());
    const float score_scale = static_cast<float>(scale);
    const float norm_eps = static_cast<float>(eps);

    // Each workgroup owns one (batch, head) state slice. Reduce the rounded
    // per-value outputs locally so RMSNorm and gate need no global scratch or
    // second launch; every lane reaches the barriers for every decode step.
    size_t lanes = 1;
    while (lanes < static_cast<size_t>(value_dim)) lanes <<= 1;
    auto cgf = [&](sycl::handler& handler) {
        sycl::local_accessor<float, 1> squares(sycl::range<1>(lanes), handler);
        handler.parallel_for<GatedDeltaDecodeKernel<T>>(
            sycl::nd_range<1>(
                sycl::range<1>(static_cast<size_t>(batch * heads) * lanes),
                sycl::range<1>(lanes)),
            [=](sycl::nd_item<1> item) {
                const int64_t dv = static_cast<int64_t>(item.get_local_id(0));
                const int64_t group = static_cast<int64_t>(item.get_group(0));
                const int64_t head = group % heads;
                const int64_t batch_index = group / heads;
                const int64_t key_head = head / head_repeat;
                for (int64_t step = 0; step < steps; ++step) {
                    float raw_value = 0.0f;
                    if (dv < value_dim) {
                        const T* input_row =
                            x_ptr + (batch_index * steps + step) * hidden;
                        float a_sum = 0.0f;
                        float b_sum = 0.0f;
                        for (int64_t feature = 0; feature < hidden; ++feature) {
                            const float value = static_cast<float>(input_row[feature]);
                            a_sum += value *
                                static_cast<float>(a_ptr[head * hidden + feature]);
                            b_sum += value *
                                static_cast<float>(b_ptr[head * hidden + feature]);
                        }
                        const float a_rounded = static_cast<float>(static_cast<T>(a_sum));
                        const float b_rounded = static_cast<float>(static_cast<T>(b_sum));
                        const float beta = static_cast<float>(static_cast<T>(
                            1.0f / (1.0f + sycl::exp(-b_rounded))));
                        const float pre_decay = a_rounded + dt_ptr[head];
                        const float softplus = pre_decay > 20.0f
                            ? pre_decay : sycl::log(1.0f + sycl::exp(pre_decay));
                        const float decay = sycl::exp(decay_ptr[head] * softplus);

                        float q_norm_square = 0.0f;
                        float k_norm_square = 0.0f;
                        for (int64_t d = 0; d < head_dim; ++d) {
                            const int64_t q_channel = key_head * head_dim + d;
                            const int64_t k_channel = key_dim + q_channel;
                            const float q_value = static_cast<float>(
                                qkv_ptr[(batch_index * channels + q_channel) * steps + step]);
                            const float k_value = static_cast<float>(
                                qkv_ptr[(batch_index * channels + k_channel) * steps + step]);
                            q_norm_square += q_value * q_value;
                            k_norm_square += k_value * k_value;
                        }
                        const float q_inverse = 1.0f /
                            sycl::sqrt(sycl::fmax(q_norm_square, 1e-24f));
                        const float k_inverse = 1.0f /
                            sycl::sqrt(sycl::fmax(k_norm_square, 1e-24f));

                        float memory = 0.0f;
                        for (int64_t d = 0; d < head_dim; ++d) {
                            const int64_t state_offset =
                                ((batch_index * heads + head) * head_dim + d) *
                                value_dim + dv;
                            const float updated = state_ptr[state_offset] * decay;
                            state_ptr[state_offset] = updated;
                            const int64_t k_channel = key_dim + key_head * head_dim + d;
                            const float k_value = static_cast<float>(
                                qkv_ptr[(batch_index * channels + k_channel) * steps + step]) *
                                k_inverse;
                            memory += k_value * updated;
                        }
                        const int64_t v_channel = 2 * key_dim + head * value_dim + dv;
                        const float value = static_cast<float>(
                            qkv_ptr[(batch_index * channels + v_channel) * steps + step]);
                        const float delta = (value - memory) * beta;
                        for (int64_t d = 0; d < head_dim; ++d) {
                            const int64_t state_offset =
                                ((batch_index * heads + head) * head_dim + d) *
                                value_dim + dv;
                            const int64_t base_channel = key_head * head_dim + d;
                            const float k_value = static_cast<float>(
                                qkv_ptr[(batch_index * channels + key_dim + base_channel) *
                                        steps + step]) * k_inverse;
                            const float updated = state_ptr[state_offset] + k_value * delta;
                            state_ptr[state_offset] = updated;
                            const float q_value = static_cast<float>(
                                qkv_ptr[(batch_index * channels + base_channel) * steps + step]) *
                                q_inverse * score_scale;
                            raw_value += q_value * updated;
                            if (snapshot_ptr && step + 1 < steps) {
                                const int64_t snapshot_offset =
                                    (((step * batch + batch_index) * heads + head) *
                                     head_dim + d) * value_dim + dv;
                                snapshot_ptr[snapshot_offset] = updated;
                            }
                        }
                    }
                    const float rounded = dv < value_dim
                        ? static_cast<float>(static_cast<T>(raw_value)) : 0.0f;
                    squares[dv] = rounded * rounded;
                    item.barrier(sycl::access::fence_space::local_space);
                    for (size_t stride = lanes / 2; stride > 0; stride >>= 1) {
                        if (static_cast<size_t>(dv) < stride) {
                            squares[dv] += squares[dv + stride];
                        }
                        item.barrier(sycl::access::fence_space::local_space);
                    }
                    if (dv < value_dim) {
                        const int64_t index =
                            ((batch_index * steps + step) * heads + head) *
                            value_dim + dv;
                        const float inverse = 1.0f / sycl::sqrt(
                            squares[0] / static_cast<float>(value_dim) + norm_eps);
                        const float normalized = static_cast<float>(
                            static_cast<T>(rounded * inverse *
                                           static_cast<float>(norm_ptr[dv])));
                        const float gate = static_cast<float>(z_ptr[index]);
                        const float activated = static_cast<float>(
                            static_cast<T>(gate / (1.0f + sycl::exp(-gate))));
                        output_ptr[index] = static_cast<T>(normalized * activated);
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }
            });
    };
    utils::submit_kernel(cgf, x.device(), "kitchen_gated_delta_decode_fused");
    return output;
}

}  // namespace

torch::Tensor deltanet_conv_step(
    torch::Tensor proj,
    torch::Tensor conv_state,
    torch::Tensor conv_w,
    std::optional<torch::Tensor> conv_b,
    std::optional<torch::Tensor> snapshots) {
    TORCH_CHECK(proj.device().is_xpu(), "proj must be on XPU");
    TORCH_CHECK(proj.dim() == 3 && proj.is_contiguous(),
                "proj must be contiguous [B,S,C]");
    const int64_t batch = proj.size(0);
    const int64_t steps = proj.size(1);
    const int64_t channels = proj.size(2);
    TORCH_CHECK(batch > 0 && channels > 0 && steps >= 1 && steps <= 8,
                "deltanet_conv_step requires B,C>0 and 1<=S<=8");
    TORCH_CHECK(conv_state.device() == proj.device() &&
                    conv_state.scalar_type() == proj.scalar_type() &&
                    conv_state.dim() == 3 && conv_state.size(0) == batch &&
                    conv_state.size(1) == channels &&
                    conv_state.size(2) >= 1 && conv_state.is_contiguous(),
                "conv_state must be contiguous [B,C,KS-1] on proj device/dtype");
    const int64_t kernel_size = conv_state.size(2) + 1;
    TORCH_CHECK(conv_w.device() == proj.device() &&
                    conv_w.scalar_type() == proj.scalar_type() &&
                    conv_w.is_contiguous() &&
                    conv_w.numel() == channels * kernel_size,
                "conv_w must hold contiguous [C,1,KS] weights");
    if (conv_b.has_value()) {
        TORCH_CHECK(conv_b->device() == proj.device() &&
                        conv_b->scalar_type() == proj.scalar_type() &&
                        conv_b->is_contiguous() && conv_b->numel() == channels,
                    "conv_b must hold contiguous [C] values");
    }
    if (snapshots.has_value()) {
        TORCH_CHECK(snapshots->device() == proj.device() &&
                        snapshots->scalar_type() == proj.scalar_type() &&
                        snapshots->is_contiguous() &&
                        snapshots->sizes() ==
                            at::IntArrayRef({steps - 1, batch, channels,
                                             kernel_size - 1}),
                    "snapshots must be contiguous [S-1,B,C,KS-1]");
    }
    switch (proj.scalar_type()) {
        case at::kHalf:
            return launch_delta_conv_step<sycl::half>(
                proj, conv_state, conv_w, conv_b, snapshots);
        case at::kBFloat16:
            return launch_delta_conv_step<sycl::ext::oneapi::bfloat16>(
                proj, conv_state, conv_w, conv_b, snapshots);
        case at::kFloat:
            return launch_delta_conv_step<float>(
                proj, conv_state, conv_w, conv_b, snapshots);
        default:
            TORCH_CHECK(false, "deltanet_conv_step requires fp16, bf16 or fp32");
    }
}

torch::Tensor gated_delta_decode_fused(
    torch::Tensor mixed_qkv,
    torch::Tensor x,
    torch::Tensor w_a,
    torch::Tensor w_b,
    torch::Tensor dt_bias,
    torch::Tensor g_decay,
    torch::Tensor state,
    int64_t key_dim,
    int64_t key_heads,
    double scale,
    torch::Tensor z,
    torch::Tensor norm_weight,
    double eps,
    std::optional<torch::Tensor> snapshots) {
    TORCH_CHECK(x.device().is_xpu() && x.dim() == 3 && x.is_contiguous(),
                "x must be contiguous XPU [B,S,HD]");
    const int64_t batch = x.size(0);
    const int64_t steps = x.size(1);
    const int64_t hidden = x.size(2);
    TORCH_CHECK(batch > 0 && hidden > 0 && steps >= 1 && steps <= 8,
                "gated_delta_decode_fused requires B,HD>0 and 1<=S<=8");
    TORCH_CHECK(state.device() == x.device() && state.scalar_type() == at::kFloat &&
                    state.dim() == 4 && state.is_contiguous() &&
                    state.size(0) == batch,
                "state must be contiguous FP32 XPU [B,HV,DK,DV]");
    const int64_t heads = state.size(1);
    const int64_t head_dim = state.size(2);
    const int64_t value_dim = state.size(3);
    TORCH_CHECK(heads > 0 && head_dim > 0 && value_dim > 0 &&
                    key_heads > 0 && heads % key_heads == 0 &&
                    key_dim == key_heads * head_dim,
                "head and key dimensions are inconsistent");
    const int64_t channels = 2 * key_dim + heads * value_dim;
    TORCH_CHECK(mixed_qkv.device() == x.device() &&
                    mixed_qkv.scalar_type() == x.scalar_type() &&
                    mixed_qkv.dim() == 3 && mixed_qkv.is_contiguous() &&
                    mixed_qkv.size(0) == batch &&
                    mixed_qkv.size(1) == channels &&
                    mixed_qkv.size(2) == steps,
                "mixed_qkv must be contiguous [B,2*KEY_DIM+HV*DV,S]");
    for (const auto& weight : {w_a, w_b}) {
        TORCH_CHECK(weight.device() == x.device() &&
                        weight.scalar_type() == x.scalar_type() &&
                        weight.dim() == 2 && weight.is_contiguous() &&
                        weight.size(0) == heads && weight.size(1) == hidden,
                    "w_a and w_b must be contiguous [HV,HD]");
    }
    for (const auto& vector : {dt_bias, g_decay}) {
        TORCH_CHECK(vector.device() == x.device() &&
                        vector.scalar_type() == at::kFloat &&
                        vector.dim() == 1 && vector.is_contiguous() &&
                        vector.numel() == heads,
                    "dt_bias and g_decay must be contiguous FP32 [HV]");
    }
    TORCH_CHECK(z.device() == x.device() && z.scalar_type() == x.scalar_type() &&
                    z.dim() == 3 && z.is_contiguous() &&
                    z.size(0) == batch && z.size(1) == steps &&
                    z.size(2) == heads * value_dim,
                "z must be contiguous [B,S,HV*DV]");
    TORCH_CHECK(norm_weight.device() == x.device() &&
                    norm_weight.scalar_type() == x.scalar_type() &&
                    norm_weight.dim() == 1 && norm_weight.is_contiguous() &&
                    norm_weight.numel() == value_dim,
                "norm_weight must be contiguous [DV]");
    TORCH_CHECK(std::isfinite(scale) && std::isfinite(eps) && eps >= 0.0,
                "scale and eps must be finite and eps nonnegative");
    if (snapshots.has_value()) {
        TORCH_CHECK(snapshots->device() == x.device() &&
                        snapshots->scalar_type() == at::kFloat &&
                        snapshots->dim() == 5 && snapshots->is_contiguous() &&
                        snapshots->size(0) == steps - 1 &&
                        snapshots->size(1) == batch &&
                        snapshots->size(2) == heads &&
                        snapshots->size(3) == head_dim &&
                        snapshots->size(4) == value_dim,
                    "snapshots must be contiguous FP32 [S-1,B,HV,DK,DV]");
    }
    switch (x.scalar_type()) {
        case at::kHalf:
            return launch_gated_delta_decode<sycl::half>(
                mixed_qkv, x, w_a, w_b, dt_bias, g_decay, state,
                key_dim, key_heads, scale, z, norm_weight, eps, snapshots);
        case at::kBFloat16:
            return launch_gated_delta_decode<sycl::ext::oneapi::bfloat16>(
                mixed_qkv, x, w_a, w_b, dt_bias, g_decay, state,
                key_dim, key_heads, scale, z, norm_weight, eps, snapshots);
        case at::kFloat:
            return launch_gated_delta_decode<float>(
                mixed_qkv, x, w_a, w_b, dt_bias, g_decay, state,
                key_dim, key_heads, scale, z, norm_weight, eps, snapshots);
        default:
            TORCH_CHECK(false, "gated_delta_decode_fused requires fp16, bf16 or fp32");
    }
}

}  // namespace omni_xpu::kitchen
