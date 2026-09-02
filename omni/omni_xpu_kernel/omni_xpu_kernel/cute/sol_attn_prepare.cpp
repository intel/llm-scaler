// Copyright 2026
// SPDX-License-Identifier: Apache-2.0
//
// Triton-free Sol-Attn preprocessing for Intel XPU.  The output contract is
// shared by the correctness reference and the CUTE/DPAS forward mainloop:
// K centroids, V block means, and one exact/approximate route byte per Q/K
// block pair.  The CUTE mainloop adds log2(block_length) to each approximate
// score, so V means reproduce the V-sum numerator while the ordinary softmax
// denominator reproduces the block-length weight.

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/library.h>

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>

#include "sol_attn_config.h"

namespace omni_xpu_sol_attn {
namespace {

#ifndef SOL_ATTN_INLINE_ROUTE
#define SOL_ATTN_INLINE_ROUTE 0
#endif

using bf16 = sycl::ext::oneapi::bfloat16;

constexpr int64_t kBlock = 64;
constexpr int64_t kHeadDim = 128;
constexpr int64_t kWorkGroup = 128;
constexpr float kLog2E = 1.4426950408889634f;

template <typename KernelFunc>
void submit(sycl::queue& queue, int64_t groups, KernelFunc&& kernel) {
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(static_cast<size_t>(groups * kWorkGroup)),
            sycl::range<1>(static_cast<size_t>(kWorkGroup))),
        std::forward<KernelFunc>(kernel));
  });
}

struct TopKThresholdKernel;

void check_input(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.device().is_xpu(), name, " must be on XPU");
  TORCH_CHECK(tensor.scalar_type() == at::kBFloat16, name, " must be bfloat16");
  TORCH_CHECK(tensor.dim() == 4, name, " must be BTHD");
  TORCH_CHECK(tensor.size(1) > 0, name, " must have a non-empty sequence");
  TORCH_CHECK(tensor.size(3) == kHeadDim, name, " must have head_dim 128");
  TORCH_CHECK(tensor.stride(3) == 1, name, " must have a contiguous head dimension");
}

#if SOL_ATTN_INLINE_ROUTE
std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor> prepare(
#else
std::tuple<at::Tensor, at::Tensor, at::Tensor> prepare(
#endif
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    double scale_value,
    double tau_value,
    int64_t sink_start,
    int64_t sink_end,
    int64_t sink_q_start,
    int64_t sink_q_end,
    int64_t topk_count,
    const at::Tensor& block_len) {
  check_input(q, "q");
  check_input(k, "k");
  check_input(v, "v");
  TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(),
              "q, k, and v must share BTHD shape");
  TORCH_CHECK(q.device() == k.device() && q.device() == v.device(),
              "q, k, and v must share an XPU device");
  TORCH_CHECK(std::isfinite(scale_value), "scale must be finite");
  TORCH_CHECK(std::isfinite(tau_value), "tau must be finite");

  const int64_t batch = q.size(0);
  const int64_t tokens = q.size(1);
  const int64_t heads = q.size(2);
  const int64_t blocks = (tokens + kBlock - 1) / kBlock;
  TORCH_CHECK(
      block_len.device() == q.device() &&
          block_len.scalar_type() == at::kInt &&
          block_len.is_contiguous() &&
          (block_len.numel() == 0 ||
           (block_len.dim() == 1 && block_len.numel() == blocks)),
      "block_len must be an empty int32 XPU tensor or contiguous int32 (N,)");
  TORCH_CHECK(0 <= sink_start && sink_start <= sink_end && sink_end <= blocks,
              "sink_blocks must be an ordered range inside the block count");
  TORCH_CHECK(0 <= sink_q_start && sink_q_start <= sink_q_end && sink_q_end <= blocks,
              "sink_q must be an ordered range inside the block count");
  const int64_t sink_count = sink_end - sink_start;
  const int64_t selectable_blocks = blocks - sink_count;
  const int64_t max_topk_count = selectable_blocks > 0
      ? selectable_blocks - 1
      : 0;
  TORCH_CHECK(topk_count == -1 ||
                  (topk_count >= 0 &&
                   topk_count <= max_topk_count),
              "topk_count must be -1 or inside the non-sink block budget");

  const float scale_log2 = static_cast<float>(scale_value) * kLog2E;
  const float tau = static_cast<float>(tau_value);
  const int64_t summary_groups = batch * heads * blocks;

  auto float_options = q.options().dtype(at::kFloat);
  auto q_centroids = at::empty({batch, heads, blocks, kHeadDim}, float_options);
  auto k_centroids = at::empty({batch, heads, blocks, kHeadDim}, q.options());
  auto v_means = at::empty_like(k_centroids);
  auto k_mean = at::empty({batch, heads, kHeadDim}, float_options);
  auto k_variance = at::empty_like(k_mean);
  auto thresholds = at::empty({batch, heads, blocks}, float_options);
#if SOL_ATTN_INLINE_ROUTE
  auto key_sinks = at::empty(
      {batch, heads, blocks}, q.options().dtype(at::kByte));
  auto topk_routes = topk_count >= 0
      ? at::empty(
            {batch, heads, blocks, blocks},
            q.options().dtype(at::kByte))
      : at::empty({0}, q.options().dtype(at::kByte));
#else
  auto routes = at::empty(
      {batch, heads, blocks, blocks}, q.options().dtype(at::kByte));
#endif
  auto route_scores = topk_count >= 0
      ? at::empty({batch, heads, blocks, blocks}, float_options)
      : at::Tensor{};

  const auto* q_ptr = reinterpret_cast<const bf16*>(q.data_ptr());
  const auto* k_ptr = reinterpret_cast<const bf16*>(k.data_ptr());
  const auto* v_ptr = reinterpret_cast<const bf16*>(v.data_ptr());
  auto* qc_ptr = q_centroids.data_ptr<float>();
  auto* kc_ptr = reinterpret_cast<bf16*>(k_centroids.data_ptr());
  auto* vm_ptr = reinterpret_cast<bf16*>(v_means.data_ptr());
  auto* mean_ptr = k_mean.data_ptr<float>();
  auto* variance_ptr = k_variance.data_ptr<float>();
  auto* threshold_ptr = thresholds.data_ptr<float>();
#if SOL_ATTN_INLINE_ROUTE
  auto* key_sink_ptr = key_sinks.data_ptr<uint8_t>();
  auto* topk_route_ptr = topk_count >= 0
      ? topk_routes.data_ptr<uint8_t>()
      : nullptr;
#else
  auto* route_ptr = routes.data_ptr<uint8_t>();
#endif
  auto* route_score_ptr = topk_count >= 0
      ? route_scores.data_ptr<float>()
      : nullptr;
  const auto* block_len_ptr = block_len.numel() == 0
      ? nullptr
      : block_len.data_ptr<int32_t>();

  const int64_t sq_b = q.stride(0);
  const int64_t sq_t = q.stride(1);
  const int64_t sq_h = q.stride(2);
  const int64_t sk_b = k.stride(0);
  const int64_t sk_t = k.stride(1);
  const int64_t sk_h = k.stride(2);
  const int64_t sv_b = v.stride(0);
  const int64_t sv_t = v.stride(1);
  const int64_t sv_h = v.stride(2);

  auto& queue = c10::xpu::getCurrentXPUStream(q.device().index()).queue();

  submit(queue, summary_groups, [=](sycl::nd_item<1> item) {
    const int64_t group = item.get_group_linear_id();
    const int64_t dim = item.get_local_linear_id();
    const int64_t block = group % blocks;
    const int64_t batch_head = group / blocks;
    const int64_t head = batch_head % heads;
    const int64_t batch_index = batch_head / heads;
    const int64_t start = block * kBlock;
    const int64_t physical_length = sycl::min(kBlock, tokens - start);
    const int64_t requested_length = block_len_ptr == nullptr
        ? physical_length
        : static_cast<int64_t>(block_len_ptr[block]);
    const int64_t length = requested_length < 1
        ? 1
        : (requested_length > physical_length
               ? physical_length
               : requested_length);

    float q_sum = 0.0f;
    float k_sum = 0.0f;
    float v_sum = 0.0f;
    for (int64_t offset = 0; offset < length; ++offset) {
      const int64_t token = start + offset;
      q_sum += static_cast<float>(
          q_ptr[batch_index * sq_b + token * sq_t + head * sq_h + dim]);
      k_sum += static_cast<float>(
          k_ptr[batch_index * sk_b + token * sk_t + head * sk_h + dim]);
      v_sum += static_cast<float>(
          v_ptr[batch_index * sv_b + token * sv_t + head * sv_h + dim]);
    }
    const int64_t summary = (batch_head * blocks + block) * kHeadDim + dim;
    qc_ptr[summary] = q_sum / static_cast<float>(length);
    kc_ptr[summary] = static_cast<bf16>(k_sum / static_cast<float>(length));
    vm_ptr[summary] = static_cast<bf16>(v_sum / static_cast<float>(length));
  });

  if (topk_count >= 0) {
    submit(queue, summary_groups,
           [=](sycl::nd_item<1> item)
               [[sycl::reqd_sub_group_size(16)]] {
      const int64_t group = item.get_group_linear_id();
      const int64_t query_block = group % blocks;
      const int64_t batch_head = group / blocks;
      const int64_t summary_offset = batch_head * blocks * kHeadDim;
      const auto subgroup = item.get_sub_group();
      const int64_t subgroup_id = subgroup.get_group_linear_id();
      const int64_t subgroup_count = item.get_local_range(0) / 16;
      const int64_t lane = subgroup.get_local_linear_id();

      for (int64_t key_block = subgroup_id; key_block < blocks;
           key_block += subgroup_count) {
        float partial = 0.0f;
        for (int64_t dim = lane; dim < kHeadDim; dim += 16) {
          partial +=
              qc_ptr[summary_offset + query_block * kHeadDim + dim] *
              static_cast<float>(
                  kc_ptr[summary_offset + key_block * kHeadDim + dim]);
        }
        const float score = sycl::reduce_over_group(
            subgroup, partial, sycl::plus<float>()) * scale_log2;
        if (lane == 0) {
          route_score_ptr[
              (batch_head * blocks + query_block) * blocks + key_block] =
                  score;
        }
      }
    });

  }

  submit(queue, batch * heads, [=](sycl::nd_item<1> item) {
    const int64_t batch_head = item.get_group_linear_id();
    const int64_t dim = item.get_local_linear_id();
    float total = 0.0f;
    float total_square = 0.0f;
    for (int64_t block = 0; block < blocks; ++block) {
      const float value = static_cast<float>(
          kc_ptr[(batch_head * blocks + block) * kHeadDim + dim]);
      total += value;
      total_square += value * value;
    }
    const float mean = total / static_cast<float>(blocks);
    mean_ptr[batch_head * kHeadDim + dim] = mean;
    variance_ptr[batch_head * kHeadDim + dim] =
        sycl::fmax(total_square / static_cast<float>(blocks) - mean * mean, 0.0f);
  });

  submit(queue, summary_groups, [=](sycl::nd_item<1> item) {
    const int64_t group = item.get_group_linear_id();
    const int64_t dim = item.get_local_linear_id();
    const int64_t block = group % blocks;
    const int64_t batch_head = group / blocks;
    const int64_t summary = (batch_head * blocks + block) * kHeadDim + dim;
    const float centroid = qc_ptr[summary];
    const float raw_mean = sycl::reduce_over_group(
        item.get_group(), centroid * mean_ptr[batch_head * kHeadDim + dim],
        sycl::plus<float>());
    const float raw_variance = sycl::reduce_over_group(
        item.get_group(), centroid * centroid *
            variance_ptr[batch_head * kHeadDim + dim],
        sycl::plus<float>());
    if (dim == 0) {
      const float mean = raw_mean * scale_log2;
      const float variance = raw_variance * scale_log2 * scale_log2;
      const bool query_sink =
          block >= sink_q_start && block < sink_q_end;
      threshold_ptr[batch_head * blocks + block] = query_sink
          ? -std::numeric_limits<float>::infinity()
          : mean + tau * sycl::sqrt(
              sycl::fmax(variance, 0.0f) + 1.0e-6f);
#if SOL_ATTN_INLINE_ROUTE
      key_sink_ptr[batch_head * blocks + block] = static_cast<uint8_t>(
          block >= sink_start && block < sink_end);
#endif
    }
  });

  if (topk_count >= 0) {
    int64_t padded_blocks = 1;
    while (padded_blocks < blocks) {
      padded_blocks *= 2;
    }
    const auto local_memory =
        queue.get_device().get_info<sycl::info::device::local_mem_size>();
    TORCH_CHECK(
        padded_blocks * static_cast<int64_t>(sizeof(float)) <=
            static_cast<int64_t>(local_memory),
        "top-k Sol-Attn block count exceeds XPU local-memory capacity");

    queue.submit([&](sycl::handler& cgh) {
      sycl::local_accessor<float, 1> sorted_scores(
          sycl::range<1>(static_cast<size_t>(padded_blocks)), cgh);
      cgh.parallel_for<TopKThresholdKernel>(
          sycl::nd_range<1>(
              sycl::range<1>(
                  static_cast<size_t>(summary_groups * kWorkGroup)),
              sycl::range<1>(static_cast<size_t>(kWorkGroup))),
          [=](sycl::nd_item<1> item) {
            const int64_t group = item.get_group_linear_id();
            const int64_t lane = item.get_local_linear_id();
            const int64_t query_block = group % blocks;
            const int64_t batch_head = group / blocks;
            const int64_t row_offset =
                (batch_head * blocks + query_block) * blocks;

            for (int64_t index = lane; index < padded_blocks;
                 index += kWorkGroup) {
              sorted_scores[index] =
                  index < blocks &&
                      !(index >= sink_start && index < sink_end)
                  ? route_score_ptr[row_offset + index]
                  : -std::numeric_limits<float>::infinity();
            }
            sycl::group_barrier(item.get_group());

            for (int64_t width = 2; width <= padded_blocks; width *= 2) {
              for (int64_t stride = width / 2; stride > 0; stride /= 2) {
                for (int64_t index = lane; index < padded_blocks;
                     index += kWorkGroup) {
                  const int64_t peer = index ^ stride;
                  if (peer <= index) {
                    continue;
                  }
                  const float left = sorted_scores[index];
                  const float right = sorted_scores[peer];
                  const bool descending = (index & width) == 0;
                  const bool swap = descending ? left < right : left > right;
                  if (swap) {
                    sorted_scores[index] = right;
                    sorted_scores[peer] = left;
                  }
                }
                sycl::group_barrier(item.get_group());
              }
            }

            if (lane == 0 &&
                !(query_block >= sink_q_start && query_block < sink_q_end)) {
              float threshold = topk_count == 0
                  ? std::numeric_limits<float>::infinity()
                  : sorted_scores[topk_count - 1];
              threshold_ptr[batch_head * blocks + query_block] = threshold;
            }
          });
    });

#if SOL_ATTN_INLINE_ROUTE
    const int64_t route_elements = summary_groups * blocks;
    const int64_t route_groups =
        (route_elements + kWorkGroup - 1) / kWorkGroup;
    submit(queue, route_groups, [=](sycl::nd_item<1> item) {
      const int64_t index = item.get_global_linear_id();
      if (index < route_elements) {
        const int64_t query_row = index / blocks;
        topk_route_ptr[index] = static_cast<uint8_t>(
            route_score_ptr[index] >= threshold_ptr[query_row]);
      }
    });
#endif
  }

#if !SOL_ATTN_INLINE_ROUTE
  submit(queue, summary_groups, [=](sycl::nd_item<1> item) {
    const int64_t group = item.get_group_linear_id();
    const int64_t dim = item.get_local_linear_id();
    const int64_t query_block = group % blocks;
    const int64_t batch_head = group / blocks;
    const float q_value =
        qc_ptr[(batch_head * blocks + query_block) * kHeadDim + dim];
    const bool query_sink =
        query_block >= sink_q_start && query_block < sink_q_end;
    const float threshold = threshold_ptr[batch_head * blocks + query_block];

    for (int64_t key_block = 0; key_block < blocks; ++key_block) {
      const float partial = q_value * static_cast<float>(
          kc_ptr[(batch_head * blocks + key_block) * kHeadDim + dim]);
      const float route_score = sycl::reduce_over_group(
          item.get_group(), partial, sycl::plus<float>()) * scale_log2;
      if (dim == 0) {
        const int64_t distance = query_block > key_block
            ? query_block - key_block
            : key_block - query_block;
        const bool key_sink = key_block >= sink_start && key_block < sink_end;
        const bool exact = query_sink || key_sink || distance <= 1 ||
            (topk_count >= 0 ? route_score >= threshold
                             : route_score > threshold);
        route_ptr[(batch_head * blocks + query_block) * blocks + key_block] =
            static_cast<uint8_t>(exact);
      }
    }
  });
#endif

#if SOL_ATTN_INLINE_ROUTE
  return {
      k_centroids,
      v_means,
      q_centroids,
      thresholds,
      key_sinks,
      topk_routes};
#else
  return {k_centroids, v_means, routes};
#endif
}

at::Tensor materialize_routes(
    const at::Tensor& k_centroids,
    const at::Tensor& q_centroids,
    const at::Tensor& thresholds,
    const at::Tensor& key_sinks,
    double scale_value) {
  TORCH_CHECK(k_centroids.device().is_xpu(),
              "k_centroids must be on XPU");
  TORCH_CHECK(q_centroids.device().is_xpu(),
              "q_centroids must be on XPU");
  TORCH_CHECK(thresholds.device().is_xpu(),
              "thresholds must be on XPU");
  TORCH_CHECK(key_sinks.device().is_xpu(),
              "key_sinks must be on XPU");
  TORCH_CHECK(k_centroids.scalar_type() == at::kBFloat16,
              "k_centroids must be bfloat16");
  TORCH_CHECK(q_centroids.scalar_type() == at::kFloat,
              "q_centroids must be float32");
  TORCH_CHECK(thresholds.scalar_type() == at::kFloat,
              "thresholds must be float32");
  TORCH_CHECK(key_sinks.scalar_type() == at::kByte,
              "key_sinks must be uint8");
  TORCH_CHECK(k_centroids.dim() == 4 && q_centroids.dim() == 4,
              "centroids must be BHBD tensors");
  TORCH_CHECK(k_centroids.sizes() == q_centroids.sizes(),
              "q/k centroids must share shape");
  TORCH_CHECK(k_centroids.size(3) == kHeadDim,
              "centroids must have head_dim 128");
  TORCH_CHECK(thresholds.sizes() == k_centroids.sizes().slice(0, 3),
              "thresholds must be BHB");
  TORCH_CHECK(key_sinks.sizes() == thresholds.sizes(),
              "key_sinks must match thresholds");
  TORCH_CHECK(k_centroids.is_contiguous() && q_centroids.is_contiguous() &&
                  thresholds.is_contiguous() && key_sinks.is_contiguous(),
              "route materialization inputs must be contiguous");
  TORCH_CHECK(k_centroids.device() == q_centroids.device() &&
                  k_centroids.device() == thresholds.device() &&
                  k_centroids.device() == key_sinks.device(),
              "route materialization inputs must share an XPU device");
  TORCH_CHECK(std::isfinite(scale_value), "scale must be finite");

  const int64_t batch = k_centroids.size(0);
  const int64_t heads = k_centroids.size(1);
  const int64_t blocks = k_centroids.size(2);
  TORCH_CHECK(blocks > 0, "route materialization needs a non-empty block axis");
  const float scale_log2 = static_cast<float>(scale_value) * kLog2E;
  auto routes = at::empty(
      {batch, heads, blocks, blocks}, key_sinks.options());

  const auto* kc_ptr = reinterpret_cast<const bf16*>(
      k_centroids.data_ptr());
  const auto* qc_ptr = q_centroids.data_ptr<float>();
  const auto* threshold_ptr = thresholds.data_ptr<float>();
  const auto* key_sink_ptr = key_sinks.data_ptr<uint8_t>();
  auto* route_ptr = routes.data_ptr<uint8_t>();
  auto& queue = c10::xpu::getCurrentXPUStream(
      k_centroids.device().index()).queue();

  submit(queue, batch * heads * blocks, [=](sycl::nd_item<1> item) {
    const int64_t group = item.get_group_linear_id();
    const int64_t dim = item.get_local_linear_id();
    const int64_t query_block = group % blocks;
    const int64_t batch_head = group / blocks;
    const int64_t summary_offset = batch_head * blocks * kHeadDim;
    const float q_value =
        qc_ptr[summary_offset + query_block * kHeadDim + dim];
    const float threshold = threshold_ptr[batch_head * blocks + query_block];

    for (int64_t key_block = 0; key_block < blocks; ++key_block) {
      const float partial = q_value * static_cast<float>(
          kc_ptr[summary_offset + key_block * kHeadDim + dim]);
      const float score = sycl::reduce_over_group(
          item.get_group(), partial, sycl::plus<float>()) * scale_log2;
      if (dim == 0) {
        const int64_t distance = query_block > key_block
            ? query_block - key_block
            : key_block - query_block;
        const bool exact =
            key_sink_ptr[batch_head * blocks + key_block] != 0 ||
            distance <= 1 || score > threshold;
        route_ptr[(batch_head * blocks + query_block) * blocks + key_block] =
            static_cast<uint8_t>(exact);
      }
    }
  });
  return routes;
}

}  // namespace
}  // namespace omni_xpu_sol_attn

TORCH_LIBRARY_FRAGMENT(omni_xpu_sol_attn, m) {
#if SOL_ATTN_INLINE_ROUTE
  m.def(
      "prepare(Tensor q, Tensor k, Tensor v, float scale, float tau, "
      "int sink_start, int sink_end, int sink_q_start, int sink_q_end, "
      "int topk_count, Tensor block_len) "
      "-> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
#else
  m.def(
      "prepare(Tensor q, Tensor k, Tensor v, float scale, float tau, "
      "int sink_start, int sink_end, int sink_q_start, int sink_q_end, "
      "int topk_count, Tensor block_len) "
      "-> (Tensor, Tensor, Tensor)");
#endif
  m.def(
      "materialize_routes(Tensor k_centroids, Tensor q_centroids, "
      "Tensor thresholds, Tensor key_sinks, float scale) -> Tensor");
}

TORCH_LIBRARY_IMPL(omni_xpu_sol_attn, XPU, m) {
  m.impl("prepare", &omni_xpu_sol_attn::prepare);
  m.impl("materialize_routes", &omni_xpu_sol_attn::materialize_routes);
}
