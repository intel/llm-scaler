#pragma once

#include <cstdint>

#include "utils.h"

namespace qwen38_ngram_embedding {

constexpr int kRows = 16;
constexpr int kRowWidth = 160;
constexpr int kBlockWidth = 32;
constexpr int kBlocksPerRow = kRowWidth / kBlockWidth;
constexpr int kDwordBlockWidth = 16;

class QwenNg2aGather16x5PackedDword;

inline void launch(
    sycl::queue& q,
    const std::int64_t* ngram_ids,
    const sycl::half* local_weight,
    const std::int64_t* local_vocab_start,
    const std::int64_t* local_num_rows,
    sycl::half* local_partial) {
  q.submit([&](sycl::handler& h) {
    h.parallel_for<QwenNg2aGather16x5PackedDword>(
        sycl::range<2>(kRows, kBlocksPerRow),
        [=](sycl::id<2> item) SYCL_ESIMD_KERNEL {
          using namespace sycl::ext::intel::esimd;

          const int row = static_cast<int>(item[0]);
          const int block = static_cast<int>(item[1]);
          const std::int64_t global_id = ngram_ids[row];
          const std::int64_t shard_start = local_vocab_start[0];
          const std::int64_t shard_end = shard_start + local_num_rows[0];
          const bool valid =
              (global_id >= shard_start) && (global_id < shard_end);

          const simd_mask<1> valid_mask(valid);
          simd<std::int64_t, 1> local_index(global_id - shard_start);
          local_index.merge(simd<std::int64_t, 1>(0), !valid_mask);

          const std::int64_t row_base = local_index[0] * kRowWidth;
          const int block_offset = block * kBlockWidth;
          const sycl::half* src = local_weight + row_base + block_offset;
          sycl::half* dst = local_partial + row * kRowWidth + block_offset;

          // 32 FP16 values occupy 16 dwords. This preserves every FP16 bit.
          const auto* src_dword =
              reinterpret_cast<const std::uint32_t*>(src);
          auto* dst_dword = reinterpret_cast<std::uint32_t*>(dst);
          const simd<std::uint32_t, kDwordBlockWidth> zero(0u);
          const simd<std::uint32_t, kDwordBlockWidth> values =
              block_load<std::uint32_t, kDwordBlockWidth>(
                  src_dword, valid_mask, zero);
          block_store<std::uint32_t, kDwordBlockWidth>(dst_dword, values);
        });
  });
}

}  // namespace qwen38_ngram_embedding
