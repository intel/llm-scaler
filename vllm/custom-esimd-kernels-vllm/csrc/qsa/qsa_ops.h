// Copyright 2026
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ATen/ATen.h>

#include <tuple>

namespace qsa {

at::Tensor select_paged_tokens(
    const at::Tensor& q,
    const at::Tensor& compressed_key_cache,
    const at::Tensor& page_table,
    const at::Tensor& token_to_req,
    const at::Tensor& query_positions,
    const at::Tensor& sequence_lengths,
    int64_t token_topk,
    int64_t compress_ratio,
    at::Tensor out);

at::Tensor select_paged_tokens_v2(
    const at::Tensor& q,
    const at::Tensor& compressed_key_cache,
    const at::Tensor& page_table,
    const at::Tensor& token_to_req,
    const at::Tensor& query_positions,
    const at::Tensor& sequence_lengths,
    int64_t token_topk,
    int64_t compress_ratio,
    int64_t compressed_page_size,
    at::Tensor out);

at::Tensor store_cache_rows_v3(
    at::Tensor cache,
    const at::Tensor& slot_mapping,
    const at::Tensor& rows);

std::tuple<at::Tensor, at::Tensor> store_cache_rows_v4(
    at::Tensor cache,
    const at::Tensor& slot_mapping,
    const at::Tensor& rows,
    at::Tensor receipt);

at::Tensor store_cache_rows_r_aware_v1(
    at::Tensor cache,
    const at::Tensor& slot_mapping,
    const at::Tensor& rows,
    bool unique_slots_proven);

at::Tensor group_compress_v1(
    const at::Tensor& raw_keys,
    const at::Tensor& raw_positions,
    const at::Tensor& compressor_state_cache,
    const at::Tensor& rope_position_cache,
    const at::Tensor& compressor_state_block_table,
    const at::Tensor& token_to_req,
    const at::Tensor& query_start_loc,
    const at::Tensor& logical_positions,
    const at::Tensor& compressed_slots,
    at::Tensor pooled,
    at::Tensor first_positions,
    int64_t compress_ratio,
    int64_t compressed_capacity,
    bool historical_ring_proven);

at::Tensor indexer_norm_rope_v1(
    const at::Tensor& input,
    at::Tensor output,
    const at::Tensor& weight,
    const at::Tensor& positions,
    const at::Tensor& cos_sin_cache,
    bool mrope,
    bool positions_bounds_proven);

}  // namespace qsa

at::Tensor sparse_paged_attention_v2(
    const at::Tensor& q,
    const at::Tensor& k_cache,
    const at::Tensor& v_cache,
    const at::Tensor& logical_indices,
    const at::Tensor& block_table,
    const at::Tensor& token_to_req,
    int64_t main_page_size,
    at::Tensor out);

at::Tensor sparse_paged_attention_bounded_v2(
    const at::Tensor& q,
    const at::Tensor& k_cache,
    const at::Tensor& v_cache,
    const at::Tensor& logical_indices,
    const at::Tensor& block_table,
    const at::Tensor& token_to_req,
    int64_t max_valid_width,
    int64_t main_page_size,
    at::Tensor out);
