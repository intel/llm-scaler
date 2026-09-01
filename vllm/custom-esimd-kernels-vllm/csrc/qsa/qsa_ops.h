// Copyright 2026
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ATen/ATen.h>

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
