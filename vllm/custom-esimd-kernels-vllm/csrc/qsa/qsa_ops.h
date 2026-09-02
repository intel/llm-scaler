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

}  // namespace qsa
