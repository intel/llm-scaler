// Copyright 2026
// SPDX-License-Identifier: Apache-2.0
//
// Production Sol-Attn policy admitted by the BMG kernel and canonical
// workflow gates. Experimental alternatives remain in omni-xpu-kernel-tuning.
#pragma once

#if !defined(OMNI_XPU_ARCH_BMG) || defined(OMNI_XPU_ARCH_PTL_H)
#error "The packaged Sol-Attn kernel is currently validated for BMG only"
#endif

#define SOL_ATTN_Q_TILE 256
#define SOL_ATTN_SUBGROUP_LAYOUT_Q 32
#define SOL_ATTN_GRF_SIZE 256

// Physical Arc B580 keeps the same route/cache policy but uses the smaller
// query tile admitted by the local multi-shape and real-activation gates.
// Other BMG products retain the packaged default above.
#define SOL_ATTN_B580_Q_TILE 128
#define SOL_ATTN_B580_SUBGROUP_LAYOUT_Q 16
#define SOL_ATTN_B580_GRF_SIZE 256

#define SOL_ATTN_INLINE_ROUTE 1
#define SOL_ATTN_SHARED_INLINE_ROUTE 1
#define SOL_ATTN_PREFETCH_ROUTED_KV 1
#define SOL_ATTN_DOUBLE_BUFFER_ROUTE_MASKS 1
#define SOL_ATTN_PARALLEL_SHARED_INLINE_ROUTE 1
#define SOL_ATTN_GROUP_LOAD_ROUTE_K_CENTROID 1

#define SOL_ATTN_BMG_CACHEABLE_EXACT_KV_LOADS 1
#define SOL_ATTN_HOIST_APPROXIMATE_LOG2 1
#define SOL_ATTN_BMG_CACHEABLE_SUMMARY_KV_LOADS 1
