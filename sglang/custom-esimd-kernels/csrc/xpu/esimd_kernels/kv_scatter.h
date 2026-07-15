/* kv_scatter.h — Fused KV-cache scatter write (pure copy, no math).
 *
 * Replaces the XPU naive fallback in memory_pool._set_kv_buffer_impl:
 *   k_cache[indices] = k      (advanced-index scatter #1)
 *   v_cache[indices] = v      (advanced-index scatter #2)
 * with a single ESIMD launch that writes both K and V. Per decode step this
 * cuts 2 scatter launches/layer -> 1 (120 -> 60 for gemma4's 60 layers).
 *
 * Layout (both K and V share the same [T]-indexed slot map):
 *   k_ptr / v_ptr:     [T, row_dim] fp16, contiguous (per-token source rows)
 *   k_cache / v_cache: [S, row_dim] fp16 (destination pool, S = #slots)
 *   idx_ptr:           [T] int64 (out_cache_loc: destination slot per token)
 *   row_dim = kv_heads * head_dim  (gemma4: 16*256=4096 SWA, 4*512=2048 full)
 *
 * Grid: T WGs, 1 thread each (one token per work-group), mirroring
 * fused_add_rms_norm_batched.h. Pure block_load->block_store fp16 copy, so
 * the write is bit-identical to the advanced-index scatter it replaces.
 */

#pragma once
#include "utils.h"

template<int VL>
struct KVScatter_kernel {
    const fp16*    k_ptr;      // [T, row_dim]
    const fp16*    v_ptr;      // [T, row_dim]
    fp16*          k_cache;    // [S, row_dim]
    fp16*          v_cache;    // [S, row_dim]
    const int64_t* idx_ptr;    // [T]
    int T;
    int row_dim;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        int t = item.get_group(0);
        if (t >= T) return;

        int64_t dst = idx_ptr[t];
        const fp16* ks = k_ptr + (int64_t)t * row_dim;
        const fp16* vs = v_ptr + (int64_t)t * row_dim;
        fp16*       kd = k_cache + dst * (int64_t)row_dim;
        fp16*       vd = v_cache + dst * (int64_t)row_dim;

        int n_chunks = row_dim / VL;
        for (int c = 0; c < n_chunks; c++) {
            int o = c * VL;
            block_store<fp16, VL>(kd + o, block_load<fp16, VL>(ks + o));
            block_store<fp16, VL>(vd + o, block_load<fp16, VL>(vs + o));
        }
    }
};

inline void kv_scatter_host(
    const fp16* k_ptr,
    const fp16* v_ptr,
    fp16* k_cache,
    fp16* v_cache,
    const int64_t* idx_ptr,
    int T,
    int row_dim,
    sycl::queue& q)
{
    if (T <= 0) return;

    #define LAUNCH_KVS(V)                                                   \
        q.submit([&](sycl::handler& cgh) {                                  \
            cgh.parallel_for(                                               \
                sycl::nd_range<1>({(size_t)T}, {1}),                        \
                KVScatter_kernel<V>{                                        \
                    k_ptr, v_ptr, k_cache, v_cache, idx_ptr, T, row_dim});  \
        });

    if      (row_dim % 512 == 0) { LAUNCH_KVS(512) }
    else if (row_dim % 256 == 0) { LAUNCH_KVS(256) }
    else if (row_dim % 128 == 0) { LAUNCH_KVS(128) }
    else if (row_dim % 64  == 0) { LAUNCH_KVS(64)  }
    else                         { LAUNCH_KVS(32)  }

    #undef LAUNCH_KVS
}
