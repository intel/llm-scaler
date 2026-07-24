// SPDX-License-Identifier: BSD-3-Clause
//
// sglang_decode_attn.h
// =====================
// 2-phase split-K decode SDPA for sglang's flat NHD KV-cache layout.
// Targets Qwen3.5-MoE on Intel BMG: head_dim=256, GQA, fp16/bf16 KV.
//
// Layout:
//   q          : [batches, num_q_heads, 256]            fp16
//   k_buf      : [n_slots,  num_kv_heads, 256]          fp16/bf16
//   v_buf      : [n_slots,  num_kv_heads, 256]          fp16/bf16
//   kv_indptr  : [batches+1]                            int32 (CSR)
//   kv_indices : [sum_seq_lens]                         int32 (slot ids)
//   out        : [batches, num_q_heads, 256]            fp16
//
// Phase 1: Each work-item handles one (batch, q_head, split) where
//   split partitions kv-seq into chunks of SPLIT_TILE tokens.
//   For each split it produces (m_part, l_part, o_part[D]) = local online-softmax stats.
//
// Phase 2: Each work-item handles one (batch, q_head). It reads up to N_SPLITS
//   partials, merges them via online softmax, normalises, writes fp16 out.
//
// Numerics: fp32 accumulate everywhere, online softmax for numerical stability.
// Radix-cache friendly: every token is addressed via kv_indices[kv_start+i] —
// non-contiguous slots are handled correctly.
//
#pragma once

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/esimd/memory.hpp>

#include <cstdint>

namespace sgl_esimd {

namespace esimd = sycl::ext::intel::esimd;
namespace xesimd = sycl::ext::intel::experimental::esimd;
using fp16 = sycl::half;

// Compile-time tile/split parameters.
constexpr uint32_t HEAD_DIM    = 256;
constexpr uint32_t SPLIT_TILE  = 64;   // tokens per phase-1 work-item
constexpr uint32_t MAX_SPLITS  = 256;  // phase-2 reduction cap (== max seq / SPLIT_TILE)

// Phase 1 — partial softmax stats over [base, base + SPLIT_TILE) of one head.
template <typename KvT>
SYCL_ESIMD_FUNCTION inline void
sglang_decode_attn_phase1_impl(
    const fp16* q_ptr,                  // [B, Hq, D]
    const KvT*  k_buf,                  // [n_slots, Hkv, D]
    const KvT*  v_buf,                  // [n_slots, Hkv, D]
    const uint32_t* kv_indptr,          // [B+1]
    const uint32_t* kv_indices,         // [sum_lens]
    float* m_part,                      // [B, Hq, S]
    float* l_part,                      // [B, Hq, S]
    float* o_part,                      // [B, Hq, S, D]
    uint32_t batch,
    uint32_t q_head,
    uint32_t split,                     // index along kv-seq splits
    uint32_t Hq,
    uint32_t Hkv,
    uint32_t gqa_ratio,
    uint32_t n_splits,                  // S
    float    sm_scale)
{
    const uint32_t kv_head  = q_head / gqa_ratio;
    const uint32_t kv_start = kv_indptr[batch];
    const uint32_t kv_end   = kv_indptr[batch + 1];
    const uint32_t kv_len   = kv_end - kv_start;

    const uint32_t base   = split * SPLIT_TILE;
    if (base >= kv_len) {
        // Empty split: write neutral state.
        const size_t  hidx  = (batch * Hq + q_head) * (size_t)n_splits + split;
        m_part[hidx] = -1e30f;
        l_part[hidx] = 0.f;
        // o_part left untouched; phase 2 will multiply by 0 sum anyway.
        esimd::simd<float, HEAD_DIM> zero(0.f);
        float* op = o_part + hidx * HEAD_DIM;
        #pragma unroll
        for (uint32_t i = 0; i < HEAD_DIM / 16; ++i) {
            esimd::block_store<float, 16>(op + i * 16,
                zero.template select<16, 1>(i * 16));
        }
        return;
    }
    const uint32_t actual = ((kv_len - base) < SPLIT_TILE)
        ? (kv_len - base) : SPLIT_TILE;

    // Load q for this head once.
    esimd::simd<fp16, HEAD_DIM> q_vec;
    {
        const fp16* q_row = q_ptr + (batch * Hq + q_head) * HEAD_DIM;
        #pragma unroll
        for (uint32_t i = 0; i < HEAD_DIM / 16; ++i) {
            q_vec.template select<16, 1>(i * 16) =
                esimd::block_load<fp16, 16>(q_row + i * 16);
        }
    }

    float    m_local = -1e30f;
    float    l_local = 0.f;
    esimd::simd<float, HEAD_DIM> o_local(0.f);

    // Walk SPLIT_TILE tokens. (Sequential within each work-item; many work-items
    // run in parallel so HW occupancy is still good.)
    for (uint32_t t = 0; t < actual; ++t) {
        const uint32_t slot = kv_indices[kv_start + base + t];
        const KvT* k_row = k_buf + (size_t)slot * Hkv * HEAD_DIM
                                   + (size_t)kv_head * HEAD_DIM;

        esimd::simd<float, HEAD_DIM> qk;
        if constexpr (std::is_same_v<KvT, fp16>) {
            esimd::simd<fp16, HEAD_DIM> kv;
            #pragma unroll
            for (uint32_t i = 0; i < HEAD_DIM / 16; ++i) {
                kv.template select<16, 1>(i * 16) =
                    esimd::block_load<fp16, 16>(k_row + i * 16);
            }
            qk = esimd::convert<float>(q_vec * kv);
        } else if constexpr (std::is_same_v<KvT, sycl::ext::oneapi::bfloat16>) {
            esimd::simd<sycl::ext::oneapi::bfloat16, HEAD_DIM> kv;
            #pragma unroll
            for (uint32_t i = 0; i < HEAD_DIM / 16; ++i) {
                kv.template select<16, 1>(i * 16) =
                    esimd::block_load<sycl::ext::oneapi::bfloat16, 16>(k_row + i * 16);
            }
            esimd::simd<float, HEAD_DIM> kv_f = esimd::convert<float>(kv);
            esimd::simd<float, HEAD_DIM> q_f  = esimd::convert<float>(q_vec);
            qk = q_f * kv_f;
        } else {
            qk = 0.f;
        }
        float dot = esimd::detail::sum<float, float, HEAD_DIM>(qk);
        float logit = dot * sm_scale;

        // Online softmax update.
        float new_m = (m_local > logit) ? m_local : logit;
        float scale_old;
        {
            esimd::simd<float, 1> tmp(m_local - new_m);
            scale_old = esimd::exp(tmp)[0];
        }
        float p;
        {
            esimd::simd<float, 1> tmp(logit - new_m);
            p = esimd::exp(tmp)[0];
        }

        // Scale prior accumulator and add this token's contribution.
        o_local = o_local * scale_old;
        l_local = l_local * scale_old + p;

        const KvT* v_row = v_buf + (size_t)slot * Hkv * HEAD_DIM
                                   + (size_t)kv_head * HEAD_DIM;
        esimd::simd<float, HEAD_DIM> v_f;
        if constexpr (std::is_same_v<KvT, fp16>) {
            esimd::simd<fp16, HEAD_DIM> v_vec;
            #pragma unroll
            for (uint32_t i = 0; i < HEAD_DIM / 16; ++i) {
                v_vec.template select<16, 1>(i * 16) =
                    esimd::block_load<fp16, 16>(v_row + i * 16);
            }
            v_f = esimd::convert<float>(v_vec);
        } else if constexpr (std::is_same_v<KvT, sycl::ext::oneapi::bfloat16>) {
            esimd::simd<sycl::ext::oneapi::bfloat16, HEAD_DIM> v_vec;
            #pragma unroll
            for (uint32_t i = 0; i < HEAD_DIM / 16; ++i) {
                v_vec.template select<16, 1>(i * 16) =
                    esimd::block_load<sycl::ext::oneapi::bfloat16, 16>(v_row + i * 16);
            }
            v_f = esimd::convert<float>(v_vec);
        }
        o_local = o_local + v_f * p;
        m_local = new_m;
    }

    // Store partial state for phase 2.
    const size_t  hidx = (batch * Hq + q_head) * (size_t)n_splits + split;
    m_part[hidx] = m_local;
    l_part[hidx] = l_local;
    float* op = o_part + hidx * HEAD_DIM;
    #pragma unroll
    for (uint32_t i = 0; i < HEAD_DIM / 16; ++i) {
        esimd::block_store<float, 16>(op + i * 16,
            o_local.template select<16, 1>(i * 16));
    }
}

// Phase 2 — merge n_splits partials per (batch, q_head), normalise, write fp16.
SYCL_ESIMD_FUNCTION inline void
sglang_decode_attn_phase2_impl(
    const float* m_part,                // [B, Hq, S]
    const float* l_part,                // [B, Hq, S]
    const float* o_part,                // [B, Hq, S, D]
    fp16*  out_ptr,                     // [B, Hq, D]
    uint32_t batch,
    uint32_t q_head,
    uint32_t Hq,
    uint32_t n_splits)
{
    const size_t head_base = (batch * Hq + q_head) * (size_t)n_splits;

    float    m_running = -1e30f;
    float    l_running = 0.f;
    esimd::simd<float, HEAD_DIM> o_acc(0.f);

    for (uint32_t s = 0; s < n_splits; ++s) {
        float m_s = m_part[head_base + s];
        float l_s = l_part[head_base + s];
        if (l_s <= 0.f) continue;  // empty split

        float new_m = (m_running > m_s) ? m_running : m_s;
        float scale_old, scale_new;
        {
            esimd::simd<float, 1> tmp(m_running - new_m);
            scale_old = esimd::exp(tmp)[0];
        }
        {
            esimd::simd<float, 1> tmp(m_s - new_m);
            scale_new = esimd::exp(tmp)[0];
        }

        // Load partial output for split s.
        esimd::simd<float, HEAD_DIM> o_s;
        const float* op = o_part + (head_base + s) * HEAD_DIM;
        #pragma unroll
        for (uint32_t i = 0; i < HEAD_DIM / 16; ++i) {
            o_s.template select<16, 1>(i * 16) =
                esimd::block_load<float, 16>(op + i * 16);
        }

        o_acc = o_acc * scale_old + o_s * scale_new;
        l_running = l_running * scale_old + l_s * scale_new;
        m_running = new_m;
    }

    if (l_running > 0.f) {
        o_acc = o_acc * (1.f / l_running);
    }
    esimd::simd<fp16, HEAD_DIM> o_fp16 = esimd::convert<fp16>(o_acc);
    fp16* o_row = out_ptr + (batch * Hq + q_head) * HEAD_DIM;
    #pragma unroll
    for (uint32_t i = 0; i < HEAD_DIM / 16; ++i) {
        esimd::block_store<fp16, 16>(o_row + i * 16,
            o_fp16.template select<16, 1>(i * 16));
    }
}

}  // namespace sgl_esimd
