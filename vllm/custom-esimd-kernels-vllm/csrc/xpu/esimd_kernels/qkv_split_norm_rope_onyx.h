#pragma once
#include "utils.h"
#include <cmath>

// Fused QKV Split + parameterless RMSNorm + RoPE + q_scale
// for the Onyx (day0) architecture:
//   - head_dim = 128
//   - qk_norm is parameterless (`x / rms(x)` — no weight)
//   - after norm on Q: multiply by a q_scale scalar (q_scale_factor / sqrt(head_dim))
//   - RoPE supports both interleaved-pair and half-split (NEOX) layouts.
//   - V heads: copy only (no norm, no RoPE)
//   - no gate head (Onyx's output_gate_proj is a separate ColumnParallelLinear
//     driven from hidden_states, not shared with QKV)
//
// Work decomposition: 2D dispatch (totalHeads, nTokens), one WG per (head, token)
//   totalHeads = qHead + 2 * kvHead
//
// cos_sin_cache layout: [max_pos, rotary_dim] fp16, per row = concat(cos(half), sin(half))
// with rotary_dim == head_dim == 128, half = 64. Kernel loads 64 cos then 64 sin
// per token position from the cache.

#define ONYX_QKV_LOC_Q 0
#define ONYX_QKV_LOC_K 1
#define ONYX_QKV_LOC_V 2

template <typename PositionT>
ESIMD_INLINE void qkv_split_norm_rope_onyx_kernel(
    uint8_t* qkvState,
    uint8_t* qState,
    uint8_t* kState,
    uint8_t* vState,
    PositionT* ropePos,
    fp16* ropeCosSinCache,   // [max_pos, 128] fp16 (cos[64] || sin[64])
    uint32_t hiddenDim,
    uint32_t qHead,
    uint32_t kvHead,
    float qScale,            // Onyx: qk_scale_factor / sqrt(head_dim)
    float eps,
    bool neoxStyle,
    sycl::nd_item<2>& ndi) {

    constexpr uint32_t headDim = 128;
    constexpr uint32_t halfDim = 64;

    int32_t headIdx = ndi.get_group(0);
    int32_t tokIdx  = ndi.get_group(1);

    // Head partitioning: [Q0..Qq-1, K0..Kk-1, V0..Vk-1]
    uint32_t outHead;
    uint32_t whereAmI;
    if ((uint32_t)headIdx < qHead) {
        whereAmI = ONYX_QKV_LOC_Q;
        outHead = headIdx;
    } else if ((uint32_t)headIdx < qHead + kvHead) {
        whereAmI = ONYX_QKV_LOC_K;
        outHead = headIdx - qHead;
    } else {
        whereAmI = ONYX_QKV_LOC_V;
        outHead = headIdx - qHead - kvHead;
    }

    uint32_t inputOffset = tokIdx * hiddenDim + headIdx * headDim;
    simd<fp16, 128> activation = block_load<fp16, 128>((fp16*)qkvState + inputOffset);

    if (whereAmI == ONYX_QKV_LOC_V) {
        // V: pure copy
        uint32_t outputOffset = kvHead * headDim * tokIdx + outHead * headDim;
        block_store<fp16, 128>((fp16*)vState + outputOffset, activation);
        return;
    }

    // Q or K: parameterless RMSNorm then interleaved-pair RoPE
    simd<float, 128> outputTemp = activation;
    simd<float, 128> outputSq = outputTemp * outputTemp;
    float acc = sycl::ext::intel::esimd::detail::sum<float, float, 128>(outputSq) / (float)headDim;
    float scale = __ESIMD_NS::rsqrt(acc + eps);
    outputTemp = outputTemp * scale;

    if (whereAmI == ONYX_QKV_LOC_Q) {
        // Fold q_scale scalar into normed Q before RoPE (RoPE is a rotation,
        // linear in x, so folding scale before or after is equivalent).
        outputTemp = outputTemp * qScale;
    }

    // The cache is [cos[64] || sin[64]] for both layouts.
    uint32_t rowOffset = static_cast<uint32_t>(ropePos[tokIdx]) * headDim;
    // cos: [rowOffset .. +64), sin: [rowOffset+64 .. +128)
    simd<fp16, 64> cos16;
    simd<fp16, 64> sin16;
    cos16.select<32, 1>(0)  = block_load<fp16, 32>(ropeCosSinCache + rowOffset);
    cos16.select<32, 1>(32) = block_load<fp16, 32>(ropeCosSinCache + rowOffset + 32);
    sin16.select<32, 1>(0)  = block_load<fp16, 32>(ropeCosSinCache + rowOffset + 64);
    sin16.select<32, 1>(32) = block_load<fp16, 32>(ropeCosSinCache + rowOffset + 96);
    simd<float, 64> fcos = cos16;
    simd<float, 64> fsin = sin16;

    if (neoxStyle) {
        simd<float, 64> x1 = outputTemp.select<64, 1>(0);
        simd<float, 64> x2 = outputTemp.select<64, 1>(64);
        simd<float, 64> o1 = x1 * fcos - x2 * fsin;
        simd<float, 64> o2 = x2 * fcos + x1 * fsin;
        outputTemp.select<64, 1>(0) = o1;
        outputTemp.select<64, 1>(64) = o2;
    } else {
        simd<float, 64> x1 = outputTemp.select<64, 2>(0);
        simd<float, 64> x2 = outputTemp.select<64, 2>(1);
        simd<float, 64> o1 = x1 * fcos - x2 * fsin;
        simd<float, 64> o2 = x2 * fcos + x1 * fsin;
        outputTemp.select<64, 2>(0) = o1;
        outputTemp.select<64, 2>(1) = o2;
    }

    activation = outputTemp;
    if (whereAmI == ONYX_QKV_LOC_Q) {
        uint32_t outputOffset = qHead * headDim * tokIdx + outHead * headDim;
        block_store<fp16, 128>((fp16*)qState + outputOffset, activation);
    } else {
        uint32_t outputOffset = kvHead * headDim * tokIdx + outHead * headDim;
        block_store<fp16, 128>((fp16*)kState + outputOffset, activation);
    }
}

template <typename PositionT>
inline void qkv_split_norm_rope_onyx_host(
    uint8_t* qkvState,
    uint8_t* qState,
    uint8_t* kState,
    uint8_t* vState,
    PositionT* ropePos,
    fp16* ropeCosSinCache,
    uint32_t ntoks,
    uint32_t hiddenDim,
    uint32_t qHead,
    uint32_t kvHead,
    float qScale,
    float eps,
    bool neoxStyle,
    sycl::queue& q) {

    uint32_t totalHeads = qHead + 2 * kvHead;
    sycl::range<2> globalRange(totalHeads, ntoks);
    sycl::range<2> localRange(1, 1);
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<2>(globalRange, localRange),
            [=](sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL {
                qkv_split_norm_rope_onyx_kernel(
                    qkvState, qState, kState, vState, ropePos, ropeCosSinCache,
                    hiddenDim, qHead, kvHead, qScale, eps, neoxStyle, ndi);
            });
    });
}
