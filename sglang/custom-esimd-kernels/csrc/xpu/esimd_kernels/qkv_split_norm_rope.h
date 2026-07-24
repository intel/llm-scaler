#pragma once
#include "utils.h"
#include <cmath>

// Fused QKV Split + RMSNorm + RoPE kernel
// Adapted from qkv.split/qkv.split.h for project conventions.
//
// Operations per head:
//   Q heads: RMSNorm(weight+1.0, eps=1e-6) → RoPE(theta=10M)
//   K heads: RMSNorm(weight+1.0, eps=1e-6) → RoPE(theta=10M)
//   G heads (gating): copy only
//   V heads: copy only
//
// Work decomposition: 2D dispatch (totalHeads, nTokens), one WG per (head, token)
//   With gating:    totalHeads = 2*qHead + 2*kvHead  (Q,G interleaved)
//   Without gating: totalHeads = qHead + 2*kvHead

#define QKV_LOCATION_Q 0
#define QKV_LOCATION_G 1
#define QKV_LOCATION_K 2
#define QKV_LOCATION_V 3

ESIMD_INLINE void qkv_split_norm_rope_kernel(
    uint8_t* qkvState,
    uint8_t* qState,
    uint8_t* gateState,
    uint8_t* kState,
    uint8_t* vState,
    uint8_t* normWq,
    uint8_t* normWk,
    uint32_t* ropePos,
    fp16* ropeCosSinCache,  // [max_pos, rotaryDim] fp16 — first half cos, second half sin
    uint32_t ntoks,
    uint32_t hiddenDim,
    uint32_t headDim,
    uint32_t qHead,
    uint32_t kvHead,
    uint32_t rotaryDim,
    sycl::nd_item<2>& ndi) {

    constexpr float eps = 1e-6f;
    uint32_t rotaryHalf = rotaryDim / 2;

    int32_t headIdx = ndi.get_group(0);
    int32_t tokIdx  = ndi.get_group(1);

    uint32_t outHead = headIdx;
    uint32_t hdimQkv  = headDim * (qHead + 2 * kvHead);
    uint32_t hdimQgkv = headDim * (2 * qHead + 2 * kvHead);
    uint32_t outputGate = (hiddenDim == hdimQgkv) ? 1 : 0;
    uint32_t whereAmI = QKV_LOCATION_Q;

    uint32_t inputOffset = tokIdx * hiddenDim + headIdx * headDim;
    uint32_t outputOffset;

    uint32_t i32RopeCoord = ropePos[tokIdx];
    float fp32RopeCoord = (float)i32RopeCoord;

    simd<fp16, 256> activation;

    if (headDim != 256) {
        return;
    }

    // Load 256 fp16 from QKV buffer
    activation = block_load<fp16, 256>((fp16*)qkvState + inputOffset);

    // Determine which output this head maps to
    if (outputGate == 1) {
        // With gating: heads are [Q0,G0, Q1,G1, ..., K0, K1, ..., V0, V1, ...]
        if ((uint32_t)headIdx < 2 * qHead) {
            whereAmI = headIdx & 0x1;  // 0=Q, 1=G
            outHead = headIdx >> 1;
        } else if ((uint32_t)headIdx < (2 * qHead + kvHead)) {
            whereAmI = QKV_LOCATION_K;
            outHead = headIdx - 2 * qHead;
        } else {
            whereAmI = QKV_LOCATION_V;
            outHead = headIdx - 2 * qHead - kvHead;
        }
    } else {
        // Without gating: heads are [Q0, Q1, ..., K0, K1, ..., V0, V1, ...]
        if ((uint32_t)headIdx < qHead) {
            whereAmI = QKV_LOCATION_Q;
            outHead = headIdx;
        } else if ((uint32_t)headIdx < (qHead + kvHead)) {
            whereAmI = QKV_LOCATION_K;
            outHead = headIdx - qHead;
        } else {
            whereAmI = QKV_LOCATION_V;
            outHead = headIdx - qHead - kvHead;
        }
    }

    if (whereAmI == QKV_LOCATION_Q) {
        // RMSNorm + partial RoPE for Q
        simd<fp16, 256> fp16RmsWeights;
        simd<float, 256> fp32RmsWeights;
        simd<float, 256> outputTemp;

        outputOffset = qHead * headDim * tokIdx + outHead * headDim;
        outputTemp = activation;
        simd<float, 256> outputSq = outputTemp * outputTemp;

        // RMSNorm: x * (weight + 1.0) / rms
        fp16RmsWeights = block_load<fp16, 256>((fp16*)normWq);
        float acc = sycl::ext::intel::esimd::detail::sum<float, float, 256>(outputSq) / (float)headDim;
        float scale = __ESIMD_NS::rsqrt(acc + eps);
        fp32RmsWeights = fp16RmsWeights + 1.0f;
        outputTemp = outputTemp * fp32RmsWeights;
        outputTemp.select<256, 1>(0) = outputTemp.select<256, 1>(0) * scale;

        // RoPE: read from rotary_emb.cos_sin_cache [max_pos, rotaryDim]
        // Layout: [cos(rotaryHalf), sin(rotaryHalf)] per row
        {
            uint32_t rH = rotaryDim / 2;
            // Row offset in fp16 elements: position * rotaryDim
            uint32_t row_offset = i32RopeCoord * rotaryDim;
            if (rotaryDim == 64) {
                // Load 64 fp16: [cos(32), sin(32)]
                simd<fp16, 64> cs16;
                cs16.select<32, 1>(0) = block_load<fp16, 32>(ropeCosSinCache + row_offset);
                cs16.select<32, 1>(32) = block_load<fp16, 32>(ropeCosSinCache + row_offset + 32);
                simd<float, 32> pcos = cs16.select<32, 1>(0);
                simd<float, 32> psin = cs16.select<32, 1>(32);

                simd<float, 32> x1 = outputTemp.select<32, 1>(0);
                simd<float, 32> x2 = outputTemp.select<32, 1>(32);
                outputTemp.select<32, 1>(0)  = x1 * pcos - x2 * psin;
                outputTemp.select<32, 1>(32) = x2 * pcos + x1 * psin;
            } else {
                // Full rotation: load 256 fp16
                simd<float, 128> fcos, fsin;
#pragma unroll
                for (int kk = 0; kk < 4; kk++) {
                    simd<fp16, 32> c16 = block_load<fp16, 32>(ropeCosSinCache + row_offset + 32 * kk);
                    fcos.select<32, 1>(32 * kk) = c16;
                }
#pragma unroll
                for (int kk = 0; kk < 4; kk++) {
                    simd<fp16, 32> s16 = block_load<fp16, 32>(ropeCosSinCache + row_offset + 128 + 32 * kk);
                    fsin.select<32, 1>(32 * kk) = s16;
                }

                simd<float, 128> x1 = outputTemp.select<128, 1>(0);
                simd<float, 128> x2 = outputTemp.select<128, 1>(128);
                outputTemp.select<128, 1>(0)   = x1 * fcos - x2 * fsin;
                outputTemp.select<128, 1>(128) = x2 * fcos + x1 * fsin;
            }
        }

        activation = outputTemp;
        block_store<fp16, 256>((fp16*)qState + outputOffset, activation);
    }
    else if (whereAmI == QKV_LOCATION_G) {
        // Gate: sigmoid(x) = 1 / (1 + e^(-x))
        outputOffset = qHead * headDim * tokIdx + outHead * headDim;
        simd<float, 256> gateTemp = activation;
        gateTemp = 1.0f / (1.0f + exp<float, 256>(-gateTemp));
        activation = gateTemp;
        block_store<fp16, 256>((fp16*)gateState + outputOffset, activation);
    }
    else if (whereAmI == QKV_LOCATION_K) {
        // RMSNorm + partial RoPE for K
        simd<fp16, 256> fp16RmsWeights;
        simd<float, 256> fp32RmsWeights;
        simd<float, 256> outputTemp;

        outputOffset = kvHead * headDim * tokIdx + outHead * headDim;
        outputTemp = activation;
        simd<float, 256> kOutputSq = outputTemp * outputTemp;

        fp16RmsWeights = block_load<fp16, 256>((fp16*)normWk);
        float acc = sycl::ext::intel::esimd::detail::sum<float, float, 256>(kOutputSq) / (float)headDim;
        float scale = __ESIMD_NS::rsqrt(acc + eps);
        fp32RmsWeights = fp16RmsWeights + 1.0f;
        outputTemp = outputTemp * fp32RmsWeights;
        outputTemp.select<256, 1>(0) = outputTemp.select<256, 1>(0) * scale;

        // RoPE: read from cos_sin_cache (same as Q)
        {
            uint32_t krow_offset = i32RopeCoord * rotaryDim;
            if (rotaryDim == 64) {
                simd<fp16, 64> kcs16;
                kcs16.select<32, 1>(0) = block_load<fp16, 32>(ropeCosSinCache + krow_offset);
                kcs16.select<32, 1>(32) = block_load<fp16, 32>(ropeCosSinCache + krow_offset + 32);
                simd<float, 32> kpcos = kcs16.select<32, 1>(0);
                simd<float, 32> kpsin = kcs16.select<32, 1>(32);

                simd<float, 32> kx1 = outputTemp.select<32, 1>(0);
                simd<float, 32> kx2 = outputTemp.select<32, 1>(32);
                outputTemp.select<32, 1>(0)  = kx1 * kpcos - kx2 * kpsin;
                outputTemp.select<32, 1>(32) = kx2 * kpcos + kx1 * kpsin;
            } else {
                simd<float, 128> kfcos, kfsin;
#pragma unroll
                for (int kk = 0; kk < 4; kk++) {
                    simd<fp16, 32> kc16 = block_load<fp16, 32>(ropeCosSinCache + krow_offset + 32 * kk);
                    kfcos.select<32, 1>(32 * kk) = kc16;
                }
#pragma unroll
                for (int kk = 0; kk < 4; kk++) {
                    simd<fp16, 32> ks16 = block_load<fp16, 32>(ropeCosSinCache + krow_offset + 128 + 32 * kk);
                    kfsin.select<32, 1>(32 * kk) = ks16;
                }

                simd<float, 128> kx1 = outputTemp.select<128, 1>(0);
                simd<float, 128> kx2 = outputTemp.select<128, 1>(128);
                outputTemp.select<128, 1>(0)   = kx1 * kfcos - kx2 * kfsin;
                outputTemp.select<128, 1>(128) = kx2 * kfcos + kx1 * kfsin;
            }
        }

        activation = outputTemp;
        block_store<fp16, 256>((fp16*)kState + outputOffset, activation);
    }
    else if (whereAmI == QKV_LOCATION_V) {
        // V: copy only
        outputOffset = kvHead * headDim * tokIdx + outHead * headDim;
        block_store<fp16, 256>((fp16*)vState + outputOffset, activation);
    }
}

// Host dispatcher: submits the 2D kernel to the SYCL queue
inline void qkv_split_norm_rope_host(
    uint8_t* qkvState,
    uint8_t* qState,
    uint8_t* gateState,
    uint8_t* kState,
    uint8_t* vState,
    uint8_t* normWq,
    uint8_t* normWk,
    uint32_t* ropePos,
    fp16* ropeCosSinCache,
    uint32_t ntoks,
    uint32_t hiddenDim,
    uint32_t qHead,
    uint32_t kvHead,
    bool attnOutputGate,
    uint32_t rotaryDim,
    sycl::queue& q) {

    constexpr uint32_t headDim = 256;
    uint32_t totalHeads = attnOutputGate
        ? (2 * qHead + 2 * kvHead)
        : (qHead + 2 * kvHead);

    sycl::range<2> globalRange(totalHeads, ntoks);
    sycl::range<2> localRange(1, 1);

    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<2>(globalRange, localRange),
            [=](sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL {
                qkv_split_norm_rope_kernel(
                    qkvState, qState, gateState, kState, vState,
                    normWq, normWk, ropePos, ropeCosSinCache,
                    ntoks, hiddenDim, headDim, qHead, kvHead, rotaryDim, ndi);
            });
    });
}
