// FP8 KV-cache variant of page-attention decode (eagle_ops.page_attn_decode).
//
// Two FP8 layouts are supported via a tag:
//   - Fp8Variant::E5M2    : fp16_bits = byte << 8                        (lossless)
//   - Fp8Variant::E4M3FN  : fp16_bits = sign | (exp+8)<<10 | mant<<7     (subnormal -> 0)
//
// Bytes-per-load are exactly half of the fp16 kernel: gather/2d-load drop NElts/2,
// strides switch from sizeof(fp16) to sizeof(uint8_t), so the kkCache/vvCache layout
// after dequant matches the fp16 kernel byte-for-byte and the downstream reshuffle +
// MAC stays untouched.  k_scale/v_scale are folded into matMulQuantCoeff and the
// final softmax-divide respectively, costing zero extra multiplies in the hot loop.

#pragma once

enum class Fp8Variant : uint32_t { E4M3FN = 0, E5M2 = 1 };

// Convert N fp8 bytes (in a simd<uint8_t,N>) to fp16 in-place.  Returns simd<fp16,N>.
template <Fp8Variant V, int N>
ESIMD_INLINE simd<fp16, N> dequantFp8(simd<uint8_t, N> bytes) {
  static_assert(N % 2 == 0, "dequantFp8 expects an even N");
  if constexpr (V == Fp8Variant::E5M2) {
    // bit-pattern: sign(1)|exp(5)|mant(2)  ==  fp16 high 8 bits.
    // Pack: u16 lane i = bytes[i] << 8.  Avoid u8->u16 implicit cast (sign?)
    // by writing each byte into the high half of a freshly zero-init u16
    // through a 2x-wide u8 view.
    simd<uint16_t, N> out16 = 0;
    out16.template bit_cast_view<uint8_t>().template select<N, 2>(1) = bytes;
    return out16.template bit_cast_view<fp16>();
  } else {
    // E4M3FN: sign(1)|exp(4)|mant(3).  fp16: sign|exp(5,bias15)|mant(10,bias7+8).
    //
    // Layout:
    //   in_byte = s eeee mmm
    //   fp16     = s eeeee mmmmmmmmmm  with exp = e4 + 8, mant = m << 7
    //
    // Implementation reuses two simd<u16,N> live values (`b16` and `out`)
    // instead of materialising sign16/exp4/mant3/norm separately — keeps
    // dequant pressure off the GRF file (was ~2.5KB live for N=256).
    //
    //   out = (b16 & 0x7F) << 7      // (mant<<7) | (exp4<<10)
    //   out += 0x2000                // bias bump: exp += 8 (8<<10 = 0x2000)
    //   out |= (b16 & 0x80) << 8     // sign bit
    //   if (exp4 == 0) out = sign    // subnormal/zero collapse
    simd<uint16_t, N> b16 = 0;
    b16.template bit_cast_view<uint8_t>().template select<N, 2>(0) = bytes;     // low half
    simd<uint16_t, N> out = (b16 & 0x7F) << 7;
    out += 0x2000;
    out |= (b16 & 0x80) << 8;
    // subnormal path: ((b16 & 0x78) == 0) → keep only sign bit.
    out.merge((b16 & 0x80) << 8, (b16 & 0x78) == 0);
    return out.template bit_cast_view<fp16>();
  }
}

template <Fp8Variant V>
ESIMD_INLINE void sdpaDecodeGqa4Phase1Fp8(
  uint8_t* qState,
  uint8_t* kState,
  uint8_t* pState,
  float* pGroupMax,
  float* pGlobalMax,
  uint32_t* pPollP,
  uint32_t* pageTable,
  uint32_t* batchKvSeqLen,
  uint32_t batchSize,
  uint32_t kvCacheBatchStride0,
  uint32_t kvCacheBatchStride1,
  uint32_t pageTableBatchStride,
  uint32_t pageTableSizeLog2,
  uint32_t pStride,
  uint32_t maxStride,
  uint32_t headKv,
  uint32_t gqaRatio,
  float k_scale,
  sycl::nd_item<3>& ndi) {
  constexpr uint32_t baseOffsetInc16[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
  constexpr float matMulQuantCoeffBase = 0.0625f;
  const float matMulQuantCoeff = matMulQuantCoeffBase * k_scale;     // <-- fold scale
  constexpr uint32_t headDim = 256;
  constexpr uint32_t maxPGroupOut = 128;
  constexpr uint32_t outputPerGroup = 64;
  constexpr uint32_t loopCount = outputPerGroup / 16;
  constexpr uint32_t slmSizePhase1 = 4 * maxPGroupOut * 4 * sizeof(float);
  __ESIMD_NS::slm_init<slmSizePhase1>();
  int localLinearId = ndi.get_local_id(0);
  int globalLinearId0 = ndi.get_group(0);
  int globalLinearId1 = ndi.get_group(1);
  int globalLinearId2 = ndi.get_group(2);
  int hh = localLinearId & 0x3;
  int vv = localLinearId >> 2;
  int batchIdx = globalLinearId2;
  uint32_t gqaGroups = gqaRatio / 4;
  int kvHeadIdx = globalLinearId0 / gqaGroups;
  int qGroupIdx = globalLinearId0 % gqaGroups;
  int qHeadIdx = kvHeadIdx * gqaRatio + qGroupIdx * 4;
  uint32_t pageTableSize = 1 << pageTableSizeLog2;
  uint32_t pageTableLoopMask = pageTableSize - 1;
  uint32_t kvSeqLen = batchKvSeqLen[batchIdx];
  uint32_t pageTableBase = pageTableBatchStride * batchIdx;
  simd<uint32_t, 16> simdOffsetsK;

  simd<uint32_t, 16> baseOffsetInc16AsSimd(baseOffsetInc16);
  simd<float, 64 * 4> qqFp32;
  simd<fp16, 64 * 4> qqFp16;
  simd<float, 16 * 4> ppFp32;
  simd<fp16, 16 * 64> kkCache;            // dequanted layout, identical to fp16 kernel
  simd<float, 1> ppMax;

  uint32_t headQ = headKv * gqaRatio;
  uint64_t outputOffset = qHeadIdx * pStride + globalLinearId1 * outputPerGroup + hh * pStride + batchIdx * headQ * pStride;
  uint32_t outputMaxOffset = qHeadIdx * maxStride + globalLinearId1 + hh * maxStride + batchIdx * headQ * maxStride;
  // Q is still fp16 in HBM.
  uint32_t offsetQ = qHeadIdx * headDim * sizeof(fp16) + hh * 32 * sizeof(fp16) + batchIdx * headQ * headDim * sizeof(fp16);
  uint32_t baseCoordK = globalLinearId1 * outputPerGroup;
  // K is fp8 in HBM: all element strides drop to sizeof(uint8_t)==1.
  uint32_t offsetBaseK = hh * 32 * sizeof(uint8_t) + headDim * kvHeadIdx * sizeof(uint8_t);
  uint32_t kvSeqOffset = baseCoordK;

  if (baseCoordK >= kvSeqLen) {
    return;
  }

#pragma unroll
  for (int qn = 0; qn < 4; qn++) {
#pragma unroll
    for (int qk = 0; qk < 2; qk++) {
      qqFp16.template bit_cast_view<uint8_t>().select<64, 1>(128 * qn + 64 * qk) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        64,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)qState + offsetQ + qn * headDim * sizeof(fp16) + qk * 4 * 32 * sizeof(fp16));
    }
  }

  qqFp32 = qqFp16;

#pragma unroll
  for (int loopIdx = 0; loopIdx < loopCount; loopIdx++) {
    ppFp32 = 0.0f;
    if (kvSeqOffset < kvSeqLen) {
      uint32_t pageTableIdx = kvSeqOffset >> pageTableSizeLog2;
      uint32_t pageTableOffset = kvSeqOffset & pageTableLoopMask;
      uint32_t pageIdx = pageTable[pageTableBase + pageTableIdx];
      simd<uint16_t, 16> mask;
      simd<uint16_t, 16> maskNeg;
      simd<float, 16 * 32> kkFp32;
      simd<uint32_t, 16> logicSimdOffsetK;
      logicSimdOffsetK = baseOffsetInc16AsSimd + kvSeqOffset;
      mask = logicSimdOffsetK < kvSeqLen;
      maskNeg = logicSimdOffsetK >= kvSeqLen;
      simdOffsetsK = baseOffsetInc16AsSimd;

      // ── K-load (fp8) ─────────────────────────────────────────────
      // The fp16 K-load uses lsc_gather<u32, NElts=4, simd_size=16> which
      // returns SoA layout: raw[NElt*16 + lane] = lane's NElt-th u32.  Each
      // u32 == 4 bytes == 2 fp16, so a gather yields 16 lanes × 8 fp16 =
      // 128 fp16 = 256 byte / 64 u32.  fp16 then runs 4 inner kkk × 2 outer
      // kk = 8 gathers, totalling 1024 fp16 in kkCache.
      //
      // For fp8 we keep NElts=4 (same 16 byte/lane = 16 fp8/lane), but
      // *each gather now yields 256 fp8 = 256 dequanted fp16 = 512 byte =
      // 128 u32* — exactly DOUBLE the fp16 path's per-gather payload.  To
      // keep kkCache size unchanged we therefore halve the inner kkk loop
      // (0..1 instead of 0..3) and reshuffle with a different stride.
      //
      // Byte step: inner +16 byte = +16 fp8; outer +96 byte = +96 fp8.
      // (fp16's +192 byte outer is +96 fp16, same element count.)
      simdOffsetsK =
        simdOffsetsK * headKv * headDim * sizeof(uint8_t) +
        offsetBaseK +
        pageIdx * kvCacheBatchStride1 * sizeof(uint8_t) +
        pageTableOffset * headKv * headDim * sizeof(uint8_t)
        ;

      // Pull the entire chunk's 64-col footprint with ONE gather:
      //   NElts=16 -> 16 lane * 64 byte = 1024 fp8 / gather.  But hh's two
      //   32-col segments are NOT contiguous (gap of 96 byte in the middle);
      //   so the "single 64-col gather" version is invalid.  Stay with
      //   NElts=8 × 2 segments per chunk (proven path below).
#pragma unroll
      for (int kk = 0; kk < 2; kk++) {
        // u32 NElts=8 -> 16 lane * 32 byte = 512 fp8 bytes / gather
        // (halves gather-message count vs the original NElts=4 × 2 inner).
        simd<uint32_t, 16 * 8> raw =
          __ESIMD_ENS::lsc_gather<
          uint32_t,
          8,
          __ESIMD_ENS::lsc_data_size::u32,
          __ESIMD_ENS::cache_hint::cached,
          __ESIMD_ENS::cache_hint::cached,
          16,
          uint32_t
          >((uint32_t*)kState, simdOffsetsK, mask);
        simd<uint8_t, 512> rawBytes = raw.template bit_cast_view<uint8_t>();
        // dequant in two halves: keeps per-call live u16 footprint same
        // as the proven path (avoids extra spill inside dequantFp8).
        kkCache.template select<256, 1>(512 * kk + 0)   = dequantFp8<V, 256>(rawBytes.template select<256, 1>(0));
        kkCache.template select<256, 1>(512 * kk + 256) = dequantFp8<V, 256>(rawBytes.template select<256, 1>(256));

        simdOffsetsK += 128 * sizeof(uint8_t);   // jump to next 32-col segment

        // SoA → col-major reshuffle.  s ∈ [0,8), p ∈ [0,4), c = s*4+p.
#pragma unroll
        for (int s = 0; s < 8; s++) {
#pragma unroll
          for (int p = 0; p < 4; p++) {
            int c = s * 4 + p;
            kkFp32.select<16, 1>(16 * c) =
              kkCache.template select<16, 4>(512 * kk + 64 * s + p);
          }
        }

#pragma unroll
        for (int kkk = 0; kkk < 32; kkk++) {
#pragma unroll
          for (int pn = 0; pn < 4; pn++) {
            ppFp32.select<16, 1>(16 * pn) = ppFp32.select<16, 1>(16 * pn) + kkFp32.select<16, 1>(16 * kkk) * qqFp32[64 * pn + 32 * kk + kkk];
          }
        }
      }

#pragma unroll
      for (int pn = 0; pn < 4; pn++) {
        ppFp32.select<16, 1>(pn * 16).merge(FP32_MIN, maskNeg);
      }
    }
    else {
      ppFp32 = FP32_MIN;
    }

#pragma unroll
    for (int pn = 0; pn < 4; pn++) {
      slm_block_store<float, 16>(
        loopIdx * 4 * 64 * sizeof(float) +
        pn * 64 * sizeof(float) +
        hh * 16 * sizeof(float),
        ppFp32.select<16, 1>(16 * pn));
    }
    kvSeqOffset += 16;
  }

  barrier();

  {
    simd<float, 16 * 4> ppOut;
    simd<float, 16> maxTemp;
    simd<float, 64> ppTemp;
#pragma unroll
    for (int pk = 0; pk < 4; pk++) {
      ppTemp = slm_block_load<float, 64>(pk * 64 * 4 * sizeof(float) + hh * 64 * sizeof(float));
#pragma unroll
      for (int pn = 0; pn < 1; pn++) {
        ppOut.select<16, 1>(4 * 16 * pn + 16 * pk) = ppTemp.select<16, 1>(64 * pn) + ppTemp.select<16, 1>(64 * pn + 16);
        ppOut.select<16, 1>(4 * 16 * pn + 16 * pk) = ppOut.select<16, 1>(4 * 16 * pn + 16 * pk) + ppTemp.select<16, 1>(64 * pn + 32);
        ppOut.select<16, 1>(4 * 16 * pn + 16 * pk) = ppOut.select<16, 1>(4 * 16 * pn + 16 * pk) + ppTemp.select<16, 1>(64 * pn + 48);
      }
    }

    ppOut = ppOut * matMulQuantCoeff;        // includes k_scale
    maxTemp = ppOut.select<16, 1>(0);
#pragma unroll
    for (int pk = 1; pk < 4; pk++) {
      maxTemp = __ESIMD_NS::max<float, 16, float>(maxTemp, ppOut.select<16, 1>(16 * pk));
    }

    maxTemp.select<8, 1>(0) = __ESIMD_NS::max<float, 8, float>(maxTemp.select<8, 1>(0), maxTemp.select<8, 1>(8));
    maxTemp.select<4, 1>(0) = __ESIMD_NS::max<float, 4, float>(maxTemp.select<4, 1>(0), maxTemp.select<4, 1>(4));
    maxTemp.select<2, 1>(0) = __ESIMD_NS::max<float, 2, float>(maxTemp.select<2, 1>(0), maxTemp.select<2, 1>(2));
    ppMax[0] = __ESIMD_NS::max<float>(maxTemp[0], maxTemp[1]);
    ppOut.select<64, 1>(0) = ppOut.select<64, 1>(0) - ppMax[0];
    ppOut = __ESIMD_NS::pow<float, 64, float>(2.718281828459f, ppOut);

    {
      block_store<float, 64>((float*)pState + outputOffset, ppOut.select<64, 1>(0));
      simd<float, 1> zeros = 0.0f;
      pGroupMax[outputMaxOffset] = ppMax[0];
      uint32_t atomicOffset = (batchIdx * gqaRatio * headKv + qHeadIdx + hh) * sizeof(float);
      uint32_t arrivalId =
        atomic_update<
        __ESIMD_NS::atomic_op::inc,
        uint32_t,
        1,
        uint32_t>(pPollP, atomicOffset);

      if (0 == arrivalId) {
        atomic_update<
          __ESIMD_NS::atomic_op::fcmpxchg,
          float,
          1,
          uint32_t>(pGlobalMax, atomicOffset, ppMax, zeros);
      }
      else {
        atomic_update<
          __ESIMD_NS::atomic_op::fmax,
          float,
          1,
          uint32_t>(pGlobalMax, atomicOffset, ppMax);
      }

      outputOffset += pStride;
      outputMaxOffset += maxStride;
    }
  }
}

template <Fp8Variant V>
ESIMD_INLINE void sdpaDecodeGqa4Phase2Fp8(
  uint8_t* pState,
  float* pGroupMax,
  uint8_t* vState,
  float* pGlobalMax,
  float* pTempOut,
  float* pSoftmaxSum,
  uint8_t* out,
  uint32_t* pageTable,
  uint32_t* batchKvSeqLen,
  uint32_t batchSize,
  uint32_t kvCacheBatchStride0,
  uint32_t kvCacheBatchStride1,
  uint32_t pageTableBatchStride,
  uint32_t pageTableSizeLog2,
  uint32_t pStride,
  uint32_t maxStride,
  uint32_t headKv,
  uint32_t longestBatch,
  uint32_t flag,
  uint32_t gqaRatio,
  float v_scale,
  sycl::nd_item<3>& ndi) {
  constexpr uint32_t headDim = 256;
  constexpr uint32_t maxPGroupOut = 128;
  constexpr uint32_t outputPerGroup = 64;
  constexpr float matMulQuantCoeff = 0.0625f;
  constexpr uint32_t slmSizeOut = 2 * 4 * 16 * sizeof(float);
  constexpr uint32_t slmSizeSoftmaxSum = 2 * 4 * sizeof(float);
  constexpr uint32_t slmSize = slmSizeOut + slmSizeSoftmaxSum;
  __ESIMD_NS::slm_init<slmSize>();
  constexpr uint32_t baseOffsetInc16[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
  int localLinearId = ndi.get_local_id(0);
  int globalLinearId0 = ndi.get_group(0);
  int globalLinearId1 = ndi.get_group(1);
  int globalLinearId2 = ndi.get_group(2);
  int hh = localLinearId & 0x1;
  int vv = localLinearId >> 1;
  int batchIdx = globalLinearId2;
  uint32_t gqaGroups = gqaRatio / 4;
  int kvHeadIdx = globalLinearId1 / gqaGroups;
  int qGroupIdx = globalLinearId1 % gqaGroups;
  uint32_t kvSeqLen = batchKvSeqLen[batchIdx];
  int kvSeqOutGroup = (kvSeqLen + 1023) >> 10;
  uint32_t pageTableSize = 1 << pageTableSizeLog2;
  uint32_t pageTableLoopMask = (1 << pageTableSizeLog2) - 1;
  uint32_t pageTableBase = pageTableBatchStride * batchIdx;
  uint32_t channelOffset = globalLinearId0 & 0xf;
  uint32_t groupIdx = globalLinearId0 >> 4;

  simd<uint32_t, 16> simdOffsets(baseOffsetInc16);
  simd<uint32_t, 16> pageTableIndice;
  simd<uint32_t, 16> pageTableOffsets;

  simd<uint32_t, 16> pageAlignedOffsets;
  simd<uint32_t, 16> pageOffsetsInner;
  simd<uint32_t, 16> pageNumbers;

  simd<float, 4 * 32> ppFp32;
  simd<float, 4> historicMax;
  simd<fp16, 32 * 16> vvCache;             // dequanted, same layout as fp16 kernel
  simd<float, 16 * 16> vvFp32;
  simd<float, 4 * 16> softmaxSum;
  simd<float, 4 * 16> outputFp32;
  simd<float, 32> vvScale;

  if (groupIdx >= kvSeqOutGroup) {
    return;
  }
  uint32_t reduceCount = (longestBatch + 1023) >> 10;
  uint32_t headQ = headKv * gqaRatio;
  uint32_t qBaseIdx = kvHeadIdx * gqaRatio + qGroupIdx * 4;
  uint32_t outputOffset = qBaseIdx * headDim + channelOffset * 16 + hh * 2 * headDim + batchIdx * headQ * headDim;
  uint32_t outTempOffset = qBaseIdx * headDim + channelOffset * 16 + hh * 2 * headDim + groupIdx * headQ * headDim + batchIdx * reduceCount * headQ * headDim;
  uint32_t outOffsetSoftmaxSum = qBaseIdx + hh * 2 + groupIdx * headQ + batchIdx * reduceCount * headQ;
  uint32_t offsetP = qBaseIdx * pStride + hh * 32 + groupIdx * 1024 + batchIdx * headQ * pStride;
  uint32_t offsetMax = qBaseIdx * maxStride + batchIdx * headQ * maxStride + groupIdx * 16;
  uint32_t offsetGlobalMax = batchIdx * headQ + qBaseIdx;
  // V row width / X stride switch to fp8 (1 byte/element).
  uint32_t widthV = headKv * headDim * sizeof(uint8_t) - 1;
  uint32_t heightV = pageTableSize - 1;
  uint32_t vX = channelOffset * 16 + headDim * kvHeadIdx;
  uint32_t vY = 32 * hh;
  uint32_t totalPages = (kvSeqLen + pageTableSize - 1) >> pageTableSizeLog2;
  uint32_t lastPageHeight = (kvSeqLen - 1) & (pageTableSize - 1);
  if (kvSeqLen == 0) {
    lastPageHeight = 0;
  }
  uint8_t* vPtrBase = (uint8_t*)vState + kvCacheBatchStride0;     // K|V stride still in elements

  historicMax = FP32_MIN * matMulQuantCoeff;
  softmaxSum = 0;
  outputFp32 = 0;

  simd<float, 4> globalMax;
  simd<float, 4> currMax;
  simd<float, 4> compensationP;
  simd<uint16_t, 16> mask;
  simd<uint16_t, 16> maskNeg;

  uint32_t kvSeqOffset = groupIdx * 1024;
  simdOffsets = simdOffsets * 64 + kvSeqOffset;
  mask = simdOffsets < kvSeqLen;
  maskNeg = simdOffsets >= kvSeqLen;
  pageTableIndice = (simdOffsets >> pageTableSizeLog2);
  pageTableOffsets = pageTableIndice * sizeof(uint32_t) + pageTableBase * sizeof(uint32_t);
  pageOffsetsInner = simdOffsets & pageTableLoopMask;
  pageNumbers =
    gather<
    uint32_t,
    16,
    1,
    uint32_t>(pageTable, pageTableOffsets, mask);

  pageAlignedOffsets = pageNumbers * kvCacheBatchStride1;
  pageAlignedOffsets.merge(0, maskNeg);
  globalMax = block_load<float, 4>(pGlobalMax + offsetGlobalMax);

#pragma unroll
  for (uint32_t loop = 0; loop < 16; loop++) {
    if (kvSeqOffset + loop * 64 < kvSeqLen) {
      uint8_t* currPtrV = vPtrBase + pageAlignedOffsets[loop];
      if (pageTableIndice[loop] + 1 < totalPages) {
        heightV = pageTableSize - 1;
      }
      else {
        heightV = lastPageHeight;
      }
      vY = pageOffsetsInner[loop] + 32 * hh;

#pragma unroll
      for (int pn = 0; pn < 4; pn++) {
        ppFp32.select<32, 1>(32 * pn) = block_load<float, 32>((float*)pState + offsetP + pStride * pn);
        currMax[pn] = pGroupMax[offsetMax + maxStride * pn];
      }

      // Merged 2D-load: a single 16-byte × 32-row message replaces the two
      // 16×16 loads (1024 byte / 1 message vs 512 byte × 2 messages).
      // dequant is still applied in two 256-wide halves so the live
      // simd<u16,N> footprint inside dequantFp8 stays at the same size as
      // the per-half version (avoids extra GRF spill).
      simd<uint8_t, 512> vRawAll =
        __ESIMD_ENS::lsc_load_2d<
        uint8_t,
        16,
        32,
        1,
        false,
        false,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached
        >(currPtrV, widthV, heightV, widthV, vX, vY);
      vvCache.select<256, 1>(0 * 256) = dequantFp8<V, 256>(vRawAll.select<256, 1>(0));
      vvCache.select<256, 1>(1 * 256) = dequantFp8<V, 256>(vRawAll.select<256, 1>(256));

      compensationP = currMax - globalMax;
      compensationP = exp(compensationP);

#pragma unroll
      for (int oc = 0; oc < 4; oc++) {
        ppFp32.select<32, 1>(32 * oc) = ppFp32.select<32, 1>(32 * oc) * compensationP[oc];
      }

#pragma unroll
      for (int oc = 0; oc < 4; oc++) {
#pragma unroll
        for (int pk = 0; pk < 2; pk++) {
          softmaxSum.select<16, 1>(16 * oc) = softmaxSum.select<16, 1>(16 * oc) + ppFp32.select<16, 1>(32 * oc + 16 * pk);
        }
      }

#pragma unroll
      for (int vk = 0; vk < 2; vk++) {
        vvFp32 = vvCache.select<256, 1>(256 * vk);
#pragma unroll
        for (int oc = 0; oc < 4; oc++) {
#pragma unroll
          for (int pk = 0; pk < 16; pk++) {
            outputFp32.select<16, 1>(16 * oc) += vvFp32.select<16, 1>(16 * pk) * ppFp32[32 * oc + 16 * vk + pk];
          }
        }
      }

      offsetP += 64;
      offsetMax += 1;
    }
  }

#pragma unroll
  for (int32_t kk = 0; kk < 4; kk++) {
    slm_block_store<float, 16>(hh * 16 * sizeof(float) + 32 * kk * sizeof(float), outputFp32.select<16, 1>(16 * kk));
  }

#pragma unroll
  for (int32_t kk = 0; kk < 4; kk++) {
    float slmTempSoftmaxSum;
    slmTempSoftmaxSum = __ESIMD_DNS::sum<float, float, 16>(softmaxSum.select<16, 1>(16 * kk));
    slm_block_store<float, 1>(slmSizeOut + hh * sizeof(float) + 2 * kk * sizeof(float), slmTempSoftmaxSum);
  }

  barrier();

  {
    simd<float, 4> dividor;
    simd<float, 2> softmaxMul;
    simd<float, 64> outputTempFp32;
    dividor = slm_block_load<float, 4>(slmSizeOut + hh * 4 * sizeof(float));
    softmaxMul = dividor.select<2, 2>(0) + dividor.select<2, 2>(1);
    outputTempFp32 = slm_block_load<float, 64>(hh * 4 * 16 * sizeof(float));

#pragma unroll
    for (int oc = 0; oc < 2; oc++) {
      outputFp32.select<16, 1>(16 * oc) = outputTempFp32.select<16, 1>(32 * oc) + outputTempFp32.select<16, 1>(32 * oc + 16);
    }

    if ((flag & 0x1) == 1) {
      // Final fused divide -> fold v_scale here (one extra mul, then nothing).
      softmaxMul = v_scale / softmaxMul;
#pragma unroll
      for (int oc = 0; oc < 2; oc++) {
        outputFp32.select<16, 1>(16 * oc) = outputFp32.select<16, 1>(16 * oc) * softmaxMul[oc];
      }
      simd<fp16, 32> outputTemp = outputFp32.select<32, 1>(0);
#pragma unroll
      for (int oc = 0; oc < 2; oc++) {
        block_store<fp16, 16>((fp16*)out + outputOffset + oc * headDim, outputTemp.select<16, 1>(16 * oc));
      }
    }
    else {
      // Phase3 will divide; pre-scale outputFp32 once with v_scale so phase3 stays
      // dtype-agnostic (zero changes there).
#pragma unroll
      for (int oc = 0; oc < 2; oc++) {
        outputFp32.select<16, 1>(16 * oc) = outputFp32.select<16, 1>(16 * oc) * v_scale;
      }
#pragma unroll
      for (int oc = 0; oc < 2; oc++) {
        block_store<float, 16>((float*)pTempOut + outTempOffset + oc * headDim, outputFp32.select<16, 1>(16 * oc));
      }

      block_store<float, 2>((float*)pSoftmaxSum + outOffsetSoftmaxSum, softmaxMul);
    }
  }
}
