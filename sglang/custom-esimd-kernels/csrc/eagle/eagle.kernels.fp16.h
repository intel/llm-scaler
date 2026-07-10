
ESIMD_INLINE void causalConv1dUpdateFp16(
  uint8_t* qkvzState,
  uint8_t* zOut,
  uint8_t* convW,
  uint8_t* convB,
  uint8_t* convState,
  uint32_t* convStateIdx,
  uint32_t* acceptedTokens,
  uint32_t nTokens,
  uint32_t headQk,
  uint32_t headV,
  uint32_t headDim,
  uint32_t hiddenDim,
  uint32_t conv_stride0,
  sycl::nd_item<2>& ndi) {
  constexpr uint32_t baseOffsetInc16[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
  constexpr int32_t convKernelSize = 4;
  constexpr float inv128 = 1.0f / 128.0f;
  constexpr float eps = 1e-6;
  uint32_t totalHeads = headQk * 2 + headV;
  uint32_t convDim = totalHeads * headDim;
  uint32_t cachedConvStates = convKernelSize - 1 + nTokens - 1;
  uint32_t qkDim = headDim * headQk;
  uint32_t vzDim = headDim * headV;
  uint32_t hsPerBatchSz = hiddenDim * nTokens;
  uint32_t csPerBatchSz = conv_stride0;
  uint32_t headIdx = ndi.get_group(0);
  constexpr uint32_t tokenIdx = 0;
  uint32_t batchIdx = ndi.get_group(1);
  uint32_t convStateOffset = convStateIdx[batchIdx * nTokens];
  uint32_t updateTokens = acceptedTokens[batchIdx];
  uint32_t offsetB;
  uint32_t offsetW;
  uint32_t blockAddress = headIdx >> 2;
  uint32_t whereAmI = headIdx & 0x3; // 0: q, 1: k, 2~3: v
  uint32_t qkvBase = tokenIdx * hiddenDim + blockAddress * 6 * headDim + whereAmI * headDim + batchIdx * nTokens * hiddenDim;
  uint32_t zBase = qkvBase + 2 * headDim;
  uint32_t offsetCs = csPerBatchSz * convStateOffset + tokenIdx * convDim;
  uint32_t flatHead;

  if (whereAmI < 2) {
    flatHead = whereAmI * headQk;
    flatHead = flatHead + blockAddress;
  }
  else {
    flatHead = whereAmI - 2;
    flatHead = flatHead + blockAddress * 2;
    flatHead = flatHead + 2 * headQk;
  }

  offsetCs = offsetCs + flatHead * headDim;
  offsetB = flatHead * headDim;
  offsetW = convKernelSize * flatHead * headDim;

  simd<fp16, 6 * 128> fp16InputHistoric;
  simd<fp16, 7 * 128> fp16InputCurrent;
  simd<fp16, 4 * 128> fp16Weight;
  simd<float, 4 * 128> fp32Output;
  simd<float, 128> fp32W;
  simd<float, 128> fp32In;
  {
    simd<fp16, 128> fp16Bias = 0;
    if (nullptr != convB) {
      fp16Bias.select<128, 1>(0) = block_load<fp16, 128>((fp16*)convB + offsetB);
    }
#pragma unroll
    for (int32_t kk = 0; kk < 4; kk++) {
      fp32Output.select<128, 1>(128 * kk) = fp16Bias;
    }
  }
  fp16Weight = block_load<fp16, 512>((fp16*)convW + offsetW);
#pragma unroll
  for (int32_t kk = 0; kk < 6; kk++) {
    if (kk < cachedConvStates) {
      fp16InputHistoric.select<128, 1>(128 * kk) = block_load<fp16, 128>((fp16*)convState + offsetCs + convDim * kk);
    }
    else {
      fp16InputHistoric.select<128, 1>(128 * kk) = 0;
    }
  }
#pragma unroll
  for (int32_t kk = 0; kk < 4; kk++) {
    if (kk < nTokens) {
      fp16InputCurrent.select<128, 1>(128 * kk + 128 * 3) = block_load<fp16, 128>((fp16*)qkvzState + qkvBase + hiddenDim * kk);
    }
    else {
      fp16InputCurrent.select<128, 1>(128 * kk + 128 * 3) = 0;
    }
  }

  if (updateTokens == 1) {
    fp16InputCurrent.select<128 * 3, 1>(0) = fp16InputHistoric.select<128 * 3, 1>(0 * 128);
  }
  else if (updateTokens == 2) {
    fp16InputCurrent.select<128 * 3, 1>(0) = fp16InputHistoric.select<128 * 3, 1>(1 * 128);
  }
  else if (updateTokens == 3) {
    fp16InputCurrent.select<128 * 3, 1>(0) = fp16InputHistoric.select<128 * 3, 1>(2 * 128);
  }
  else if (updateTokens == 4) {
    fp16InputCurrent.select<128 * 3, 1>(0) = fp16InputHistoric.select<128 * 3, 1>(3 * 128);
  }

#pragma unroll
  for (int32_t cc = 0; cc < convKernelSize; cc++) {
    fp32W = fp16Weight.select<128, 4>(cc);
#pragma unroll
    for (int32_t kk = 0; kk < 4; kk++) {
      fp32In = fp16InputCurrent.select<128, 1>(128 * (cc + kk));
      fp32Output.select<128, 1>(128 * kk) = fp32Output.select<128, 1>(128 * kk) + fp32In * fp32W;
    }
  }

  // The rolling of conv state:
  //
  // Before forward, the conv_state is:
  // [history1, history2, ..., historyM].
  //
  // After forward, the conv_state becomes:
  // [history2, ..., historyM, draft1, draft2, ..., draftN].
  //
  // After acceptance, it becomes:
  //
  // - accept 1 tokens: [history2, ..., historyM, draft1]
  // - accept 2 tokens: [history3, ..., historyM, draft1, draft2]
  // - accept 3 tokens: [history4, ..., historyM, draft1, draft2, draft3]
  // - accept 4 tokens: [history5, ..., historyM, draft1, draft2, draft3, draft4]
  // - and so on.
  if (updateTokens == 1) {
#pragma unroll
    for (int32_t kk = 0; kk < convKernelSize - 2; kk++) {
      fp16InputHistoric.select<128, 1>(128 * kk) = fp16InputHistoric.select<128, 1>(128 * kk + 128);
    }
  }
  else if (updateTokens == 2) {
#pragma unroll
    for (int32_t kk = 0; kk < convKernelSize - 2; kk++) {
      fp16InputHistoric.select<128, 1>(128 * kk) = fp16InputHistoric.select<128, 1>(128 * kk + 128 * 2);
    }
  }
  else if (updateTokens == 3) {
#pragma unroll
    for (int32_t kk = 0; kk < convKernelSize - 2; kk++) {
      fp16InputHistoric.select<128, 1>(128 * kk) = fp16InputHistoric.select<128, 1>(128 * kk + 128 * 3);
    }
  }
  else if (updateTokens == 4) {
#pragma unroll
    for (int32_t kk = 0; kk < convKernelSize - 2; kk++) {
      fp16InputHistoric.select<128, 1>(128 * kk) = fp16InputHistoric.select<128, 1>(128 * kk + 128 * 4);
    }
  }

  fp16InputHistoric.select<128 * 4, 1>(128 * 2) = fp16InputCurrent.select<128 * 4, 1>(128 * 3);
#pragma unroll
  for (int32_t kk = 0; kk < 6; kk++) {
    if (kk < cachedConvStates) {
      block_store<fp16, 128>((fp16*)convState + offsetCs + convDim * kk, fp16InputHistoric.select<128, 1>(128 * kk));
    }
  }

  // silu
#pragma unroll
  for (int32_t kk = 0; kk < 4; kk++) {
    simd<float, 128> siluT = fp32Output.select<128, 1>(128 * kk) * (-1.0f);
    siluT = exp(siluT);
    siluT = siluT + 1.0f;
    siluT = 1.0f / siluT;
    fp32Output.select<128, 1>(128 * kk) = fp32Output.select<128, 1>(128 * kk) * siluT;
  }

  // norm QK
  if (whereAmI < 2) {
#pragma unroll
    for (int32_t bb = 0; bb < 4; bb++) {
      fp32In = fp32Output.select<128, 1>(128 * bb) * fp32Output.select<128, 1>(128 * bb);
      float acc = sycl::ext::intel::esimd::detail::sum<float, float, 128>(fp32In);
      float scale = __ESIMD_NS::rsqrt(acc + eps);
      fp32Output.select<128, 1>(128 * bb) = fp32Output.select<128, 1>(128 * bb) * scale;
    }
  }
  else { // copy z
#pragma unroll
    for (int32_t kk = 0; kk < 4; kk++) {
      if (kk < nTokens) {
        fp16Weight.select<128, 1>(128 * kk) = block_load<fp16, 128>((fp16*)qkvzState + zBase + hiddenDim * kk);
      }
    }
  }

  simd<fp16, 4 * 128> fp16Output;
  fp16Output = fp32Output;
#pragma unroll
  for (int32_t bb = 0; bb < 4; bb++) {
    if (bb < nTokens) {
      block_store<fp16, 128>((fp16*)qkvzState + qkvBase + bb * hiddenDim, fp16Output.select<128, 1>(128 * bb));
    }
  }
  if (whereAmI >= 2) {
    uint32_t zvDim = headV * headDim;
    zBase = tokenIdx * zvDim + blockAddress * 2 * headDim + (whereAmI - 2) * headDim + batchIdx * nTokens * zvDim;
    for (int32_t kk = 0; kk < 4; kk++) {
      if (kk < nTokens) {
        block_store<fp16, 128>((fp16*)zOut + zBase + kk * zvDim, fp16Weight.select<128, 1>(128 * kk));
      }
    }
  }
}

ESIMD_INLINE void gdnRecurFp16(
  uint8_t* qkvzState,
  uint8_t* baState,
  fp16* aLogW,
  fp16* dtBias,
  uint32_t* acceptedTokens,
  uint32_t* ssmStateIdx,
  uint8_t* lastRecurState,
  uint8_t* outState,
  uint32_t nTokens,
  uint32_t headQk,
  uint32_t headV,
  uint32_t headDim,
  uint32_t qkvz_stride0,
  uint32_t ssm_stride0,
  sycl::nd_item<2>& ndi) {
  constexpr uint32_t baseOffsetInc16[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
  constexpr float softPlusBeta = 1.0f;
  constexpr float softPlusInvBeta = 1.0f / softPlusBeta;
  constexpr float softPlusThr = 20.0f;
  constexpr float rsqrt128 = 0.08838834764831844f; // 1.0f / sqrt(128.0f);
  constexpr uint32_t slmSize = 16 * 128 * sizeof(float) + 128 * sizeof(float);
  constexpr uint32_t slmOffsetDelta = 16 * 128 * sizeof(float);
  __ESIMD_NS::slm_init(slmSize);

  uint32_t qkDim = headDim * headQk;
  uint32_t vzDim = headDim * headV;
  uint32_t hiddenDim = qkDim * 2 + vzDim * 2;
  uint32_t headIdx = ndi.get_group(0);
  uint32_t hh = ndi.get_local_id(0);
  uint32_t bs = ndi.get_group(1);
  uint32_t headGroup = headIdx >> 1;
  uint32_t headSubGroup = headIdx & 0x1;
  uint32_t acceptedId = acceptedTokens[bs];
  uint32_t* ssmIdxPtr = ssmStateIdx + bs * nTokens;
  simd<uint32_t, 16> simdOffsetsQk(baseOffsetInc16);
  simd<uint32_t, 32> simdOffsetsRecur;
  simd<uint32_t, 16> simdOffsetsV;
  simd<uint32_t, 16> simdOffsetsBa;
  simd<uint32_t, 16> simdOffsetsOut;
  if (acceptedId < 1) {
    return;
  }
  acceptedId = acceptedId - 1;
  uint32_t initRecurState = ssmIdxPtr[acceptedId];
  uint32_t offsetRecurStateBase;

  simd<fp16, 8 * 4> fp16Q;
  simd<fp16, 8 * 4> fp16K;
  simd<float, 8 * 4> fp32Gk;
  simd<float, 8 * 4> fp32K;
  simd<float, 8 * 4> fp32Q;

  simd<fp16, 128 * 4> fp16V;
  simd<fp16, 128 * 8> fp16InS;
  simd_mask<16> mask;
  simd<fp16, 16> fp16A;
  simd<fp16, 16>  fp16B;
  simd<float, 16> fp32A;
  simd<float, 16> fp32B;

  fp16 fp16Alog = aLogW[headIdx];
  fp16 fp16DtBias = dtBias[headIdx];
  float fp32Alog;
  float fp32DtBias;

  mask = simdOffsetsQk < nTokens;
  simdOffsetsRecur.select<16, 1>(0) = simdOffsetsQk;
  simdOffsetsRecur.select<16, 1>(16) = simdOffsetsQk + 16;
  simdOffsetsQk.merge(0, simdOffsetsQk >= nTokens);
  simdOffsetsV = simdOffsetsQk;
  simdOffsetsBa = simdOffsetsQk;
  simdOffsetsOut = simdOffsetsQk;
  simdOffsetsQk = simdOffsetsQk * hiddenDim + headGroup * 6 * headDim + bs * nTokens * hiddenDim;
  simdOffsetsV = simdOffsetsV * hiddenDim + headGroup * 6 * headDim + headSubGroup * headDim + 2 * headDim + bs * nTokens * hiddenDim;
  simdOffsetsBa = simdOffsetsBa * 2 * headV * sizeof(fp16) + headGroup * 4 * sizeof(fp16) + headSubGroup * sizeof(fp16) + bs * nTokens * 2 * headV * sizeof(fp16);
  simdOffsetsOut = simdOffsetsOut * vzDim + headIdx * headDim + bs * nTokens * vzDim;
  offsetRecurStateBase = headIdx * headDim * headDim * sizeof(fp16) + hh * 8 * sizeof(fp16) + initRecurState * ssm_stride0 * sizeof(fp16);
  simdOffsetsRecur.select<32, 1>(0) = simdOffsetsRecur.select<32, 1>(0) * 128 * sizeof(fp16);
  fp16A =
    __ESIMD_ENS::lsc_gather<
    fp16,
    1,
    __ESIMD_ENS::lsc_data_size::u16,
    __ESIMD_ENS::cache_hint::cached,
    __ESIMD_ENS::cache_hint::cached,
    16,
    uint32_t
    >((fp16*)baState, simdOffsetsBa + 2 * sizeof(fp16));

  if (hh == 0) {
#pragma unroll
    for (int32_t nn = 0; nn < 4; nn++) {
      fp16V.select<128, 1>(128 * nn) = block_load<fp16, 128>((fp16*)qkvzState + simdOffsetsV[nn]);
    }
    fp16B =
      __ESIMD_ENS::lsc_gather<
      fp16,
      1,
      __ESIMD_ENS::lsc_data_size::u16,
      __ESIMD_ENS::cache_hint::cached,
      __ESIMD_ENS::cache_hint::cached,
      16,
      uint32_t
      >((fp16*)baState, simdOffsetsBa);
  }

#pragma unroll
  for (int32_t kk = 0; kk < 4; kk++) {
    fp16InS.template bit_cast_view<uint32_t>().select<128, 1>(128 * kk) =
      __ESIMD_ENS::lsc_gather<
      uint32_t,
      4,
      __ESIMD_ENS::lsc_data_size::u32,
      __ESIMD_ENS::cache_hint::cached,
      __ESIMD_ENS::cache_hint::cached,
      32,
      uint32_t
      >((uint32_t*)lastRecurState, simdOffsetsRecur + offsetRecurStateBase + kk * 32 * 128 * sizeof(fp16));
  }

#pragma unroll
  for (int32_t nn = 0; nn < 4; nn++) {
    fp16Q.select<8, 1>(8 * nn) = block_load<fp16, 8>((fp16*)qkvzState + simdOffsetsQk[nn] + hh * 8);
    fp16K.select<8, 1>(8 * nn) = block_load<fp16, 8>((fp16*)qkvzState + simdOffsetsQk[nn] + headDim + hh * 8);
  }

  {
    simd<float, 16> softPlusTemp;
    fp32A = fp16A;
    fp32DtBias = fp16DtBias;
    fp32A.select<16, 1>(0) = fp32A.select<16, 1>(0) + fp32DtBias;

    softPlusTemp = exp(fp32A * softPlusBeta);
    softPlusTemp = softPlusInvBeta * __ESIMD_NS::log(softPlusTemp + 1.0f);
    fp32A.merge(softPlusTemp, fp32A <= softPlusThr);

    fp32Alog = fp16Alog;
    fp32Alog = exp(fp32Alog) * (-1.0f);
    fp32A.select<16, 1>(0) = fp32A.select<16, 1>(0) * fp32Alog;
    fp32A = exp(fp32A);
  }

  {
    fp32B = fp16B;
    fp32B = fp32B * (-1.0f);
    fp32B = exp(fp32B) + 1.0f;
    fp32B = 1.0f / fp32B;
  }

  fp32Gk = fp16K;
  fp32K = fp32Gk;
  fp32Q = fp16Q;
  fp32Q.select<32, 1>(0) = fp32Q.select<32, 1>(0) * rsqrt128;

#pragma unroll
  for (int32_t nn = 0; nn < 4; nn++) {
    fp32Gk.select<8, 1>(8 * nn) = fp32Gk.select<8, 1>(8 * nn) * fp32A[nn];
  }

#pragma unroll
  for (int32_t nn = 0; nn < 4; nn++) {
    if (nn < nTokens) {
      uint32_t storeRecurStateOffset = ssmIdxPtr[nn];
      simd<float, 128> kvMemTemp = 0.0f;
#pragma unroll
      for (int32_t kk = 0; kk < 4; kk++) {
#pragma unroll
        for (int32_t kkk = 0; kkk < 2; kkk++) {
          simd<float, 128> tempState1;
          tempState1.select<32, 1>(0 * 32) = fp16InS.select<32, 2>(64 * kk + 256 * 0 + kkk);
          tempState1.select<32, 1>(1 * 32) = fp16InS.select<32, 2>(64 * kk + 256 * 1 + kkk);
          tempState1.select<32, 1>(2 * 32) = fp16InS.select<32, 2>(64 * kk + 256 * 2 + kkk);
          tempState1.select<32, 1>(3 * 32) = fp16InS.select<32, 2>(64 * kk + 256 * 3 + kkk);
          kvMemTemp.select<128, 1>(0) = kvMemTemp.select<128, 1>(0) + tempState1.select<128, 1>(0) * fp32Gk[nn * 8 + 2 * kk + kkk];
        }
      }
      slm_block_store<float, 128>(hh * 128 * sizeof(float), kvMemTemp);
      barrier();

      if (hh == 0) {
        simd<float, 128> tempState1;
        simd<float, 128> kvMemTemp;

        kvMemTemp = slm_block_load<float, 128>(0 * 128 * sizeof(float));
#pragma unroll
        for (int32_t kk = 1; kk < 16; kk++) {
          tempState1 = slm_block_load<float, 128>(kk * 128 * sizeof(float));
          kvMemTemp = kvMemTemp + tempState1;
        }

        tempState1 = fp16V.select<128, 1>(128 * nn);
        tempState1 = tempState1 - kvMemTemp;

        tempState1.select<128, 1>(0) = tempState1.select<128, 1>(0) * fp32B[nn];
        slm_block_store<float, 128>(slmOffsetDelta, tempState1);
      }

      barrier();
      {
        simd<float, 128> fp32Delta;
        simd<float, 128> fp32CurrSum = 0.0f;
        fp32Delta = slm_block_load<float, 128>(slmOffsetDelta);
        uint32_t outputOffset = simdOffsetsOut[nn];

#pragma unroll
        for (int32_t kk = 0; kk < 4; kk++) {
#pragma unroll
          for (int32_t kkk = 0; kkk < 2; kkk++) {
            simd<float, 128> tempState1;
            tempState1.select<32, 1>(0 * 32) = fp16InS.select<32, 2>(64 * kk + 256 * 0 + kkk);
            tempState1.select<32, 1>(1 * 32) = fp16InS.select<32, 2>(64 * kk + 256 * 1 + kkk);
            tempState1.select<32, 1>(2 * 32) = fp16InS.select<32, 2>(64 * kk + 256 * 2 + kkk);
            tempState1.select<32, 1>(3 * 32) = fp16InS.select<32, 2>(64 * kk + 256 * 3 + kkk);
            tempState1.select<128, 1>(0) = tempState1.select<128, 1>(0) * fp32A[nn];
            tempState1.select<128, 1>(0) = tempState1.select<128, 1>(0) + fp32Delta.select<128, 1>(0) * fp32K[8 * nn + 2 * kk + kkk];
            fp16InS.select<32, 2>(64 * kk + 256 * 0 + kkk) = tempState1.select<32, 1>(0 * 32);
            fp16InS.select<32, 2>(64 * kk + 256 * 1 + kkk) = tempState1.select<32, 1>(1 * 32);
            fp16InS.select<32, 2>(64 * kk + 256 * 2 + kkk) = tempState1.select<32, 1>(2 * 32);
            fp16InS.select<32, 2>(64 * kk + 256 * 3 + kkk) = tempState1.select<32, 1>(3 * 32);
            fp32CurrSum.select<128, 1>(0) = fp32CurrSum.select<128, 1>(0) + tempState1.select<128, 1>(0) * fp32Q[nn * 8 + 2 * kk + kkk];
          }
        }

        slm_block_store<float, 128>(hh * 128 * sizeof(float), fp32CurrSum);
      }
      barrier();
      if (hh == 0) {
        simd<float, 128> fp32OutTemp;
        simd<float, 128> tempState1;

        fp32OutTemp = slm_block_load<float, 128>(0 * 128 * sizeof(float));
#pragma unroll
        for (int32_t kk = 1; kk < 16; kk++) {
          tempState1 = slm_block_load<float, 128>(kk * 128 * sizeof(float));
          fp32OutTemp = fp32OutTemp + tempState1;
        }
        simd<fp16, 128> fp16Out = fp32OutTemp;
        block_store<fp16, 128>((fp16*)outState + simdOffsetsOut[nn], fp16Out);
      }
      barrier();

      storeRecurStateOffset = storeRecurStateOffset * ssm_stride0;
      storeRecurStateOffset = storeRecurStateOffset * sizeof(fp16) + headIdx * headDim * headDim * sizeof(fp16) + hh * 8 * sizeof(fp16);
#pragma unroll
      for (int32_t kk = 0; kk < 4; kk++) {
        __ESIMD_ENS::lsc_scatter<
          uint32_t,
          4,
          __ESIMD_ENS::lsc_data_size::u32,
          __ESIMD_ENS::cache_hint::write_back,
          __ESIMD_ENS::cache_hint::write_back,
          32,
          uint32_t
        >((uint32_t*)lastRecurState, simdOffsetsRecur + storeRecurStateOffset + kk * 32 * 128 * sizeof(fp16), fp16InS.bit_cast_view<uint32_t>().select<128, 1>(128 * kk));
      }
    }
  }
}
