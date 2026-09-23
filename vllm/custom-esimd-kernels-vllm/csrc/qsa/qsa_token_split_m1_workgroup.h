// SPDX-License-Identifier: Apache-2.0
// Included inside qsa_token_split_candidate. Preserve the 33-partial ABI;
// each workgroup cooperates on one original 64-slot interval.
template <int QueryHeads, int PageSize, int Workers>
class QsaTokenSplitM1Workgroup;

template <int QueryHeads, int PageSize, int Workers = 4>
void phase0_m1_workgroup(
    sycl::queue& queue, const fp16* q_in, const fp16* packed_kv,
    const int* logical_indices, const int* block_table,
    const int* token_to_req, int block_table_stride, int block_table_rows,
    int physical_pages, int partial_head_stride, fp32* partials) {
  static_assert(64 % Workers == 0);
  constexpr int Parts = (SLOT_COUNT + 63) / 64;
  constexpr int LocalStride = 272; // 64B-aligned independent SLM records.
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<QsaTokenSplitM1Workgroup<QueryHeads, PageSize, Workers>>(
        sycl::nd_range<1>(QueryHeads * Parts * Workers, Workers),
        [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
          slm_init<Workers * LocalStride * sizeof(float)>();
          const int group = item.get_group(0);
          const int worker = item.get_local_id(0);
          const int head = group / Parts;
          const int partial = group % Parts;
          const int start = partial * 64 + worker * (64 / Workers);
          const int request = token_to_req[0];
          const bool request_ok = request >= 0 && request < block_table_rows;
          const int safe_request = request_ok ? request : 0;
          simd<float, 256> q = block_load<fp16, 256>(q_in + head * HD);
          simd<float, 256> accum(0.f);
          float maximum = NEG_SENTINEL, denominator = 0.f;
          for (int j = 0; j < 64 / Workers; ++j) {
            const int slot = start + j;
            const int logical = logical_indices[slot < SLOT_COUNT ? slot : SLOT_COUNT - 1];
            const bool slot_ok = slot < SLOT_COUNT && logical >= 0 && request_ok;
            const int safe_logical = slot_ok ? logical : 0;
            const int page = safe_logical / PageSize;
            const bool page_ok = page < block_table_stride;
            const int physical = block_table[safe_request * block_table_stride + (page_ok ? page : 0)];
            const bool physical_ok = physical >= 0 && physical < physical_pages;
            const bool valid = slot_ok && page_ok && physical_ok;
            const fp16* kv = packed_kv + static_cast<int64_t>(physical_ok ? physical : 0) * PageSize * 512
                + (safe_logical % PageSize) * 512;
            simd<float, 256> key = block_load<fp16, 256>(kv);
            float score = sycl::ext::intel::esimd::detail::sum<float, float, 256>(q * key) * INV_SQRT_HD;
            simd<float, 1> score_lane(score);
            score_lane.merge(simd<float, 1>(NEG_SENTINEL), simd_mask<1>(!valid));
            score = score_lane[0];
            float next = __ESIMD_NS::max<float, 1, float>(simd<float, 1>(maximum), score_lane)[0];
            float correction = sycl::exp2((maximum - next) * LOG2E);
            simd<float, 1> weight(sycl::exp2((score - next) * LOG2E));
            weight.merge(simd<float, 1>(0.f), simd_mask<1>(!valid));
            simd<float, 256> value = block_load<fp16, 256>(kv + HD);
            value.merge(simd<float, 256>(0.f), simd_mask<256>(!valid));
            const float weight_value = weight[0];
            accum = accum * correction + value * weight_value;
            denominator = denominator * correction + weight_value;
            maximum = next;
          }
          const uint32_t offset = worker * LocalStride * sizeof(float);
          slm_block_store<float, 256>(offset, accum);
          simd<float, 2> stats;
          stats[0] = maximum; stats[1] = denominator;
          slm_block_store<float, 2>(offset + HD * sizeof(float), stats);
          barrier();
          if (worker == 0) {
            float max_all = NEG_SENTINEL;
            simd<float, Workers> maxima, sums;
            #pragma unroll
            for (int w = 0; w < Workers; ++w) {
              simd<float, 2> s = slm_block_load<float, 2>((w * LocalStride + HD) * sizeof(float));
              maxima[w] = s[0]; sums[w] = s[1];
              max_all = __ESIMD_NS::max<float, 1, float>(simd<float,1>(max_all), simd<float,1>(s[0]))[0];
            }
            simd<float, 256> merged(0.f);
            float sum = 0.f;
            #pragma unroll
            for (int w = 0; w < Workers; ++w) {
              const float local_max = maxima[w];
              const float local_sum = sums[w];
              const float factor = sycl::exp2((local_max - max_all) * LOG2E);
              merged += slm_block_load<float, 256>(w * LocalStride * sizeof(float)) * factor;
              sum += local_sum * factor;
            }
            fp32* dst = partials + head * partial_head_stride + partial * PARTIAL_STRIDE;
            dst[0] = max_all; dst[1] = sum;
            block_store<float, 256>(dst + 2, merged);
          }
        });
  });
}

template <int QueryHeads>
class QsaTokenSplitM1ParallelMerge;

// Four independent output tiles per head. Gather all 33 scalar statistics at
// once, rather than serially loading/expanding them in a single work-item.
template <int QueryHeads>
void phase1_m1_parallel(sycl::queue& queue, fp16* output,
                       const fp32* partials, int partial_head_stride) {
  constexpr int Parts = (SLOT_COUNT + 63) / 64;
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<QsaTokenSplitM1ParallelMerge<QueryHeads>>(
        sycl::nd_range<1>(QueryHeads * 4, 1),
        [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
          const int head = item.get_global_id(0) / 4;
          const int tile = item.get_global_id(0) % 4;
          const float* src = partials + head * partial_head_stride;
          simd<uint32_t, 64> indices(0, 1);
          simd_mask<64> valid = indices < Parts;
          simd<uint32_t, 64> offsets = indices * (PARTIAL_STRIDE * sizeof(float));
          offsets.merge(0, !valid); // remain in-bounds even in padded lanes
          simd<float, 64> maxima = gather<float, 64>(src, offsets);
          simd<float, 64> sums = gather<float, 64>(src + 1, offsets);
          maxima.merge(simd<float, 64>(NEG_SENTINEL), !valid);
          sums.merge(simd<float, 64>(0.f), !valid);
          const float maximum = hmax<float>(maxima);
          simd<float, 64> factors = __ESIMD_NS::exp2<float>((maxima - maximum) * LOG2E);
          factors.merge(simd<float, 64>(0.f), !valid);
          float denominator = sycl::ext::intel::esimd::detail::sum<float, float, 64>(sums * factors);
          simd<float, 64> accum(0.f);
          for (int p = 0; p < Parts; ++p) {
            const float factor = factors[p];
            accum += block_load<float, 64>(src + p * PARTIAL_STRIDE + 2 + tile * 64) * factor;
          }
          const float inv = denominator > 0.f ? 1.f / denominator : 0.f;
          block_store<fp16, 64>(output + head * HD + tile * 64,
                               convert<fp16>(accum * inv));
        });
  });
}
