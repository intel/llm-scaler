// Copyright 2026
// SPDX-License-Identifier: Apache-2.0
// Shared top-k threshold ordering and quantized Sol routing/pooled tail.
// Included inside sol_attn_prepare.cpp's implementation namespace.

template <bool Descending = true, typename LocalScores>
void sol_sort_descending(LocalScores scores, int64_t padded, sycl::nd_item<1> item) {
  const int64_t lane = item.get_local_linear_id();
  for (int64_t width = 2; width <= padded; width *= 2) {
    for (int64_t stride = width / 2; stride > 0; stride /= 2) {
      for (int64_t index = lane; index < padded; index += kWorkGroup) {
        const int64_t peer = index ^ stride;
        if (peer <= index) continue;
        const auto left = scores[index], right = scores[peer];
        const bool descending = ((index & width) == 0) == Descending;
        const bool swap = descending ? left < right : left > right;
        if (swap) { scores[index] = right; scores[peer] = left; }
      }
      sycl::group_barrier(item.get_group());
    }
  }
}

struct SolPreparedRouteKernel;
struct SolPreparedTailKernel;

std::vector<at::Tensor> token_group_centroids(const at::Tensor& qmean, const at::Tensor& refs) {
  TORCH_CHECK(qmean.device().is_xpu() && qmean.scalar_type() == at::kFloat &&
      qmean.dim() == 4 && qmean.size(3) == 128 && qmean.is_contiguous() &&
      qmean.size(0)>0 && qmean.size(1)>0 && qmean.size(2)>0,
      "Sol token groups require contiguous FP32 [B,H,N,128] query means on XPU");
  const int64_t B=qmean.size(0),H=qmean.size(1),N=qmean.size(2),NG=(N+1)/2;
  TORCH_CHECK(refs.device()==qmean.device() && refs.scalar_type()==at::kFloat &&
      refs.is_contiguous() && refs.sizes()==at::IntArrayRef({B,H,N}),"Sol token reference score contract mismatch");
  auto gq=at::empty({B,H,NG,128},qmean.options().dtype(at::kChar));
  auto gs=at::empty({B,H,NG},qmean.options()), gr=at::empty_like(gs);
  const float* qp=qmean.data_ptr<float>(); const float* rp=refs.data_ptr<float>();
  auto* op=gq.data_ptr<int8_t>(); auto* sp=gs.data_ptr<float>(); auto* gp=gr.data_ptr<float>();
  auto& queue=c10::xpu::getCurrentXPUStream(qmean.device().index()).queue();
  submit(queue,B*H*NG,[=](sycl::nd_item<1> item) {
    const int64_t row=item.get_group_linear_id(),d=item.get_local_linear_id(),bh=row/NG,q=(row%NG)*2;
    float mean=qp[(bh*N+q)*128+d];
    if(q+1<N) mean=(mean+qp[(bh*N+q+1)*128+d])*0.5f;
    const float amax=sycl::reduce_over_group(item.get_group(),sycl::fabs(mean),sycl::maximum<float>());
    const float scale=sycl::fmax(amax/127.0f,1e-8f);
    op[row*128+d]=sol_q8(mean,1.0f/scale);
    if(d==0) { sp[row]=scale; gp[row]=q+1<N?sycl::fmax(rp[bh*N+q],rp[bh*N+q+1]):rp[bh*N+q]; }
  });
  return {gq,gs,gr};
}

at::Tensor token_bin_cutoff(const at::Tensor& hist, int64_t budget) {
  TORCH_CHECK(hist.device().is_xpu() && hist.scalar_type()==at::kInt && hist.dim()==4 &&
      hist.size(3)==128 && hist.is_contiguous() && hist.numel()>0,
      "Sol token histogram must be contiguous int32 [B,H,NG,128] on XPU");
  TORCH_CHECK(budget>0 && budget<=256 && budget%64==0,"Sol token budget must be a multiple of 64 through 256");
  auto cutoff=at::empty(hist.sizes().slice(0,3),hist.options());
  const auto* hp=hist.data_ptr<int32_t>(); auto* cp=cutoff.data_ptr<int32_t>();
  auto& queue=c10::xpu::getCurrentXPUStream(hist.device().index()).queue();
  submit(queue,cutoff.numel(),[=](sycl::nd_item<1> item) {
    const int64_t row=item.get_group_linear_id(),bin=127-item.get_local_linear_id();
    const int32_t cumulative=sycl::inclusive_scan_over_group(item.get_group(),hp[row*128+bin],sycl::plus<int32_t>());
    const int32_t overflow=sycl::reduce_over_group(item.get_group(),cumulative>budget?int32_t(bin):-1,sycl::maximum<int32_t>());
    if(item.get_local_linear_id()==0) cp[row]=overflow+1;
  });
  return cutoff;
}

at::Tensor sort_token_indices(const at::Tensor& indices, const at::Tensor& counts) {
  TORCH_CHECK(indices.device().is_xpu() && indices.scalar_type()==at::kInt && indices.dim()==4 &&
      indices.is_contiguous() && indices.numel()>0 && indices.size(3)<=256 && indices.size(3)%64==0,
      "Sol token indices must be contiguous int32 [B,H,NG,budget] on XPU");
  TORCH_CHECK(counts.device()==indices.device() && counts.scalar_type()==at::kInt &&
      counts.is_contiguous() && counts.sizes()==indices.sizes().slice(0,3),"Sol token count contract mismatch");
  const int64_t budget=indices.size(3),rows=counts.numel();
  int64_t padded=1; while(padded<budget) padded*=2;
  auto output=at::empty_like(indices);
  const auto* ip=indices.data_ptr<int32_t>(); const auto* cp=counts.data_ptr<int32_t>();
  auto* op=output.data_ptr<int32_t>();
  auto& queue=c10::xpu::getCurrentXPUStream(indices.device().index()).queue();
  queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<int32_t,1> sorted(sycl::range<1>(padded),cgh);
    cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(rows*kWorkGroup),sycl::range<1>(kWorkGroup)),
        [=](sycl::nd_item<1> item) {
      const int64_t row=item.get_group_linear_id(),lane=item.get_local_linear_id();
      const int32_t n=sycl::clamp(cp[row],0,int32_t(budget));
      for(int64_t i=lane;i<padded;i+=kWorkGroup) sorted[i]=i<n?ip[row*budget+i]:INT32_MAX;
      sycl::group_barrier(item.get_group());
      sol_sort_descending<false>(sorted,padded,item);
      for(int64_t i=lane;i<budget;i+=kWorkGroup) op[row*budget+i]=i<n?sorted[i]:-1;
    });
  });
  return output;
}

at::Tensor merge_token_tail(const at::Tensor& pooled, const at::Tensor& grouped) {
  TORCH_CHECK(pooled.device().is_xpu() && pooled.scalar_type()==at::kFloat && pooled.dim()==4 &&
      pooled.size(3)==130 && pooled.numel()>0 && pooled.is_contiguous(),
      "Sol pooled tail must be contiguous FP32 [B,H,N,130] on XPU");
  const int64_t B=pooled.size(0),H=pooled.size(1),N=pooled.size(2),NG=(N+1)/2;
  TORCH_CHECK(grouped.device()==pooled.device() && grouped.scalar_type()==at::kFloat &&
      grouped.is_contiguous() && grouped.sizes()==at::IntArrayRef({B,H,NG,130}),"Sol grouped tail contract mismatch");
  const auto* pp=pooled.data_ptr<float>(); const auto* gp=grouped.data_ptr<float>();
  auto output=at::empty_like(pooled); auto* op=output.data_ptr<float>();
  auto& queue=c10::xpu::getCurrentXPUStream(pooled.device().index()).queue();
  submit(queue,B*H*N,[=](sycl::nd_item<1> item) {
    const int64_t row=item.get_group_linear_id(),d=item.get_local_linear_id();
    const int64_t group=(row/N)*NG+(row%N)/2;
    const float pl=pp[row*130+1],gl=gp[group*130+1];
    const float maximum=sycl::fmax(pl>0?pp[row*130]:-INFINITY,gl>0?gp[group*130]:-INFINITY);
    const float pf=pl>0?sycl::exp2(pp[row*130]-maximum):0;
    const float gf=gl>0?sycl::exp2(gp[group*130]-maximum):0;
    op[row*130+2+d]=pp[row*130+2+d]*pf+gp[group*130+2+d]*gf;
    if(d==0) { op[row*130]=maximum; op[row*130+1]=pl*pf+gl*gf; }
  });
  return output;
}

std::vector<at::Tensor> pooled_routes(
    const at::Tensor& scores, const at::Tensor& threshold,
    const at::Tensor& v_sum, const at::Tensor& v_scale, const at::Tensor& block_len,
    int64_t tokens, int64_t sink_start, int64_t sink_end,
    int64_t sink_q_start, int64_t sink_q_end, int64_t topk, bool tail, bool token_groups) {
  TORCH_CHECK(scores.device().is_xpu() && scores.dim() == 4 &&
      scores.scalar_type() == at::kFloat && scores.is_contiguous(),
      "Sol pooled scores must be contiguous FP32 [B,H,N,N] on XPU");
  const int64_t B = scores.size(0), H = scores.size(1), N = scores.size(2), BH = B*H;
  TORCH_CHECK(B > 0 && H > 0 && tokens > 0 && N == (tokens+63)/64 && scores.size(3) == N,
      "Sol pooled scores must match the nonempty token/block dimensions");
  auto check = [&](const at::Tensor& x, at::ScalarType dtype, at::IntArrayRef sizes) {
    TORCH_CHECK(x.device() == scores.device() && x.scalar_type() == dtype &&
        x.is_contiguous() && x.sizes() == sizes, "Sol pooled summary contract mismatch");
  };
  check(threshold, at::kFloat, {B,H,N});
  check(v_sum, at::kBFloat16, {B,H,N,128});
  check(v_scale, at::kFloat, {B,H,128});
  TORCH_CHECK(block_len.device() == scores.device() && block_len.scalar_type() == at::kInt &&
      block_len.is_contiguous() && (block_len.numel() == 0 || block_len.sizes() == at::IntArrayRef({N})),
      "Sol pooled block_len must be empty or contiguous int32 [N] on the input XPU");
  TORCH_CHECK(0 <= sink_start && sink_start <= sink_end && sink_end <= N &&
      0 <= sink_q_start && sink_q_start <= sink_q_end && sink_q_end <= N,
      "Sol sink ranges must lie inside the block count");
  TORCH_CHECK(topk == -1 || (topk >= 0 && topk <= std::max(int64_t(0),N-(sink_end-sink_start)-1)),
      "Sol top-k count exceeds the non-sink block budget");
  auto routes = at::empty(scores.sizes(), scores.options().dtype(at::kByte));
  auto state = at::empty({B,H,N,130}, scores.options());
  auto refs = at::empty({B,H,N}, scores.options());
  auto common = at::empty({B,H,(N+1)/2,N}, routes.options());
  const auto* sp = scores.data_ptr<float>(); const auto* tp = threshold.data_ptr<float>();
  const auto* vp = reinterpret_cast<const bf16*>(v_sum.data_ptr());
  const auto* vsp = v_scale.data_ptr<float>();
  const auto* lengths = block_len.numel() ? block_len.data_ptr<int32_t>() : nullptr;
  auto* rp = routes.data_ptr<uint8_t>(); auto* cp = common.data_ptr<uint8_t>();
  auto* op = state.data_ptr<float>(); auto* refp = refs.data_ptr<float>();
  auto& queue = c10::xpu::getCurrentXPUStream(scores.device().index()).queue();
  int64_t padded = 1;
  if (topk >= 0) while (padded < N) padded *= 2;
  const int64_t local_bytes = std::max(padded*int64_t(sizeof(float)),N*int64_t(sizeof(bf16)));
  TORCH_CHECK(local_bytes <= int64_t(queue.get_device().get_info<sycl::info::device::local_mem_size>()),
      "Sol pooled block count exceeds XPU local-memory capacity");
  queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float,1> sorted(sycl::range<1>(padded),cgh);
    cgh.parallel_for<SolPreparedRouteKernel>(
        sycl::nd_range<1>(sycl::range<1>(BH*N*kWorkGroup),sycl::range<1>(kWorkGroup)),
        [=](sycl::nd_item<1> item) {
      const int64_t row = item.get_group_linear_id(), q = row % N, lane = item.get_local_linear_id();
      float thr = tp[row];
      if (topk >= 0) {
        for (int64_t j = lane; j < padded; j += kWorkGroup) sorted[j] =
            j < N && !(j >= sink_start && j < sink_end) ? sp[row*N+j] : -INFINITY;
        sycl::group_barrier(item.get_group());
        sol_sort_descending(sorted,padded,item);
        thr = topk == 0 ? INFINITY : sorted[topk-1];
        if (topk > 0) thr -= sycl::fabs(thr) * 1e-5f;
      }
      const bool qsink = q >= sink_q_start && q < sink_q_end;
      for (int64_t j = lane; j < N; j += kWorkGroup) rp[row*N+j] = uint8_t(
          qsink || (j >= sink_start && j < sink_end) || sycl::abs(q-j) <= 1 || sp[row*N+j] >= thr);
    });
  });
  queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<bf16,1> probs(sycl::range<1>(N),cgh);
    cgh.parallel_for<SolPreparedTailKernel>(
        sycl::nd_range<1>(sycl::range<1>(BH*N*kWorkGroup),sycl::range<1>(kWorkGroup)),
        [=](sycl::nd_item<1> item) {
      const int64_t row = item.get_group_linear_id(), q = row % N, bh = row / N;
      const int64_t lane = item.get_local_linear_id(), peer = q ^ 1;
      auto shared_candidate = [&](int64_t j) {
        return rp[row*N+j] == 0 && (peer >= N || rp[(bh*N+peer)*N+j] == 0);
      };
      float maximum = -INFINITY, reference = -INFINITY;
      for (int64_t j = lane; j < N; j += kWorkGroup) {
        const bool shared = token_groups && shared_candidate(j);
        if ((q & 1) == 0) cp[(bh*((N+1)/2)+q/2)*N+j] = uint8_t(shared);
        if (shared) reference = sycl::fmax(reference,sp[row*N+j]);
        if (tail && rp[row*N+j] == 0 && !shared) maximum = sycl::fmax(maximum,sp[row*N+j]);
      }
      maximum = sycl::reduce_over_group(item.get_group(),maximum,sycl::maximum<float>());
      reference = sycl::reduce_over_group(item.get_group(),reference,sycl::maximum<float>());
      float denominator = 0;
      for (int64_t j = lane; j < N; j += kWorkGroup) {
        const bool keep = tail && rp[row*N+j] == 0 && !(token_groups && shared_candidate(j));
        const float p = keep ? sycl::exp2(sp[row*N+j]-maximum) : 0.0f;
        probs[j] = bf16(p);
        denominator += p*float(sol_live_length(lengths,j,tokens));
      }
      denominator = sycl::reduce_over_group(item.get_group(),denominator,sycl::plus<float>());
      sycl::group_barrier(item.get_group());
      float numerator = 0;
      for (int64_t j = 0; j < N; ++j) numerator += float(probs[j])*float(vp[(bh*N+j)*128+lane]);
      op[row*130+2+lane] = numerator/vsp[bh*128+lane];
      if (lane == 0) { op[row*130] = maximum; op[row*130+1] = denominator; refp[row] = reference; }
    });
  });
  return {routes,state,refs,common};
}
