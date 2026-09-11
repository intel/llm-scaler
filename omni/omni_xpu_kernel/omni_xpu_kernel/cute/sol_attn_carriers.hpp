// Copyright 2026
// SPDX-License-Identifier: Apache-2.0
//
// Quantized Sol carriers share the maintained Sol block-reduction order and
// the subgroup row-quantization policy used by the native INT8 backend.
// Included inside sol_attn_prepare.cpp's implementation namespace.

template <typename T>
struct SolInputView {
  const T* data;
  int64_t batch_stride, token_stride, head_stride;

  float operator()(int64_t b, int64_t t, int64_t h, int64_t d) const {
    return float(data[b * batch_stride + t * token_stride + h * head_stride + d]);
  }
};

struct SolBlockSums {
  float q = 0, k = 0, v = 0, v_max = 0;
};

template <bool TrackVMax, typename T>
SolBlockSums sol_block_sums(
    SolInputView<T> q, SolInputView<T> k, SolInputView<T> v,
    int64_t batch, int64_t start, int64_t head, int64_t dim, int64_t length) {
  SolBlockSums sums;
  for (int64_t offset = 0; offset < length; ++offset) {
    const float vv = v(batch, start + offset, head, dim);
    sums.q += q(batch, start + offset, head, dim);
    sums.k += k(batch, start + offset, head, dim);
    sums.v += vv;
    if constexpr (TrackVMax) sums.v_max = sycl::fmax(sums.v_max, sycl::fabs(vv));
  }
  return sums;
}

// Sol's symmetric quantizer deliberately differs from general INT8 rowwise
// quantization: its lower endpoint is -127 and its minimum scale is explicit.
inline int8_t sol_q8(float value, float inverse_scale) {
  return int8_t(int32_t(sycl::clamp(sycl::rint(value * inverse_scale), -127.0f, 127.0f)));
}

inline int64_t sol_live_length(const int32_t* lengths, int64_t block, int64_t tokens) {
  const int64_t physical = sycl::min(int64_t(64), tokens - block * 64);
  return lengths == nullptr ? physical : sycl::min(physical, sycl::max(int64_t(1), int64_t(lengths[block])));
}

template <typename T>
SolInputView<T> sol_input_view(const at::Tensor& tensor) {
  return {reinterpret_cast<const T*>(tensor.data_ptr()),
      tensor.stride(0), tensor.stride(1), tensor.stride(2)};
}


// The first sixteen fields are the ordinary prepared attention boundary.
// Producer-only fields hold per-block V maxima/K sums and next-step statistics.
enum SolCarrierField { Q8, K8, V8, QS, KS, VS, CQ, CQS, CK, CKS,
  QM, KM, KMean, KVar, VSum, Threshold, VMax, NextKMean, NextVScale, KSum };
using SolCarriers = std::vector<at::Tensor>;

SolCarriers sol_allocate_carriers(const at::Tensor& reference, int64_t B, int64_t T, int64_t H, bool producer=false) {
  const int64_t N = (T + 63) / 64;
  const auto fp = reference.options().dtype(at::kFloat), i8 = reference.options().dtype(at::kChar);
  auto q8 = at::empty({B, H, T, 128}, i8);
  auto k8 = at::empty_like(q8), v8 = at::empty_like(q8);
  auto qs = at::empty({B, H, T}, fp), ks = at::empty_like(qs);
  auto vs = at::empty({B, H, 128}, fp);
  auto qm = at::empty({B, H, N, 128}, fp), km = at::empty_like(qm);
  auto cq = at::empty({B, H, N, 128}, i8), ck = at::empty_like(cq);
  auto cqs = at::empty({B, H, N}, fp), cks = at::empty_like(cqs);
  auto v_sum = at::empty({B, H, N, 128}, reference.options().dtype(at::kBFloat16));
  return {q8.permute({0,2,1,3}), k8.permute({0,2,1,3}), v8.permute({0,2,1,3}),
    qs, ks, vs, cq, cqs, ck, cks, qm, km, at::empty_like(vs), at::empty_like(vs),
    v_sum, at::empty_like(cqs), at::empty_like(qm), producer?at::empty_like(vs):at::empty({0},fp),
    producer?at::empty_like(vs):at::empty({0},fp), producer?at::empty_like(qm):at::empty({0},fp)};
}

template <typename T, bool SaveKSums = false>
void sol_pool_chunk(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const SolCarriers& c, int64_t offset, const at::Tensor& block_len) {
  const int64_t B=q.size(0), H=q.size(2), Toks=c[Q8].size(1), N=(Toks+63)/64;
  const int64_t chunk_blocks=(q.size(1)+63)/64;
  const auto qv=sol_input_view<T>(q), kv=sol_input_view<T>(k), vv=sol_input_view<T>(v);
  const int32_t* lengths=block_len.numel()?block_len.data_ptr<int32_t>():nullptr;
  auto* qm=c[QM].data_ptr<float>(); auto* km=c[KM].data_ptr<float>();
  auto* vmax=c[VMax].data_ptr<float>(); auto* ksum=c[KSum].data_ptr<float>();
  auto* vsum=reinterpret_cast<bf16*>(c[VSum].data_ptr());
  auto& queue=c10::xpu::getCurrentXPUStream(q.device().index()).queue();
  submit(queue,B*H*chunk_blocks,[=](sycl::nd_item<1> item) {
    const int64_t g=item.get_group_linear_id(), d=item.get_local_linear_id();
    const int64_t block=g%chunk_blocks+offset/64, bh=g/chunk_blocks;
    const int64_t length=sol_live_length(lengths,block,Toks);
    const auto sums=sol_block_sums<true>(qv,kv,vv,bh/H,block*64-offset,bh%H,d,length);
    const int64_t out=(bh*N+block)*128+d;
    qm[out]=sums.q/float(length); km[out]=sums.k/float(length);
    vsum[out]=bf16(sums.v); vmax[out]=sums.v_max;
    if constexpr(SaveKSums) ksum[out]=sums.k;
  });
}

template <bool Producer>
void sol_finish_statistics(const SolCarriers& c,const at::Tensor& block_len) {
  const int64_t B=c[Q8].size(0), Toks=c[Q8].size(1), H=c[Q8].size(2), N=(Toks+63)/64;
  const auto* km=c[KM].data_ptr<float>(); const auto* vmax=c[VMax].data_ptr<float>();
  const auto* ksum=c[KSum].data_ptr<float>();
  const int32_t* lengths=block_len.numel()?block_len.data_ptr<int32_t>():nullptr;
  auto* mean=c[KMean].data_ptr<float>(); auto* variance=c[KVar].data_ptr<float>();
  auto* vs=c[VS].data_ptr<float>(); auto* next_k=c[NextKMean].data_ptr<float>();
  auto* next_v=c[NextVScale].data_ptr<float>();
  auto& queue=c10::xpu::getCurrentXPUStream(c[Q8].device().index()).queue();
  submit(queue,B*H,[=](sycl::nd_item<1> item) {
    const int64_t bh=item.get_group_linear_id(),d=item.get_local_linear_id();
    float total=0,squares=0,amax=0,weighted=0; int64_t live=0;
    for(int64_t block=0;block<N;++block) {
      const int64_t index=(bh*N+block)*128+d;
      const float x=km[index]; total+=x; squares+=x*x;
      amax=sycl::fmax(amax,vmax[index]);
      if constexpr(Producer) { weighted+=ksum[index]; live+=sol_live_length(lengths,block,Toks); }
    }
    const float m=total/float(N);
    mean[bh*128+d]=m; variance[bh*128+d]=sycl::fmax(squares/float(N)-m*m,0.0f);
    if constexpr(Producer) {
      next_k[bh*128+d]=weighted/float(live);
      next_v[bh*128+d]=sycl::fmax(amax/127.0f*1.1f,1e-8f);
    } else vs[bh*128+d]=sycl::fmax(amax/127.0f,1e-8f);
  });
}

template <bool Producer>
void sol_prepare_centroids(const SolCarriers& c,float scale,float tau) {
  const int64_t B=c[Q8].size(0),H=c[Q8].size(2),N=c[QM].size(2);
  const auto* qm=c[QM].data_ptr<float>(); const auto* km=c[KM].data_ptr<float>();
  const auto* mean=c[KMean].data_ptr<float>(); const auto* var=c[KVar].data_ptr<float>();
  auto* cq=c[CQ].data_ptr<int8_t>(); auto* ck=c[CK].data_ptr<int8_t>();
  auto* cqs=c[CQS].data_ptr<float>(); auto* cks=c[CKS].data_ptr<float>();
  auto* threshold=c[Threshold].data_ptr<float>();
  const float log2scale=scale*kLog2E;
  auto& queue=c10::xpu::getCurrentXPUStream(c[Q8].device().index()).queue();
  submit(queue,B*H*N,[=](sycl::nd_item<1> item) {
    const int64_t group=item.get_group_linear_id(),d=item.get_local_linear_id();
    const int64_t bh=group/N,out=group*128+d;
    const float qmean=qm[out],centered_k=km[out]-mean[bh*128+d];
    const float qmax=sycl::reduce_over_group(item.get_group(),sycl::fabs(qmean),sycl::maximum<float>());
    const float kmax=sycl::reduce_over_group(item.get_group(),sycl::fabs(centered_k),sycl::maximum<float>());
    const float qscale=sycl::fmax(qmax/127.0f,1e-8f),kscale=sycl::fmax(kmax/127.0f,1e-12f);
    const int8_t quantized_q=sol_q8(qmean,1.0f/qscale);
    cq[out]=quantized_q; ck[out]=sol_q8(centered_k,1.0f/kscale);
    // The producer's threshold uses its quantized/dequantized centroid;
    // ordinary preprocessing uses the raw pooled Q mean.
    const float threshold_q=Producer?float(quantized_q)*qscale:qmean;
    const float variance=sycl::reduce_over_group(item.get_group(),
      threshold_q*threshold_q*var[bh*128+d],sycl::plus<float>());
    if(d==0) { cqs[group]=qscale; cks[group]=kscale;
      threshold[group]=tau*sycl::sqrt(variance*log2scale*log2scale+1e-6f); }
  });
}

template <typename T>
void sol_quantize_chunk(const at::Tensor& q,const at::Tensor& k,const at::Tensor& v,
    const SolCarriers& c,const at::Tensor& key_mean,int64_t offset,const at::Tensor& block_len) {
  const int64_t B=q.size(0),H=q.size(2),M=q.size(1),Toks=c[Q8].size(1);
  const auto qv=sol_input_view<T>(q),kv=sol_input_view<T>(k),vv=sol_input_view<T>(v);
  const int32_t* lengths=block_len.numel()?block_len.data_ptr<int32_t>():nullptr;
  const auto* mean=key_mean.data_ptr<float>(); const auto* vs=c[VS].data_ptr<float>();
  auto* q8=c[Q8].data_ptr<int8_t>(); auto* k8=c[K8].data_ptr<int8_t>(); auto* v8=c[V8].data_ptr<int8_t>();
  auto* qs=c[QS].data_ptr<float>(); auto* ks=c[KS].data_ptr<float>();
  constexpr int SG=32,RowsPerWG=8,WG=SG*RowsPerWG;
  const int64_t rows=B*H*M,groups=(rows+RowsPerWG-1)/RowsPerWG;
  auto& queue=c10::xpu::getCurrentXPUStream(q.device().index()).queue();
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(groups*WG),sycl::range<1>(WG)),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG)]] {
      const auto sg=item.get_sub_group(); const int lane=sg.get_local_linear_id();
      const int64_t local_row=item.get_group_linear_id()*RowsPerWG+sg.get_group_linear_id();
      if(local_row>=rows)return;
      const int64_t bh=local_row/M,token=local_row%M,global_token=offset+token;
      const int64_t row=bh*Toks+global_token;
      const bool live=global_token%64<sol_live_length(lengths,global_token/64,Toks);
      float qreg[4],kreg[4],vreg[4]; float qmax=0,kmax=0;
#pragma unroll
      for(int i=0;i<4;++i) {
        const int d=lane+i*SG;
        qreg[i]=live?qv(bh/H,token,bh%H,d):0;
        kreg[i]=live?kv(bh/H,token,bh%H,d)-mean[bh*128+d]:0;
        vreg[i]=live?vv(bh/H,token,bh%H,d):0;
        qmax=sycl::fmax(qmax,sycl::fabs(qreg[i])); kmax=sycl::fmax(kmax,sycl::fabs(kreg[i]));
      }
      qmax=sycl::reduce_over_group(sg,qmax,sycl::maximum<float>());
      kmax=sycl::reduce_over_group(sg,kmax,sycl::maximum<float>());
      const float qscale=sycl::fmax(qmax/127.0f,1e-8f),kscale=sycl::fmax(kmax/127.0f,1e-8f);
      if(lane==0) {qs[row]=live?qscale:0;ks[row]=live?kscale:0;}
#pragma unroll
      for(int i=0;i<4;++i) {
        const int d=lane+i*SG;
        q8[row*128+d]=sol_q8(qreg[i],1.0f/qscale);
        k8[row*128+d]=sol_q8(kreg[i],1.0f/kscale);
        v8[row*128+d]=sol_q8(vreg[i],1.0f/vs[bh*128+d]);
      }
    });
  });
}

template <typename T>
SolCarriers prepare_carriers_typed(const at::Tensor& q,const at::Tensor& k,const at::Tensor& v,
    float scale,float tau,const at::Tensor& block_len) {
  auto c=sol_allocate_carriers(q,q.size(0),q.size(1),q.size(2));
  sol_pool_chunk<T>(q,k,v,c,0,block_len);
  sol_finish_statistics<false>(c,block_len);
  sol_prepare_centroids<false>(c,scale,tau);
  sol_quantize_chunk<T>(q,k,v,c,c[KMean],0,block_len);
  c.resize(16);return c;
}

std::vector<at::Tensor> prepare_carriers(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    double scale, double tau, const at::Tensor& block_len) {
  TORCH_CHECK(q.device().is_xpu() && q.dim() == 4 && q.size(0) > 0 &&
      q.size(1) > 0 && q.size(2) > 0 && q.size(3) == 128 &&
      (q.scalar_type() == at::kBFloat16 || q.scalar_type() == at::kHalf),
      "Sol carriers require nonempty BF16/FP16 BTHD XPU input with D128");
  for (const auto& tensor : {q, k, v}) {
    TORCH_CHECK(tensor.sizes() == q.sizes() && tensor.scalar_type() == q.scalar_type() &&
        tensor.device() == q.device() && tensor.stride(3) == 1,
        "Sol Q/K/V must share shape, dtype, device and a contiguous D axis");
  }
  const int64_t blocks = (q.size(1) + 63) / 64;
  TORCH_CHECK(block_len.device() == q.device() && block_len.scalar_type() == at::kInt &&
      block_len.is_contiguous() && (block_len.numel() == 0 ||
      (block_len.dim() == 1 && block_len.numel() == blocks)),
      "Sol block_len must be empty or contiguous int32 [ceil(T/64)] on the input XPU");
  TORCH_CHECK(std::isfinite(scale) && std::isfinite(tau), "Sol scale and tau must be finite");
  if (q.scalar_type() == at::kHalf) return prepare_carriers_typed<sycl::half>(q, k, v, float(scale), float(tau), block_len);
  return prepare_carriers_typed<bf16>(q, k, v, float(scale), float(tau), block_len);
}

void sol_check_producer_state(const SolCarriers& c,const at::Tensor& block_len) {
  TORCH_CHECK(c.size()==20,"Sol producer requires its complete carrier workspace");
  const auto& q=c[Q8];
  TORCH_CHECK(q.device().is_xpu()&&q.dim()==4&&q.size(0)==1&&q.size(1)>0&&q.size(2)>0&&q.size(3)==128,
      "Sol producer workspace requires B=1, T/H>0 and D128");
  const int64_t T=q.size(1),H=q.size(2),N=(T+63)/64;
  for(int i=0;i<20;++i) {
    const auto& x=c[i];
    TORCH_CHECK(x.device()==q.device(),"Sol producer workspace tensors must share one XPU");
    if(i<=V8) { TORCH_CHECK(x.scalar_type()==at::kChar&&x.sizes()==q.sizes()&&
        x.permute({0,2,1,3}).is_contiguous(),"Sol producer Q/K/V require physical BHTD INT8 storage"); }
    else {
      const auto dtype=(i==CQ||i==CK)?at::kChar:(i==VSum?at::kBFloat16:at::kFloat);
      TORCH_CHECK(x.scalar_type()==dtype&&x.is_contiguous(),"invalid Sol producer workspace dtype/layout");
      std::vector<int64_t> shape;
      if(i==QS||i==KS)shape={1,H,T};
      else if(i==VS||i==KMean||i==KVar||i==NextKMean||i==NextVScale)shape={1,H,128};
      else if(i==CQS||i==CKS||i==Threshold)shape={1,H,N};
      else shape={1,H,N,128};
      TORCH_CHECK(x.sizes()==at::IntArrayRef(shape),"invalid Sol producer workspace shape at field ",i);
    }
  }
  TORCH_CHECK(block_len.device()==q.device()&&block_len.scalar_type()==at::kInt&&block_len.is_contiguous()&&
      (block_len.numel()==0||(block_len.dim()==1&&block_len.numel()==N)),"invalid producer block_len");
}

SolCarriers producer_begin(const at::Tensor& reference,int64_t tokens,int64_t heads,const at::Tensor& vscale) {
  TORCH_CHECK(reference.device().is_xpu()&&tokens>0&&heads>0,"Sol producer requires an XPU and positive T/H");
  TORCH_CHECK(vscale.device()==reference.device()&&vscale.scalar_type()==at::kFloat&&vscale.is_contiguous()&&
      vscale.sizes()==at::IntArrayRef({heads,128}),"producer vscale must be contiguous FP32 [H,128] on the XPU");
  auto c=sol_allocate_carriers(reference,1,tokens,heads,true);
  c[VS]=vscale.view({1,heads,128});
  return c;
}

void producer_chunk(const at::Tensor& q,const at::Tensor& k,const at::Tensor& v,
    const SolCarriers& c,const at::Tensor& kmean,int64_t offset,const at::Tensor& block_len) {
  sol_check_producer_state(c,block_len);
  const int64_t T=c[Q8].size(1),H=c[Q8].size(2);
  TORCH_CHECK(q.dim()==4&&q.size(0)==1&&q.size(1)>0&&q.size(2)==H&&q.size(3)==128,
      "producer chunk must be nonempty [1,M,H,128]");
  const int64_t M=q.size(1);
  TORCH_CHECK(offset>=0&&offset%64==0&&offset+M<=T&&(M%64==0||offset+M==T),
      "producer chunks must cover whole 64-token blocks, except the final ragged block");
  for(const auto& x:{q,k,v}) TORCH_CHECK(x.device()==c[Q8].device()&&x.scalar_type()==at::kBFloat16&&
      x.sizes()==q.sizes()&&x.stride(3)==1,"producer Q/K/V must share BF16 BTHD shape, XPU and contiguous D");
  TORCH_CHECK(kmean.device()==q.device()&&kmean.scalar_type()==at::kFloat&&kmean.is_contiguous()&&
      kmean.sizes()==at::IntArrayRef({H,128}),"producer kmean must be contiguous FP32 [H,128] on the XPU");
  sol_pool_chunk<bf16,true>(q,k,v,c,offset,block_len);
  sol_quantize_chunk<bf16>(q,k,v,c,kmean,offset,block_len);
}

void producer_finish(const SolCarriers& c,double scale,double tau,const at::Tensor& block_len) {
  sol_check_producer_state(c,block_len);
  TORCH_CHECK(std::isfinite(scale)&&std::isfinite(tau),"Sol scale and tau must be finite");
  sol_finish_statistics<true>(c,block_len);
  sol_prepare_centroids<true>(c,float(scale),float(tau));
}
