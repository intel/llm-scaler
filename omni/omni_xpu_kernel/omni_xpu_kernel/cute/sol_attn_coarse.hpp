// Copyright 2026
// SPDX-License-Identifier: Apache-2.0
// Optional VSA coarse branch over raw FP32 block means. The full token graph
// remains sparse; only the small pooled score matrix is materialized.
// Included inside sol_attn_prepare.cpp's implementation namespace.

struct SolCoarseScoreKernel;
struct SolCoarsePVKernel;

auto coarse_output(const at::Tensor& qm,const at::Tensor& km,const at::Tensor& vsum,
    const at::Tensor& block_len,int64_t tokens,double scale) -> at::Tensor {
  TORCH_CHECK(qm.device().is_xpu()&&qm.scalar_type()==at::kFloat&&qm.dim()==4&&
      qm.size(0)>0&&qm.size(1)>0&&qm.size(2)>0&&qm.size(3)==128&&qm.is_contiguous(),
      "coarse Q means must be contiguous FP32 [B,H,N,128] on XPU");
  const int64_t B=qm.size(0),H=qm.size(1),N=qm.size(2),rows=B*H*N;
  TORCH_CHECK(km.device()==qm.device()&&km.scalar_type()==at::kFloat&&km.sizes()==qm.sizes()&&km.is_contiguous(),
      "coarse K means must match Q means");
  TORCH_CHECK(vsum.device()==qm.device()&&vsum.scalar_type()==at::kBFloat16&&vsum.sizes()==qm.sizes()&&vsum.is_contiguous(),
      "coarse V sums must be BF16 with the pooled Q shape and device");
  TORCH_CHECK(tokens>0&&(tokens+63)/64==N&&std::isfinite(scale),"coarse token count/scale mismatch");
  TORCH_CHECK(block_len.device()==qm.device()&&block_len.scalar_type()==at::kInt&&block_len.is_contiguous()&&
      (block_len.numel()==0||block_len.sizes()==at::IntArrayRef({N})),"invalid coarse block lengths");
  auto scores=at::empty({B,H,N,N},qm.options()),output=at::empty_like(qm);
  const auto* qp=qm.data_ptr<float>();const auto* kp=km.data_ptr<float>();
  const auto* vp=reinterpret_cast<bf16*>(vsum.data_ptr());
  const auto* lengths=block_len.numel()?block_len.data_ptr<int32_t>():nullptr;
  auto* sp=scores.data_ptr<float>();auto* op=output.data_ptr<float>();
  const float factor=float(scale)*kLog2E;
  auto& queue=c10::xpu::getCurrentXPUStream(qm.device().index()).queue();
  TORCH_CHECK(N*int64_t(sizeof(float))<=int64_t(queue.get_device().get_info<sycl::info::device::local_mem_size>()),
      "coarse block count exceeds XPU local-memory capacity");
  constexpr int SG=32,WG=128,ScoresPerWG=WG/SG;
  const int64_t pairs=rows*N,groups=(pairs+ScoresPerWG-1)/ScoresPerWG;
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<SolCoarseScoreKernel>(sycl::nd_range<1>(sycl::range<1>(groups*WG),sycl::range<1>(WG)),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG)]] {
      const auto sg=item.get_sub_group();const int lane=sg.get_local_linear_id();
      const int64_t pair=item.get_group_linear_id()*ScoresPerWG+sg.get_group_linear_id();
      if(pair>=pairs)return;
      const int64_t row=pair/N,key=pair%N,bh=row/N;
      float sum=0;
#pragma unroll
      for(int i=0;i<4;++i) {const int d=lane+i*SG;sum+=qp[row*128+d]*kp[(bh*N+key)*128+d];}
      sum=sycl::reduce_over_group(sg,sum,sycl::plus<float>());
      if(lane==0)sp[pair]=sum*factor;
    });
  });
  queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float,1> weights(sycl::range<1>(N),cgh);
    cgh.parallel_for<SolCoarsePVKernel>(sycl::nd_range<1>(sycl::range<1>(rows*WG),sycl::range<1>(WG)),
        [=](sycl::nd_item<1> item) {
      const int64_t row=item.get_group_linear_id(),d=item.get_local_linear_id(),bh=row/N;
      float maximum=-INFINITY;
      for(int64_t key=d;key<N;key+=WG)maximum=sycl::fmax(maximum,sp[row*N+key]);
      maximum=sycl::reduce_over_group(item.get_group(),maximum,sycl::maximum<float>());
      float denominator=0;
      for(int64_t key=d;key<N;key+=WG){const float p=sycl::exp2(sp[row*N+key]-maximum);weights[key]=p;denominator+=p;}
      denominator=sycl::reduce_over_group(item.get_group(),denominator,sycl::plus<float>());
      sycl::group_barrier(item.get_group());
      float value=0;
      for(int64_t key=0;key<N;++key) value+=weights[key]*(float(vp[(bh*N+key)*128+d])/float(sol_live_length(lengths,key,tokens)));
      op[row*128+d]=value/denominator;
    });
  });
  return output;
}

template<typename Out,typename Gate>
void add_coarse_typed(const at::Tensor& output,const at::Tensor& coarse,const at::Tensor& gate) {
  const int64_t T=output.size(1),H=output.size(2),N=(T+63)/64,count=output.numel();
  auto* op=reinterpret_cast<Out*>(output.data_ptr());
  const auto* gp=reinterpret_cast<const Gate*>(gate.data_ptr());
  const auto* cp=coarse.data_ptr<float>();
  auto& queue=c10::xpu::getCurrentXPUStream(output.device().index()).queue();
  queue.parallel_for(sycl::range<1>(count),[=](sycl::id<1> index) {
    const int64_t i=index[0],d=i%128,h=(i/128)%H,t=(i/(128*H))%T,b=i/(128*H*T);
    op[i]=Out(float(op[i])+float(gp[i])*cp[((b*H+h)*N+t/64)*128+d]);
  });
}

void add_coarse_(const at::Tensor& output,const at::Tensor& coarse,const at::Tensor& gate) {
  TORCH_CHECK(output.device().is_xpu()&&output.dim()==4&&output.size(0)>0&&output.size(1)>0&&
      output.size(2)>0&&output.size(3)==128&&output.is_contiguous()&&
      (output.scalar_type()==at::kBFloat16||output.scalar_type()==at::kHalf),"invalid coarse output buffer");
  TORCH_CHECK(coarse.device()==output.device()&&coarse.scalar_type()==at::kFloat&&coarse.is_contiguous()&&
      coarse.sizes()==at::IntArrayRef({output.size(0),output.size(2),(output.size(1)+63)/64,128}),"invalid coarse block output");
  TORCH_CHECK(gate.device()==output.device()&&gate.sizes()==output.sizes()&&gate.is_contiguous()&&
      (gate.scalar_type()==at::kBFloat16||gate.scalar_type()==at::kHalf||gate.scalar_type()==at::kFloat),"invalid coarse gate");
  if(output.scalar_type()==at::kBFloat16) {
    if(gate.scalar_type()==at::kBFloat16)add_coarse_typed<bf16,bf16>(output,coarse,gate);
    else if(gate.scalar_type()==at::kHalf)add_coarse_typed<bf16,sycl::half>(output,coarse,gate);
    else add_coarse_typed<bf16,float>(output,coarse,gate);
  } else {
    if(gate.scalar_type()==at::kBFloat16)add_coarse_typed<sycl::half,bf16>(output,coarse,gate);
    else if(gate.scalar_type()==at::kHalf)add_coarse_typed<sycl::half,sycl::half>(output,coarse,gate);
    else add_coarse_typed<sycl::half,float>(output,coarse,gate);
  }
}
