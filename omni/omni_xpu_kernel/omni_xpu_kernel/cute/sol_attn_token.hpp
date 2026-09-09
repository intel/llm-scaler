// Copyright 2026
// SPDX-License-Identifier: Apache-2.0
// Token selection keeps the maintained CUTE QK fragment. A single subgroup
// owns its eight query-group rows and streams K64 tiles; no NG x T score
// matrix or global partial histogram is materialized.

template <bool WriteTail> struct SolTokenScanKernel;

template <bool WriteTail>
std::vector<at::Tensor> token_scan(
    const at::Tensor& q, const at::Tensor& qs, const at::Tensor& refs,
    const at::Tensor& k, const at::Tensor& ks, const at::Tensor& common, double scale,
    const at::Tensor& v, const at::Tensor& cutoff, int64_t budget, bool tail) {
  TORCH_CHECK(q.device().is_xpu() && q.scalar_type()==at::kChar && q.is_contiguous() &&
      q.dim()==4 && q.size(3)==128 && q.size(0)>0 && q.size(1)>0 && q.size(2)>0,
      "Sol token Q must be contiguous INT8 [B,H,NG,128] on XPU");
  const int B=checked_int(q.size(0),"batch"),H=checked_int(q.size(1),"heads"),NG=checked_int(q.size(2),"token groups");
  TORCH_CHECK(k.device()==q.device() && k.scalar_type()==at::kChar && k.dim()==4 &&
      k.size(0)==B && k.size(2)==H && k.size(3)==128 && k.stride(3)==1 && k.size(1)>0,
      "Sol token K must be INT8 [B,T,H,128] with contiguous D on the Q device");
  const int T=checked_int(k.size(1),"tokens"),N=(T+63)/64;
  auto check=[&](const at::Tensor& x,at::ScalarType type,at::IntArrayRef shape) {
    TORCH_CHECK(x.device()==q.device() && x.scalar_type()==type && x.is_contiguous() && x.sizes()==shape,
        "Sol token carrier contract mismatch");
  };
  check(qs,at::kFloat,{B,H,NG}); check(refs,at::kFloat,{B,H,NG}); check(ks,at::kFloat,{B,H,T});
  check(common,at::kByte,{B,H,NG,N});
  TORCH_CHECK(NG==(N+1)/2 && std::isfinite(scale),"Sol token groups or scale mismatch");
  if constexpr (WriteTail) {
    check(cutoff,at::kInt,{B,H,NG});
    TORCH_CHECK(v.device()==q.device() && v.scalar_type()==at::kChar && v.sizes()==k.sizes() && v.stride(3)==1,
        "Sol token V must match the INT8 K carrier");
    TORCH_CHECK(budget>0 && budget<=256 && budget%64==0,"Sol token budget must be a multiple of 64 through 256");
  }
  using KT=SolKernel<cutlass::bfloat16_t,SolTilePolicy<64,8,256>,true,true,false,false,int8_t,true>;
  using Base=typename KT::DenseMainloop;
  using Collective=typename KT::CollectiveMainloop;
  using MMA=typename KT::TiledMMAQK;
  using MMAPV=typename KT::TiledMMAPV;
  using CopyQ=typename Base::TiledCopyQ; using CopyK=typename Base::TiledCopyK;
  using CopyV=typename Base::TiledCopyV;
  const auto* qp=q.data_ptr<int8_t>(); const auto* kp=k.data_ptr<int8_t>();
  const auto* qsp=qs.data_ptr<float>(); const auto* ksp=ks.data_ptr<float>();
  const auto* rp=refs.data_ptr<float>(); const auto* cp=common.data_ptr<uint8_t>();
  const int64_t kb=k.stride(0),kh=k.stride(2); const int kt=checked_int(k.stride(1),"K token stride");
  auto hist=WriteTail?at::Tensor{}:at::empty({B,H,NG,128},q.options().dtype(at::kInt));
  auto indices=WriteTail?at::empty({B,H,NG,budget},q.options().dtype(at::kInt)):at::Tensor{};
  auto selected_counts=WriteTail?at::empty({B,H,NG},q.options().dtype(at::kInt)):at::Tensor{};
  auto state=WriteTail?at::empty({B,H,NG,130},q.options().dtype(at::kFloat)):at::Tensor{};
  auto* hp=WriteTail?nullptr:hist.data_ptr<int32_t>();
  auto* ip=WriteTail?indices.data_ptr<int32_t>():nullptr;
  auto* np=WriteTail?selected_counts.data_ptr<int32_t>():nullptr;
  auto* op=WriteTail?state.data_ptr<float>():nullptr;
  const auto* cutp=WriteTail?cutoff.data_ptr<int32_t>():nullptr;
  const auto* vp=WriteTail?v.data_ptr<int8_t>():kp;
  const int64_t vb=WriteTail?v.stride(0):0,vh=WriteTail?v.stride(2):0;
  const int vt=WriteTail?checked_int(v.stride(1),"V token stride"):128;
  const float log2scale=float(scale)*1.4426950408889634f;
  const int groups=(NG+7)/8;
  auto& queue=c10::xpu::getCurrentXPUStream(q.device().index()).queue();
  queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<uint32_t,1> counts(sycl::range<1>(WriteTail?8:8*128),cgh);
    cgh.parallel_for<SolTokenScanKernel<WriteTail>>(
        sycl::nd_range<1>(sycl::range<1>(int64_t(B)*H*groups*16),sycl::range<1>(16)),
        sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>},
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
      const int tile=item.get_group_linear_id(),lane=item.get_local_linear_id();
      const int bh=tile/groups,q0=(tile%groups)*8;
      auto Q=make_tensor(make_gmem_ptr(const_cast<int8_t*>(qp+(int64_t(bh)*NG+q0)*128)),
          make_layout(make_shape(NG-q0,128),make_stride(int(128),_1{})));
      auto K=make_tensor(make_gmem_ptr(const_cast<int8_t*>(kp+int64_t(bh/H)*kb+int64_t(bh%H)*kh)),
          make_layout(make_shape(T,128),make_stride(kt,_1{})));
      auto gQ=local_tile(make_identity_tensor(Q.shape()),typename KT::ShapeQK{},make_coord(0,0,_),Step<_1,X,_1>{});
      auto gK=local_tile(make_identity_tensor(K.shape()),typename KT::ShapeQK{},make_coord(_,_,_),Step<X,_1,_1>{});
      CopyQ cq{Q}; CopyK ck{K}; MMA mma{};
      auto tq=cq.get_slice(lane); auto tk=ck.get_slice(lane); auto tm=mma.get_slice(lane);
      auto sq=tq.partition_S(gQ); auto sk=tk.partition_S(gK);
      auto rq=tq.partition_sg_fragment_D(gQ(_,_,0));
      auto rk=tk.partition_sg_fragment_D(gK(_,_,0,0));
      auto aq=tm.partition_sg_fragment_A(gQ(_,_,0));
      auto ak=tm.partition_sg_fragment_B(gK(_,_,0,0));
      auto V=make_tensor(make_gmem_ptr(const_cast<int8_t*>(vp)+(WriteTail?int64_t(bh/H)*vb+int64_t(bh%H)*vh:0)),
          make_layout(make_shape(128,T),make_stride(_1{},vt)));
      auto gV=local_tile(make_identity_tensor(V.shape()),make_shape(_128{},_64{}),make_coord(0,_));
      auto gVs=local_tile(gV,typename KT::ShapePV{},make_coord(_,_,0),Step<X,_1,_1>{});
      CopyV cv{V}; MMAPV mma_pv{};
      auto tv=cv.get_slice(lane); auto tm_pv=mma_pv.get_slice(lane);
      auto sv=tv.partition_S(gVs);
      auto rv=tv.partition_sg_fragment_D(gVs(_,_,0,0));
      auto av=tm_pv.partition_sg_fragment_B(gVs(_,_,0,0));
      auto probability=tm_pv.partition_sg_fragment_A(make_identity_tensor(take<0,2>(typename KT::ShapeQK{})));
      typename Collective::FragA numerator; clear(numerator);
      typename Collective::FragARow maximum,cutbin;
      fill(maximum,std::numeric_limits<float>::lowest());
      typename Collective::FragSPartialRow partial_sum; clear(partial_sum);
      std::array<decltype(aq),4> qregs;
      CUTLASS_PRAGMA_UNROLL
      for(int d=0;d<4;++d) { copy(cq,sq(_,_,_,d),rq); reorder(rq,qregs[d]); }
      typename Collective::FragARow qscale,reference;
      qscale(0)=lane<8 && q0+lane<NG?qsp[bh*NG+q0+lane]*log2scale:0;
      reference(0)=lane<8 && q0+lane<NG?rp[bh*NG+q0+lane]:-INFINITY;
      if constexpr (WriteTail) cutbin(0)=lane<8 && q0+lane<NG?float(cutp[bh*NG+q0+lane]):128.0f;
      for(int i=lane;i<(WriteTail?8:8*128);i+=16) counts[i]=0;
      sycl::group_barrier(item.get_group());
      const auto sg=item.get_sub_group();
      for(int block=0;block<N;++block) {
        const bool candidate=lane<8 && q0+lane<NG && cp[(int64_t(bh)*NG+q0+lane)*N+block]!=0;
        if(!sycl::any_of_group(sg,candidate)) continue;
        typename Base::FragS acc; clear(acc);
        CUTLASS_PRAGMA_UNROLL
        for(int d=0;d<4;++d) { copy(ck,sk(_,_,_,block,d),rk); reorder(rk,ak); cute::gemm(mma,qregs[d],ak,acc); }
        typename Collective::FragSColumn kscale;
        CUTLASS_PRAGMA_UNROLL
        for(int i=0;i<kscale.size();++i) {
          const int key=block*64+lane+i*16;
          kscale(i)=key<T?ksp[bh*T+key]:0;
        }
        auto gc=local_tile(make_identity_tensor(make_shape(NG-q0,T)),make_shape(_64{},_64{}),make_coord(0,block));
        auto coords=tm.partition_C(gc);
        typename Collective::FragS softmax_scores;
        CUTLASS_PRAGMA_UNROLL
        for(int i=0;i<acc.size();++i) {
          const int row=get<0>(coords(i)),key=get<1>(coords(i)),query=q0+row;
          const float ks0=broadcast<1>(kscale,acc,i);
          float kept_score=-INFINITY;
          if(query<NG && key<T && ks0>0 && cp[(int64_t(bh)*NG+query)*N+block]) {
            const float score=float(acc(i))*broadcast<0>(qscale,acc,i)*ks0;
            kept_score=score;
            const float rel=score-broadcast<0>(reference,acc,i)+8.0f;
            if(rel>=0) {
              const int bin=rel<24.0f?int(rel*4.0f):sycl::min(127,96+int((rel-24.0f)*0.5f));
              if constexpr (WriteTail) {
                if(bin>=int(broadcast<0>(cutbin,acc,i))) {
                  sycl::atomic_ref<uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::work_group,
                      sycl::access::address_space::local_space> counter(counts[row]);
                  const uint32_t slot=counter.fetch_add(1);
                  if(slot<uint32_t(budget)) ip[(int64_t(bh)*NG+query)*budget+slot]=key;
                  kept_score=-INFINITY;
                }
              } else {
                sycl::atomic_ref<uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::work_group,
                    sycl::access::address_space::local_space> counter(counts[row*128+bin]);
                counter.fetch_add(1);
              }
            }
          }
          if constexpr (WriteTail) softmax_scores(i)=kept_score;
        }
        if constexpr (WriteTail) {
          if(tail) {
            auto [rescale,tile_sum]=Collective::softmax_deferred_sum(softmax_scores,maximum);
            reorder(softmax_scores,probability);
            constexpr int SumPerV=decltype(partial_sum.size())::value/4;
            CUTLASS_PRAGMA_UNROLL
            for(int vv=0;vv<4;++vv) {
              copy(cv,sv(_,_,_,vv,block),rv); reorder(rv,av);
              CUTLASS_PRAGMA_UNROLL
              for(int i=numerator.size()/4-1;i>=0;--i) numerator(_,_,_,vv)(i)*=broadcast<0>(rescale,numerator,i);
              CUTLASS_PRAGMA_UNROLL
              for(int j=0;j<SumPerV;++j) {
                const int i=vv*SumPerV+j;
                partial_sum(i)=partial_sum(i)*group_broadcast(sg,rescale(0),i)+tile_sum(i);
              }
              cute::gemm(mma_pv,probability,av,numerator(_,_,_,vv));
            }
          }
        }
      }
      sycl::group_barrier(item.get_group());
      if constexpr (WriteTail) {
        auto total=sol_reduce_horizontal(partial_sum,sycl::plus<void>{});
        auto coords=tm_pv.partition_C(make_identity_tensor(make_shape(_64{},_128{})));
        CUTLASS_PRAGMA_UNROLL
        for(int i=0;i<numerator.size();++i) {
          const int query=q0+get<0>(coords(i)),dim=get<1>(coords(i));
          if(query<NG) op[(int64_t(bh)*NG+query)*130+2+dim]=numerator(i);
        }
        if(lane<8 && q0+lane<NG) {
          const int64_t row=int64_t(bh)*NG+q0+lane;
          np[row]=int32_t(counts[lane]); op[row*130]=total(0)>0?maximum(0):-INFINITY; op[row*130+1]=total(0);
        }
      } else {
        for(int i=lane;i<8*128;i+=16) if(q0+i/128<NG) hp[(int64_t(bh)*NG+q0)*128+i]=int32_t(counts[i]);
      }
    });
  });
  if constexpr (WriteTail) return {indices,selected_counts,state};
  else return {hist};
}

at::Tensor token_histogram(const at::Tensor& q,const at::Tensor& qs,const at::Tensor& refs,
    const at::Tensor& k,const at::Tensor& ks,const at::Tensor& common,double scale) {
  return token_scan<false>(q,qs,refs,k,ks,common,scale,at::Tensor{},at::Tensor{},0,false)[0];
}

std::vector<at::Tensor> token_remainder(const at::Tensor& q,const at::Tensor& qs,const at::Tensor& refs,
    const at::Tensor& k,const at::Tensor& ks,const at::Tensor& v,const at::Tensor& common,
    const at::Tensor& cutoff,double scale,int64_t budget,bool tail) {
  return token_scan<true>(q,qs,refs,k,ks,common,scale,v,cutoff,budget,tail);
}
