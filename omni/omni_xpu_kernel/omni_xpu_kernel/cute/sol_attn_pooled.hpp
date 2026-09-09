// Copyright 2026
// SPDX-License-Identifier: Apache-2.0
// Small centroid QK uses the same CUTE copies, layouts and INT8 DPAS as the
// prepared exact mainloop. Only the pooled matrix is materialized.

struct SolCentroidScoreKernel;

at::Tensor centroid_scores(
    const at::Tensor& q, const at::Tensor& k,
    const at::Tensor& qs, const at::Tensor& ks, double scale) {
  TORCH_CHECK(q.device().is_xpu() && q.dim() == 4 &&
      q.scalar_type() == at::kChar && q.is_contiguous() && q.size(3) == 128,
      "Sol centroid Q must be contiguous INT8 [B,H,M,128] on XPU");
  const int B = checked_int(q.size(0), "batch"), H = checked_int(q.size(1), "heads");
  const int M = checked_int(q.size(2), "query centroids");
  TORCH_CHECK(k.device() == q.device() && k.scalar_type() == at::kChar &&
      k.is_contiguous() && k.dim() == 4 && k.size(0) == B && k.size(1) == H && k.size(3) == 128,
      "Sol centroid K must be contiguous INT8 [B,H,N,128] on the Q device");
  const int N = checked_int(k.size(2), "key centroids");
  TORCH_CHECK(B > 0 && H > 0 && M > 0 && N > 0 && std::isfinite(scale),
      "Sol centroid dimensions must be nonempty and scale finite");
  for (const auto& pair : {std::make_pair(qs, M), std::make_pair(ks, N)}) {
    TORCH_CHECK(pair.first.device() == q.device() && pair.first.scalar_type() == at::kFloat &&
        pair.first.is_contiguous() && pair.first.sizes() == at::IntArrayRef({B,H,pair.second}),
        "Sol centroid scales must be contiguous FP32 [B,H,rows] on the Q device");
  }
  using KT = SolKernel<cutlass::bfloat16_t, SolTilePolicy<64,8,256>,
      true, true, false, false, int8_t, true>;
  using Base = typename KT::DenseMainloop;
  using MMA = typename KT::TiledMMAQK;
  using CopyQ = typename Base::TiledCopyQ;
  using CopyK = typename Base::TiledCopyK;
  const auto* qp = q.data_ptr<int8_t>(); const auto* kp = k.data_ptr<int8_t>();
  const auto* qsp = qs.data_ptr<float>(); const auto* ksp = ks.data_ptr<float>();
  auto out = at::empty({B,H,M,N}, q.options().dtype(at::kFloat));
  auto* op = out.data_ptr<float>();
  const float log2scale = float(scale) * 1.4426950408889634f;
  const int mt = (M + 63) / 64, nt = (N + 63) / 64;
  auto& queue = c10::xpu::getCurrentXPUStream(q.device().index()).queue();
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<SolCentroidScoreKernel>(
        sycl::nd_range<1>(sycl::range<1>(int64_t(B)*H*mt*nt*128), sycl::range<1>(128)),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
      const int tile = item.get_group_linear_id(), tid = item.get_local_linear_id();
      const int nk = tile % nt, mq = (tile / nt) % mt, bh = tile / (mt * nt);
      auto Q = make_tensor(make_gmem_ptr(const_cast<int8_t*>(qp + int64_t(bh)*M*128)),
          make_layout(make_shape(M, 128), make_stride(int(128), _1{})));
      auto K = make_tensor(make_gmem_ptr(const_cast<int8_t*>(kp + int64_t(bh)*N*128)),
          make_layout(make_shape(N, 128), make_stride(int(128), _1{})));
      auto gQ = local_tile(make_identity_tensor(Q.shape()), typename KT::ShapeQK{},
          make_coord(mq, 0, _), Step<_1,X,_1>{});
      auto gK = local_tile(make_identity_tensor(K.shape()), typename KT::ShapeQK{},
          make_coord(0, nk, _), Step<X,_1,_1>{});
      CopyQ cq{Q}; CopyK ck{K}; MMA mma{};
      auto tq = cq.get_slice(tid);
      auto tk = ck.get_slice(tid);
      auto tm = mma.get_slice(tid);
      auto sq = tq.partition_S(gQ);
      auto sk = tk.partition_S(gK);
      auto rq = tq.partition_sg_fragment_D(gQ(_,_,0));
      auto rk = tk.partition_sg_fragment_D(gK(_,_,0));
      auto aq = tm.partition_sg_fragment_A(gQ(_,_,0));
      auto ak = tm.partition_sg_fragment_B(gK(_,_,0));
      typename Base::FragS acc;
      clear(acc);
      CUTLASS_PRAGMA_UNROLL
      for (int d = 0; d < 4; ++d) {
        copy(cq, sq(_,_,_,d), rq); reorder(rq, aq);
        copy(ck, sk(_,_,_,d), rk); reorder(rk, ak);
        cute::gemm(mma, aq, ak, acc);
      }
      auto gc = local_tile(make_identity_tensor(make_shape(M,N)),
          make_shape(_64{},_64{}), make_coord(mq,nk));
      auto coords = tm.partition_C(gc);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < acc.size(); ++i) {
        const int m = get<0>(coords(i)), n = get<1>(coords(i));
        if (m < M && n < N) op[(int64_t(bh)*M+m)*N+n] =
            float(acc(i)) * (qsp[bh*M+m] * log2scale) * ksp[bh*N+n];
      }
    });
  });
  return out;
}
