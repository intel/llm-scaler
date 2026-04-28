"""Insert M>1 kernels into moe_int4.sycl"""
import re

path = "/llm/models/test/llm-scaler/vllm/custom-esimd-kernels-vllm/csrc/moe_batch/moe_int4.sycl"
with open(path) as f:
    src = f.read()

new_code = open("/dev/stdin").read() if False else """
// ═══════════════════════════════════════════════════════════════════════════════
// Multi-token (M>=1) CUTLASS N-major INT4 MoE with topk + shared expert
// ═══════════════════════════════════════════════════════════════════════════════

template <typename IndexT>
void moe_multi_m_up_cutlass_int4_with_shared_fp16_kernel(
    const fp16* x, const uint8_t* w13, const fp16* w13_scales,
    const IndexT* topk_idx,
    const fp16* shared_gate_up_weight, const fp16* shared_expert_gate_weight,
    fp16* routed_intermediates, fp16* shared_intermediates,
    float* shared_gate_values,
    const int n_tokens, const int top_k, const int hidden_size,
    const int intermediate_size, const int shared_inter_size,
    const int num_shared_experts, const torch::Device& device) {

    const int two_inter = 2 * intermediate_size;
    const int k_bytes = hidden_size / 2;
    const int k_groups = hidden_size / 128;
    constexpr int WG_SIZE = 64;
    const int routed_groups = n_tokens * top_k * intermediate_size;

    auto cgf = [&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> ga(sycl::range<1>(WG_SIZE), cgh);
        sycl::local_accessor<float, 1> ua(sycl::range<1>(WG_SIZE), cgh);
        cgh.parallel_for<class MoeMultiMUpRoutedInt4<IndexT>>(
            sycl::nd_range<1>({(size_t)routed_groups * WG_SIZE}, {WG_SIZE}),
            [=](sycl::nd_item<1> item) {
                const int gid = (int)item.get_group(0);
                const int lid = (int)item.get_local_id(0);
                const int token = gid / (top_k * intermediate_size);
                const int rem = gid - token * top_k * intermediate_size;
                const int route = rem / intermediate_size;
                const int col = rem - route * intermediate_size;
                const int expert = (int)topk_idx[token * top_k + route];
                const uint8_t* gr = w13 + ((size_t)expert * two_inter + col) * k_bytes;
                const uint8_t* ur = w13 + ((size_t)expert * two_inter + intermediate_size + col) * k_bytes;
                const fp16* gs = w13_scales + ((size_t)expert * two_inter + col) * k_groups;
                const fp16* us = w13_scales + ((size_t)expert * two_inter + intermediate_size + col) * k_groups;
                const fp16* xr = x + (size_t)token * hidden_size;
                float gsum = 0.f, usum = 0.f;
                for (int kb = lid; kb < k_bytes; kb += WG_SIZE) {
                    int k0 = kb << 1; int g = kb >> 6;
                    uint8_t gp = gr[kb], up = ur[kb];
                    float x0 = (float)xr[k0], x1 = (float)xr[k0+1];
                    gsum += x0*((float)decode_s4_nibble(gp&0xF)*(float)gs[g]) + x1*((float)decode_s4_nibble((gp>>4)&0xF)*(float)gs[g]);
                    usum += x0*((float)decode_s4_nibble(up&0xF)*(float)us[g]) + x1*((float)decode_s4_nibble((up>>4)&0xF)*(float)us[g]);
                }
                ga[lid]=gsum; ua[lid]=usum;
                item.barrier(sycl::access::fence_space::local_space);
                for(int s=WG_SIZE/2;s>0;s>>=1){if(lid<s){ga[lid]+=ga[lid+s];ua[lid]+=ua[lid+s];}item.barrier(sycl::access::fence_space::local_space);}
                if(lid==0){float g=ga[0],u=ua[0];routed_intermediates[(size_t)(token*top_k+route)*intermediate_size+col]=fp16(g/(1.f+sycl::exp(-g))*u);}
            });
    };
    submit_kernel(cgf, device, "moe multi m up routed int4");

    if (num_shared_experts > 0 && shared_gate_up_weight != nullptr) {
        const int s2i = 2 * shared_inter_size;
        auto cgf2 = [&](sycl::handler& cgh) {
            sycl::local_accessor<float,1> ga2(sycl::range<1>(WG_SIZE),cgh);
            sycl::local_accessor<float,1> ua2(sycl::range<1>(WG_SIZE),cgh);
            const int sg = n_tokens * num_shared_experts * shared_inter_size;
            cgh.parallel_for<class MoeMultiMUpSharedFP16V2<IndexT>>(
                sycl::nd_range<1>({(size_t)sg*WG_SIZE},{WG_SIZE}),
                [=](sycl::nd_item<1> item){
                    const int gid=(int)item.get_group(0),lid=(int)item.get_local_id(0);
                    const int tok=gid/(num_shared_experts*shared_inter_size);
                    const int r2=gid-tok*num_shared_experts*shared_inter_size;
                    const int sid=r2/shared_inter_size,col=r2-sid*shared_inter_size;
                    const fp16*xr=x+(size_t)tok*hidden_size;
                    const fp16*gw=shared_gate_up_weight+(size_t)sid*s2i*hidden_size+(size_t)col*hidden_size;
                    const fp16*uw=shared_gate_up_weight+(size_t)sid*s2i*hidden_size+(size_t)(shared_inter_size+col)*hidden_size;
                    float gs2=0.f,us2=0.f;
                    for(int k=lid;k<hidden_size;k+=WG_SIZE){float xv=(float)xr[k];gs2+=xv*(float)gw[k];us2+=xv*(float)uw[k];}
                    ga2[lid]=gs2;ua2[lid]=us2;
                    item.barrier(sycl::access::fence_space::local_space);
                    for(int s=WG_SIZE/2;s>0;s>>=1){if(lid<s){ga2[lid]+=ga2[lid+s];ua2[lid]+=ua2[lid+s];}item.barrier(sycl::access::fence_space::local_space);}
                    if(lid==0){float g=ga2[0],u=ua2[0];shared_intermediates[(size_t)(tok*num_shared_experts+sid)*shared_inter_size+col]=fp16(g/(1.f+sycl::exp(-g))*u);}
                });
        };
        submit_kernel(cgf2, device, "moe multi m up shared fp16");

        auto cgf3 = [&](sycl::handler& cgh) {
            cgh.parallel_for<class MoeMultiMGateV2<IndexT>>(
                sycl::range<1>(n_tokens * num_shared_experts),
                [=](sycl::id<1> idx) SYCL_ESIMD_KERNEL {
                    const int f=(int)idx[0],tok=f/num_shared_experts,sid=f%num_shared_experts;
                    const fp16*xr=x+(size_t)tok*hidden_size;
                    simd<float,64> acc(0.f);
                    for(int k=0;k<hidden_size;k+=64)
                        acc+=convert<float>(block_load<fp16,64>(xr+k)*block_load<fp16,64>(shared_expert_gate_weight+(size_t)sid*hidden_size+k));
                    float d=sycl::ext::intel::esimd::detail::sum<float,float,64>(acc);
                    shared_gate_values[f]=1.f/(1.f+sycl::exp(-d));
                });
        };
        submit_kernel(cgf3, device, "moe multi m gate precompute");
    }
}

template <typename IndexT>
void moe_multi_m_down_cutlass_int4_with_shared_fp16_kernel(
    const fp16* routed_intermediates, const uint8_t* w2, const fp16* w2_scales,
    const fp16* topk_weight, const IndexT* topk_idx,
    const fp16* shared_intermediates, const float* shared_gate_values,
    const fp16* shared_down_weight, fp16* output,
    const int n_tokens, const int top_k, const int hidden_size,
    const int intermediate_size, const int shared_inter_size,
    const int num_shared_experts, const torch::Device& device) {

    const int k_bytes=intermediate_size/2,k_groups=intermediate_size/128;
    constexpr int WG_SIZE=64;
    auto cgf=[&](sycl::handler& cgh){
        sycl::local_accessor<float,1> la(sycl::range<1>(WG_SIZE),cgh);
        cgh.parallel_for<class MoeMultiMDownInt4WithSharedFP16<IndexT>>(
            sycl::nd_range<1>({(size_t)n_tokens*hidden_size*WG_SIZE},{WG_SIZE}),
            [=](sycl::nd_item<1> item){
                const int gid=(int)item.get_group(0),lid=(int)item.get_local_id(0);
                const int tok=gid/hidden_size,h=gid-tok*hidden_size;
                float sum=0.f;
                for(int r=0;r<top_k;++r){
                    int exp=(int)topk_idx[tok*top_k+r];
                    const uint8_t*wr=w2+((size_t)exp*hidden_size+h)*k_bytes;
                    const fp16*sr=w2_scales+((size_t)exp*hidden_size+h)*k_groups;
                    const fp16*ir=routed_intermediates+(size_t)(tok*top_k+r)*intermediate_size;
                    float rw=(float)topk_weight[tok*top_k+r];
                    for(int k=lid;k<intermediate_size;k+=WG_SIZE){
                        uint8_t p=wr[k>>1];uint8_t n=(k&1)?((p>>4)&0xF):(p&0xF);
                        sum+=rw*(float)ir[k]*((float)decode_s4_nibble(n)*(float)sr[k>>7]);
                    }
                }
                if(num_shared_experts>0 && shared_intermediates!=nullptr){
                    for(int sid=0;sid<num_shared_experts;sid++){
                        const fp16*hi=shared_intermediates+(size_t)(tok*num_shared_experts+sid)*shared_inter_size;
                        const fp16*dw=shared_down_weight+(size_t)sid*hidden_size*shared_inter_size+(size_t)h*shared_inter_size;
                        float g=shared_gate_values[tok*num_shared_experts+sid];
                        for(int k=lid;k<shared_inter_size;k+=WG_SIZE) sum+=g*(float)hi[k]*(float)dw[k];
                    }
                }
                la[lid]=sum;
                item.barrier(sycl::access::fence_space::local_space);
                for(int s=WG_SIZE/2;s>0;s>>=1){if(lid<s)la[lid]+=la[lid+s];item.barrier(sycl::access::fence_space::local_space);}
                if(lid==0) output[(size_t)tok*hidden_size+h]=fp16(la[0]);
            });
    };
    submit_kernel(cgf, device, "moe multi m down int4 with shared fp16");
}

torch::Tensor moe_forward_cutlass_nmajor_int4_full(
    torch::Tensor x, torch::Tensor logits,
    torch::Tensor w13, torch::Tensor w13_scales,
    torch::Tensor w2, torch::Tensor w2_scales,
    torch::Tensor shared_gu_w, torch::Tensor shared_d_w, torch::Tensor shared_gate_w,
    int64_t num_shared_experts, int64_t n_routed_experts) {

    TORCH_CHECK(x.dim()==2 && x.is_contiguous() && x.scalar_type()==torch::kHalf);
    int M=x.size(0),H=x.size(1);
    int two_I=w13.size(1),I=two_I/2,top_k=8;
    int s_I = shared_gu_w.numel()>0 ? (int)(shared_gu_w.size(0)/(2*num_shared_experts)) : 0;

    sycl::queue& q = c10::xpu::getCurrentXPUStream(x.device().index()).queue();

    auto topk_w = torch::empty({M,top_k}, torch::device(x.device()).dtype(torch::kHalf));
    auto topk_i = torch::empty({M,top_k}, torch::device(x.device()).dtype(torch::kInt));
    if(n_routed_experts==256 && top_k==8)
        moe_topk_v2_host<256,8>((const fp16*)logits.data_ptr(),(fp16*)topk_w.data_ptr(),topk_i.data_ptr<int32_t>(),M,q);
    else
        moe_topk_forward_kernel_impl<class MoeCNMFullTopK>((const fp16*)logits.data_ptr(),topk_i.data_ptr<int32_t>(),(fp16*)topk_w.data_ptr(),M,(int)n_routed_experts,top_k,true,q);

    auto inter = torch::empty({M*top_k,I}, torch::device(x.device()).dtype(torch::kHalf));
    auto output = torch::empty({M,H}, torch::device(x.device()).dtype(torch::kHalf));
    torch::Tensor s_inter, s_gate;
    if(num_shared_experts>0){
        s_inter = torch::empty({M*(int)num_shared_experts,s_I}, torch::device(x.device()).dtype(torch::kHalf));
        s_gate = torch::empty({M*(int)num_shared_experts}, torch::device(x.device()).dtype(torch::kFloat));
    }

    moe_multi_m_up_cutlass_int4_with_shared_fp16_kernel<int32_t>(
        (const fp16*)x.data_ptr(),(const uint8_t*)w13.data_ptr(),(const fp16*)w13_scales.data_ptr(),
        topk_i.data_ptr<int32_t>(),
        shared_gu_w.numel()>0?(const fp16*)shared_gu_w.data_ptr():nullptr,
        shared_gate_w.numel()>0?(const fp16*)shared_gate_w.data_ptr():nullptr,
        (fp16*)inter.data_ptr(),
        s_inter.defined()?(fp16*)s_inter.data_ptr():nullptr,
        s_gate.defined()?s_gate.data_ptr<float>():nullptr,
        M,top_k,H,I,s_I,(int)num_shared_experts,x.device());

    moe_multi_m_down_cutlass_int4_with_shared_fp16_kernel<int32_t>(
        (const fp16*)inter.data_ptr(),(const uint8_t*)w2.data_ptr(),(const fp16*)w2_scales.data_ptr(),
        (const fp16*)topk_w.data_ptr(),topk_i.data_ptr<int32_t>(),
        s_inter.defined()?(const fp16*)s_inter.data_ptr():nullptr,
        s_gate.defined()?s_gate.data_ptr<float>():nullptr,
        shared_d_w.numel()>0?(const fp16*)shared_d_w.data_ptr():nullptr,
        (fp16*)output.data_ptr(),
        M,top_k,H,I,s_I,(int)num_shared_experts,x.device());

    return output;
}
"""

# Insert before TORCH_LIBRARY_FRAGMENT
marker = "TORCH_LIBRARY_FRAGMENT(moe_int4_ops, m)"
pos = src.find(marker)
assert pos > 0, "marker not found"
src = src[:pos] + new_code + "\n" + src[pos:]

# Add def + impl
old_def = '    m.def("moe_gemm_int4_nmajor(Tensor x, Tensor weight, Tensor scale, "\n          "Tensor expert_idx, int group_size) -> Tensor");'
new_def = old_def + """
    m.def("moe_forward_cutlass_nmajor_int4_full(Tensor x, Tensor logits, "
          "Tensor w13, Tensor w13_scales, Tensor w2, Tensor w2_scales, "
          "Tensor shared_gu_w, Tensor shared_d_w, Tensor shared_gate_w, "
          "int num_shared_experts, int n_routed_experts) -> Tensor");"""
src = src.replace(old_def, new_def)

old_impl = '    m.impl("moe_gemm_int4_nmajor", &moe_gemm_int4_nmajor);'
new_impl = old_impl + '\n    m.impl("moe_forward_cutlass_nmajor_int4_full", &moe_forward_cutlass_nmajor_int4_full);'
src = src.replace(old_impl, new_impl)

with open(path, "w") as f:
    f.write(src)
print("done")
