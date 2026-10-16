// SPDX-License-Identifier: Apache-2.0
// Quantization semantics ported from Comfy Kitchen b614102f8f0fcbba00d7fe570c31eeeda4f0061d:
// sage_attention/quant_qk_int8.cu and quant_v_int8.cu.
// V quantization: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// XPU stores V in natural channel-major order; the CUDA MMA permutation is
// a physical packing detail. Rotation, scale groups and rounding are retained.
#include "kitchen_decode.h"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/math.hpp>
#include <array>
#include "utils.h"

namespace omni_xpu::kitchen {
namespace {
using BF16=sycl::ext::oneapi::bfloat16;
struct Source {
    const BF16* p;std::array<int64_t,4> stride;
    float load(int b,int h,int r,int d) const {
        return float(p[b*stride[0]+h*stride[1]+r*stride[2]+d*stride[3]]);
    }
};
Source source(const torch::Tensor& t) {
    return {static_cast<const BF16*>(t.data_ptr()),{t.stride(0),t.stride(1),t.stride(2),t.stride(3)}};
}
inline void rotate(float* x,sycl::sub_group sg,int lane) {
    for(int half=0;half<2;++half) {
        float* v=x+4*half;
        const float a0=v[0]+v[1],a1=v[0]-v[1],a2=v[2]+v[3],a3=v[2]-v[3];
        v[0]=(a0+a2)*.5f;v[1]=(a1+a3)*.5f;v[2]=(a0-a2)*.5f;v[3]=(a1-a3)*.5f;
        for(int bit=1;bit<32;bit*=2) for(int c=0;c<4;++c) {
            const float other=sycl::permute_group_by_xor(sg,v[c],bit);
            v[c]=(lane&bit)?other-v[c]:v[c]+other;
        }
        for(int c=0;c<4;++c) v[c]*=.1767766952966369f;
    }
}
inline int8_t quant(float x,float inv) {
    // The callers use fdiv_rn for inv: an approximate reciprocal changes
    // ties for common BF16 values even when the stored scale is identical.
    return int8_t(sycl::fmax(-127.f,sycl::fmin(127.f,sycl::rint(x*inv))));
}
void validate_cache(const torch::Tensor& k,const torch::Tensor& v,const torch::Tensor& ks,
                    const torch::Tensor& vs,const torch::Tensor& length,int64_t capacity) {
    TORCH_CHECK(k.device().is_xpu() && k.dim()==5 && k.size(0)>0 && k.size(2)>0
        && capacity>0 && k.size(1)==(capacity+511)/512 && k.size(3)==512 && k.size(4)==256
        && k.scalar_type()==at::kChar && k.is_contiguous(), "INT8 decode requires contiguous K[B,P,Hk,512,256]");
    TORCH_CHECK(v.device()==k.device() && v.scalar_type()==at::kChar && v.is_contiguous()
        && v.sizes()==at::IntArrayRef({k.size(0),k.size(1),k.size(2),256,512}),
        "INT8 decode requires contiguous V[B,P,Hk,256,512]");
    TORCH_CHECK(ks.device()==k.device() && vs.device()==k.device() && ks.scalar_type()==at::kFloat
        && vs.scalar_type()==at::kFloat && ks.is_contiguous() && vs.is_contiguous()
        && ks.sizes()==at::IntArrayRef({k.size(0),k.size(1),k.size(2),32})
        && vs.sizes()==at::IntArrayRef({k.size(0),k.size(1),k.size(2),256}),"INT8 decode scale shape/dtype mismatch");
    TORCH_CHECK(length.device()==k.device() && length.scalar_type()==at::kLong
        && length.sizes()==at::IntArrayRef({1}) && length.is_contiguous(),
        "INT8 decode length must be contiguous XPU int64 [1]");
}
class QuantizeDecodeK;
class QuantizeDecodeV;
class QuantizeDecodeQ;
}

void int8_decode_update(torch::Tensor key,torch::Tensor value,torch::Tensor k,torch::Tensor v,
    torch::Tensor ks,torch::Tensor vs,torch::Tensor length,int64_t capacity,bool initialize) {
    validate_cache(k,v,ks,vs,length,capacity);
    TORCH_CHECK(key.device()==k.device() && value.device()==k.device() && key.dim()==4
        && key.scalar_type()==at::kBFloat16 && value.scalar_type()==key.scalar_type()
        && value.sizes()==key.sizes() && key.size(0)==k.size(0) && key.size(1)==k.size(2)
        && key.size(2)>0 && key.size(2)<=capacity && key.size(3)==256,
        "INT8 cache sources must be BF16 [B,Hk,rows,256] on the cache device");
    const int B=k.size(0), HK=k.size(2), P=k.size(1), N=key.size(2), tasks=initialize?P:2;
    TORCH_CHECK(initialize || N==capacity || (N>=1024 && N%512==0),
        "circular sources must retain two page-aligned pages");
    const auto K=source(key),V=source(value); const int64_t* len=length.data_ptr<int64_t>();
    auto* ki=k.data_ptr<int8_t>();auto* vi=v.data_ptr<int8_t>();
    auto* sk=ks.data_ptr<float>();auto* sv=vs.data_ptr<float>();
    auto k_submit=[&](sycl::handler& cgh) {
        cgh.parallel_for<QuantizeDecodeK>(sycl::nd_range<1>(size_t(B)*HK*tasks*8*4*32,32),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                int idx=it.get_group(0);const int wid=idx%4;idx/=4;
                const int block=idx%8;idx/=8;const int task=idx%tasks;idx/=tasks;
                const int h=idx%HK,b=idx/HK,lane=it.get_local_id(0);
                const int64_t raw=len[0];const int live_length=raw<0?0:(raw>capacity?capacity:raw);
                const int prev=(live_length? (live_length-1)/512:0)-1;
                const int page=initialize?task:sycl::max(0,prev)+task;
                if(page>=P || page*512>=live_length) return;
                const int live=sycl::min(512,live_length-page*512);
                float values[16][8],mx=0;
                for(int j=0;j<16;++j) {
                    const int r=block*64+(j/2)*8+wid*2+j%2;
                    for(int d=0;d<8;++d) values[j][d]=r<live?K.load(b,h,(page*512+r)%N,(d/4)*128+lane*4+d%4):0.f;
                    rotate(values[j],it.get_sub_group(),lane);
                    for(int d=0;d<8;++d) mx=sycl::fmax(mx,sycl::fabs(values[j][d]));
                }
                mx=sycl::reduce_over_group(it.get_sub_group(),mx,sycl::maximum<float>());
                const float scale=sycl::ext::intel::math::fdiv_rn(mx,127.f)+1e-7f;const int64_t slab=(int64_t(b)*P+page)*HK+h;
                if(lane==0) sk[slab*32+block*4+wid]=scale;
                for(int j=0;j<16;++j) {
                    const int r=block*64+(j/2)*8+wid*2+j%2;
                    for(int d=0;d<8;++d) ki[slab*512*256+r*256+(d/4)*128+lane*4+d%4]=quant(values[j][d],sycl::ext::intel::math::fdiv_rn(1.f,scale));
                }
            });
    };
    utils::submit_kernel(k_submit,key.device(),"kitchen_decode_quant_k");
    auto v_submit=[&](sycl::handler& cgh) {
        cgh.parallel_for<QuantizeDecodeV>(sycl::range<1>(size_t(B)*HK*tasks*256),[=](sycl::id<1> it) {
            int idx=it[0],d=idx%256;idx/=256;const int task=idx%tasks;idx/=tasks;
            const int h=idx%HK,b=idx/HK;const int64_t raw=len[0];
            const int n=raw<0?0:(raw>capacity?capacity:raw), prev=(n?(n-1)/512:0)-1;
            const int page=initialize?task:sycl::max(0,prev)+task;
            if(page>=P || page*512>=n) return;
            const int live=sycl::min(512,n-page*512);float mx=0;
            for(int r=0;r<live;++r) mx=sycl::fmax(mx,sycl::fabs(V.load(b,h,(page*512+r)%N,d)));
            const float scale=sycl::fmax(sycl::ext::intel::math::fdiv_rn(mx,127.f),1e-12f);
            const int64_t slab=(int64_t(b)*P+page)*HK+h;sv[slab*256+d]=scale;
            for(int r=0;r<512;++r) vi[slab*512*256+d*512+r]=r<live?quant(V.load(b,h,(page*512+r)%N,d),sycl::ext::intel::math::fdiv_rn(1.f,scale)):0;
        });
    };
    utils::submit_kernel(v_submit,key.device(),"kitchen_decode_quant_v");
}

std::vector<torch::Tensor> int8_decode(torch::Tensor query,torch::Tensor k,torch::Tensor v,
    torch::Tensor ks,torch::Tensor vs,torch::Tensor length,int64_t capacity) {
    validate_cache(k,v,ks,vs,length,capacity);
    TORCH_CHECK(query.device()==k.device() && query.scalar_type()==at::kBFloat16 && query.dim()==4
        && query.size(0)==k.size(0) && query.size(1)>0 && query.size(1)%k.size(2)==0
        && query.size(2)>=1 && query.size(2)<=8 && query.size(3)==256
        && query.size(1)/k.size(2)*query.size(2)<=64,
        "INT8 query requires BF16 [B,H,S,256], 1<=S<=8 and (H/Hk)*S<=64");
    const int B=query.size(0), H=query.size(1), S=query.size(2), HK=k.size(2), groups=H/HK;
    auto qi=torch::empty(query.sizes(),query.options().dtype(at::kChar));
    auto qs=torch::empty({B,HK,64},query.options().dtype(at::kFloat));
    const auto Q=source(query);auto* out=qi.data_ptr<int8_t>();auto* scales=qs.data_ptr<float>();
    auto submit=[&](sycl::handler& cgh) {
        cgh.parallel_for<QuantizeDecodeQ>(sycl::nd_range<1>(size_t(B)*HK*64*32,32),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                int idx=it.get_group(0),entry=idx%64;idx/=64;
                const int hk=idx%HK,b=idx/HK,lane=it.get_local_id(0),base=(entry/8)*16+entry%8;
                float x[2][8],mx=0;
                for(int j=0;j<2;++j) {
                    const int r=base+j*8,h=hk*groups+r/S,s=r%S;
                    for(int d=0;d<8;++d) x[j][d]=r<groups*S?Q.load(b,h,s,(d/4)*128+lane*4+d%4):0.f;
                    rotate(x[j],it.get_sub_group(),lane);
                    for(int d=0;d<8;++d) mx=sycl::fmax(mx,sycl::fabs(x[j][d]));
                }
                mx=sycl::reduce_over_group(it.get_sub_group(),mx,sycl::maximum<float>());
                const float scale=sycl::ext::intel::math::fdiv_rn(mx,127.f)+1e-7f;
                if(lane==0) scales[(int64_t(b)*HK+hk)*64+entry]=scale;
                for(int j=0;j<2;++j) {
                    const int r=base+j*8,h=hk*groups+r/S,s=r%S;
                    if(r<groups*S) for(int d=0;d<8;++d)
                        out[((int64_t(b)*H+h)*S+s)*256+(d/4)*128+lane*4+d%4]=quant(x[j][d],sycl::ext::intel::math::fdiv_rn(1.f,scale));
                }
            });
    };
    utils::submit_kernel(submit,query.device(),"kitchen_decode_quant_q");
    return decode_quantized(query,qi,qs,k,v,ks,vs,length,capacity);
}
}
