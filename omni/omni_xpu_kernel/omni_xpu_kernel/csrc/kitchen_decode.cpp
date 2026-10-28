// SPDX-License-Identifier: MIT
// Chunked QK / softmax / PV and stable merge follow Strata's SYCL QSA decode:
// Copyright (c) 2026 Niko1221 and the Strata contributors.
// Source: 0928ec3bbdef1716e060e215fe0ef0beeb1a090f,
// sycl/src/kernels/cuda/qsa_decode_attn.dp.cpp.
// Adapted for Kitchen's dense strided cache, arbitrary GQA ratio, short verify
// rows, empty prefixes and natural-log LSE. No serving scheduler dependency.
#include "kitchen_decode.h"
#include <ATen/MemoryOverlap.h>
#include <sycl/sycl.hpp>
#include <array>
#include <cmath>
#include "utils.h"

namespace omni_xpu::kitchen {
namespace {
using BF16 = sycl::ext::oneapi::bfloat16;
constexpr int WG = 256, SG = 32, TILE = 64, ROWS = 4;
struct View {
    const BF16* data;
    std::array<int64_t, 4> stride;
    float load(int b, int h, int s, int d) const {
        return float(data[b*stride[0]+h*stride[1]+s*stride[2]+d*stride[3]]);
    }
};
View view(const torch::Tensor& t) {
    return {static_cast<const BF16*>(t.data_ptr()),
            {t.stride(0), t.stride(1), t.stride(2), t.stride(3)}};
}
void validate_qkv(const torch::Tensor& q, const torch::Tensor& k,
                  const torch::Tensor& v) {
    TORCH_CHECK(q.is_xpu() && k.device() == q.device() && v.device() == q.device(),
                "decode Q/K/V must share one XPU device");
    TORCH_CHECK(q.scalar_type() == at::kBFloat16 && k.scalar_type() == q.scalar_type()
                && v.scalar_type() == q.scalar_type(), "decode requires BF16 Q/K/V");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.sizes() == k.sizes(),
                "decode expects Q[B,H,S,D] and K/V[B,Hk,L,D]");
    TORCH_CHECK(q.size(0) > 0 && q.size(1) > 0 && q.size(2) >= 1 && q.size(2) <= 8
                && k.size(1) > 0 && k.size(2) > 0 && k.size(0) == q.size(0)
                && q.size(1) % k.size(1) == 0 && k.size(3) == q.size(3)
                && (q.size(3) == 64 || q.size(3) == 128 || q.size(3) == 256),
                "decode requires B,H,Hk,L>0, 1<=S<=8, D=64/128/256 and H divisible by Hk");
}
struct DenseReader {
    View Q,K,V;
    float query(int b,int h,int s,int d) const { return Q.load(b,h,s,d); }
    float key(int b,int h,int s,int d) const { return K.load(b,h,s,d); }
    float value(int b,int h,int s,int d) const { return V.load(b,h,s,d); }
};
struct Int8Reader {
    const int8_t *Q,*K,*V; const float *qs,*ks,*vs;
    int H,HK,S,pages;
    float query(int b,int h,int s,int d) const {
        const int kh=h/(H/HK), r=(h%(H/HK))*S+s;
        return float(Q[((int64_t(b)*H+h)*S+s)*256+d]) * qs[(int64_t(b)*HK+kh)*64+(r/16)*8+r%8];
    }
    float key(int b,int h,int row,int d) const {
        const int p=row/512, r=row%512;
        const int64_t slab=(int64_t(b)*pages+p)*HK+h;
        return float(K[slab*512*256+r*256+d])*ks[slab*32+(r/64)*4+(r%8)/2];
    }
    float value(int b,int h,int row,int d) const {
        const int p=row/512,r=row%512;const int64_t slab=(int64_t(b)*pages+p)*HK+h;
        return float(V[slab*512*256+d*512+r])*vs[slab*256+d];
    }
};
template<typename Length,typename Reader> class DecodeChunks;
template<typename Length,typename Reader> class DecodeCombine;
class DecodeStepMerge;

template<typename Length,typename Reader>
void launch_decode(const torch::Tensor& q, const torch::Tensor& lengths,
                   bool causal, torch::Tensor& output, torch::Tensor& lse,
                   int HK,int cap, Reader data,bool shared_length=false) {
    const int B=q.size(0), H=q.size(1), S=q.size(2), D=q.size(3);
    const int groups=H/HK, rows=groups*S;
    const int packs=(rows+ROWS-1)/ROWS;
    const int splits=std::min(32, std::max(1, (cap+255)/256));
    const int split_tiles=((cap+TILE-1)/TILE+splits-1)/splits;
    auto partial=torch::empty({B,H,S,splits,D}, q.options().dtype(at::kFloat));
    auto statistics=torch::empty({B,H,S,splits}, q.options().dtype(at::kFloat));
    const Length* lens=lengths.data_ptr<Length>();
    float* part=partial.data_ptr<float>();
    float* stats=statistics.data_ptr<float>();
    const float scale=1.0f/std::sqrt(float(D));
    auto submit=[&](sycl::handler& cgh) {
        sycl::local_accessor<float,2> sq(sycl::range<2>(ROWS,256),cgh), sp(sycl::range<2>(ROWS,TILE),cgh);
        sycl::local_accessor<float,1> bm(ROWS,cgh), bs(ROWS,cgh);
        cgh.parallel_for<DecodeChunks<Length,Reader>>(
            sycl::nd_range<1>(size_t(B)*HK*packs*splits*WG,WG),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                int idx=it.get_group(0), split=idx%splits; idx/=splits;
                const int pack=idx%packs; idx/=packs;
                const int kh=idx%HK, b=idx/HK;
                const int t=it.get_local_id(0), lane=t%SG, warp=t/SG;
                const auto sg=it.get_sub_group();
                const int64_t raw_length=lens[shared_length?0:b];
                const int64_t length=raw_length<0?0:(raw_length>cap?cap:raw_length);
                int limit[ROWS]; int max_limit=0;
                for(int g=0;g<ROWS;++g) {
                    const int row=pack*ROWS+g, h=kh*groups+row/S, s=row%S;
                    const int64_t end_row=length-(causal?S-s-1:0);
                    limit[g]=row<rows && end_row>0 ? int(end_row) : 0;
                    max_limit=sycl::max(max_limit,limit[g]);
                    if(t<D) sq[g][t]=row<rows ? data.query(b,h,s,t) : 0.f;
                }
                it.barrier(sycl::access::fence_space::local_space);
                float acc[ROWS]={}, maximum[ROWS], denominator[ROWS]={};
                for(int g=0;g<ROWS;++g) maximum[g]=-INFINITY;
                const int begin=split*split_tiles*TILE;
                const int end=sycl::min(cap,begin+split_tiles*TILE);
                for(int base=begin;base<end && base<max_limit;base+=TILE) {
                    // One subgroup owns one key row; its loaded key is shared
                    // across the packed GQA/query rows before the next load.
                    for(int r=warp;r<TILE;r+=WG/SG) {
                        const bool live=base+r<max_limit;
                        float kv[8];
                        for(int d=0;d<D/SG;++d) kv[d]=live?data.key(b,kh,base+r,lane+SG*d):0.f;
                        for(int g=0;g<ROWS;++g) {
                            float dot=0.f;
                            for(int d=0;d<D/SG;++d) dot+=kv[d]*sq[g][lane+SG*d];
                            dot=sycl::reduce_over_group(sg,dot,sycl::plus<float>());
                            if(lane==0) sp[g][r]=base+r<limit[g]?dot*scale:-INFINITY;
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                    if(warp<ROWS) {
                        const float a=sp[warp][lane], c=sp[warp][lane+SG];
                        const float m=sycl::reduce_over_group(sg,sycl::fmax(a,c),sycl::maximum<float>());
                        const float pa=m==-INFINITY?0.f:sycl::exp(a-m);
                        const float pc=m==-INFINITY?0.f:sycl::exp(c-m);
                        sp[warp][lane]=pa; sp[warp][lane+SG]=pc;
                        const float sum=sycl::reduce_over_group(sg,pa+pc,sycl::plus<float>());
                        if(lane==0) { bm[warp]=m; bs[warp]=sum; }
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                    if(t<D) {
                        float factor[ROWS];
                        for(int g=0;g<ROWS;++g) {
                            const float m=sycl::fmax(maximum[g],bm[g]);
                            const float old=maximum[g]==-INFINITY?0.f:sycl::exp(maximum[g]-m);
                            factor[g]=bm[g]==-INFINITY?0.f:sycl::exp(bm[g]-m);
                            acc[g]*=old;
                            denominator[g]=denominator[g]*old+bs[g]*factor[g]; maximum[g]=m;
                        }
                        for(int r=0;r<TILE && base+r<max_limit;++r) {
                            const float value=data.value(b,kh,base+r,t);
                            for(int g=0;g<ROWS;++g) {
                                const float weight=sp[g][r]*factor[g];
                                if(weight>0) acc[g]+=weight*value;
                            }
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                }
                for(int g=0;g<ROWS;++g) {
                    const int row=pack*ROWS+g;
                    if(row>=rows) continue;
                    const int h=kh*groups+row/S, s=row%S;
                    const int64_t pos=((int64_t(b)*H+h)*S+s)*splits+split;
                    if(t<D) part[pos*D+t]=denominator[g]>0?acc[g]/denominator[g]:0.f;
                    if(t==0) stats[pos]=denominator[g]>0?maximum[g]+sycl::log(denominator[g]):-INFINITY;
                }
            });
    };
    utils::submit_kernel(submit,q.device(),"kitchen_decode_chunks");
    BF16* out=static_cast<BF16*>(output.data_ptr()); float* ls=lse.data_ptr<float>();
    auto combine=[&](sycl::handler& cgh) {
        cgh.parallel_for<DecodeCombine<Length,Reader>>(sycl::nd_range<1>(size_t(B)*H*S*WG,WG),
            [=](sycl::nd_item<1> it) {
                const int index=it.get_group(0), d=it.get_local_id(0);
                const int s=index%S, h=index/S%H, b=index/(S*H);
                if(d>=D) return;
                float m=-INFINITY, sum=0, value=0;
                for(int i=0;i<splits;++i) m=sycl::fmax(m,stats[int64_t(index)*splits+i]);
                for(int i=0;i<splits;++i) {
                    const int64_t p=int64_t(index)*splits+i;
                    if(stats[p]==-INFINITY) continue;
                    const float w=sycl::exp(stats[p]-m); sum+=w; value+=w*part[p*D+d];
                }
                out[((int64_t(b)*S+s)*H+h)*D+d]=BF16(sum>0?value/sum:0.f);
                if(d==0) ls[index]=sum>0?m+sycl::log(sum):-INFINITY;
            });
    };
    utils::submit_kernel(combine,q.device(),"kitchen_decode_combine");
}
}

std::vector<torch::Tensor> decode_attention(torch::Tensor q, torch::Tensor k,
    torch::Tensor v, torch::Tensor lengths, bool causal) {
    validate_qkv(q,k,v);
    TORCH_CHECK(lengths.device()==q.device() && lengths.dim()==1
                && lengths.size(0)==q.size(0) && lengths.is_contiguous()
                && (lengths.scalar_type()==at::kInt || lengths.scalar_type()==at::kLong),
                "decode lengths must be contiguous XPU int32/int64 [B]");
    auto out=torch::empty({q.size(0),q.size(2),q.size(1)*q.size(3)},q.options());
    auto lse=torch::empty({q.size(0),q.size(1),q.size(2)},q.options().dtype(at::kFloat));
    if(lengths.scalar_type()==at::kInt) launch_decode<int32_t>(q,lengths,causal,out,lse,k.size(1),k.size(2),DenseReader{view(q),view(k),view(v)});
    else launch_decode<int64_t>(q,lengths,causal,out,lse,k.size(1),k.size(2),DenseReader{view(q),view(k),view(v)});
    return {out,lse};
}

std::vector<torch::Tensor> decode_quantized(torch::Tensor q,torch::Tensor qi,torch::Tensor qs,
    torch::Tensor k,torch::Tensor v,torch::Tensor ks,torch::Tensor vs,
    torch::Tensor length,int64_t capacity) {
    auto out=torch::empty({q.size(0),q.size(2),q.size(1)*256},q.options());
    auto lse=torch::empty({q.size(0),q.size(1),q.size(2)},q.options().dtype(at::kFloat));
    Int8Reader data{qi.data_ptr<int8_t>(),k.data_ptr<int8_t>(),v.data_ptr<int8_t>(),
        qs.data_ptr<float>(),ks.data_ptr<float>(),vs.data_ptr<float>(),int(q.size(1)),int(k.size(2)),int(q.size(2)),int(k.size(1))};
    launch_decode<int64_t>(q,length,false,out,lse,k.size(2),capacity,data,true);
    return {out,lse};
}

static void decode_merge(torch::Tensor out, torch::Tensor lse, torch::Tensor q,
    torch::Tensor k, torch::Tensor v, std::optional<torch::Tensor> mask, torch::Tensor merged) {
    validate_qkv(q,k,v);
    const int B=q.size(0), H=q.size(1), S=q.size(2), D=q.size(3), HK=k.size(1);
    TORCH_CHECK(k.size(2)==S,"step merge requires S key/value rows");
    TORCH_CHECK(out.sizes()==at::IntArrayRef({B,S,int64_t(H)*D}) && merged.sizes()==out.sizes()
                && lse.sizes()==at::IntArrayRef({B,H,S}) && lse.scalar_type()==at::kFloat
                && out.scalar_type()==q.scalar_type() && merged.scalar_type()==q.scalar_type()
                && out.device()==q.device() && merged.device()==q.device() && lse.device()==q.device()
                && out.is_contiguous() && merged.is_contiguous() && lse.is_contiguous(),
                "step merge requires contiguous BF16 out/merged[B,S,H*D] and FP32 LSE[B,H,S]");
    at::assert_no_internal_overlap(merged); at::assert_no_partial_overlap(merged,out);
    for(const auto& input:{q,k,v,lse}) at::assert_no_overlap(merged,input);
    if(mask) {
        TORCH_CHECK(mask->device()==q.device() && mask->scalar_type()==at::kInt
                    && mask->sizes()==at::IntArrayRef({S}) && mask->is_contiguous(),
                    "tree mask must be contiguous XPU int32 [S]");
        at::assert_no_overlap(merged,*mask);
    }
    const int32_t* masks=mask?mask->data_ptr<int32_t>():nullptr;
    const auto Q=view(q), K=view(k), V=view(v);
    const BF16* prefix=static_cast<const BF16*>(out.data_ptr());
    BF16* dest=static_cast<BF16*>(merged.data_ptr());
    const float* ls=lse.data_ptr<float>(); const float scale=1.f/std::sqrt(float(D));
    auto submit=[&](sycl::handler& cgh) {
        sycl::local_accessor<float,1> score(8,cgh);
        cgh.parallel_for<DecodeStepMerge>(sycl::nd_range<1>(size_t(B)*H*S*WG,WG),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                const int index=it.get_group(0), t=it.get_local_id(0), lane=t%SG, row=t/SG;
                const int s=index%S, h=index/S%H, b=index/(S*H), kh=h/(H/HK);
                float dot=0;
                const bool allowed=row<S && (masks?((masks[s]>>row)&1):(row<=s));
                if(allowed) for(int d=lane;d<D;d+=SG) dot+=Q.load(b,h,s,d)*K.load(b,kh,row,d);
                dot=sycl::reduce_over_group(it.get_sub_group(),dot,sycl::plus<float>());
                if(lane==0) score[row]=allowed?dot*scale:-INFINITY;
                it.barrier(sycl::access::fence_space::local_space);
                if(t>=D) return;
                float m=ls[index]; for(int i=0;i<S;++i) m=sycl::fmax(m,score[i]);
                const float wp=ls[index]==-INFINITY?0.f:sycl::exp(ls[index]-m);
                const int64_t pos=((int64_t(b)*S+s)*H+h)*D+t;
                float sum=wp, value=wp>0?wp*float(prefix[pos]):0.f;
                for(int i=0;i<S;++i) {
                    if(score[i]==-INFINITY) continue;
                    const float w=sycl::exp(score[i]-m);sum+=w;value+=w*V.load(b,kh,i,t);
                }
                dest[pos]=BF16(sum>0?value/sum:0.f);
            });
    };
    utils::submit_kernel(submit,q.device(),"kitchen_decode_step_merge");
}

void decode_step_merge(torch::Tensor out,torch::Tensor lse,torch::Tensor q,
    torch::Tensor k,torch::Tensor v,torch::Tensor merged) {
    decode_merge(out,lse,q,k,v,std::nullopt,merged);
}

void decode_tree_merge(torch::Tensor out,torch::Tensor lse,torch::Tensor q,
    torch::Tensor k,torch::Tensor v,torch::Tensor mask,torch::Tensor merged) {
    decode_merge(out,lse,q,k,v,mask,merged);
}
}
