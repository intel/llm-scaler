// SPDX-License-Identifier: MIT
// Register-resident verify / state-only accepted-prefix replay follows Strata:
// Copyright (c) 2026 Niko1221 and the Strata contributors.
// verify_kernels.dp.cpp @3052aef22aea8067b056e9d0ed2332d0d0639ea6.
// Kitchen has contiguous [B,Hv,DK,DV] FP32 state, contiguous GQA groups,
// dtype-rounded gates/SiLU and device {pending,parity}, adapted here explicitly.
#include <ATen/ops/linear.h>
#include <torch/extension.h>
#include <sycl/sycl.hpp>
#include <cmath>
#include "utils.h"

namespace omni_xpu::kitchen {
namespace {
constexpr int SLOT=8, D=128, RG=4, RPG=D/RG;
template<typename T> class DeferredConv;
template<typename T> class DeferredGates;
template<typename T> class DeferredSums;
template<typename T> class DeferredDelta;
void ctl_check(const torch::Tensor& ctl,const torch::Tensor& ref) {
    TORCH_CHECK(ctl.device()==ref.device() && ctl.scalar_type()==at::kInt
        && ctl.sizes()==at::IntArrayRef({2}) && ctl.is_contiguous(),"ctl must be contiguous XPU int32 [2]");
}
void tensor_check(const torch::Tensor& t,const torch::Tensor& ref,
                  at::ScalarType dtype,at::IntArrayRef shape,const char* label) {
    TORCH_CHECK(t.device()==ref.device() && t.scalar_type()==dtype && t.is_contiguous()
        && t.sizes()==shape,label," shape/device/dtype mismatch");
}

template<typename T>
void conv_launch(torch::Tensor proj,torch::Tensor state,torch::Tensor weight,
    std::optional<torch::Tensor> bias,torch::Tensor prev,torch::Tensor qkv,torch::Tensor ctl) {
    const int B=proj.size(0),S=proj.size(1),C=proj.size(2),W=state.size(2);
    const auto* x=static_cast<const T*>(proj.data_ptr());auto* st=static_cast<T*>(state.data_ptr());
    const auto* w=static_cast<const T*>(weight.data_ptr());const T* b=bias?static_cast<const T*>(bias->data_ptr()):nullptr;
    auto* pb=static_cast<T*>(prev.data_ptr());auto* qb=static_cast<T*>(qkv.data_ptr());const auto* control=ctl.data_ptr<int>();
    auto submit=[&](sycl::handler& cgh) {
        cgh.parallel_for<DeferredConv<T>>(sycl::range<1>(size_t(B)*C),[=](sycl::id<1> it) {
            const int batch=it[0]/C,c=it[0]%C,p=control[1];const int n=control[0];
            if(p<0 || p>1 || n<0 || n>SLOT) return;
            const int64_t current=int64_t(p)*B*SLOT*C,previous=int64_t(1-p)*B*SLOT*C;
            T* history=st+(int64_t(batch)*C+c)*W;
            // Forward traversal is safe: each old source index is >= its destination.
            for(int r=0;r<W;++r) history[r]=n+r<W?history[n+r]:pb[previous+(int64_t(batch)*SLOT+n+r-W)*C+c];
            for(int t=0;t<S;++t) {
                float sum=b?float(b[c]):0.f;
                for(int tap=0;tap<=W;++tap) {
                    const int r=t+tap;
                    const float v=r<W?float(history[r]):float(x[(int64_t(batch)*S+r-W)*C+c]);
                    sum+=v*float(w[int64_t(c)*(W+1)+tap]);
                }
                // Kitchen rounds the convolution before applying SiLU.
                const float y=float(T(sum));
                qb[((int64_t(p)*B+batch)*C+c)*SLOT+t]=T(y/(1.f+sycl::exp(-y)));
            }
            for(int t=0;t<SLOT;++t) {
                pb[current+(int64_t(batch)*SLOT+t)*C+c]=t<S?x[(int64_t(batch)*S+t)*C+c]:T(0.f);
                if(t>=S) qb[((int64_t(p)*B+batch)*C+c)*SLOT+t]=T(0.f);
            }
        });
    };
    utils::submit_kernel(submit,proj.device(),"kitchen_deferred_conv");
}

template<typename T>
torch::Tensor delta_launch(torch::Tensor x,torch::Tensor wa,torch::Tensor wb,
    torch::Tensor dt,torch::Tensor gd,torch::Tensor state,int HK,double scale,
    torch::Tensor z,torch::Tensor nw,double eps,torch::Tensor qkv,
    torch::Tensor gates,torch::Tensor sums,torch::Tensor ctl) {
    const int B=x.size(0),S=x.size(1),HV=state.size(1),C=qkv.size(2),KD=HK*D;
    // Project once per row/head with the existing XPU GEMM, preserving its
    // BF16/FP16 output rounding. No repeated projection for every value lane.
    auto a=at::linear(x,wa),b=at::linear(x,wb);
    const auto* ap=static_cast<const T*>(a.data_ptr());const auto* bp=static_cast<const T*>(b.data_ptr());
    const auto* qp=static_cast<const T*>(qkv.data_ptr());const auto* zp=static_cast<const T*>(z.data_ptr());
    const auto* np=static_cast<const T*>(nw.data_ptr());const auto* dp=dt.data_ptr<float>();const auto* gp=gd.data_ptr<float>();
    auto* gatesp=gates.data_ptr<float>();auto* sumsp=sums.data_ptr<float>();auto* sp=state.data_ptr<float>();
    const auto* control=ctl.data_ptr<int>();auto output=torch::empty({B,S,HV,D},x.options());
    auto* out=static_cast<T*>(output.data_ptr());const float sc=scale,epsilon=eps;
    auto prepare=[&](sycl::handler& cgh) {
        cgh.parallel_for<DeferredGates<T>>(sycl::range<1>(size_t(B)*SLOT*HV),[=](sycl::id<1> it) {
            const int h=it[0]%HV,t=it[0]/HV%SLOT,batch=it[0]/(HV*SLOT),p=control[1];
            if(p<0 || p>1) return;
            const int64_t dst=((int64_t(p)*B+batch)*SLOT+t)*HV+h;
            float decay=0,beta=0;
            if(t<S) {
                const int64_t src=(int64_t(batch)*S+t)*HV+h;
                const float av=float(ap[src])+dp[h],bv=float(bp[src]);
                const float softplus=av>20.f?av:sycl::log(1.f+sycl::exp(av));
                decay=sycl::exp(gp[h]*softplus);beta=float(T(1.f/(1.f+sycl::exp(-bv))));
            }
            gatesp[dst*2]=decay;gatesp[dst*2+1]=beta;
        });
    };
    utils::submit_kernel(prepare,x.device(),"kitchen_deferred_gates");
    auto norm=[&](sycl::handler& cgh) {
        cgh.parallel_for<DeferredSums<T>>(sycl::range<1>(size_t(B)*SLOT*HK),[=](sycl::id<1> it) {
            const int h=it[0]%HK,t=it[0]/HK%SLOT,batch=it[0]/(HK*SLOT),p=control[1];
            if(p<0 || p>1) return;
            float qs=0,ks=0;
            if(t<S) for(int d=0;d<D;++d) {
                const int64_t base=(int64_t(p)*B+batch)*C*SLOT+(h*D+d)*SLOT+t;
                const float q=float(qp[base]),k=float(qp[base+KD*SLOT]);qs+=q*q;ks+=k*k;
            }
            const int64_t dst=((int64_t(p)*B+batch)*SLOT+t)*HK+h;
            sumsp[dst*2]=qs;sumsp[dst*2+1]=ks;
        });
    };
    utils::submit_kernel(norm,x.device(),"kitchen_deferred_sums");
    auto recurrent=[&](sycl::handler& cgh) {
        sycl::local_accessor<float,2> qk(sycl::range<2>(2,D),cgh),red(sycl::range<2>(RG,D),cgh);
        sycl::local_accessor<float,1> square(D,cgh);
        cgh.parallel_for<DeferredDelta<T>>(sycl::nd_range<1>(size_t(B)*HV*RG*D,RG*D),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                const int head=it.get_group(0)%HV,batch=it.get_group(0)/HV;
                const int col=it.get_local_id(0)%D,rg=it.get_local_id(0)/D,kh=head/(HV/HK);
                const int p=control[1],pending=control[0];
                if(p<0 || p>1 || pending<0 || pending>SLOT) return;
                const int64_t base=((int64_t(batch)*HV+head)*D+rg*RPG)*D+col;
                float st[RPG];for(int r=0;r<RPG;++r) st[r]=sp[base+r*D];
                // One register-resident state serves both previous accepted
                // replay and current read-only verification; commit exactly once.
                for(int phase=0;phase<2;++phase) {
                    const int parity=phase?p:1-p,count=phase?S:pending;
                    for(int t=0;t<count;++t) {
                        const int64_t qbbase=(int64_t(parity)*B+batch)*C*SLOT;
                        const int64_t ss=((int64_t(parity)*B+batch)*SLOT+t)*HK+kh;
                        if(rg==0) {
                            const float qi=1.f/sycl::fmax(sycl::sqrt(sumsp[ss*2]),1e-12f);
                            const float ki=1.f/sycl::fmax(sycl::sqrt(sumsp[ss*2+1]),1e-12f);
                            qk[0][col]=phase?float(qp[qbbase+(kh*D+col)*SLOT+t])*qi*sc:0.f;
                            qk[1][col]=float(qp[qbbase+(KD+kh*D+col)*SLOT+t])*ki;
                        }
                        it.barrier(sycl::access::fence_space::local_space);
                        const int64_t gate=((int64_t(parity)*B+batch)*SLOT+t)*HV+head;
                        const float decay=gatesp[gate*2],beta=gatesp[gate*2+1];
                        float memory=0;
                        for(int r=0;r<RPG;++r) {st[r]*=decay;memory+=st[r]*qk[1][rg*RPG+r];}
                        red[rg][col]=memory;it.barrier(sycl::access::fence_space::local_space);
                        const float kv=red[0][col]+red[1][col]+red[2][col]+red[3][col];
                        const float value=float(qp[qbbase+(2*KD+head*D+col)*SLOT+t]);
                        const float delta=(value-kv)*beta;float result=0;
                        for(int r=0;r<RPG;++r) {st[r]+=qk[1][rg*RPG+r]*delta;if(phase)result+=st[r]*qk[0][rg*RPG+r];}
                        it.barrier(sycl::access::fence_space::local_space);
                        if(phase) {
                            red[rg][col]=result;it.barrier(sycl::access::fence_space::local_space);
                            const float raw=float(T(red[0][col]+red[1][col]+red[2][col]+red[3][col]));
                            if(rg==0) square[col]=raw*raw;
                            it.barrier(sycl::access::fence_space::local_space);
                            for(int stride=D/2;stride>0;stride/=2) {
                                if(rg==0 && col<stride) square[col]+=square[col+stride];
                                it.barrier(sycl::access::fence_space::local_space);
                            }
                            if(rg==0) {
                                const int64_t dst=((int64_t(batch)*S+t)*HV+head)*D+col;
                                const float normed=float(T(raw*sycl::rsqrt(square[0]/D+epsilon)*float(np[col])));
                                const float zz=float(zp[dst]),gate_z=float(T(zz/(1.f+sycl::exp(-zz))));
                                out[dst]=T(normed*gate_z);
                            }
                        }
                        it.barrier(sycl::access::fence_space::local_space);
                    }
                    if(!phase && pending>0) for(int r=0;r<RPG;++r) sp[base+r*D]=st[r];
                }
            });
    };
    utils::submit_kernel(recurrent,x.device(),"kitchen_deferred_replay_verify");
    return output;
}
}

void deltanet_conv_deferred(torch::Tensor proj,torch::Tensor state,torch::Tensor weight,
    std::optional<torch::Tensor> bias,torch::Tensor proj_buf,torch::Tensor qkv_buf,torch::Tensor ctl) {
    TORCH_CHECK(proj.device().is_xpu() && proj.dim()==3 && proj.is_contiguous()
        && proj.size(0)>0 && proj.size(1)>0 && proj.size(1)<=SLOT && proj.size(2)>0,"deferred conv requires XPU [B,S,C], 1<=S<=8");
    const int B=proj.size(0),C=proj.size(2);
    TORCH_CHECK(state.dim()==3 && state.size(0)==B && state.size(1)==C,"conv state must be [B,C,KS-1]");
    tensor_check(state,proj,proj.scalar_type(),state.sizes(),"conv_state");
    TORCH_CHECK(weight.numel()==C*(state.size(2)+1) && weight.device()==proj.device()
        && weight.scalar_type()==proj.scalar_type() && weight.is_contiguous(),"conv weight mismatch");
    if(bias) tensor_check(*bias,proj,proj.scalar_type(),{C},"conv bias");
    tensor_check(proj_buf,proj,proj.scalar_type(),{2,B,SLOT,C},"proj buffer");
    tensor_check(qkv_buf,proj,proj.scalar_type(),{2,B,C,SLOT},"qkv buffer");ctl_check(ctl,proj);
    if(proj.scalar_type()==at::kBFloat16) conv_launch<sycl::ext::oneapi::bfloat16>(proj,state,weight,bias,proj_buf,qkv_buf,ctl);
    else if(proj.scalar_type()==at::kHalf) conv_launch<sycl::half>(proj,state,weight,bias,proj_buf,qkv_buf,ctl);
    else if(proj.scalar_type()==at::kFloat) conv_launch<float>(proj,state,weight,bias,proj_buf,qkv_buf,ctl);
    else TORCH_CHECK(false,"deferred conv requires BF16/FP16/FP32");
}

torch::Tensor gated_delta_decode_deferred(torch::Tensor x,torch::Tensor wa,torch::Tensor wb,
    torch::Tensor dt,torch::Tensor gd,torch::Tensor state,int64_t key_dim,int64_t hk,double scale,
    torch::Tensor z,torch::Tensor nw,double eps,torch::Tensor qkv,torch::Tensor gates,torch::Tensor sums,torch::Tensor ctl) {
    TORCH_CHECK(x.device().is_xpu() && x.dim()==3 && x.is_contiguous() && x.size(0)>0
        && x.size(1)>0 && x.size(1)<=SLOT && x.size(2)>0,"deferred GDN requires XPU [B,S,HD], 1<=S<=8");
    TORCH_CHECK(state.dim()==4 && state.size(1)>0 && state.size(2)==D && state.size(3)==D
        && hk>0 && state.size(1)%hk==0 && key_dim==hk*D,"deferred GDN requires DK=DV=128 and valid head groups");
    const int B=x.size(0),S=x.size(1),HV=state.size(1),C=2*key_dim+HV*D;
    tensor_check(state,x,at::kFloat,{B,HV,D,D},"FP32 state");
    tensor_check(wa,x,x.scalar_type(),{HV,x.size(2)},"w_a");tensor_check(wb,x,x.scalar_type(),wa.sizes(),"w_b");
    tensor_check(dt,x,at::kFloat,{HV},"dt_bias");tensor_check(gd,x,at::kFloat,{HV},"g_decay");
    tensor_check(z,x,x.scalar_type(),{B,S,HV*D},"z");tensor_check(nw,x,x.scalar_type(),{D},"norm weight");
    tensor_check(qkv,x,x.scalar_type(),{2,B,C,SLOT},"qkv buffer");
    tensor_check(gates,x,at::kFloat,{2,B,SLOT,HV,2},"gates buffer");
    tensor_check(sums,x,at::kFloat,{2,B,SLOT,hk,2},"sumsq buffer");ctl_check(ctl,x);
    TORCH_CHECK(std::isfinite(scale) && std::isfinite(eps) && eps>=0,"invalid deferred scale/eps");
    if(x.scalar_type()==at::kBFloat16) return delta_launch<sycl::ext::oneapi::bfloat16>(x,wa,wb,dt,gd,state,hk,scale,z,nw,eps,qkv,gates,sums,ctl);
    if(x.scalar_type()==at::kHalf) return delta_launch<sycl::half>(x,wa,wb,dt,gd,state,hk,scale,z,nw,eps,qkv,gates,sums,ctl);
    TORCH_CHECK(x.scalar_type()==at::kFloat,"deferred GDN requires BF16/FP16/FP32");
    return delta_launch<float>(x,wa,wb,dt,gd,state,hk,scale,z,nw,eps,qkv,gates,sums,ctl);
}
}
