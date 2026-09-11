"""Independent XPU semantics for indexed tokens and the full native Sol path."""
import math

import pytest
import torch


def ops():
    from omni_xpu_kernel import cute
    cute._ensure_loaded()
    return torch.ops.omni_xpu_sol_attn


def reference(q,k,v,qs,ks,vs,routes,tail,scale,indices=None,counts=None,bias=None,fp16=False):
    b,t,h,_=q.shape
    key=torch.arange(t,device=q.device)
    allowed=routes.bool().index_select(2,key//64).index_select(3,key//64)
    if indices is not None:
        valid=(torch.arange(indices.shape[-1],device=q.device)<counts.unsqueeze(-1)) & (indices>=0) & (indices<t)
        mask=torch.zeros((*counts.shape,t),device=q.device,dtype=torch.int32)
        mask.scatter_add_(-1,indices.clamp(0,t-1).long(),valid.to(torch.int32))
        allowed |= mask.index_select(2,key//128)>0
    allowed &= (ks>0).unsqueeze(-2)
    scores=(q.permute(0,2,1,3).float() @ k.permute(0,2,3,1).float())
    scores *= qs.unsqueeze(-1)*ks.unsqueeze(-2)*(scale*math.log2(math.e))
    if bias is not None: scores += bias.reshape(b,1,1,t)
    scores.masked_fill_(~allowed,float("-inf"))
    state=tail.index_select(2,key//64)
    maximum=torch.maximum(scores.amax(-1),torch.where(state[...,1]>0,state[...,0],float("-inf")))
    maximum=torch.where(torch.isfinite(maximum),maximum,0)
    weights=torch.exp2(scores-maximum.unsqueeze(-1))
    factor=torch.where(state[...,1]>0,torch.exp2(state[...,0]-maximum),0)
    numerator=weights @ v.permute(0,2,1,3).float()+factor.unsqueeze(-1)*state[...,2:]
    denominator=weights.sum(-1)+factor*state[...,1]
    out=numerator/denominator.clamp_min(1e-30).unsqueeze(-1)*vs.unsqueeze(-2)
    return out.permute(0,2,1,3).to(torch.float16 if fp16 else torch.bfloat16)


@pytest.mark.parametrize("tokens,layout",[(65,"bthd"),(257,"packed"),(513,"bhtd")])
@pytest.mark.parametrize("fp16",[False,True])
@pytest.mark.parametrize("with_bias",[False,True])
@pytest.mark.parametrize("augmented",[False,True])
def test_indexed_fragment_mapping_and_controls(tokens,layout,fp16,with_bias,augmented):
    gen=torch.Generator(device="xpu").manual_seed(14033+tokens)
    b,h,d=2,3,128; n=(tokens+63)//64; ng=(n+1)//2; budget=256
    if layout=="packed":
        packed=torch.randint(-128,128,(b,tokens,3,h,d),device="xpu",dtype=torch.int8,generator=gen)
        q,k,v=packed.unbind(2)
    else:
        shape=(b,tokens,h,d) if layout=="bthd" else (b,h,tokens,d)
        q,k,v=(torch.randint(-128,128,shape,device="xpu",dtype=torch.int8,generator=gen) for _ in range(3))
        if layout=="bhtd": q,k,v=(x.permute(0,2,1,3) for x in (q,k,v))
    qs=torch.rand((b,h,tokens),device="xpu",generator=gen)*0.015+0.001
    ks=torch.rand((b,h,tokens),device="xpu",generator=gen)*0.015+0.001
    ks[...,::67]=0
    vs=torch.rand((b,h,128),device="xpu",generator=gen)*0.1+0.001
    routes=torch.zeros((b,h,n,n),device="xpu",dtype=torch.uint8)
    indices=counts=None
    if augmented:
        indices=((torch.arange(b*h*ng,device="xpu").reshape(b,h,ng,1)*11+
                  torch.arange(budget,device="xpu")*7)%tokens).to(torch.int32)
        cases=torch.tensor([0,1,63,64,65,127,128,129,191,192,193,255,256],device="xpu",dtype=torch.int32)
        counts=cases[torch.arange(b*h*ng,device="xpu")%cases.numel()].reshape(b,h,ng).clamp_max(tokens)
    else:
        routes=(torch.rand(routes.shape,device="xpu",generator=gen)<0.4).to(torch.uint8)
    tail=torch.randn((b,h,n,130),device="xpu",generator=gen)
    tail[...,0]*=5; tail[...,1]=tail[...,1].abs()+0.1
    tail[:,0]=0
    bias=torch.randn((b,tokens),device="xpu",generator=gen)*2 if with_bias else None
    if with_bias: bias[0].fill_(float("-inf")); bias[1,::13]=float("-inf")
    scale=128**-0.5
    actual=ops().forward_cute_prepared(q,k,v,qs,ks,vs,routes,tail,scale,indices,counts,bias,fp16)
    expected=reference(q,k,v,qs,ks,vs,routes,tail,scale,indices,counts,bias,fp16)
    assert actual.dtype==expected.dtype and actual.shape==q.shape and actual.is_contiguous()
    assert bool(torch.isfinite(actual).all().item())
    torch.testing.assert_close(actual,expected,rtol=0.02,atol=0.02)


def merged_reference(pooled,grouped):
    group=grouped.index_select(2,torch.arange(pooled.shape[2],device=pooled.device)//2)
    maximum=torch.maximum(torch.where(pooled[...,1]>0,pooled[...,0],float("-inf")),
                          torch.where(group[...,1]>0,group[...,0],float("-inf")))
    pf=torch.where(pooled[...,1]>0,torch.exp2(pooled[...,0]-maximum),0)
    gf=torch.where(group[...,1]>0,torch.exp2(group[...,0]-maximum),0)
    return torch.cat((maximum.unsqueeze(-1),(pooled[...,1]*pf+group[...,1]*gf).unsqueeze(-1),
                      pooled[...,2:]*pf.unsqueeze(-1)+group[...,2:]*gf.unsqueeze(-1)),-1)


@pytest.mark.parametrize("tokens",[257,577])
@pytest.mark.parametrize("budget",[64,256])
@pytest.mark.parametrize("tail_enabled",[False,True])
def test_complete_native_augmented_composition(tokens,budget,tail_enabled):
    from test_cute_sol_carriers_correctness import NAMES,quantize
    from test_cute_sol_pooled_correctness import route_reference
    from test_cute_sol_token_correctness import histogram_reference
    gen=torch.Generator(device="xpu").manual_seed(15031+tokens)
    packed=torch.randn((2,tokens,3,3,128),device="xpu",dtype=torch.bfloat16,generator=gen)
    q,k,v=packed.unbind(2); n=(tokens+63)//64; ng=(n+1)//2
    scale,tau=128**-0.5,1.3
    lengths=torch.empty(0,device="xpu",dtype=torch.int32)
    c=dict(zip(NAMES,ops().prepare_carriers(q,k,v,scale,tau,lengths)))
    scores=ops().centroid_scores(c['cq'],c['ck'],c['cqs'],c['cks'],scale)
    sinks=(0,1,0,1)
    routes,pooled,refs,common=ops().pooled_routes(scores,c['threshold'],c['vsum'],c['vs'],lengths,tokens,*sinks,-1,tail_enabled,True)
    gq,gqs,gref=ops().token_group_centroids(c['qmean'],refs)
    hist=ops().token_histogram(gq,gqs,gref,c['k'],c['ks'],common,scale)
    cut=ops().token_bin_cutoff(hist,budget)
    indices,counts,grouped=ops().token_remainder(gq,gqs,gref,c['k'],c['ks'],c['v'],common,cut,scale,budget,tail_enabled)
    indices=ops().sort_token_indices(indices,counts)
    state=ops().merge_token_tail(pooled,grouped)
    args=[c[x] for x in ('q','k','v','qs','ks','vs')]
    output=ops().forward_cute_prepared(*args,routes,state,scale,indices,counts)
    # Independent group-centroid and histogram selection, and FP32 token tail.
    rr,rp,refs,rc=route_reference(scores,c['threshold'],c['vsum'],c['vs'],lengths,tokens,sinks,-1,tail_enabled,True)
    gm=c['qmean'][:,:,::2].clone()
    gm[:,:,:n//2]=(c['qmean'][:,:,:2*(n//2):2]+c['qmean'][:,:,1::2])*0.5
    rq,rqs=quantize(gm,1e-8)
    rref=refs[:,:,::2].clone(); rref[:,:,:n//2]=torch.maximum(refs[:,:,:2*(n//2):2],refs[:,:,1::2])
    rh,bins,valid,rs=histogram_reference(rq,rqs,rref,c['k'],c['ks'],rc,scale)
    bin_axis=torch.arange(128,device="xpu")
    cutoff=torch.where(rh.flip(-1).cumsum(-1).flip(-1)>budget,bin_axis,-1).amax(-1)+1
    chosen=valid & (bins>=cutoff.unsqueeze(-1)); rcounts=chosen.sum(-1).to(torch.int32)
    key=torch.arange(tokens,device="xpu")
    rind=torch.where(chosen,key,2**31-1).sort(-1).values[...,:budget]
    rind=torch.where(rind==2**31-1,-1,rind).to(torch.int32)
    torch.testing.assert_close(indices,rind,rtol=0,atol=0)
    torch.testing.assert_close(counts,rcounts,rtol=0,atol=0)
    keep=rc.index_select(-1,key//64).bool() & (c['ks']>0).unsqueeze(-2) & ~chosen
    if not tail_enabled: keep.zero_()
    rs=rs.masked_fill(~keep,float("-inf")); maximum=rs.amax(-1)
    weights=torch.where(keep,torch.exp2(rs-maximum.unsqueeze(-1)),0)
    rgroup=torch.cat((maximum.unsqueeze(-1),weights.sum(-1).unsqueeze(-1),weights @ c['v'].permute(0,2,1,3).float()),-1)
    rstate=merged_reference(rp,rgroup)
    expected=reference(*args,rr,rstate,scale,rind,rcounts)
    torch.testing.assert_close(output,expected,rtol=0.02,atol=0.02)
