"""Independent XPU checks of selected-row state and its routed merge."""
import math

import pytest
import torch

from test_cute_sol_prepared_extras_correctness import ops, reference


@pytest.mark.parametrize("tokens,layout", [(1,"bthd"),(63,"bhtd"),(65,"packed"),
                                          (257,"packed"),(513,"bhtd"),(577,"bthd")])
@pytest.mark.parametrize("fp16", [False,True])
@pytest.mark.parametrize("with_bias", [False,True])
def test_selected_row_state_and_ordinary_merge(tokens,layout,fp16,with_bias):
    b,h,d,budget=2,3,128,256
    n=(tokens+63)//64; ng=(n+1)//2
    gen=torch.Generator(device="xpu").manual_seed(19037+tokens)
    if layout=="packed":
        packed=torch.randint(-128,128,(b,tokens,3,h,d),device="xpu",dtype=torch.int8,generator=gen)
        q,k,v=packed.unbind(2)
    else:
        shape=(b,tokens,h,d) if layout=="bthd" else (b,h,tokens,d)
        q,k,v=(torch.randint(-128,128,shape,device="xpu",dtype=torch.int8,generator=gen) for _ in range(3))
        if layout=="bhtd":q,k,v=(x.permute(0,2,1,3) for x in (q,k,v))
    qs=torch.rand((b,h,tokens),device="xpu",generator=gen)*0.015+0.001
    ks=torch.rand((b,h,tokens),device="xpu",generator=gen)*0.015+0.001
    ks[...,::67]=0
    vs=torch.rand((b,h,128),device="xpu",generator=gen)*0.1+0.001
    routes=(torch.rand((b,h,n,n),device="xpu",generator=gen)<0.15).to(torch.uint8)
    routes[:,0]=0
    common=~routes[:,:,::2].bool()
    common[:,:,:n//2] &= ~routes[:,:,1::2].bool()
    key=torch.arange(tokens,device="xpu")
    candidates=common.index_select(-1,key//64)
    requested=torch.tensor([0,1,63,64,65,127,128,129,191,192,193,255,256],device="xpu",dtype=torch.int32)
    counts=requested[torch.arange(b*h*ng,device="xpu")%requested.numel()].reshape(b,h,ng)
    counts=torch.minimum(counts,candidates.sum(-1).to(torch.int32))
    # Ascending unique keys, selected only from both query blocks' complement.
    sorted_keys=torch.where(candidates,key,2**31-1).sort(-1).values
    indices=torch.full((b,h,ng,budget),-1,device="xpu",dtype=torch.int32)
    take=min(tokens,budget)
    indices[...,:take]=sorted_keys[...,:take].to(torch.int32)
    indices=torch.where(torch.arange(budget,device="xpu")<counts.unsqueeze(-1),indices,-1)
    tail=torch.randn((b,h,n,130),device="xpu",generator=gen)
    tail[...,0]*=5;tail[...,1]=tail[...,1].abs()+0.1;tail[:,0]=0
    bias=torch.randn((b,tokens),device="xpu",generator=gen)*2 if with_bias else None
    if with_bias:bias[0].fill_(float("-inf"));bias[1,::13]=float("-inf")
    scale=128**-0.5
    values=(q,k,v,qs,ks,vs,routes,tail)
    state=ops().forward_cute_selected(*values,scale,indices,counts,bias)
    assert state.shape==(b,h,tokens,144) and state.dtype==torch.float32 and state.is_contiguous()
    selected=torch.zeros_like(candidates,dtype=torch.int32)
    valid=(torch.arange(budget,device="xpu")<counts.unsqueeze(-1)) & (indices>=0) & (indices<tokens)
    selected.scatter_add_(-1,indices.clamp(0,tokens-1).long(),valid.to(torch.int32))
    allowed=selected.index_select(2,key//128).bool() & (ks>0).unsqueeze(-2)
    scores=q.permute(0,2,1,3).float() @ k.permute(0,2,3,1).float()
    scores *= qs.unsqueeze(-1)*ks.unsqueeze(-2)*(scale*math.log2(math.e))
    if bias is not None:scores+=bias[:,None,None,:]
    scores.masked_fill_(~allowed,float("-inf"))
    maximum=scores.amax(-1);live=torch.isfinite(maximum)
    weights=torch.where(allowed,torch.exp2(scores-torch.where(live,maximum,0).unsqueeze(-1)),0)
    total=weights.sum(-1)
    expected_selected=(weights @ v.permute(0,2,1,3).float())/total.clamp_min(1e-30).unsqueeze(-1)*vs.unsqueeze(-2)
    torch.testing.assert_close(state[...,0][live],maximum[live],rtol=2e-5,atol=2e-5)
    torch.testing.assert_close(state[...,1],total,rtol=2e-5,atol=2e-5)
    torch.testing.assert_close(state[...,16:],expected_selected,rtol=0.02,atol=0.02)
    output=ops().forward_cute_prepared_split(*values,state,scale,bias,fp16)
    expected=reference(*values,scale,indices,counts,bias,fp16)
    assert output.dtype==expected.dtype and output.shape==q.shape and output.is_contiguous()
    assert torch.isfinite(output).all().item()
    torch.testing.assert_close(output,expected,rtol=0.02,atol=0.02)
    delta=output.float()-expected.float()
    assert (delta.square().sum()/expected.float().square().sum().clamp_min(1)).sqrt().item()<0.01


@pytest.mark.parametrize("strong",["ordinary","selected","tail"])
@pytest.mark.parametrize("fp16",[False,True])
def test_split_retains_weak_source_mass(strong,fp16):
    q=torch.zeros((1,256,1,128),device="xpu",dtype=torch.int8);q[...,0]=1
    k=torch.zeros_like(q);v=torch.full_like(q,127)
    qs=torch.ones((1,1,256),device="xpu");ks=torch.ones_like(qs);vs=torch.ones((1,1,128),device="xpu")
    routes=torch.zeros((1,1,4,4),device="xpu",dtype=torch.uint8);routes[...,0]=1
    indices=torch.arange(64,128,device="xpu",dtype=torch.int32).expand(1,1,2,64).contiguous()
    counts=torch.full((1,1,2),64,device="xpu",dtype=torch.int32)
    tail=torch.zeros((1,1,4,130),device="xpu")
    if strong=="tail":tail[...,0]=10;tail[...,1]=64
    else:
        source=slice(0,64) if strong=="ordinary" else slice(64,128)
        k[:,source,:,0]=10;v[:,source]=0
    values=(q,k,v,qs,ks,vs,routes,tail);scale=math.log(2)
    state=ops().forward_cute_selected(*values,scale,indices,counts)
    output=ops().forward_cute_prepared_split(*values,state,scale,None,fp16)
    expected=reference(*values,scale,indices,counts,None,fp16)
    torch.testing.assert_close(output,expected,rtol=0.008,atol=0.001)
    assert output.float().amin().item()>0.1
