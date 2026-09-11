"""XPU-only token-group and whole-bin selection oracles."""
import math

import pytest
import torch


def ops():
    from omni_xpu_kernel import cute
    cute._ensure_loaded()
    return torch.ops.omni_xpu_sol_attn


def histogram_reference(q,qs,refs,k,ks,common,scale):
    tokens=k.shape[1]
    key=torch.arange(tokens,device=k.device)
    # Integer inputs are exact in FP32; the D128 dot remains inside its exact
    # integer range. All score construction and histogramming stays on XPU.
    log2s=torch.tensor(scale,device=k.device,dtype=torch.float32)*torch.tensor(math.log2(math.e),device=k.device,dtype=torch.float32)
    scores=(q.float() @ k.permute(0,2,3,1).float())*(qs*log2s).unsqueeze(-1)*ks.unsqueeze(-2)
    valid=common.index_select(-1,key//64).bool() & (ks>0).unsqueeze(-2)
    rel=scores-refs.unsqueeze(-1)+8
    bins=torch.where(rel<24,(rel*4).floor(),96+((rel-24)*0.5).floor()).clamp(0,127).long()
    valid &= rel>=0
    hist=torch.zeros((*q.shape[:3],128),device=q.device,dtype=torch.int32)
    hist.scatter_add_(-1,bins,valid.to(torch.int32))
    return hist,bins,valid,scores


@pytest.mark.parametrize("n",[1,3,16,33])
def test_token_group_centroids(n):
    from test_cute_sol_carriers_correctness import quantize
    gen=torch.Generator(device="xpu").manual_seed(10019+n)
    qm=torch.randn((2,3,n,128),device="xpu",generator=gen)
    refs=torch.randn((2,3,n),device="xpu",generator=gen)
    actual=ops().token_group_centroids(qm,refs)
    means=qm[:,:,::2].clone()
    means[:,:,:n//2]=(qm[:,:,:2*(n//2):2]+qm[:,:,1::2])*0.5
    q,qs=quantize(means,1e-8)
    expected_refs=refs[:,:,::2].clone()
    expected_refs[:,:,:n//2]=torch.maximum(refs[:,:,:2*(n//2):2],refs[:,:,1::2])
    torch.testing.assert_close(actual[0],q,rtol=0,atol=0)
    torch.testing.assert_close(actual[1],qs,rtol=1e-7,atol=1e-9)
    torch.testing.assert_close(actual[2],expected_refs,rtol=0,atol=0)


@pytest.mark.parametrize("n",[1,3,17,33])
@pytest.mark.parametrize("layout",["bhtd","packed"])
@pytest.mark.parametrize("pattern",["random","flat","empty"])
def test_streaming_token_histogram(n,layout,pattern):
    gen=torch.Generator(device="xpu").manual_seed(11027+n)
    tokens=n*64-9; ng=(n+1)//2
    q=torch.randint(-128,128,(2,3,ng,128),device="xpu",dtype=torch.int8,generator=gen)
    if layout=="packed":
        packed=torch.randint(-128,128,(2,tokens,3,3,128),device="xpu",dtype=torch.int8,generator=gen)
        k=packed[:,:,1]
    else:
        k=torch.randint(-128,128,(2,3,tokens,128),device="xpu",dtype=torch.int8,generator=gen).permute(0,2,1,3)
    qs=torch.rand((2,3,ng),device="xpu",generator=gen)*0.01+0.001
    ks=torch.rand((2,3,tokens),device="xpu",generator=gen)*0.01+0.001
    ks[...,::61]=0
    refs=torch.randn((2,3,ng),device="xpu",generator=gen)
    common=(torch.rand((2,3,ng,n),device="xpu",generator=gen)<0.35).to(torch.uint8)
    if pattern=="flat": q.zero_(); refs.zero_(); common.fill_(1)
    if pattern=="empty": common.zero_()
    scale=128**-0.5
    expected,_,_,_=histogram_reference(q,qs,refs,k,ks,common,scale)
    actual=ops().token_histogram(q,qs,refs,k,ks,common,scale)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    for budget in (64,128,192,256):
        cut=ops().token_bin_cutoff(actual,budget)
        cumulative=expected.flip(-1).cumsum(-1).flip(-1)
        bins=torch.arange(128,device="xpu")
        wanted=torch.where(cumulative>budget,bins,-1).amax(-1).to(torch.int32)+1
        torch.testing.assert_close(cut,wanted,rtol=0,atol=0)
        admitted=torch.where(bins>=cut.unsqueeze(-1),actual,0).sum(-1)
        assert bool((admitted<=budget).all().item())


def test_whole_bin_cutoff_includes_overflow_bin():
    hist=torch.zeros((1,1,4,128),device="xpu",dtype=torch.int32)
    hist[:,:,0,127]=257  # Even saturated top-bin scores must not bypass budget.
    hist[:,:,1,32]=1000
    hist[:,:,2,32]=256
    actual=ops().token_bin_cutoff(hist,256)
    expected=torch.tensor([[[128,33,0,0]]],device="xpu",dtype=torch.int32)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)


@pytest.mark.parametrize("n",[3,17])
@pytest.mark.parametrize("layout",["bhtd","packed"])
@pytest.mark.parametrize("budget",[64,192,256])
@pytest.mark.parametrize("pattern",["random","flat","empty"])
def test_token_selection_and_remaining_centroid_tail(n,layout,budget,pattern):
    gen=torch.Generator(device="xpu").manual_seed(12037+n)
    tokens=n*64-9; ng=(n+1)//2
    q=torch.randint(-128,128,(2,3,ng,128),device="xpu",dtype=torch.int8,generator=gen)
    if layout=="packed":
        packed=torch.randint(-128,128,(2,tokens,3,3,128),device="xpu",dtype=torch.int8,generator=gen)
        k,v=packed[:,:,1],packed[:,:,2]
    else:
        k,v=(torch.randint(-128,128,(2,3,tokens,128),device="xpu",dtype=torch.int8,generator=gen).permute(0,2,1,3) for _ in range(2))
    qs=torch.rand((2,3,ng),device="xpu",generator=gen)*0.03+0.001
    ks=torch.rand((2,3,tokens),device="xpu",generator=gen)*0.03+0.001
    ks[...,::61]=0
    refs=torch.randn((2,3,ng),device="xpu",generator=gen)
    common=(torch.rand((2,3,ng,n),device="xpu",generator=gen)<0.7).to(torch.uint8)
    if pattern=="flat":
        q.zero_(); refs.zero_(); common.fill_(1)
        v.copy_(torch.arange(-128,0,device="xpu",dtype=torch.int8).expand_as(v))
    if pattern=="empty": common.zero_()
    scale=128**-0.5
    hist,bins,in_window,scores=histogram_reference(q,qs,refs,k,ks,common,scale)
    cutoff=ops().token_bin_cutoff(hist,budget)
    selected=in_window & (bins>=cutoff.unsqueeze(-1))
    wanted_counts=selected.sum(-1).to(torch.int32)
    key=torch.arange(tokens,device="xpu")
    wanted_indices=torch.where(selected,key,2**31-1).sort(-1).values[...,:budget]
    if tokens<budget:
        wanted_indices=torch.nn.functional.pad(wanted_indices,(0,budget-tokens),value=2**31-1)
    wanted_indices=torch.where(wanted_indices==2**31-1,-1,wanted_indices).to(torch.int32)
    for tail in (False,True):
        indices,counts,state=ops().token_remainder(q,qs,refs,k,ks,v,common,cutoff,scale,budget,tail)
        sorted_indices=ops().sort_token_indices(indices,counts)
        torch.testing.assert_close(counts,wanted_counts,rtol=0,atol=0)
        torch.testing.assert_close(sorted_indices,wanted_indices,rtol=0,atol=0)
        assert bool((counts<=budget).all().item())
        keep=common.index_select(-1,key//64).bool() & (ks>0).unsqueeze(-2) & ~selected
        if not tail: keep.zero_()
        masked=scores.masked_fill(~keep,float("-inf"))
        maximum=masked.amax(-1)
        probability=torch.where(keep,torch.exp2(masked-maximum.unsqueeze(-1)),0)
        denominator=probability.sum(-1)
        expected=(probability @ v.permute(0,2,1,3).float())/denominator.clamp_min(1e-30).unsqueeze(-1)
        actual=state[...,2:]/state[...,1].clamp_min(1e-30).unsqueeze(-1)
        torch.testing.assert_close(state[...,0],maximum,rtol=1e-5,atol=1e-5)
        torch.testing.assert_close(state[...,1],denominator,rtol=2e-5,atol=2e-5)
        torch.testing.assert_close(actual,expected,rtol=0.02,atol=0.02)
        assert bool(torch.isfinite(actual).all().item())


def test_merge_query_block_and_group_tail():
    gen=torch.Generator(device="xpu").manual_seed(13121)
    pooled=torch.randn((2,3,5,130),device="xpu",generator=gen)
    grouped=torch.randn((2,3,3,130),device="xpu",generator=gen)
    pooled[...,0]*=10; grouped[...,0]*=10
    pooled[...,1]=pooled[...,1].abs(); grouped[...,1]=grouped[...,1].abs()
    pooled[:,0,:,1]=0; grouped[:,1,:,1]=0
    pooled[:,2,0,1]=0; grouped[:,2,0,1]=0
    groups=torch.arange(5,device="xpu")//2
    g=grouped.index_select(2,groups)
    maximum=torch.maximum(torch.where(pooled[...,1]>0,pooled[...,0],float("-inf")),
                          torch.where(g[...,1]>0,g[...,0],float("-inf")))
    pf=torch.where(pooled[...,1]>0,torch.exp2(pooled[...,0]-maximum),0)
    gf=torch.where(g[...,1]>0,torch.exp2(g[...,0]-maximum),0)
    expected=torch.cat((maximum.unsqueeze(-1),
                        (pooled[...,1]*pf+g[...,1]*gf).unsqueeze(-1),
                        pooled[...,2:]*pf.unsqueeze(-1)+g[...,2:]*gf.unsqueeze(-1)),-1)
    actual=ops().merge_token_tail(pooled,grouped)
    torch.testing.assert_close(actual,expected,rtol=2e-6,atol=2e-6)
