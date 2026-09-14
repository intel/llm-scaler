"""Optional call-local QK score reuse preserves defined token outputs exactly."""
import pytest
import torch

from test_cute_sol_token_correctness import ops
from test_cute_sol_token_boundaries import boundary_inputs


def values(tokens, layout, pattern):
    b,h,d=2,3,128;n=(tokens+63)//64;ng=(n+1)//2
    gen=torch.Generator(device='xpu').manual_seed(51200+tokens)
    q=torch.randint(-128,128,(b,h,ng,d),dtype=torch.int8,device='xpu',generator=gen)
    if layout=='packed':
        kv=torch.randint(-128,128,(b,tokens,2,h,d),dtype=torch.int8,device='xpu',generator=gen)
        k,v=kv.unbind(2)
    else:
        k,v=[torch.randint(-128,128,(b,h,tokens,d),dtype=torch.int8,device='xpu',generator=gen).permute(0,2,1,3) for _ in range(2)]
    qs=torch.rand((b,h,ng),device='xpu',generator=gen)*0.02+0.001
    ks=torch.rand((b,h,tokens),device='xpu',generator=gen)*0.02+0.001
    ks[...,::13]=0
    refs=torch.randn((b,h,ng),device='xpu',generator=gen)
    common=(torch.rand((b,h,ng,n),device='xpu',generator=gen)>0.3).to(torch.uint8)
    if pattern=='none':common.zero_()
    if pattern=='flat':q.zero_();refs.zero_();common.fill_(1)
    return q,qs,refs,k,ks,v,common


def baseline(values, budget, tail):
    q,qs,refs,k,ks,v,common=values
    hist=ops().token_histogram(q,qs,refs,k,ks,common,128**-0.5)
    cut=ops().token_bin_cutoff(hist,budget)
    return ops().token_remainder(q,qs,refs,k,ks,v,common,cut,128**-0.5,budget,tail)


def equal(a,b,budget):
    ai,ac,ast=a;bi,bc,bst=b
    assert torch.equal(ac,bc)
    assert bool(((ac>=0)&(ac<=budget)).all().item())
    assert torch.equal(ops().sort_token_indices(ai,ac),ops().sort_token_indices(bi,bc))
    assert torch.equal(ast.view(torch.uint8),bst.view(torch.uint8))


@pytest.mark.parametrize('tokens',[1,63,65,257,1025])
@pytest.mark.parametrize('layout',['bhtd','packed'])
@pytest.mark.parametrize('pattern',['random','flat','none'])
@pytest.mark.parametrize('budget',[64,256])
@pytest.mark.parametrize('tail',[False,True])
def test_score_cache_matches_streaming(monkeypatch,tokens,layout,pattern,budget,tail):
    data=values(tokens,layout,pattern);original=[t.clone() for t in data]
    expected=baseline(data,budget,tail)
    monkeypatch.delenv('OMNI_XPU_FORCE_SKU',raising=False)
    actual=ops().token_select_remainder(*data,128**-0.5,budget,tail)
    equal(actual,expected,budget)
    for x,y in zip(data,original):assert torch.equal(x,y)


@pytest.mark.parametrize('budget',[64,128,192,256])
@pytest.mark.parametrize('layout',['bhtd','packed'])
def test_cached_bin_boundary_keeps_exact_budget(monkeypatch,budget,layout):
    data=boundary_inputs(budget+1,layout);data[4][...,:budget]*=2
    monkeypatch.delenv('OMNI_XPU_FORCE_SKU',raising=False)
    for tail in (False,True):
        result=ops().token_select_remainder(*data,128**-0.5,budget,tail)
        equal(result,baseline(data,budget,tail),budget)
        assert bool((result[1]==budget).all().item())


def test_cached_dispatch_and_forced_streaming_are_distinct(monkeypatch):
    data=values(65,'packed','random')
    monkeypatch.delenv('OMNI_XPU_FORCE_SKU',raising=False)
    expected=baseline(data,64,True)
    ops().token_select_remainder(*data,128**-0.5,64,True);torch.xpu.synchronize()
    for forced in (False,True):
        with monkeypatch.context() as context:
            if forced:context.setenv('OMNI_XPU_FORCE_SKU','generic')
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.XPU]) as prof:
                actual=ops().token_select_remainder(*data,128**-0.5,64,True);torch.xpu.synchronize()
            equal(actual,expected,64)
            names=[e.name for e in prof.events()]
            assert any(('SolTokenScanKernel' if forced else 'SolTokenScoreCacheKernel') in n for n in names)
            if forced:assert not any('SolTokenScoreCacheKernel' in n for n in names)


@pytest.mark.parametrize('invalid',['budget','common_dtype','v_shape','scale'])
def test_composite_validates_before_cache_allocation(invalid):
    data=list(values(65,'bhtd','random'));budget=64;scale=128**-0.5
    if invalid=='budget':budget=65
    if invalid=='common_dtype':data[6]=data[6].bool()
    if invalid=='v_shape':data[5]=data[5][:,:-1]
    if invalid=='scale':scale=float('nan')
    with pytest.raises(RuntimeError):ops().token_select_remainder(*data,scale,budget,True)
