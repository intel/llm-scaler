"""Independent XPU oracles for pooled DPAS, routing and tail composition."""
import math

import pytest
import torch


def ops():
    from omni_xpu_kernel import cute
    cute._ensure_loaded()
    return torch.ops.omni_xpu_sol_attn


@pytest.mark.parametrize("m,n", [(1,1), (7,3), (63,65), (64,64), (65,63), (247,247)])
@pytest.mark.parametrize("pattern", ["random", "extremes"])
def test_centroid_dpas(m, n, pattern):
    gen = torch.Generator(device="xpu").manual_seed(7327 + m + n)
    q = torch.randint(-128,128,(2,3,m,128),device="xpu",dtype=torch.int8,generator=gen)
    k = torch.randint(-128,128,(2,3,n,128),device="xpu",dtype=torch.int8,generator=gen)
    if pattern == "extremes":
        q.fill_(-128); k.fill_(127)
        q[...,1::2] = 127; k[...,::3] = -128
    qs = torch.rand((2,3,m),device="xpu",generator=gen)*0.1
    ks = torch.rand((2,3,n),device="xpu",generator=gen)*0.1
    scale = 128**-0.5
    actual = ops().centroid_scores(q,k,qs,ks,scale)
    expected = (q.float() @ k.float().transpose(-1,-2)) * (qs.unsqueeze(-1)*float(scale*math.log2(math.e))) * ks.unsqueeze(-2)
    torch.testing.assert_close(actual,expected,rtol=5e-7,atol=5e-7)


def route_reference(scores, threshold, vsum, vs, lengths, tokens, sinks, topk, tail, groups):
    n = scores.shape[-1]
    j = torch.arange(n,device=scores.device)
    sink_s,sink_e,sink_qs,sink_qe = sinks
    qsink = (j >= sink_qs) & (j < sink_qe)
    ksink = (j >= sink_s) & (j < sink_e)
    threshold = threshold.unsqueeze(-1)
    if topk >= 0:
        ranked = scores.masked_fill(ksink,float("-inf"))
        threshold = ranked.topk(topk,dim=-1).values[...,-1:] if topk else torch.full_like(threshold,float("inf"))
        if topk: threshold = threshold - threshold.abs()*1e-5
    routes = (scores >= threshold) | qsink.reshape(1,1,n,1) | ksink | ((j[:,None]-j[None,:]).abs()<=1)
    partner = (j ^ 1).clamp_max(n-1)
    common_rows = ~routes & ~routes.index_select(2,partner) if groups else torch.zeros_like(routes)
    common = common_rows[:,:,::2].to(torch.uint8)
    refs = scores.masked_fill(~common_rows,float("-inf")).amax(-1)
    keep = ~routes & ~common_rows if tail else torch.zeros_like(routes)
    masked = scores.masked_fill(~keep,float("-inf"))
    maximum = masked.amax(-1)
    p = torch.where(keep,torch.exp2(masked-maximum.unsqueeze(-1)),0)
    live_lengths = (tokens-64*j).clamp_max(64)
    if lengths.numel(): live_lengths = torch.minimum(live_lengths,lengths.clamp_min(1))
    denominator = (p*live_lengths).sum(-1)
    numerator = (p.to(torch.bfloat16).float() @ vsum.float()) / vs.unsqueeze(-2)
    state = torch.cat((maximum.unsqueeze(-1),denominator.unsqueeze(-1),numerator),-1)
    return routes.to(torch.uint8),state,refs,common


@pytest.mark.parametrize("n", [1,3,7,65])
@pytest.mark.parametrize("topk", [-1,0,1])
@pytest.mark.parametrize("groups", [False,True])
def test_routes_tail_and_common_candidates(n,topk,groups):
    if n == 1 and topk == 1: pytest.skip("no selectable non-sink block")
    gen = torch.Generator(device="xpu").manual_seed(8447+n)
    # Discrete scores exercise inclusive ties and top-k ties exactly.
    scores = torch.randint(-16,17,(2,3,n,n),device="xpu",generator=gen).float()*0.25
    threshold = torch.zeros((2,3,n),device="xpu")
    vsum = torch.randn((2,3,n,128),device="xpu",generator=gen).to(torch.bfloat16)*8
    vs = torch.rand((2,3,128),device="xpu",generator=gen)*0.1+0.01
    tokens = n*64-17
    lengths = torch.arange(n,device="xpu",dtype=torch.int32)*17-3
    sinks = (0,min(n,1),1,min(n,2)) if n > 1 else (0,0,0,0)
    for tail in (False,True):
        actual = ops().pooled_routes(scores,threshold,vsum,vs,lengths,tokens,*sinks,topk,tail,groups)
        expected = route_reference(scores,threshold,vsum,vs,lengths,tokens,sinks,topk,tail,groups)
        for i,(got,want) in enumerate(zip(actual,expected)):
            torch.testing.assert_close(got,want,rtol=2e-5 if i==1 else 0,atol=2e-4 if i==1 else 0)


@pytest.mark.parametrize("tokens", [65,257,521])
@pytest.mark.parametrize("tail", [False,True])
def test_native_nonaugmented_composition(tokens,tail):
    from test_cute_sol_carriers_correctness import NAMES
    from test_cute_sol_prepared_correctness import reference as attention_reference
    gen = torch.Generator(device="xpu").manual_seed(9437+tokens)
    packed = torch.randn((2,tokens,3,3,128),device="xpu",dtype=torch.bfloat16,generator=gen)
    q,k,v = packed.unbind(2)
    scale,tau = 128**-0.5,1.3
    lengths = torch.empty(0,device="xpu",dtype=torch.int32)
    c = dict(zip(NAMES,ops().prepare_carriers(q,k,v,scale,tau,lengths)))
    scores = ops().centroid_scores(c['cq'],c['ck'],c['cqs'],c['cks'],scale)
    expected_scores = (c['cq'].float() @ c['ck'].float().transpose(-1,-2)) * (c['cqs'].unsqueeze(-1)*float(scale*math.log2(math.e))) * c['cks'].unsqueeze(-2)
    sinks = (0,1,0,1)
    routes,state,*_ = ops().pooled_routes(scores,c['threshold'],c['vsum'],c['vs'],lengths,tokens,*sinks,-1,tail,False)
    rr,rs,*_ = route_reference(expected_scores,c['threshold'],c['vsum'],c['vs'],lengths,tokens,sinks,-1,tail,False)
    torch.testing.assert_close(routes,rr,rtol=0,atol=0)
    args = [c[name] for name in ('q','k','v','qs','ks','vs')]
    output = ops().forward_cute_prepared(*args,routes,state,scale)
    expected = attention_reference(*args,rr,rs,scale)
    torch.testing.assert_close(output,expected,rtol=0.02,atol=0.02)


def test_topk_backoff_includes_near_threshold_scores():
    n=7
    scores=torch.full((1,1,n,n),-4.0,device="xpu")
    scores[...,3]=2.0
    scores[...,4]=2.0-1e-5
    scores[...,5]=2.0-4e-5
    threshold=torch.zeros((1,1,n),device="xpu")
    vsum=torch.zeros((1,1,n,128),device="xpu",dtype=torch.bfloat16)
    vs=torch.ones((1,1,128),device="xpu")
    lengths=torch.empty(0,device="xpu",dtype=torch.int32)
    routes,*_=ops().pooled_routes(scores,threshold,vsum,vs,lengths,n*64,0,0,0,0,1,False,False)
    assert routes[0,0,0,3].item()==1
    assert routes[0,0,0,4].item()==1
    assert routes[0,0,0,5].item()==0
