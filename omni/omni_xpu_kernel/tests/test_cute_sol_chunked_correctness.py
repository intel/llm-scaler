"""Chunk orchestration, RMS/RoPE composition, bootstrap and input ownership."""
import pytest
import torch
from test_cute_sol_producer_correctness import producer_reference,check


def norm_rope_reference(q,k,freqs,weights,rot):
    def run(x,w):
        xf=x.float();normalized=(xf*torch.rsqrt(xf.square().mean(-1,keepdim=True)+1e-6)*w.float()).to(torch.bfloat16)
        a,b=normalized[...,:rot//2].float(),normalized[...,rot//2:rot].float()
        out=normalized.clone()
        out[...,:rot//2]=(freqs[...,0,0].float()*a+freqs[...,0,1].float()*b).to(torch.bfloat16)
        out[...,rot//2:rot]=(freqs[...,1,0].float()*a+freqs[...,1,1].float()*b).to(torch.bfloat16)
        return out
    return run(q,weights[0]),run(k,weights[1])


@pytest.mark.parametrize('tokens,chunk_size',[(257,64),(577,192)])
@pytest.mark.parametrize('heads,rot',[(3,8),(3,128),(56,96)])
@pytest.mark.parametrize('bootstrap',[False,True])
@pytest.mark.parametrize('callable_input',[False,True])
def test_chunked_norm_carriers_bootstrap_and_replay(tokens,chunk_size,heads,rot,bootstrap,callable_input):
    from omni_xpu_kernel.cute.sol_attn_v2 import _prepare_chunked
    from omni_xpu_kernel import rotary
    gen=torch.Generator(device='xpu').manual_seed(17531+tokens+heads+rot)
    source=torch.randn((tokens,3*heads*128),device='xpu',dtype=torch.bfloat16,generator=gen)
    snapshot=source.clone()
    angles=torch.randn((1,tokens,1,rot//2),device='xpu',generator=gen)
    co,si=angles.cos(),angles.sin()
    freqs=torch.stack((co,-si,si,co),-1).reshape(1,tokens,1,rot//2,2,2).to(torch.bfloat16)
    weights=tuple(torch.randn(128,device='xpu',dtype=torch.bfloat16,generator=gen)*0.1+1 for _ in range(2))
    lengths=torch.arange((tokens+63)//64,device='xpu',dtype=torch.int32)*31-1
    chunks=[source[i:i+chunk_size] for i in range(0,tokens,chunk_size)]
    invocations=[]
    def factory():
        invocations.append(1)
        return iter(chunks)
    km=torch.randn((heads,128),device='xpu',generator=gen)*0.1
    vs=torch.rand((heads,128),device='xpu',generator=gen)*0.02+0.005
    c,returned_lengths=_prepare_chunked(factory if callable_input else iter(chunks),tokens,heads,freqs,weights,
        None if bootstrap else km,None if bootstrap else vs,128**-0.5,1.3,1e-6,lengths)
    if callable_input:assert len(invocations)==(2 if bootstrap else 1)
    torch.testing.assert_close(source,snapshot,rtol=0,atol=0)
    torch.testing.assert_close(returned_lengths,lengths,rtol=0,atol=0)
    # The maintained native RMS/RoPE primitive has its own independent numerical
    # check; compare chunk carrier composition against its whole-input result.
    packed=source.clone().view(1,tokens,3,heads,128);q,k,v=packed.unbind(2)
    qr,kr=norm_rope_reference(q,k,freqs,weights,rot)
    rotary.rms_kitchen_rope_split_half_(q,k,freqs,*weights,rot_dim=rot)
    torch.testing.assert_close(q,qr,rtol=0.02,atol=0.02)
    torch.testing.assert_close(k,kr,rtol=0.02,atol=0.02)
    if bootstrap:km,vs=c[17][0],c[18][0]
    expected=producer_reference(q,k,v,km,vs,128**-0.5,1.3,lengths)
    check(c,expected)


@pytest.mark.parametrize('kind',['short','overflow','unaligned','width','dtype','rank'])
def test_chunked_rejects_bad_coverage_and_projection(kind):
    from omni_xpu_kernel.cute.sol_attn_v2 import _prepare_chunked
    t,h=65,3
    q=torch.zeros((t,3*h*128),device='xpu',dtype=torch.bfloat16)
    freqs=torch.eye(2,device='xpu',dtype=torch.bfloat16).expand(1,t,1,48,2,2).contiguous()
    chunks=[q]
    if kind=='short':chunks=[q[:64]]
    if kind=='overflow':chunks=[q[:64],q[:64]]
    if kind=='unaligned':chunks=[q[:63],q[:2]]
    if kind=='width':chunks=[q[:,:-1]]
    if kind=='dtype':chunks=[q.float()]
    if kind=='rank':chunks=[q.unsqueeze(0)]
    with pytest.raises(ValueError):
        _prepare_chunked(chunks,t,h,freqs,(torch.ones(128,device='xpu'),)*2,
            torch.zeros((h,128),device='xpu'),torch.ones((h,128),device='xpu'),128**-0.5,1.3,1e-6,None)
    assert bool((q==0).all().item())
