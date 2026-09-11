"""Bounded native producer statistics/carriers against independent XPU math."""
import math
import pytest
import torch
import torch.nn.functional as F
from test_cute_sol_carriers_correctness import NAMES,reference,quantize


def native():
    from omni_xpu_kernel import cute
    cute._ensure_loaded()
    return torch.ops.omni_xpu_sol_attn


def producer_reference(q,k,v,kmean,vscale,scale,tau,lengths):
    expected=reference(q,k,v,scale,tau,lengths)
    b,t,h,d=q.shape;n=(t+63)//64
    live=(t-torch.arange(n,device=q.device)*64).clamp_max(64)
    if lengths.numel():live=torch.minimum(live,lengths.clamp_min(1))
    token=torch.arange(t,device=q.device);valid=token%64<live[token//64]
    qf,kf,vf=(x.permute(0,2,1,3).float() for x in (q,k,v))
    centered=kf-kmean.reshape(1,h,1,d)
    kr,ks=quantize(centered,1e-8)
    kr.masked_fill_(~valid.reshape(1,1,t,1),0);ks.masked_fill_(~valid.reshape(1,1,t),0)
    vf=vf*valid.reshape(1,1,t,1)
    vscaled=vf*vscale.reciprocal().reshape(1,h,1,d)
    vr=vscaled.round().clamp(-127,127).to(torch.int8)
    expected['k']=kr.permute(0,2,1,3);expected['ks']=ks
    expected['v']=vr.permute(0,2,1,3);expected['vs']=vscale.reshape(1,h,d)
    qmean=expected['cq'].float()*expected['cqs'].unsqueeze(-1)
    expected['threshold']=tau*((qmean.square()*expected['kvar'].unsqueeze(-2)).sum(-1)*(scale*math.log2(math.e))**2+1e-6).sqrt()
    expected['_scaled']['k']=(centered*ks.clamp_min(1e-8).reciprocal().unsqueeze(-1)).permute(0,2,1,3)
    expected['_scaled']['v']=vscaled.permute(0,2,1,3)
    ksum=F.pad(kf*valid.reshape(1,1,t,1),(0,0,0,n*64-t)).reshape(b,h,n,64,d).sum(-2)
    vmax=F.pad(vf.abs(),(0,0,0,n*64-t)).reshape(b,h,n,64,d).amax(-2)
    expected.update(vmax=vmax,ksum=ksum,next_k=ksum.sum(-2)/live.sum(),next_v=(vmax.amax(-2)/127*1.1).clamp_min(1e-8))
    return expected


def check(c,expected):
    for index,name in enumerate((*NAMES,'vmax','next_k','next_v','ksum')):
        actual=c[index];wanted=expected[name]
        if name=='threshold':
            # CQ was independently admitted above, including only adjacent
            # values at explicit FP32 half-integer ties. Propagate that admitted
            # discrete value into an independent FP32 threshold calculation.
            qm=c[6].float()*expected['cqs'].unsqueeze(-1)
            wanted=1.3*((qm.square()*expected['kvar'].unsqueeze(-2)).sum(-1)*
                        (128**-0.5*math.log2(math.e))**2+1e-6).sqrt()
        assert actual.device.type=='xpu' and actual.dtype==wanted.dtype and actual.shape==wanted.shape
        if actual.dtype==torch.int8:
            torch.testing.assert_close(actual,wanted,rtol=0,atol=1)
            scaled=expected['_scaled'][name]
            distance=(0.5-(scaled-scaled.round()).abs()).abs()
            budget=8*torch.finfo(torch.float32).eps*scaled.abs().clamp_min(1)
            assert bool(((actual==wanted)|(distance<=budget)).all().item()),name
        else:torch.testing.assert_close(actual,wanted,rtol=2e-5,atol=1e-7)


@pytest.mark.parametrize('tokens,chunk_size',[(1,64),(65,64),(257,128),(577,192)])
@pytest.mark.parametrize('packed',[False,True])
@pytest.mark.parametrize('masked',[False,True])
@pytest.mark.parametrize('zeros',[False,True])
def test_bounded_producer_carriers_and_next_stats(tokens,chunk_size,packed,masked,zeros):
    gen=torch.Generator(device='xpu').manual_seed(16631+tokens)
    if packed:
        storage=torch.randn((1,tokens,3,3,128),device='xpu',dtype=torch.bfloat16,generator=gen)
        q,k,v=storage.unbind(2)
    else:
        q,k,v=(torch.randn((1,3,tokens,128),device='xpu',dtype=torch.bfloat16,generator=gen).permute(0,2,1,3) for _ in range(3))
    if zeros:q.zero_();k.zero_();v.zero_()
    before=[x.clone() for x in (q,k,v)]
    lengths=torch.arange((tokens+63)//64,device='xpu',dtype=torch.int32)*31-1 if masked else torch.empty(0,device='xpu',dtype=torch.int32)
    km=torch.randn((3,128),device='xpu',generator=gen)*0.3
    vs=torch.rand((3,128),device='xpu',generator=gen)*0.02+0.005
    km_before,vs_before=km.clone(),vs.clone()
    scale,tau=128**-0.5,1.3
    c=native().producer_begin(q,tokens,3,vs)
    for begin in range(0,tokens,chunk_size):
        native().producer_chunk(*(x[:,begin:begin+chunk_size] for x in (q,k,v)),c,km,begin,lengths)
    native().producer_finish(c,scale,tau,lengths)
    expected=producer_reference(q,k,v,km,vs,scale,tau,lengths)
    check(c,expected)
    for actual,wanted in zip((q,k,v,km,vs),(*before,km_before,vs_before)):
        torch.testing.assert_close(actual,wanted,rtol=0,atol=0)
    assert c[5].data_ptr()==vs.data_ptr()


@pytest.mark.parametrize('bad', ['offset','width','dtype','workspace','lengths','kmean'])
def test_producer_rejects_invalid_metadata(bad):
    q=torch.zeros((1,65,3,128),device='xpu',dtype=torch.bfloat16)
    vs=torch.ones((3,128),device='xpu');km=torch.zeros_like(vs)
    c=native().producer_begin(q,65,3,vs);lengths=torch.empty(0,device='xpu',dtype=torch.int32)
    offset=0
    if bad=='offset':offset=1
    if bad=='width':q=q[...,:127]
    if bad=='dtype':q=q.float()
    if bad=='workspace':c=c[:16]
    if bad=='lengths':lengths=torch.ones(1,device='xpu',dtype=torch.int32)
    if bad=='kmean':km=km[:,:127].contiguous()
    with pytest.raises(RuntimeError):native().producer_chunk(q,q,q,c,km,offset,lengths)
