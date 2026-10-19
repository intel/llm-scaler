"""Upstream codec reference, distinct from native kernel indexing/parallelism."""
import pytest
import torch
from omni_xpu_kernel import kitchen
from test_kitchen_decode import reference

pytestmark=pytest.mark.skipif(not torch.xpu.is_available(),reason="requires XPU")


def rotate(x):
    # Dense normalized Walsh-Hadamard matrix, independently of native shuffles.
    h=torch.ones(1,1,device=x.device,dtype=torch.float64)
    while h.shape[0]<128:
        h=torch.cat([torch.cat([h,h],1),torch.cat([h,-h],1)],0)
    coefficient=torch.tensor(128**-.5,dtype=torch.float32).double().item()
    return (x.double().reshape(*x.shape[:-1],2,128) @ (h*coefficient)).float().reshape(x.shape)


def query_codec(q,hk):
    b,h,s,d=q.shape
    rows=h//hk*s
    x=rotate(q).reshape(b,hk,rows,d)
    x=torch.nn.functional.pad(x,(0,0,0,128-rows)).reshape(b,hk,8,2,8,256)
    scale=x.abs().amax((3,5),keepdim=True)/127+1e-7
    return ((x*scale.double().reciprocal().float()).round().clamp(-127,127)*scale).reshape(b,hk,128,d)[:,:,:rows].reshape(q.shape)


def cache(device="xpu",capacity=1536):
    p=(capacity+511)//512
    return (torch.zeros(2,p,2,512,256,device=device,dtype=torch.int8),
            torch.zeros(2,p,2,256,512,device=device,dtype=torch.int8),
            torch.ones(2,p,2,32,device=device),torch.ones(2,p,2,256,device=device))


def decoded(k,v,ks,vs):
    b,p,h,_,d=k.shape
    key=k.float().reshape(b,p,h,8,8,4,2,d)*ks.reshape(b,p,h,8,1,4,1,1)
    key=key.reshape(b,p,h,512,d).permute(0,2,1,3,4).reshape(b,h,p*512,d)
    value=(v.float()*vs[...,None]).transpose(-1,-2).permute(0,2,1,3,4).reshape_as(key)
    return key,value


@pytest.mark.parametrize("seq",[1,4,8])
def test_native_cache_codec_pages_circular_and_rollback(seq):
    torch.manual_seed(163)
    k,v,ks,vs=cache()
    sk=torch.randn(2,2,1536,256,device="xpu",dtype=torch.bfloat16)
    sv=torch.randn_like(sk)
    q=torch.randn(2,4,seq,256,device="xpu",dtype=sk.dtype)
    for i,n in enumerate([0,511,513,1023,1030,1025,1024,1536]):
        length=torch.tensor([n],device="xpu",dtype=torch.int64)
        if not i:source_key,source_value=sk,sv
        else:
            source_key=torch.full((2,2,1024,256),torch.nan,device="xpu",dtype=sk.dtype)
            source_value=torch.full_like(source_key,torch.nan)
            start=max(0,((max(n-1,0)//512)-1)*512)
            ids=torch.arange(start,n,device="xpu")
            source_key[:,:,ids%1024]=sk[:,:,ids]
            source_value[:,:,ids%1024]=sv[:,:,ids]
        kitchen.int8_decode_update(source_key,source_value,k,v,ks,vs,length,1536,i==0)
        out,lse=kitchen.int8_decode(q,k,v,ks,vs,length,1536)
        kk,vv=decoded(k,v,ks,vs)
        expected,el=reference(query_codec(q,2),kk,vv,length.expand(2),False)
        torch.testing.assert_close(out,expected.to(out.dtype),rtol=.01,atol=.002)
        torch.testing.assert_close(lse,el,rtol=1e-5,atol=1e-5)
        for page in range((n+511)//512):
            live=min(512,n-page*512)
            rotated=rotate(sk[:,:,page*512:page*512+live])
            padded=torch.nn.functional.pad(rotated,(0,0,0,512-live)).reshape(2,2,8,8,4,2,256)
            scale=padded.abs().amax((3,5,6))/127+1e-7
            torch.testing.assert_close(ks[:,page],scale.reshape(2,2,32),rtol=1e-6,atol=1e-9)
            # Validate codes against the stored FP32 scale, independently from
            # the scale's small cross-compiler arithmetic rounding above.
            stored=ks[:,page].reshape(2,2,8,1,4,1,1)
            codes=(padded*stored.double().reciprocal().float()).round().clamp(-127,127).reshape(2,2,512,256).to(torch.int8)
            assert torch.equal(k[:,page],codes)
            value=sv[:,:,page*512:page*512+live].float()
            sc=(value.abs().amax(2)/127).clamp_min(1e-12)
            torch.testing.assert_close(vs[:,page],sc,rtol=1e-6,atol=1e-9)
            assert torch.equal(v[:,page,:,:,:live],(value*vs[:,page,:,None].double().reciprocal().float()).round().clamp(-127,127).to(torch.int8).transpose(-1,-2))


def test_cache_update_attend_graph_dynamic_prefix():
    k,v,ks,vs=cache(capacity=1024)
    sk=torch.randn(2,2,1024,256,device="xpu",dtype=torch.bfloat16);sv=torch.randn_like(sk)
    q=torch.randn(2,4,1,256,device="xpu",dtype=sk.dtype)
    length=torch.tensor([513],device="xpu",dtype=torch.int64)
    kitchen.int8_decode_update(sk,sv,k,v,ks,vs,length,1024,True)
    kitchen.int8_decode_update(sk,sv,k,v,ks,vs,length,1024)
    kitchen.int8_decode(q,k,v,ks,vs,length,1024)
    torch.xpu.synchronize();graph=torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        kitchen.int8_decode_update(sk,sv,k,v,ks,vs,length,1024)
        out,lse=kitchen.int8_decode(q,k,v,ks,vs,length,1024)
    for n in [0,511,513,1024,1023]:
        length.fill_(n);graph.replay();torch.xpu.synchronize()
        kk,vv=decoded(k,v,ks,vs)
        expected,el=reference(query_codec(q,2),kk,vv,length.expand(2),False)
        torch.testing.assert_close(out,expected.to(out.dtype),rtol=.01,atol=.002)
        torch.testing.assert_close(lse,el,rtol=1e-5,atol=1e-5)


def test_cache_partial_last_page_and_length_clamp():
    capacity=641
    k,v,ks,vs=cache(capacity=capacity)
    sk=torch.randn(2,2,capacity,256,device="xpu",dtype=torch.bfloat16);sv=torch.randn_like(sk)
    q=torch.randn(2,6,8,256,device="xpu",dtype=sk.dtype)
    length=torch.tensor([capacity],device="xpu",dtype=torch.int64)
    kitchen.int8_decode_update(sk,sv,k,v,ks,vs,length,capacity,True)
    length.fill_(capacity+8)
    actual,lse=kitchen.int8_decode(q,k,v,ks,vs,length,capacity)
    kk,vv=decoded(k,v,ks,vs)
    expected,el=reference(query_codec(q,2),kk,vv,length.clamp_max(capacity).expand(2),False)
    torch.testing.assert_close(actual,expected.to(actual.dtype),rtol=.01,atol=.002)
    torch.testing.assert_close(lse,el,rtol=1e-5,atol=1e-5)
