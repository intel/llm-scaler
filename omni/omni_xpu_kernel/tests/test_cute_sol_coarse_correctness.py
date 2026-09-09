"""Optional coarse branch remains FP32 until one final output rounding."""
import pytest
import torch


def native():
    from omni_xpu_kernel import cute
    cute._ensure_loaded()
    return torch.ops.omni_xpu_sol_attn


@pytest.mark.parametrize('tokens',[1,65,1025])
@pytest.mark.parametrize('masked',[False,True])
@pytest.mark.parametrize('zeros',[False,True])
def test_fp32_coarse_attention(tokens,masked,zeros):
    n=(tokens+63)//64;gen=torch.Generator(device='xpu').manual_seed(18831+tokens)
    q,k=(torch.randn((2,3,n,128),device='xpu',generator=gen) for _ in range(2))
    v=torch.randn((2,3,n,128),device='xpu',generator=gen).mul(64).to(torch.bfloat16)
    if zeros:q.zero_();k.zero_();v.zero_()
    lengths=torch.arange(n,device='xpu',dtype=torch.int32)*31-1 if masked else torch.empty(0,device='xpu',dtype=torch.int32)
    live=(tokens-torch.arange(n,device='xpu')*64).clamp_max(64)
    if masked:live=torch.minimum(live,lengths.clamp_min(1))
    scale=128**-0.5
    expected=torch.softmax(q @ k.transpose(-2,-1)*scale,-1) @ (v.float()/live.reshape(1,1,n,1))
    output=native().coarse_output(q,k,v,lengths,tokens,scale)
    assert output.dtype==torch.float32 and output.device.type=='xpu'
    torch.testing.assert_close(output,expected,rtol=2e-5,atol=2e-5)


@pytest.mark.parametrize('tokens',[1,65,257])
@pytest.mark.parametrize('dtype',[torch.bfloat16,torch.float16])
@pytest.mark.parametrize('gate_dtype',[torch.bfloat16,torch.float16,torch.float32])
def test_coarse_add_single_rounding(tokens,dtype,gate_dtype):
    gen=torch.Generator(device='xpu').manual_seed(18913+tokens)
    output=torch.randn((2,tokens,3,128),device='xpu',dtype=dtype,generator=gen)
    coarse=torch.randn((2,3,(tokens+63)//64,128),device='xpu',generator=gen)
    gate=torch.randn(output.shape,device='xpu',dtype=gate_dtype,generator=gen)
    wanted=(output.float()+gate.float()*coarse.index_select(2,torch.arange(tokens,device='xpu')//64).permute(0,2,1,3)).to(dtype)
    pointer=output.data_ptr()
    native().add_coarse_(output,coarse,gate)
    assert output.data_ptr()==pointer and output.dtype==dtype
    torch.testing.assert_close(output,wanted,rtol=0.008,atol=0.002)
