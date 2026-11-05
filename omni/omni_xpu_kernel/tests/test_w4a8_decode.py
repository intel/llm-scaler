"""Grouped packed W4A8 arithmetic and public native API contracts."""
import pytest
import torch
from omni_xpu_kernel import int8

pytestmark = pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU required")


def packet(rows, columns, width, group, scale_dtype, with_codebook=True):
    generator = torch.Generator().manual_seed(31971 + rows + width)
    x = torch.randint(-128, 128, (rows, width), generator=generator, dtype=torch.int8)
    unsigned = torch.randint(0, 16, (columns, width), generator=generator, dtype=torch.int32)
    packed = (unsigned[:, 0::2] | (unsigned[:, 1::2] << 4)).to(torch.int8)
    scales = (torch.rand(columns, width // group, generator=generator) * 180).to(scale_dtype)
    codebook = torch.linspace(-1, 1, 16) if with_codebook else None
    values = codebook[unsigned.long()] if codebook is not None else unsigned.float() - 8
    weight = (values.reshape(columns, -1, group) * scales.float().unsqueeze(-1)).reshape(columns, width)
    weight = weight.round().clamp(-127, 127).to(torch.int8)
    exact = x.long() @ weight.long().T
    # Integer GEMM has int32 wraparound; the CPU oracle accumulates in int64.
    exact = exact.to(torch.int32)
    return x, packed, scales, codebook, exact


@pytest.mark.parametrize("rows,columns,width,group", [
    (1, 13, 128, 4), (2, 31, 256, 8), (4, 64, 768, 16),
    (8, 17, 512, 32), (1, 257, 4096, 64), (3, 7, 8192, 256),
    (1, 19, 1024, 1024), (1, 3, 384, 48),
    (5, 23, 128, 16), (7, 19, 512, 64),
    (9, 13, 256, 8), (17, 7, 128, 4), (8, 11, 96, 6),
])
@pytest.mark.parametrize("scale_dtype", [torch.float32, torch.float8_e4m3fn])
@pytest.mark.parametrize("with_codebook", [False, True])
def test_native_matches_independent_integer_reference(rows, columns, width, group, scale_dtype, with_codebook):
    x, packed, scales, codebook, exact = packet(rows, columns, width, group, scale_dtype, with_codebook)
    actual = int8.w4a8_linear_prequantized(
        x.to("xpu"), torch.ones(rows, device="xpu"), packed.to("xpu"),
        scales.to("xpu"), torch.ones(columns, device="xpu"),
        codebook.to("xpu") if codebook is not None else None,
        group_size=group, out_dtype=torch.float32)
    torch.testing.assert_close(actual.cpu(), exact.float(), rtol=0, atol=0)


@pytest.mark.parametrize("out_dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_scale_bias_and_leading_dimensions(out_dtype):
    x, packed, scales, codebook, exact = packet(6, 37, 256, 16, torch.float8_e4m3fn)
    xs = torch.linspace(.002, .02, 6)
    ws = torch.linspace(.01, .03, 37)
    bias = torch.linspace(-1, 1, 37, dtype=torch.float32)
    expected = (exact.float() * xs[:, None] * ws + bias).to(out_dtype).reshape(2, 3, 37)
    actual = int8.w4a8_linear_prequantized(
        x.reshape(2, 3, 256).to("xpu"), xs.to("xpu"), packed.to("xpu"),
        scales.to("xpu"), ws.to("xpu"), codebook.to("xpu"), bias.to("xpu"),
        out_dtype=out_dtype)
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-6 if out_dtype==torch.float32 else .008,
                               atol=1e-4 if out_dtype==torch.float32 else .02)


def test_rounding_ties_clamp_and_uniform_zero():
    codebook = torch.tensor([-.5,.5,-1.5,1.5,-2.5,2.5,-127.5,127.5,-200,200,0,-0.,-3.5,3.5,126.5,-126.5])
    codes = torch.arange(16, dtype=torch.int32).repeat(4, 4)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.int8)
    x = torch.arange(-32, 32, dtype=torch.int8).repeat(2, 1)
    scales = torch.ones(4, 4);scales[0]=0
    weight=(codebook[codes.long()].reshape(4,4,16)*scales[:,:,None]).round().clamp(-127,127).to(torch.int8).reshape(4,64)
    expected = (x.long() @ weight.long().T).float()
    actual=int8.w4a8_linear_prequantized(x.to("xpu"),torch.ones(2,device="xpu"),packed.to("xpu"),scales.to("xpu"),torch.ones(4,device="xpu"),codebook.to("xpu"),out_dtype=torch.float32)
    torch.testing.assert_close(actual.cpu(),expected,rtol=0,atol=0)


def test_graph_stream_compile_and_empty_rows():
    x,packed,scales,codebook,exact=packet(1,19,256,16,torch.float32)
    args=[x.to("xpu"),torch.ones(1,device="xpu"),packed.to("xpu"),scales.to("xpu"),torch.ones(19,device="xpu"),codebook.to("xpu")]
    def run(a):return int8.w4a8_linear_prequantized(a,*args[1:],out_dtype=torch.float32)
    expected=run(args[0]);actual=torch.compile(run,backend="eager",fullgraph=True)(args[0])
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    graph=torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):captured=run(args[0])
    args[0].fill_(0);graph.replay();torch.xpu.synchronize()
    assert torch.count_nonzero(captured).item()==0
    stream=torch.xpu.Stream()
    with torch.xpu.stream(stream):
        args[0].copy_(x.to("xpu"));stream_result=run(args[0])
    stream.synchronize();torch.testing.assert_close(stream_result.cpu(),exact.float(),rtol=0,atol=0)
    result=int8.w4a8_linear_prequantized(args[0][:0],args[1][:0],*args[2:],out_dtype=torch.bfloat16)
    assert result.shape==(0,19) and result.dtype==torch.bfloat16


def test_invalid_metadata_rejected_before_work():
    x,packed,scales,codebook,_=packet(1,19,256,16,torch.float32)
    args=[x.to("xpu"),torch.ones(1,device="xpu"),packed.to("xpu"),scales.to("xpu"),torch.ones(19,device="xpu"),codebook.to("xpu")]
    with pytest.raises(RuntimeError,match="one value per"):
        int8.w4a8_linear_prequantized(args[0],args[1].expand(2),*args[2:])
    with pytest.raises(RuntimeError,match="group scales"):
        int8.w4a8_linear_prequantized(*args[:3],args[3][:,:-1],*args[4:])
    with pytest.raises(RuntimeError,match="same-device FP32"):
        int8.w4a8_linear_prequantized(*args[:5],codebook)


def test_packed_rows_allow_unaligned_contiguous_storage():
    x, packed, scales, codebook, expected = packet(8, 13, 128, 16, torch.float8_e4m3fn)
    def shifted(value):
        storage = torch.empty(value.numel() + 1, dtype=value.dtype, device="xpu")
        result = storage[1:].view(value.shape)
        result.copy_(value)
        return result
    actual = int8.w4a8_linear_prequantized(
        shifted(x), torch.ones(8, device="xpu"), shifted(packed), scales.to("xpu"),
        torch.ones(13, device="xpu"), codebook.to("xpu"), group_size=16, out_dtype=torch.float32)
    assert torch.equal(actual.cpu(), expected.float())
