"""B70 Qwen row reuse preserves the existing native RMS/RoPE contract."""
import pytest
import torch

import omni_xpu_kernel
from omni_xpu_kernel import rotary


@pytest.fixture(autouse=True)
def b70(monkeypatch):
    if not torch.xpu.is_available():
        pytest.skip("XPU unavailable")
    if omni_xpu_kernel.__xpu_target__ != "bmg" or torch.xpu.get_device_properties(0).device_id != 0xE223:
        pytest.skip("Qwen row-cache regression requires physical B70")
    monkeypatch.delenv("OMNI_XPU_FORCE_SKU", raising=False)


def inputs(sequence=17, freq_layout="normal", dtype=torch.bfloat16, heads=32):
    torch.manual_seed(127)
    q = torch.randn(1, sequence, heads, 128, device="xpu", dtype=dtype) * 7
    k = torch.randn_like(q) * 3
    n = 1 if freq_layout == "time_broadcast" else sequence
    h = heads if freq_layout == "per_head" else 1
    pairs = 1 if freq_layout == "pair_broadcast" else 64
    if freq_layout == "strided":
        pairs *= 2
    f = torch.randn(1, n, h, pairs, 2, 2, device="xpu", dtype=torch.float32)
    if freq_layout == "strided":
        f = f[:, :, :, ::2]
    qs = torch.randn(128, device="xpu", dtype=torch.bfloat16)
    ks = torch.randn_like(qs)
    return q, k, f, qs, ks


def generic(monkeypatch, tensors):
    with monkeypatch.context() as scope:
        scope.setenv("OMNI_XPU_FORCE_SKU", "generic")
        return rotary.rms_kitchen_rope(*tensors)


def assert_bits(actual, expected):
    for a, b in zip(actual, expected):
        assert torch.equal(a.view(torch.int16), b.view(torch.int16))


@pytest.mark.parametrize("sequence", [0, 1, 17, 335, 4096])
@torch.inference_mode()
def test_qwen_sequence_bits_match_generic(monkeypatch, sequence):
    tensors = inputs(sequence, "time_broadcast" if sequence == 0 else "normal")
    assert_bits(rotary.rms_kitchen_rope(*tensors), generic(monkeypatch, tensors))


@pytest.mark.parametrize("layout", ["time_broadcast", "per_head", "pair_broadcast", "strided"])
@torch.inference_mode()
def test_qwen_frequency_broadcast_and_strides(monkeypatch, layout):
    tensors = inputs(freq_layout=layout)
    assert_bits(rotary.rms_kitchen_rope(*tensors), generic(monkeypatch, tensors))


@pytest.mark.parametrize("case", ["zeros", "tiny", "outliers", "nonfinite"])
@torch.inference_mode()
def test_qwen_adversarial_bits(monkeypatch, case):
    tensors = inputs()
    for x in tensors[:2]:
        if case == "zeros":
            x.zero_(); x.flatten()[1::2] = -0.0
        elif case == "tiny":
            x.mul_(1e-20)
        elif case == "outliers":
            x.mul_(1e4); x.flatten()[::97] = -3e6
        else:
            x.flatten()[:3] = torch.tensor([float("inf"), -float("inf"), float("nan")], device="xpu", dtype=x.dtype)
    assert_bits(rotary.rms_kitchen_rope(*tensors), generic(monkeypatch, tensors))


@torch.inference_mode()
def test_qwen_inplace_alias_and_noncontiguous_fallback(monkeypatch):
    tensors = inputs()
    expected = generic(monkeypatch, tensors)
    q, k, f, qs, ks = tensors
    q_out, k_out = rotary.rms_kitchen_rope_(q.clone(), k.clone(), f, qs, ks)
    assert_bits((q_out, k_out), expected)
    for index, x in enumerate((q, k)):
        storage = torch.empty(*x.shape[:-1], 256, device="xpu", dtype=x.dtype)
        storage[..., ::2].copy_(x)
        noncontiguous = storage[..., ::2]
        values = list(tensors)
        values[index] = noncontiguous
        assert not noncontiguous.is_contiguous()
        assert_bits(rotary.rms_kitchen_rope(*values), generic(monkeypatch, values))
    q0, k0 = q.clone(), k.clone()
    actual = rotary.rms_kitchen_rope_(q0, k0, f, qs, ks)
    assert actual[0].data_ptr() == q0.data_ptr()
    assert actual[1].data_ptr() == k0.data_ptr()


@pytest.mark.parametrize("sku", ["generic", "b60", "b70"])
@torch.inference_mode()
def test_qwen_forced_policy_uses_original_kernel(monkeypatch, sku):
    tensors = inputs()
    expected = generic(monkeypatch, tensors)
    monkeypatch.setenv("OMNI_XPU_FORCE_SKU", sku)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.XPU]) as profile:
        actual = rotary.rms_kitchen_rope(*tensors)
        torch.xpu.synchronize()
    assert_bits(actual, expected)
    names = [event.name for event in profile.events()]
    assert any("launch_rms_rope" in name for name in names)
    assert not any("launch_qwen_d128_rms_rope" in name for name in names)


@torch.inference_mode()
def test_qwen_default_kernel_compile_and_nondefault_stream(monkeypatch):
    tensors = inputs()
    expected = generic(monkeypatch, tensors)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.XPU]) as profile:
        actual = rotary.rms_kitchen_rope(*tensors)
        torch.xpu.synchronize()
    assert any("launch_qwen_d128_rms_rope" in event.name for event in profile.events())
    assert_bits(actual, expected)


    compiled = torch.compile(rotary.rms_kitchen_rope, backend="eager", fullgraph=True)
    assert_bits(compiled(*tensors), expected)
    stream = torch.xpu.Stream()
    stream.wait_stream(torch.xpu.current_stream())
    with torch.xpu.stream(stream):
        actual = rotary.rms_kitchen_rope(*tensors)
    stream.synchronize()
    assert_bits(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("heads", [31, 32])
@torch.inference_mode()
def test_qwen_other_dtype_and_heads_preserve_fallback(monkeypatch, dtype, heads):
    tensors = inputs(dtype=dtype, heads=heads)
    assert_bits(rotary.rms_kitchen_rope(*tensors), generic(monkeypatch, tensors))


@torch.inference_mode()
def test_qwen_graph_replay_observes_changed_inputs(monkeypatch):
    tensors = inputs()
    for _ in range(3):
        rotary.rms_kitchen_rope(*tensors)
    torch.xpu.synchronize()
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        actual = rotary.rms_kitchen_rope(*tensors)
    tensors[0].mul_(1.25)
    expected = generic(monkeypatch, tensors)
    graph.replay()
    torch.xpu.synchronize()
    assert_bits(actual, expected)
