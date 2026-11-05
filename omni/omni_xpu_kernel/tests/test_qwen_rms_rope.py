"""Capability-selected paired row reuse preserves the native RMS/RoPE contract."""
import pytest
import torch

from omni_xpu_kernel import rotary


@pytest.fixture(autouse=True)
def xpu_runtime(monkeypatch):
    if not torch.xpu.is_available():
        pytest.skip("XPU unavailable")
    monkeypatch.delenv("OMNI_XPU_FORCE_SKU", raising=False)


@pytest.fixture
def row_reuse_device():
    properties = torch.xpu.get_device_properties()
    if (32 not in properties.sub_group_sizes or properties.max_work_group_size < 128
            or properties.max_num_sub_groups < 4):
        pytest.skip("Qwen row reuse requires four 32-wide subgroups")


def inputs(sequence=17, freq_layout="normal", dtype=torch.bfloat16, heads=32, batch=1):
    torch.manual_seed(127)
    q = torch.randn(batch, sequence, heads, 128, device="xpu", dtype=dtype) * 7
    k = torch.randn_like(q) * 3
    n = 1 if freq_layout == "time_broadcast" else sequence
    h = heads if freq_layout in ("per_head", "full", "batch_strided") else 1
    b = batch if freq_layout in ("per_batch", "full", "batch_strided") else 1
    pairs = 1 if freq_layout == "pair_broadcast" else 64
    if freq_layout in ("strided", "batch_strided"):
        pairs *= 2
    if freq_layout == "batch_strided":
        b *= 2
    f = torch.randn(b, n, h, pairs, 2, 2, device="xpu", dtype=torch.float32)
    if freq_layout in ("strided", "batch_strided"):
        f = f[:, :, :, ::2]
    if freq_layout == "batch_strided":
        f = f[::2]
    qs = torch.randn(128, device="xpu", dtype=torch.bfloat16)
    ks = torch.randn_like(qs)
    return q, k, f, qs, ks


def generic(monkeypatch, tensors, *, split_half=False):
    with monkeypatch.context() as scope:
        scope.setenv("OMNI_XPU_FORCE_SKU", "generic")
        fn = rotary.rms_kitchen_rope_split_half if split_half else rotary.rms_kitchen_rope
        return fn(*tensors)


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
def test_qwen_default_kernel_compile_and_nondefault_stream(monkeypatch, row_reuse_device):
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
def test_qwen_other_dtypes_preserve_fallback(monkeypatch, dtype, heads):
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


@pytest.mark.parametrize("batch,heads,layout", [
    (1, 1, "normal"), (1, 16, "full"), (1, 24, "per_head"), (1, 31, "pair_broadcast"),
    (2, 24, "normal"), (2, 31, "per_batch"), (2, 32, "full"),
    (3, 7, "batch_strided"), (3, 40, "time_broadcast"),
])
@torch.inference_mode()
def test_batch_and_heads_use_row_reuse_with_correct_frequency_indexing(
        monkeypatch, row_reuse_device, batch, heads, layout):
    tensors = inputs(sequence=5, freq_layout=layout, batch=batch, heads=heads)
    expected = generic(monkeypatch, tensors)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.XPU]) as profile:
        actual = rotary.rms_kitchen_rope(*tensors)
        torch.xpu.synchronize()
    assert any("launch_qwen_d128_rms_rope" in e.name for e in profile.events())
    assert_bits(actual, expected)


@pytest.mark.parametrize("batch,sequence,heads", [(0, 5, 24), (2, 0, 24), (2, 5, 0)])
@torch.inference_mode()
def test_empty_outer_dimensions(monkeypatch, batch, sequence, heads):
    tensors = inputs(sequence=sequence, freq_layout="time_broadcast", batch=batch, heads=heads)
    assert_bits(rotary.rms_kitchen_rope(*tensors), generic(monkeypatch, tensors))


@pytest.mark.parametrize("sequence", [23, 294, 4119, 4538])
@torch.inference_mode()
def test_moss_packed_split_half_uses_row_reuse(monkeypatch, sequence):
    # Prefix lengths and packed-QKV row stride from the existing MOSS trace.
    q, k, f, qs, ks = inputs(sequence=sequence, heads=24)
    packed = torch.empty(1, sequence, 3, 24, 128, device="xpu", dtype=q.dtype)
    packed[:, :, 0].copy_(q)
    packed[:, :, 1].copy_(k)
    q, k = packed[:, :, 0], packed[:, :, 1]
    assert q.stride(1) == k.stride(1) == 9216
    tensors = (q, k, f, qs, ks)
    expected = generic(monkeypatch, tensors, split_half=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.XPU]) as profile:
        actual = rotary.rms_kitchen_rope_split_half(*tensors)
        torch.xpu.synchronize()
    assert_bits(actual, expected)
    names = [e.name for e in profile.events()]
    assert any("launch_qwen_d128_rms_rope" in name for name in names)


@pytest.mark.parametrize("batch,heads,layout", [(1,1,"normal"),(1,24,"per_head"),(2,7,"batch_strided"),(3,31,"full")])
@pytest.mark.parametrize("packed", [False,True])
@pytest.mark.parametrize("split_half", [False,True])
@torch.inference_mode()
def test_paired_layout_row_reuse(monkeypatch,row_reuse_device,batch,heads,layout,packed,split_half):
    q,k,f,qs,ks=inputs(sequence=7,batch=batch,heads=heads,freq_layout=layout)
    if packed:
        storage=torch.empty(batch,7,3,heads,128,device=q.device,dtype=q.dtype)
        storage[:,:,0].copy_(q);storage[:,:,1].copy_(k)
        q,k=storage[:,:,0],storage[:,:,1]
    tensors=(q,k,f,qs,ks)
    expected=generic(monkeypatch,tensors,split_half=split_half)
    fn=rotary.rms_kitchen_rope_split_half if split_half else rotary.rms_kitchen_rope
    assert_bits(fn(*tensors),expected)


@pytest.mark.parametrize("case", ["zeros","tiny","outliers","nonfinite"])
@torch.inference_mode()
def test_split_half_adversarial_bits(monkeypatch,case):
    tensors=inputs(heads=24)
    for x in tensors[:2]:
        if case=="zeros":x.zero_();x.flatten()[1::2]=-0.0
        elif case=="tiny":x.mul_(1e-20)
        elif case=="outliers":x.mul_(1e4);x.flatten()[::97]=-3e6
        else:x.flatten()[:3]=torch.tensor([float("inf"),-float("inf"),float("nan")],device="xpu",dtype=x.dtype)
    assert_bits(rotary.rms_kitchen_rope_split_half(*tensors),generic(monkeypatch,tensors,split_half=True))
