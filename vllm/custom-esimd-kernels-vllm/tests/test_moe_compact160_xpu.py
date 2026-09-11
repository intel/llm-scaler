"""TP4 独立 S4 数学 golden；不运行计时，GPU 由主 agent 串行调度。"""

import os

import pytest
import torch
from test_moe_m1_asymmetric_v1_xpu import OUTPUT_ATOL, _dso_path


def unpack_s4(packed, scales):
    raw = packed.cpu().view(torch.uint8).int()
    values = torch.stack((raw & 15, raw >> 4), dim=-1).flatten(-2)
    values = torch.where(values >= 8, values - 16, values).float()
    # FP32 dequant for native decode; prefill explicitly rounds this to FP16.
    groups = torch.arange(values.shape[-1]) // 128
    return values * scales.cpu().float()[..., groups]


@pytest.fixture(scope="module")
def case():
    if not torch.xpu.is_available():
        pytest.skip("需要主 agent 分配 XPU 窗口")
    torch.ops.load_library(str(_dso_path()))
    torch.set_num_threads(4)
    g = torch.Generator().manual_seed(16020260908)
    q13 = torch.randint(256, (512, 320, 1280), generator=g, dtype=torch.uint8)
    q2 = torch.randint(256, (512, 2560, 80), generator=g, dtype=torch.uint8)
    s13 = torch.full((512, 320, 20), 1 / 512, dtype=torch.float16)
    s2 = torch.empty(512, 2560, 2, dtype=torch.float16)
    s2[..., 0], s2[..., 1] = 1 / 64, 1 / 16
    shared = [
        (torch.randn(*shape, generator=g) * 0.025).half()
        for shape in ((320, 2560), (2560, 160), (1, 2560))
    ]
    cpu = (q13, s13, q2, s2, *shared)
    device = tuple(t.to("xpu") for t in cpu)
    torch.xpu.synchronize()
    return cpu, device


def inputs(m):
    g = torch.Generator().manual_seed(160 + m)
    x = (torch.randn(m, 2560, generator=g) * 0.3).half()
    # Include the final expert and cross 64/128/256 routing boundaries.
    selected = torch.tensor([0, 63, 64, 127, 128, 255, 256, 383, 510, 511])
    logits = torch.full((m, 512), -9.0, dtype=torch.float16)
    for t in range(m):
        logits[t, selected.roll(t)] = torch.arange(10, dtype=torch.float16) * 0.125
    return x, logits


@pytest.fixture(scope="module", params=[80, 160])
def workspace_case(case, request):
    _, weights = case
    compact = request.param
    if compact == 80:
        def half_gate_up(t):
            axis = 1 if t.ndim == 3 else 0
            return torch.cat((t.narrow(axis, 0, 80), t.narrow(axis, 160, 80)), axis)
        weights = (half_gate_up(weights[0]), half_gate_up(weights[1]),
                   weights[2][..., :40].contiguous(), weights[3][..., :1].contiguous(),
                   half_gate_up(weights[4]), weights[5][:, :80].contiguous(), weights[6])
    router = torch.randint(256, (512, 1280), dtype=torch.uint8, device="xpu")
    scale = torch.full((512, 20), 1 / 512, dtype=torch.float16, device="xpu")
    x = inputs(1)[0].to("xpu")
    torch.xpu.synchronize()
    return compact, weights, router, scale, x


def workspace_reference(compact, x, router, scale, weights):
    name = ("moe_forward_compact160_router_out_v1" if compact == 160 else
            "moe_forward_m1_cutlass_nmajor_int4_fp16_shared_compact80_router_out_v1")
    return getattr(torch.ops.moe_int4_ops, name)(
        x, router, scale, *weights, torch.empty_like(x), 10, 1, 512)


def test_m1_workspace_matches_existing_kernels_and_live_rebind(workspace_case):
    compact, weights, router, scale, x = workspace_case
    workspace = torch.classes.moe_int4_ops.Qwen38M1WorkspaceV1()
    live = list(weights)
    live[0] = torch.nn.Parameter(weights[0].view(torch.int8), requires_grad=False)
    for rebind in (False, True):
        if rebind:
            live[0].data = torch.zeros_like(live[0])
        reference_weights = (live[0].view(torch.uint8), *live[1:])
        expected = workspace_reference(compact, x, router, scale, reference_weights)
        actual = workspace.try_run(x, router, scale, live, compact)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_m1_workspace_isolates_streams_and_replaces_aliased_output(workspace_case):
    compact, weights, router, scale, x = workspace_case
    workspace = torch.classes.moe_int4_ops.Qwen38M1WorkspaceV1()
    streams = [torch.xpu.Stream(), torch.xpu.Stream()]
    observations = []
    for _ in range(4):
        for stream in streams:
            with torch.xpu.stream(stream):
                expected = workspace_reference(compact, x, router, scale, weights)
                actual = workspace.try_run(x, router, scale, weights, compact)
                observations.append((actual.clone(), expected, actual.data_ptr()))
    torch.xpu.synchronize()
    assert observations[0][2] != observations[1][2]
    for actual, expected, _ in observations:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    previous = workspace.try_run(x, router, scale, weights, compact)
    expected = workspace_reference(compact, previous, router, scale, weights)
    actual = workspace.try_run(previous, router, scale, weights, compact)
    assert actual.data_ptr() != previous.data_ptr()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("failure", ["m2", "weight_shape", "weight_dtype", "device"])
def test_m1_workspace_unsupported_contract_does_not_modify_output(workspace_case, failure):
    compact, weights, router, scale, x = workspace_case
    workspace = torch.classes.moe_int4_ops.Qwen38M1WorkspaceV1()
    output = workspace.try_run(x, router, scale, weights, compact)
    before = output.clone()
    bad = list(weights)
    if failure == "m2":
        x = x.repeat(2, 1)
    elif failure == "weight_shape":
        bad[0] = bad[0][:, :-1]
    elif failure == "weight_dtype":
        bad[1] = bad[1].view(torch.int16)
    else:
        x = x.cpu()
    assert workspace.try_run(x, router, scale, bad, compact) is None
    torch.testing.assert_close(output, before, rtol=0, atol=0)


def test_m1_workspace_corrupt_output_is_hard_error(workspace_case):
    compact, weights, router, scale, x = workspace_case
    workspace = torch.classes.moe_int4_ops.Qwen38M1WorkspaceV1()
    output = workspace.try_run(x, router, scale, weights, compact)
    torch.xpu.synchronize()
    output.resize_(2, 2560)
    with pytest.raises(RuntimeError, match="output cache is inconsistent"):
        workspace.try_run(x, router, scale, weights, compact)


def test_m1_workspace_defers_lazy_views_to_legacy_dispatcher(workspace_case):
    compact, weights, router, scale, x = workspace_case
    workspace = torch.classes.moe_int4_ops.Qwen38M1WorkspaceV1()
    assert workspace.try_run(torch._neg_view(x), router, scale, weights, compact) is None
    negative_weights = (*weights[:1], torch._neg_view(weights[1]), *weights[2:])
    assert workspace.try_run(x, router, scale, negative_weights, compact) is None


def golden(x, logits, cpu, *, prefill=False, include_shared=True):
    q13, s13, q2, s2, su, sd, sg = cpu
    probability = logits.float().softmax(-1)
    rw, ids = probability.topk(10, dim=-1)
    rw = rw / rw.sum(-1, keepdim=True)
    if not prefill:
        rw = rw.half().float()
    result = torch.zeros_like(x, dtype=torch.float32)
    for t in range(x.shape[0]):
        for r, e in enumerate(ids[t].tolist()):
            up = unpack_s4(q13[e], s13[e])
            down = unpack_s4(q2[e], s2[e])
            if prefill:
                up, down = up.half().float(), down.half().float()
            gu = up @ x[t].float()
            if prefill:
                gu = gu.half().float()
            act = (torch.nn.functional.silu(gu[:160]) * gu[160:]).half().float()
            route = down @ act
            if prefill:
                route = route.half().float()
            result[t] += rw[t, r] * route
        if include_shared:
            gu = su.float() @ x[t].float()
            act = (torch.nn.functional.silu(gu[:160]) * gu[160:]).half().float()
            gate = torch.sigmoid(sg.float() @ x[t].float())
            result[t] += gate * (sd.float() @ act)
    return result.half(), ids, rw


@pytest.mark.parametrize("m", [*range(1, 9), 9, 32])
def test_native_matches_independent_fp32_routed_golden(case, m):
    cpu, weights = case
    x, logits = inputs(m)
    expected, _, _ = golden(x, logits, cpu)
    output = torch.empty_like(x, device="xpu")
    actual = torch.ops.moe_int4_ops.moe_forward_compact160_out_v1(
        x.to("xpu"), logits.to("xpu"), *weights, output, 10, 1, 512
    )
    assert actual.data_ptr() == output.data_ptr()
    torch.testing.assert_close(actual.cpu(), expected, atol=OUTPUT_ATOL, rtol=0)


def test_native_preserves_stream_isolation_and_output_storage(case):
    cpu, weights = case
    data = [inputs(2), inputs(8)]
    expected = [golden(x, logits, cpu)[0] for x, logits in data]
    device_data = [(x.to("xpu"), logits.to("xpu")) for x, logits in data]
    streams = [torch.xpu.Stream(), torch.xpu.Stream()]
    parent = torch.xpu.current_stream()
    for stream in streams:
        stream.wait_stream(parent)
    outputs = []
    for _ in range(12):
        for i, stream in enumerate(streams):
            with torch.xpu.stream(stream):
                x, logits = device_data[i]
                out = torch.empty_like(x)
                torch.ops.moe_int4_ops.moe_forward_compact160_out_v1(
                    x, logits, *weights, out, 10, 1, 512
                )
                outputs.append((i, out))
    for stream in streams:
        parent.wait_stream(stream)
    for i, out in outputs:
        torch.testing.assert_close(out.cpu(), expected[i], atol=OUTPUT_ATOL, rtol=0)


def test_router_hostchain_matches_separate_canonical_gemv(case):
    from custom_esimd_kernels_vllm import esimd_gemv_int4

    _, weights = case
    x = inputs(1)[0].to("xpu")
    g = torch.Generator().manual_seed(1604)
    router = torch.randint(256, (512, 1280), generator=g, dtype=torch.uint8).to("xpu")
    scale = torch.full((512, 20), 1 / 256, dtype=torch.float16, device="xpu")
    logits = torch.empty((1, 512), dtype=torch.float16, device="xpu")
    esimd_gemv_int4(x, router, scale, logits)
    expected = torch.empty_like(x)
    actual = torch.empty_like(x)
    torch.ops.moe_int4_ops.moe_forward_compact160_out_v1(
        x, logits, *weights, expected, 10, 1, 512
    )
    torch.ops.moe_int4_ops.moe_forward_compact160_router_out_v1(
        x, router, scale, *weights, actual, 10, 1, 512
    )
    assert torch.equal(actual.cpu(), expected.cpu())


def test_router_hostchain_lazy_output_is_rejected_before_submit(case):
    _, weights = case
    x = inputs(1)[0].to("xpu")
    router = torch.zeros((512, 1280), dtype=torch.uint8, device="xpu")
    scale = torch.ones((512, 20), dtype=torch.float16, device="xpu")
    output = torch.full_like(x, 123)
    with pytest.raises(RuntimeError, match="lazy output"):
        torch.ops.moe_int4_ops.moe_forward_compact160_router_out_v1(
            x, router, scale, *weights, torch._neg_view(output), 10, 1, 512
        )
    assert torch.equal(output.cpu(), torch.full((1, 2560), 123, dtype=torch.float16))


def test_prefill_lazy_input_is_rejected_without_materialization(case):
    _, weights = case
    x = torch.ones((1, 160), dtype=torch.float16, device="xpu")
    counts = torch.zeros(512, dtype=torch.int32, device="xpu")
    counts[0] = 1
    with pytest.raises(RuntimeError, match="lazy neg/conj"):
        torch.ops.moe_int4_ops.moe_compact160_down_grouped_gemm(
            torch._neg_view(x), weights[2].view(torch.int8), weights[3], counts
        )
    assert torch.equal(x.cpu(), torch.ones(1, 160, dtype=torch.float16))


@pytest.mark.parametrize(
    "failure",
    [
        "scale",
        "stride",
        "alias",
        "shared",
        "misaligned",
        "lazy_x",
        "lazy_output",
        "lazy_scale",
    ],
)
def test_native_rejects_bad_contract_before_output_write(case, failure):
    _, weights = case
    x, logits = (t.to("xpu") for t in inputs(2))
    values = list(weights)
    output = torch.full_like(x, 123)
    if failure == "scale":
        values[3] = values[3][..., :1].contiguous()
    elif failure == "shared":
        values[5] = values[5][:, :80].contiguous()
    elif failure == "stride":
        x = torch.empty((2, 2560, 2), device="xpu", dtype=torch.float16)[..., 0]
    elif failure == "misaligned":
        x = torch.empty(5121, device="xpu", dtype=torch.float16)[1:].view(2, 2560)
    elif failure == "lazy_x":
        x = torch._neg_view(x)
    elif failure == "lazy_output":
        output = torch._neg_view(output)
    elif failure == "lazy_scale":
        values[3] = torch._neg_view(values[3])
    else:
        x = output
    with pytest.raises(RuntimeError):
        torch.ops.moe_int4_ops.moe_forward_compact160_out_v1(
            x, logits, *values, output, 10, 1, 512
        )
    assert torch.equal(
        output.cpu(),
        torch.full(
            (2, 2560), -123 if failure == "lazy_output" else 123, dtype=torch.float16
        ),
    )


@pytest.mark.parametrize("m", [1, 17, 33, 129])
def test_prefill_full_compact_wrapper_against_fp16_boundary_golden(case, m):
    from vllm.model_executor.layers.quantization._qwen38_compact_moe import (
        guard_compact80_scales,
        make_xpu_fused_moe,
    )

    cpu, weights = case
    x, logits = inputs(m)
    expected, ids, rw = golden(x, logits, cpu, prefill=True, include_shared=False)
    # Bytes here are already signed S4. The wrapper must not implement_zp twice.
    q13, s13, q2, s2 = weights[:4]
    q13 = q13.view(torch.int8)
    q13._qwen38_compact160 = True
    impl = make_xpu_fused_moe(
        w13=q13,
        w13_scales=guard_compact80_scales(s13, 160),
        w13_bias=None,
        w2=q2.view(torch.int8),
        w2_scales=s2,
        w2_bias=None,
        n_experts_per_token=10,
        activation="silu",
        num_experts=512,
    )
    output = torch.empty_like(x, device="xpu")
    impl.apply(output, x.to("xpu"), rw.to("xpu"), ids.to("xpu"))
    torch.testing.assert_close(output.cpu(), expected, atol=OUTPUT_ATOL, rtol=0)


def test_signed_unpack_has_independent_group_boundary():
    packed = torch.full((1, 80), 0xF8, dtype=torch.uint8)
    actual = unpack_s4(packed, torch.tensor([[0.125, 0.5]], dtype=torch.float16))
    assert torch.equal(actual[0, :128:2], torch.full((64,), -1.0))
    assert torch.equal(actual[0, 1:128:2], torch.full((64,), -0.125))
    assert torch.equal(actual[0, 128::2], torch.full((16,), -4.0))
    assert torch.equal(actual[0, 129::2], torch.full((16,), -0.5))


def test_cpu_metadata_contract_and_additive_abi():
    if not os.environ.get("MOE_INT4_DSO"):
        pytest.skip("CPU ABI 检查需要显式指定 build-only DSO")
    torch.ops.load_library(str(_dso_path()))
    shapes = [
        (512, 320, 1280),
        (512, 320, 20),
        (512, 2560, 80),
        (512, 2560, 2),
        (320, 2560),
        (2560, 160),
        (1, 2560),
    ]
    # CPU empty storage: metadata only, no XPU initialization or tensor upload.
    weights = [
        torch.empty(shape, dtype=torch.uint8 if i in (0, 2) else torch.float16)
        for i, shape in enumerate(shapes)
    ]
    contract = torch.ops.moe_int4_ops.qwen38_moe_compact160_weight_contract_v1
    assert contract(weights, torch.device("cpu")) == 0
    bad = list(weights)
    bad[3] = torch.empty((512, 2560, 1), dtype=torch.float16)
    assert contract(bad, torch.device("cpu")) == 3
    bad[3] = torch.empty((512, 2560, 2), dtype=torch.float32)
    assert contract(bad, torch.device("cpu")) == 2
    for name in (
        "moe_forward_compact160_out_v1",
        "moe_forward_compact160_router_out_v1",
        "moe_compact160_down_grouped_gemm",
        "moe_forward_m1_cutlass_nmajor_int4_fp16_shared_compact80_out_v1",
        "moe_forward_multi_m_cutlass_nmajor_int4_fp16_shared_compact80_out_v1",
        "moe_forward_multi_m_cutlass_nmajor_int4_fp16_shared_compact80_grouped_out_v1",
    ):
        assert torch._C._jit_get_schemas_for_operator("moe_int4_ops::" + name)
    for name in (
        "moe_compact160_down_grouped_gemm",
        "moe_forward_compact160_out_v1",
        "moe_forward_compact160_router_out_v1",
    ):
        for key in ("Negative", "Conjugate"):
            assert torch._C._dispatch_has_kernel_for_dispatch_key(
                "moe_int4_ops::" + name, key
            )
