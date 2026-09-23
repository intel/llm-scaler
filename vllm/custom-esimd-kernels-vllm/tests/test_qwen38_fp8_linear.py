"""Independent reference tests for the Qwen3.8 narrow FP8 linear DSO.

Set ``QWEN38_FP8_LINEAR_SO`` to an isolated build when testing the new module
without installing it.  The reference deliberately dequantizes FP8 to FP32,
accumulates the matrix product in FP32, applies the scalar in FP32, and only
then rounds to FP16.
"""

import importlib
import importlib.util
import os
import sys

import pytest
import torch


SHAPES = [(12, 2560), (24, 2560), (160, 2560), (2560, 80)]


@pytest.fixture(scope="module")
def op():
    if not torch.xpu.is_available():
        pytest.skip("XPU required")
    torch.set_num_threads(1)
    path = os.getenv("QWEN38_FP8_LINEAR_SO")
    if path:
        spec = importlib.util.spec_from_file_location(
            "qwen38_fp8_linear_ops", path
        )
        if spec is None or spec.loader is None:
            pytest.fail(f"cannot load Qwen3.8 FP8 linear DSO: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[spec.name] = module
        return module.qwen38_fp8_linear
    try:
        return importlib.import_module(
            "custom_esimd_kernels_vllm.qwen38_fp8_linear_ops"
        ).qwen38_fp8_linear
    except (ImportError, OSError) as exc:
        pytest.skip(f"Qwen3.8 FP8 linear DSO is not available: {exc}")


@pytest.fixture(scope="module", params=SHAPES)
def weights(request):
    n, k = request.param
    generator = torch.Generator().manual_seed(3817 + n + k)
    # The small range avoids overflow while retaining nontrivial FP32 sums.
    weight = (torch.randn((n, k), generator=generator) * 2.0).to(
        torch.float8_e4m3fn
    )
    scale = torch.tensor([0.125], dtype=torch.float32)
    return n, k, weight, scale, weight.to("xpu"), scale.to("xpu")


def reference(input_cpu, weight_cpu, scale_cpu):
    accumulated = input_cpu.float() @ weight_cpu.float().T
    return (accumulated * scale_cpu.reshape(1).float()).half()


@pytest.mark.parametrize("m", range(1, 9))
def test_narrow_shapes_and_all_decode_m(op, weights, m):
    n, k, weight_cpu, scale_cpu, weight_xpu, scale_xpu = weights
    generator = torch.Generator().manual_seed(3800 + n + m)
    input_cpu = (torch.randn((m, k), generator=generator) * 0.02).half()

    actual = op(input_cpu.to("xpu"), weight_xpu, scale_xpu).cpu()
    expected = reference(input_cpu, weight_cpu, scale_cpu)
    assert actual.shape == (m, n)
    assert actual.dtype == torch.float16
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=3e-3, rtol=3e-3)


def test_try_api_rejects_invalid_input_without_changing_live_output(op):
    run = sys.modules[op.__module__].try_qwen38_fp8_linear
    x = torch.ones(1, 2560, dtype=torch.float16, device="xpu")
    w = torch.ones(12, 2560, device="xpu").to(torch.float8_e4m3fn)
    scale = torch.ones(1, device="xpu")
    out = run(x, w, scale)
    saved = out.clone()
    assert run(x, w, scale.double()) is None
    assert run(x[:, ::2], w, scale) is None
    assert run(x, w, scale.cpu()) is None
    torch.testing.assert_close(out, saved, atol=0, rtol=0)


def test_e4m3_subnormals_and_signed_values(op):
    n, k = SHAPES[0]
    codes = torch.tensor(
        [0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
         0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
         0x08, 0x78, 0x7E],
        dtype=torch.uint8,
    )
    raw = codes.repeat((n * k + codes.numel() - 1) // codes.numel())[: n * k]
    weight_cpu = raw.reshape(n, k).view(torch.float8_e4m3fn)
    scale_cpu = torch.tensor([0.75], dtype=torch.float32)
    input_cpu = torch.linspace(-0.02, 0.02, k).repeat(n // 12, 1).half()

    actual = op(
        input_cpu.to("xpu"),
        weight_cpu.to("xpu"),
        scale_cpu.to("xpu"),
    ).cpu()
    expected = reference(input_cpu, weight_cpu, scale_cpu)
    torch.testing.assert_close(actual, expected, atol=3e-3, rtol=3e-3)


def test_retained_outputs_are_independent_across_streams(op):
    n, k = SHAPES[1]
    generator = torch.Generator().manual_seed(3881)
    weight_cpu = (torch.randn((n, k), generator=generator) * 2.0).to(
        torch.float8_e4m3fn
    )
    scale_cpu = torch.tensor([0.125], dtype=torch.float32)
    weight_xpu = weight_cpu.to("xpu")
    scale_xpu = scale_cpu.to("xpu")
    streams = [torch.xpu.current_stream(), torch.xpu.Stream()]
    streams[1].wait_stream(streams[0])
    retained = []

    for step, m in enumerate([1, 8, 2, 7, 1]):
        input_cpu = (
            torch.randn(
                (m, k), generator=torch.Generator().manual_seed(3890 + step)
            )
            * 0.02
        ).half()
        with torch.xpu.stream(streams[step % 2]):
            output = op(input_cpu.to("xpu"), weight_xpu, scale_xpu)
            retained.append((output, reference(input_cpu, weight_cpu, scale_cpu)))

    torch.xpu.synchronize()
    assert len({output.data_ptr() for output, _ in retained}) == len(retained)
    for output, expected in retained:
        torch.testing.assert_close(
            output.cpu(), expected, atol=3e-3, rtol=3e-3
        )


def test_rejects_unsupported_shape_before_submit(op):
    input_xpu = torch.zeros((1, 2560), device="xpu", dtype=torch.float16)
    weight_xpu = torch.zeros(
        (32, 2560), device="xpu", dtype=torch.float8_e4m3fn
    )
    scale_xpu = torch.ones((1,), device="xpu", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="unsupported shape"):
        op(input_xpu, weight_xpu, scale_xpu)
