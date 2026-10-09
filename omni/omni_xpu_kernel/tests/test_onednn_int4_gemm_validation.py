"""Invalid INT4 grouping must raise an exception without killing the process."""

import subprocess
import sys
import textwrap

import pytest


@pytest.fixture(scope="module")
def native_xpu():
    torch = pytest.importorskip("torch")
    if not torch.xpu.is_available():
        pytest.skip("XPU is unavailable")
    from omni_xpu_kernel import _C

    return _C.svdq


@pytest.mark.parametrize("entry", [
    "onednn_int4_gemm",
    "onednn_int4_gemm_preconverted",
    "onednn_int4_gemm_add_to_output",
])
@pytest.mark.parametrize("K,num_groups,error", [
    (64, 0, "group_size must be positive"),
    (0, 0, "group_size must be positive"),
    (0, 1, "group_size must be positive"),
    (64, 128, "group_size must be positive"),
    (64, 3, "must be divisible by num_groups"),
])
def test_invalid_grouping_raises_in_subprocess(native_xpu, entry, K, num_groups, error):
    # A regression may deliver SIGFPE, so each native call runs in a child.
    code = textwrap.dedent("""
        import sys

        if sys.platform != "win32":
            import resource
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

        import torch
        from omni_xpu_kernel import _C

        entry, K, num_groups, expected = sys.argv[1:]
        K, num_groups = int(K), int(num_groups)
        M, N = 1, 64
        act = torch.empty((M, K), dtype=torch.float16, device="xpu")
        packed = torch.empty((N, K // 2), dtype=torch.uint8, device="xpu")
        scale_dtype = torch.bfloat16 if entry == "onednn_int4_gemm" else torch.float16
        scales = torch.empty((num_groups, N), dtype=scale_dtype, device="xpu")
        args = [act, packed, scales]
        if entry == "onednn_int4_gemm_add_to_output":
            args.append(torch.empty((M, N), dtype=torch.bfloat16, device="xpu"))

        try:
            getattr(_C.svdq, entry)(*args)
        except RuntimeError as error:
            assert expected in str(error), str(error)
        else:
            raise AssertionError("Invalid grouping was accepted")
    """)
    result = subprocess.run(
        [sys.executable, "-c", code, entry, str(K), str(num_groups), error],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"{entry} exited with {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


@pytest.mark.parametrize("entry", [
    "onednn_int4_gemm",
    "onednn_int4_gemm_preconverted",
    "onednn_int4_gemm_add_to_output",
])
@pytest.mark.parametrize("num_groups", [1, 2])
def test_valid_grouping_matches_reference(native_xpu, entry, num_groups):
    import torch

    M, N, K = 2, 64, 128
    act = ((torch.arange(M * K, device="xpu").reshape(M, K) % 17 - 8) / 16).half()
    quantized = (
        (torch.arange(N * K, device="xpu").reshape(N, K) * 5 + 3) % 16 - 8
    ).to(torch.int16)
    packed = (
        (quantized[:, 0::2] & 0x0F) | ((quantized[:, 1::2] & 0x0F) << 4)
    ).to(torch.uint8)
    scales = (
        (torch.arange(num_groups * N, device="xpu").reshape(num_groups, N) % 3 + 1) / 16
    ).half()
    weights = (
        quantized.float().reshape(N, num_groups, K // num_groups)
        * scales.float().T.unsqueeze(-1)
    ).reshape(N, K)
    expected = act.float() @ weights.T

    if entry == "onednn_int4_gemm":
        actual = native_xpu.onednn_int4_gemm(act, packed, scales.bfloat16())
        assert actual.dtype == torch.float16
    elif entry == "onednn_int4_gemm_preconverted":
        actual = native_xpu.onednn_int4_gemm_preconverted(act, packed ^ 0x88, scales)
        assert actual.dtype == torch.float16
    else:
        actual = torch.full((M, N), 0.25, dtype=torch.bfloat16, device="xpu")
        pointer = actual.data_ptr()
        result = native_xpu.onednn_int4_gemm_add_to_output(act, packed ^ 0x88, scales, actual)
        assert result is None
        assert actual.data_ptr() == pointer
        expected = expected + 0.25

    torch.testing.assert_close(actual, expected.to(actual.dtype), rtol=1e-2, atol=1e-2)
