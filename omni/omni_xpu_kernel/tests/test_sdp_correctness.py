"""Correctness and contract tests for standalone SDP kernel."""

import os
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX")
SIDE_CAR_SUFFIX = EXT_SUFFIX if isinstance(EXT_SUFFIX, str) and EXT_SUFFIX else ".so"
SIDE_CAR_PATH = PROJECT_ROOT / "omni_xpu_kernel" / "lgrf_uni" / f"lgrf_sdp{SIDE_CAR_SUFFIX}"


def has_xpu():
    try:
        return torch.xpu.is_available()
    except AttributeError:
        return False


def has_lgrf_sidecar():
    return SIDE_CAR_PATH.exists()


@pytest.fixture
def xpu_device():
    if not has_xpu():
        pytest.skip("XPU not available")
    return torch.device("xpu")


def make_qkv(device, *, batch=1, q_len=64, kv_len=None, heads=8, dim=128, dtype=torch.float16):
    if kv_len is None:
        kv_len = q_len
    q = torch.randn(batch, q_len, heads, dim, device=device, dtype=dtype)
    k = torch.randn(batch, kv_len, heads, dim, device=device, dtype=dtype)
    v = torch.randn(batch, kv_len, heads, dim, device=device, dtype=dtype)
    return q, k, v


def torch_sdpa_reference(q, k, v):
    q_bhld = q.permute(0, 2, 1, 3).contiguous()
    k_bhld = k.permute(0, 2, 1, 3).contiguous()
    v_bhld = v.permute(0, 2, 1, 3).contiguous()
    out = torch.nn.functional.scaled_dot_product_attention(
        q_bhld,
        k_bhld,
        v_bhld,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
    )
    return out.permute(0, 2, 1, 3).contiguous()


class TestStandaloneSDPKernel:
    @pytest.mark.skipif(not has_xpu(), reason="XPU not available")
    def test_import_sdp_submodule(self):
        from omni_xpu_kernel import sdp  # noqa: F401

    @pytest.mark.skipif(not has_xpu(), reason="XPU not available")
    @pytest.mark.skipif(not has_lgrf_sidecar(), reason="lgrf sidecar not available")
    def test_sdp_requires_lgrf_sidecar_runtime(self):
        backup = SIDE_CAR_PATH.with_suffix(SIDE_CAR_PATH.suffix + ".bak")
        SIDE_CAR_PATH.rename(backup)
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import torch; "
                        "from omni_xpu_kernel import sdp; "
                        "q=torch.randn(1, 16, 4, 128, device='xpu', dtype=torch.float16); "
                        "k=torch.randn(1, 16, 4, 128, device='xpu', dtype=torch.float16); "
                        "v=torch.randn(1, 16, 4, 128, device='xpu', dtype=torch.float16); "
                        "sdp.sdp(q, k, v)"
                    ),
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
            )
        finally:
            backup.rename(SIDE_CAR_PATH)

        assert result.returncode != 0, (
            "expected SDP call to fail when the lgrf sidecar artifact is unavailable\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
        combined = f"{result.stdout}\n{result.stderr}"
        assert "lgrf" in combined.lower() or "sidecar" in combined.lower()

    @pytest.mark.skipif(not has_xpu(), reason="XPU not available")
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_sdp_matches_torch_sdpa_reference(self, xpu_device, dtype):
        from omni_xpu_kernel import sdp

        q, k, v = make_qkv(xpu_device, q_len=64, kv_len=64, heads=8, dim=128, dtype=dtype)

        out_kernel = sdp.sdp(q, k, v)
        out_ref = torch_sdpa_reference(q, k, v)

        rtol, atol = (5e-2, 5e-2) if dtype == torch.bfloat16 else (1e-2, 1e-2)
        torch.testing.assert_close(out_kernel, out_ref, rtol=rtol, atol=atol)

    @pytest.mark.skipif(not has_xpu(), reason="XPU not available")
    def test_sdp_rejects_cpu_tensors(self):
        from omni_xpu_kernel import sdp

        q, k, v = make_qkv(torch.device("cpu"), dtype=torch.float16)

        with pytest.raises(RuntimeError, match="XPU|xpu"):
            sdp.sdp(q, k, v)

    @pytest.mark.skipif(not has_xpu(), reason="XPU not available")
    def test_sdp_rejects_noncontiguous_input(self, xpu_device):
        from omni_xpu_kernel import sdp

        q, k, v = make_qkv(xpu_device, dtype=torch.float16)
        q_bad = q.transpose(1, 2)

        with pytest.raises(RuntimeError, match="contiguous"):
            sdp.sdp(q_bad, k, v)

    @pytest.mark.skipif(not has_xpu(), reason="XPU not available")
    def test_sdp_rejects_non_4d_input(self, xpu_device):
        from omni_xpu_kernel import sdp

        q, k, v = make_qkv(xpu_device, dtype=torch.float16)
        q_bad = q.squeeze(0)
        k_bad = k.squeeze(0)
        v_bad = v.squeeze(0)

        with pytest.raises(RuntimeError, match="4-D|4D"):
            sdp.sdp(q_bad, k_bad, v_bad)

    @pytest.mark.skipif(not has_xpu(), reason="XPU not available")
    def test_sdp_rejects_batch_not_one(self, xpu_device):
        from omni_xpu_kernel import sdp

        q, k, v = make_qkv(xpu_device, batch=2, dtype=torch.float16)

        with pytest.raises(RuntimeError, match="batch size must be 1|B == 1|B=1"):
            sdp.sdp(q, k, v)

    @pytest.mark.skipif(not has_xpu(), reason="XPU not available")
    def test_sdp_rejects_unsupported_head_dim(self, xpu_device):
        from omni_xpu_kernel import sdp

        q, k, v = make_qkv(xpu_device, dim=96, dtype=torch.float16)

        with pytest.raises(RuntimeError, match="64 or 128|head_dim"):
            sdp.sdp(q, k, v)

    @pytest.mark.skipif(not has_xpu(), reason="XPU not available")
    def test_sdp_rejects_mismatched_dtypes(self, xpu_device):
        from omni_xpu_kernel import sdp

        q, k, v = make_qkv(xpu_device, dtype=torch.float16)
        k = k.to(torch.bfloat16)

        with pytest.raises(RuntimeError, match="same dtype"):
            sdp.sdp(q, k, v)

    @pytest.mark.skipif(not has_xpu(), reason="XPU not available")
    def test_sdp_rejects_unsupported_dtype(self, xpu_device):
        from omni_xpu_kernel import sdp

        q, k, v = make_qkv(xpu_device, dtype=torch.float32)

        with pytest.raises(RuntimeError, match="FP16|BF16|dtype"):
            sdp.sdp(q, k, v)

    @pytest.mark.skipif(not has_xpu(), reason="XPU not available")
    def test_sdp_handles_cross_attention_shape(self, xpu_device):
        from omni_xpu_kernel import sdp

        q, k, v = make_qkv(xpu_device, q_len=128, kv_len=32, heads=12, dim=128, dtype=torch.bfloat16)

        out_kernel = sdp.sdp(q, k, v)
        out_ref = torch_sdpa_reference(q, k, v)

        torch.testing.assert_close(out_kernel, out_ref, rtol=5e-2, atol=5e-2)

    @pytest.mark.skipif(not has_xpu(), reason="XPU not available")
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_sdp_hd64_matches_torch_sdpa_reference(self, xpu_device, dtype):
        from omni_xpu_kernel import sdp

        q, k, v = make_qkv(xpu_device, q_len=64, kv_len=64, heads=8, dim=64, dtype=dtype)

        out_kernel = sdp.sdp(q, k, v)
        out_ref = torch_sdpa_reference(q, k, v)

        rtol, atol = (5e-2, 5e-2) if dtype == torch.bfloat16 else (1e-2, 1e-2)
        torch.testing.assert_close(out_kernel, out_ref, rtol=rtol, atol=atol)

    @pytest.mark.skipif(not has_xpu(), reason="XPU not available")
    def test_sdp_hd64_cross_attention(self, xpu_device):
        from omni_xpu_kernel import sdp

        q, k, v = make_qkv(xpu_device, q_len=128, kv_len=32, heads=12, dim=64, dtype=torch.bfloat16)

        out_kernel = sdp.sdp(q, k, v)
        out_ref = torch_sdpa_reference(q, k, v)

        torch.testing.assert_close(out_kernel, out_ref, rtol=5e-2, atol=5e-2)

    @pytest.mark.skipif(not has_xpu(), reason="XPU not available")
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("q_len", [64, 256, 512])
    def test_sdp_hd64_various_seq_lengths(self, xpu_device, dtype, q_len):
        from omni_xpu_kernel import sdp

        q, k, v = make_qkv(xpu_device, q_len=q_len, kv_len=q_len, heads=8, dim=64, dtype=dtype)

        out_kernel = sdp.sdp(q, k, v)
        out_ref = torch_sdpa_reference(q, k, v)

        rtol, atol = (5e-2, 5e-2) if dtype == torch.bfloat16 else (1e-2, 1e-2)
        torch.testing.assert_close(out_kernel, out_ref, rtol=rtol, atol=atol)
