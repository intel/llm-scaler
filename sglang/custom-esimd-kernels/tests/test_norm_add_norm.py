import pytest
import torch

import custom_esimd_kernels_sglang


EPS1 = 1e-6
EPS2 = 1e-6


def _rms_norm(x, w, eps):
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w.float()


def _make(k, device="xpu"):
    h2_raw = torch.randn(1, k, dtype=torch.float16, device=device)
    h1 = torch.randn(1, k, dtype=torch.float16, device=device)
    w1 = torch.randn(k, dtype=torch.float16, device=device)
    w2 = torch.randn(k, dtype=torch.float16, device=device)
    out = torch.empty(1, k, dtype=torch.float16, device=device)
    return h2_raw, h1, w1, w2, out


@torch.no_grad()
@pytest.mark.parametrize("k", [1536, 2560, 2816, 5376])
def test_norm_add_norm_matches_reference(k):
    h2_raw, h1, w1, w2, out = _make(k)
    h1_ref = (_rms_norm(h2_raw, w1, EPS1) + h1.float()).half()
    out_ref = _rms_norm(h1_ref, w2, EPS2).half()

    custom_esimd_kernels_sglang.esimd_norm_add_norm(
        h2_raw, h1, w1, w2, out, EPS1, EPS2
    )
    torch.xpu.synchronize()

    torch.testing.assert_close(h1, h1_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(out, out_ref, atol=2e-2, rtol=2e-2)


def _replace(i, t, k=2816):
    args = list(_make(k))
    args[i] = t(args[i]) if callable(t) else t
    return args


BAD_INPUTS = {
    "out_too_small": lambda: _replace(4, torch.empty(1, 64, dtype=torch.float16, device="xpu")),
    "h1_too_small": lambda: _replace(1, torch.empty(1, 64, dtype=torch.float16, device="xpu")),
    "w1_too_small": lambda: _replace(2, torch.empty(64, dtype=torch.float16, device="xpu")),
    "multi_row": lambda: _replace(0, torch.randn(4, 2816, dtype=torch.float16, device="xpu")),
    "out_fp32": lambda: _replace(4, lambda t: t.float()),
    "h1_cpu": lambda: _replace(1, lambda t: t.cpu()),
    "non_contiguous": lambda: _replace(
        0, torch.randn(2816, 2, dtype=torch.float16, device="xpu").t()[:1]
    ),
    "k_not_multiple_of_64": lambda: list(_make(100)),
}


@torch.no_grad()
@pytest.mark.parametrize("case", list(BAD_INPUTS))
def test_norm_add_norm_rejects_bad_inputs(case):
    h2_raw, h1, w1, w2, out = BAD_INPUTS[case]()
    with pytest.raises(RuntimeError):
        custom_esimd_kernels_sglang.esimd_norm_add_norm(
            h2_raw, h1, w1, w2, out, EPS1, EPS2
        )
