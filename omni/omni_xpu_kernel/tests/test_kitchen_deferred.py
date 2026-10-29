"""Kitchen 0.2.40 deferred control ABI admission on the native XPU backend."""

import pytest
import torch

from omni_xpu_kernel import kitchen


pytestmark = pytest.mark.skipif(
    not hasattr(torch, "xpu") or not torch.xpu.is_available(),
    reason="XPU is unavailable",
)


@pytest.mark.parametrize("api", ["deltanet_conv_deferred", "gated_delta_decode_deferred"])
@pytest.mark.parametrize("length", [2, 25, 27])
def test_deferred_rejects_non_upstream_control_length_without_mutation(api, length):
    assert kitchen.supports_gated_delta_deferred()

    def tensor(shape):
        return torch.ones(shape, device="xpu", dtype=torch.float32)

    ctl = torch.zeros(length, device="xpu", dtype=torch.int32)
    if api == "deltanet_conv_deferred":
        args = (
            tensor((1, 2, 16)), tensor((1, 16, 3)), tensor((16, 4)), None,
            tensor((2, 1, 8, 16)), tensor((2, 1, 16, 8)), ctl,
        )
    else:
        args = (
            tensor((1, 2, 32)), tensor((2, 32)), tensor((2, 32)),
            tensor((2,)), -tensor((2,)), tensor((1, 2, 128, 128)),
            128, 1, 128 ** -0.5, tensor((1, 2, 256)), tensor((128,)), 1e-6,
            tensor((2, 1, 512, 8)), tensor((2, 1, 8, 2, 2)),
            tensor((2, 1, 8, 1, 2)), ctl,
        )
    tensors = [arg for arg in args if isinstance(arg, torch.Tensor)]
    before = [arg.clone() for arg in tensors]
    with pytest.raises(RuntimeError, match=r"ctl must be contiguous XPU int32 \[26\]"):
        getattr(kitchen, api)(*args)
    for actual, expected in zip(tensors, before):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
