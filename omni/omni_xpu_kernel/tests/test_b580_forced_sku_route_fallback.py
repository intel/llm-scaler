"""Physical-B580 route checks for forced effective-SKU profiles."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest
import torch

from omni_xpu_kernel import device


_PROBE = textwrap.dedent(
    r"""
    import json
    import re

    import torch

    from omni_xpu_kernel import cute, device, rotary


    def xpu_names(operation):
        operation()
        torch.xpu.synchronize()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.XPU,
            ]
        ) as profiler:
            operation()
            torch.xpu.synchronize()
        return [
            event.name
            for event in profiler.events()
            if event.device_type == torch.autograd.DeviceType.XPU
        ]


    q = torch.randn(
        (1, 129, 2, 128), device="xpu", dtype=torch.bfloat16
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    sol_names = xpu_names(lambda: cute.sol_attn(q, k, v, tau=1.3))
    sol_kernel = next(
        name for name in sol_names if "SolCuteKernelTag" in name
    )
    tile_shapes = re.findall(
        r"cute::tuple<cute::C<(\d+)>, cute::C<(\d+)> >",
        sol_kernel,
    )
    if not tile_shapes:
        raise RuntimeError("Sol kernel name did not expose a tile shape")

    sequence, heads, head_dim, rot_dim = 31, 56, 128, 96
    inner = heads * head_dim
    packed = torch.randn(
        (sequence, 3 * inner), device="xpu", dtype=torch.bfloat16
    )
    rms_q = packed[:, :inner].view(1, sequence, heads, head_dim)
    rms_k = packed[:, inner : 2 * inner].view(
        1, sequence, heads, head_dim
    )
    freqs = torch.randn(
        (1, sequence, 1, rot_dim // 2, 2, 2),
        device="xpu",
        dtype=torch.bfloat16,
    )
    scale = torch.ones(head_dim, device="xpu", dtype=torch.bfloat16)

    def rms_rope():
        rotary.rms_kitchen_rope_split_half_(
            rms_q,
            rms_k,
            freqs,
            scale,
            scale,
            epsilon=1e-5,
            rot_dim=rot_dim,
        )

    rms_names = xpu_names(rms_rope)
    rms_kernel = next(
        name for name in rms_names if "launch_minimax_h3_rms_rope" in name
    )
    observed = device.info(torch.xpu.current_device())
    print(
        json.dumps(
            {
                "physical_sku": observed["physical_bmg_sku"],
                "effective_sku": observed["bmg_sku"],
                "kernel_profile": observed["kernel_profile"],
                "forced": observed["sku_forced"],
                "sol_q_tile": int(tile_shapes[-1][0]),
                "rms_b580": "launch_minimax_h3_rms_rope_b580" in rms_kernel,
            },
            sort_keys=True,
        )
    )
    """
)


def _require_physical_b580() -> None:
    if not torch.xpu.is_available():
        pytest.skip("XPU is unavailable")
    if device.info(torch.xpu.current_device())["physical_bmg_sku"] != "b580":
        pytest.skip("a physical B580 is required")


def _probe(tmp_path: Path, forced_sku: str | None) -> dict[str, object]:
    environment = os.environ.copy()
    if forced_sku is None:
        environment.pop("OMNI_XPU_FORCE_SKU", None)
    else:
        environment["OMNI_XPU_FORCE_SKU"] = forced_sku
    completed = subprocess.run(
        [sys.executable, "-c", _PROBE],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_unforced_physical_b580_uses_local_sol_and_rms_rope(tmp_path):
    _require_physical_b580()
    result = _probe(tmp_path, None)

    assert result == {
        "effective_sku": "b580",
        "forced": False,
        "kernel_profile": "b580",
        "physical_sku": "b580",
        "rms_b580": True,
        "sol_q_tile": 128,
    }


@pytest.mark.parametrize(
    "forced_sku,effective_sku,kernel_profile",
    [
        ("b60", "b60", "b60"),
        ("b70", "b70", "b70"),
        ("generic", "unknown", "generic-bmg"),
    ],
)
def test_forced_sku_uses_generic_sol_and_rms_rope_routes(
    tmp_path,
    forced_sku,
    effective_sku,
    kernel_profile,
):
    _require_physical_b580()
    result = _probe(tmp_path, forced_sku)

    assert result == {
        "effective_sku": effective_sku,
        "forced": True,
        "kernel_profile": kernel_profile,
        "physical_sku": "b580",
        "rms_b580": False,
        "sol_q_tile": 256,
    }
