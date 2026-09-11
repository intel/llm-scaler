"""Regression for the N*HV <= WG_SIZE inline conv-shift race.

The N=2 branch (48 workgroups) previously shifted state in the fused launch,
while padded N=3 (72 workgroups) used a second ordered shift launch. The first
two rows must be bit-exact for both supported conv-state layouts.
"""

import pytest
import torch


if not torch.xpu.is_available():
    pytest.skip("requires an Intel XPU", allow_module_level=True)

from custom_esimd_kernels_sglang import esimd_gdn_conv_fused_seq


H = 8
HV = 24
K = 128
V = 128
SLOTS = 5
TRIALS = 4
STEPS = 8
DIM = 2 * H * K + HV * V
QDIM = DIM + HV * V
STATE_IDS = torch.tensor([3, 1], dtype=torch.int32)
STATE_IDS_PADDED = torch.tensor([3, 1, 4], dtype=torch.int32)


def _random(generator: torch.Generator, shape, scale: float, device: str):
    return (torch.randn(shape, generator=generator) * scale).to(
        device=device, dtype=torch.float16
    )


@pytest.mark.parametrize("layout", ["native", "transposed"])
def test_gdn_seq_conv_shift_n2_matches_padded_n3(layout):
    device = "xpu"
    generator = torch.Generator(device="cpu")

    for trial in range(TRIALS):
        generator.manual_seed(20260910 + trial)
        conv_shape = (
            (SLOTS, DIM, 3) if layout == "native" else (SLOTS, 3, DIM)
        )
        conv_initial = _random(generator, conv_shape, 0.1, device)
        ssm_initial = _random(generator, (SLOTS, HV, V, K), 0.02, device)
        conv_n2 = conv_initial.clone()
        conv_n3 = conv_initial.clone()
        ssm_n2 = ssm_initial.clone()
        ssm_n3 = ssm_initial.clone()
        conv_weight = _random(generator, (DIM, 4), 0.2, device)
        conv_bias = torch.zeros(DIM, dtype=torch.float16, device=device)
        a_log = _random(generator, (HV,), 0.2, device)
        dt_bias = _random(generator, (HV,), 0.2, device)
        ids_n2 = STATE_IDS.to(device)
        ids_n3 = STATE_IDS_PADDED.to(device)
        selected = ids_n2.to(torch.long)
        conv_oracle = conv_initial.index_select(0, selected).clone()

        for _step in range(STEPS):
            qkvz_n3 = _random(generator, (3, QDIM), 0.5, device)
            ba_n3 = _random(generator, (3, 2 * HV), 0.5, device)
            qkvz_n2 = qkvz_n3[:2]
            ba_n2 = ba_n3[:2]
            out_n2 = torch.empty((2, HV, V), dtype=torch.float16, device=device)
            z_n2 = torch.empty_like(out_n2)
            out_n3 = torch.empty((3, HV, V), dtype=torch.float16, device=device)
            z_n3 = torch.empty_like(out_n3)

            esimd_gdn_conv_fused_seq(
                qkvz_n2, conv_n2, conv_weight, conv_bias, ids_n2,
                a_log, dt_bias, ba_n2, ssm_n2, ids_n2, out_n2, z_n2,
                2, H, HV, K, V, K**-0.5,
            )
            torch.xpu.synchronize()
            esimd_gdn_conv_fused_seq(
                qkvz_n3, conv_n3, conv_weight, conv_bias, ids_n3,
                a_log, dt_bias, ba_n3, ssm_n3, ids_n3, out_n3, z_n3,
                3, H, HV, K, V, K**-0.5,
            )
            torch.xpu.synchronize()

            new_x = qkvz_n2[:, :DIM]
            if layout == "native":
                conv_oracle = torch.cat(
                    (conv_oracle[:, :, 1:], new_x.unsqueeze(-1)), dim=2
                )
            else:
                conv_oracle = torch.cat(
                    (conv_oracle[:, 1:, :], new_x.unsqueeze(1)), dim=1
                )

            conv_n2_rows = conv_n2.index_select(0, selected)
            conv_n3_rows = conv_n3.index_select(0, selected)
            ssm_n2_rows = ssm_n2.index_select(0, selected)
            ssm_n3_rows = ssm_n3.index_select(0, selected)
            for actual, expected in (
                (out_n2, out_n3[:2]),
                (z_n2, z_n3[:2]),
                (conv_n2_rows, conv_n3_rows),
                (ssm_n2_rows, ssm_n3_rows),
                (conv_n2_rows, conv_oracle),
                (conv_n3_rows, conv_oracle),
            ):
                assert torch.isfinite(actual).all()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
