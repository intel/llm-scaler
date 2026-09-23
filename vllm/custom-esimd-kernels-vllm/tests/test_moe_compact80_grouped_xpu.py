import os
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

HIDDEN_SIZE = 2560
NUM_EXPERTS = 512
TOP_K = 10
ROUTED_SIZE = 80
SHARED_SIZE = 80
MAX_TOKENS = 8
OUTPUT_ATOL = 5e-3


def _dso_path() -> Path:
    configured = os.environ.get("MOE_INT4_DSO")
    if configured:
        return Path(configured)
    package_dir = Path(__file__).parents[1] / "python" / "custom_esimd_kernels_vllm"
    matches = tuple(package_dir.glob("moe_int4_ops*.so"))
    if len(matches) != 1:
        raise RuntimeError(
            "set MOE_INT4_DSO or build exactly one focused moe_int4_ops DSO"
        )
    return matches[0]


@pytest.fixture(scope="module", autouse=True)
def _load_focused_dso() -> None:
    if not torch.xpu.is_available():
        pytest.skip("XPU is unavailable")
    dso = _dso_path()
    if not dso.is_file():
        raise RuntimeError(f"focused DSO does not exist: {dso}")
    torch.ops.load_library(str(dso))


def _nonzero_s4(shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
    low = torch.randint(1, 16, shape, dtype=torch.uint8, generator=generator)
    high = torch.randint(1, 16, shape, dtype=torch.uint8, generator=generator)
    return low | (high << 4)


@dataclass
class FullInputs:
    x_cpu: torch.Tensor
    x: torch.Tensor
    w13_cpu: torch.Tensor
    w13_scales_cpu: torch.Tensor
    w13: torch.Tensor
    w13_scales: torch.Tensor
    w2_cpu: torch.Tensor
    w2_scales_cpu: torch.Tensor
    w2: torch.Tensor
    w2_scales: torch.Tensor
    shared_gate_up_cpu: torch.Tensor
    shared_gate_up: torch.Tensor
    shared_down_cpu: torch.Tensor
    shared_down: torch.Tensor
    shared_gate_cpu: torch.Tensor
    shared_gate: torch.Tensor

    def args(
        self,
        n_tokens: int,
        logits: torch.Tensor,
        output: torch.Tensor,
    ) -> list[object]:
        return [
            self.x[:n_tokens],
            logits,
            self.w13,
            self.w13_scales,
            self.w2,
            self.w2_scales,
            self.shared_gate_up,
            self.shared_down,
            self.shared_gate,
            output,
            TOP_K,
            1,
            NUM_EXPERTS,
        ]


def build_full_inputs() -> FullInputs:
    generator = torch.Generator().manual_seed(20260907)
    x_cpu = (torch.randn(MAX_TOKENS, HIDDEN_SIZE, generator=generator) * 0.03).half()
    w13_cpu = _nonzero_s4(
        (NUM_EXPERTS, 2 * ROUTED_SIZE, HIDDEN_SIZE // 2), generator
    )
    w13_scales_cpu = (
        0.006
        + 0.018
        * torch.rand(
            NUM_EXPERTS,
            2 * ROUTED_SIZE,
            HIDDEN_SIZE // 128,
            generator=generator,
        )
    ).half()
    w2_cpu = _nonzero_s4(
        (NUM_EXPERTS, HIDDEN_SIZE, ROUTED_SIZE // 2), generator
    )
    w2_scales_cpu = (
        0.006
        + 0.018
        * torch.rand(NUM_EXPERTS, HIDDEN_SIZE, 1, generator=generator)
    ).half()
    shared_gate_up_cpu = (
        torch.randn(2 * SHARED_SIZE, HIDDEN_SIZE, generator=generator) * 0.12
    ).half()
    shared_down_cpu = (
        torch.randn(HIDDEN_SIZE, SHARED_SIZE, generator=generator) * 0.12
    ).half()
    shared_gate_cpu = (
        torch.randn(1, HIDDEN_SIZE, generator=generator) * 0.04
    ).half()

    result = FullInputs(
        x_cpu=x_cpu,
        x=x_cpu.to("xpu"),
        w13_cpu=w13_cpu,
        w13_scales_cpu=w13_scales_cpu,
        w13=w13_cpu.to("xpu"),
        w13_scales=w13_scales_cpu.to("xpu"),
        w2_cpu=w2_cpu,
        w2_scales_cpu=w2_scales_cpu,
        w2=w2_cpu.to("xpu"),
        w2_scales=w2_scales_cpu.to("xpu"),
        shared_gate_up_cpu=shared_gate_up_cpu,
        shared_gate_up=shared_gate_up_cpu.to("xpu"),
        shared_down_cpu=shared_down_cpu,
        shared_down=shared_down_cpu.to("xpu"),
        shared_gate_cpu=shared_gate_cpu,
        shared_gate=shared_gate_cpu.to("xpu"),
    )
    torch.xpu.synchronize()
    return result


@pytest.fixture(scope="module")
def full_inputs() -> FullInputs:
    return build_full_inputs()


def _make_logits(n_tokens: int, variant: int = 0) -> torch.Tensor:
    rows = []
    for token in range(n_tokens):
        row = torch.full((NUM_EXPERTS,), -6.0, dtype=torch.float16)
        base = (variant * 37 + (token % 3) * 3) % NUM_EXPERTS
        expert_ids = (base + 3 * torch.arange(TOP_K)) % NUM_EXPERTS
        row[expert_ids] = torch.arange(TOP_K, 0, -1, dtype=torch.float16)
        rows.append(row)
    return torch.stack(rows)


def _topk_reference(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities = torch.softmax(logits.float(), dim=-1)
    topk_idx = torch.argsort(probabilities, dim=-1, descending=True, stable=True)[:, :TOP_K]
    topk_weight = probabilities.gather(1, topk_idx)
    topk_weight = (topk_weight / topk_weight.sum(dim=-1, keepdim=True)).half()
    return topk_weight, topk_idx


def _dequant_w13(data: FullInputs, expert_ids: torch.Tensor) -> torch.Tensor:
    packed = data.w13_cpu[expert_ids].to(torch.int16)
    low = (packed & 0xF).float()
    high = ((packed >> 4) & 0xF).float()
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)
    unpacked = torch.empty(
        (*packed.shape[:-1], packed.shape[-1] * 2), dtype=torch.float32
    )
    unpacked[..., 0::2] = low
    unpacked[..., 1::2] = high
    scales = data.w13_scales_cpu[expert_ids].float().repeat_interleave(128, dim=-1)
    return unpacked * scales


def _dequant_w2(data: FullInputs, expert_ids: torch.Tensor) -> torch.Tensor:
    packed = data.w2_cpu[expert_ids].to(torch.int16)
    low = (packed & 0xF).float()
    high = ((packed >> 4) & 0xF).float()
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)
    unpacked = torch.empty(
        (*packed.shape[:-1], packed.shape[-1] * 2), dtype=torch.float32
    )
    unpacked[..., 0::2] = low
    unpacked[..., 1::2] = high
    return unpacked * data.w2_scales_cpu[expert_ids].float()


def _reference(data: FullInputs, n_tokens: int, logits: torch.Tensor) -> torch.Tensor:
    topk_weight, topk_idx = _topk_reference(logits)
    used = torch.unique(topk_idx)
    w13 = {
        int(expert): weight
        for expert, weight in zip(used.tolist(), _dequant_w13(data, used))
    }
    w2 = {
        int(expert): weight
        for expert, weight in zip(used.tolist(), _dequant_w2(data, used))
    }
    output = torch.zeros(n_tokens, HIDDEN_SIZE, dtype=torch.float32)
    for token in range(n_tokens):
        x = data.x_cpu[token].float()
        for route in range(TOP_K):
            expert = int(topk_idx[token, route])
            projected = torch.mv(w13[expert], x)
            intermediate = (F.silu(projected[:ROUTED_SIZE]) * projected[ROUTED_SIZE:]).half()
            output[token] += float(topk_weight[token, route]) * torch.mv(
                w2[expert], intermediate.float()
            )

    shared_projected = F.linear(
        data.x_cpu[:n_tokens].float(), data.shared_gate_up_cpu.float()
    )
    shared_intermediate = (
        F.silu(shared_projected[:, :SHARED_SIZE])
        * shared_projected[:, SHARED_SIZE:]
    ).half()
    shared_gate = torch.sigmoid(
        F.linear(data.x_cpu[:n_tokens].float(), data.shared_gate_cpu.float())
    )
    output += shared_gate * F.linear(
        shared_intermediate.float(), data.shared_down_cpu.float()
    )
    return output.half()


def _m1_op(data: FullInputs, logits: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
    return torch.ops.moe_int4_ops.moe_forward_m1_cutlass_nmajor_int4_fp16_shared_compact80_out_v1(
        *data.args(1, logits, output)
    )


def _grouped_op(
    data: FullInputs, n_tokens: int, logits: torch.Tensor, output: torch.Tensor
) -> torch.Tensor:
    return torch.ops.moe_int4_ops.moe_forward_multi_m_cutlass_nmajor_int4_fp16_shared_compact80_grouped_out_v1(
        *data.args(n_tokens, logits, output)
    )


def _legacy_compact_multi_op(
    data: FullInputs, n_tokens: int, logits: torch.Tensor, output: torch.Tensor
) -> torch.Tensor:
    return torch.ops.moe_int4_ops.moe_forward_multi_m_cutlass_nmajor_int4_fp16_shared_compact80_out_v1(
        *data.args(n_tokens, logits, output)
    )


def test_compact80_m1_m8_full_expert_correctness(
    full_inputs: FullInputs,
) -> None:
    for round_id in range(3):
        for n_tokens in range(1, 9):
            logits_cpu = _make_logits(n_tokens, round_id)
            logits = logits_cpu.to("xpu")
            output = torch.empty(
                n_tokens, HIDDEN_SIZE, dtype=torch.float16, device="xpu"
            )
            returned = (
                _m1_op(full_inputs, logits, output)
                if n_tokens == 1
                else _grouped_op(full_inputs, n_tokens, logits, output)
            )
            assert returned.data_ptr() == output.data_ptr()
            torch.testing.assert_close(
                output.cpu(),
                _reference(full_inputs, n_tokens, logits_cpu),
                rtol=0,
                atol=OUTPUT_ATOL,
            )

def test_compact80_grouped_matches_legacy_for_m2_m8(
    full_inputs: FullInputs,
) -> None:
    for round_id in range(3):
        for n_tokens in range(2, 9):
            logits = _make_logits(n_tokens, round_id + 11).to("xpu")
            old_output = torch.empty(
                n_tokens, HIDDEN_SIZE, dtype=torch.float16, device="xpu"
            )
            grouped_output = torch.empty_like(old_output)

            old_returned = _legacy_compact_multi_op(
                full_inputs, n_tokens, logits, old_output
            )
            grouped_returned = _grouped_op(
                full_inputs, n_tokens, logits, grouped_output
            )
            assert old_returned.data_ptr() == old_output.data_ptr()
            assert grouped_returned.data_ptr() == grouped_output.data_ptr()
            torch.testing.assert_close(
                grouped_output, old_output, rtol=0, atol=OUTPUT_ATOL
            )

def test_compact80_old_m1_and_grouped_m6_are_two_stream_safe(
    full_inputs: FullInputs,
) -> None:
    stream_a = torch.xpu.Stream()
    stream_b = torch.xpu.Stream()
    logits_a_m1 = _make_logits(1, 3).to("xpu")
    logits_b_m1 = _make_logits(1, 4).to("xpu")
    logits_a_m6 = _make_logits(6, 5).to("xpu")
    logits_b_m6 = _make_logits(6, 6).to("xpu")
    outputs = {}
    with torch.xpu.stream(stream_a):
        outputs["a_m1"] = torch.empty(1, HIDDEN_SIZE, dtype=torch.float16, device="xpu")
        outputs["a_m6"] = torch.empty(6, HIDDEN_SIZE, dtype=torch.float16, device="xpu")
        assert _m1_op(full_inputs, logits_a_m1, outputs["a_m1"]).data_ptr() == outputs["a_m1"].data_ptr()
        assert _grouped_op(full_inputs, 6, logits_a_m6, outputs["a_m6"]).data_ptr() == outputs["a_m6"].data_ptr()
    with torch.xpu.stream(stream_b):
        outputs["b_m1"] = torch.empty(1, HIDDEN_SIZE, dtype=torch.float16, device="xpu")
        outputs["b_m6"] = torch.empty(6, HIDDEN_SIZE, dtype=torch.float16, device="xpu")
        assert _m1_op(full_inputs, logits_b_m1, outputs["b_m1"]).data_ptr() == outputs["b_m1"].data_ptr()
        assert _grouped_op(full_inputs, 6, logits_b_m6, outputs["b_m6"]).data_ptr() == outputs["b_m6"].data_ptr()
    torch.xpu.synchronize()

    for key, n_tokens, logits_cpu in (
        ("a_m1", 1, _make_logits(1, 3)),
        ("b_m1", 1, _make_logits(1, 4)),
        ("a_m6", 6, _make_logits(6, 5)),
        ("b_m6", 6, _make_logits(6, 6)),
    ):
        torch.testing.assert_close(
            outputs[key].cpu(),
            _reference(full_inputs, n_tokens, logits_cpu),
            rtol=0,
            atol=OUTPUT_ATOL,
        )
