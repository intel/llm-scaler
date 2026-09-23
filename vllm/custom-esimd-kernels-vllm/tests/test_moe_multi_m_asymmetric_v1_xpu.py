import re
import runpy
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from test_moe_m1_asymmetric_v1_xpu import (
    EXPERT_SLOT_BY_ID,
    HIDDEN_SIZE,
    NUM_EXPERTS,
    NUM_SHARED_EXPERTS,
    OUTPUT_ATOL,
    ROUTED_SIZE,
    SELECTED_EXPERT_IDS,
    SHARED_SIZE,
    TOP_K,
    UNIQUE_EXPERT_IDS,
    _dequant_selected_w2,
    _dequant_selected_w13,
    _dso_path,
    _topk_reference,
    build_inputs,
)

MAX_TOKENS = 32
TOKEN_COUNTS = (2, 3, 4, 16, 31, 32)


@pytest.fixture(scope="module", autouse=True)
def _load_focused_dso() -> None:
    dso = _dso_path()
    if not dso.is_file():
        raise RuntimeError(f"focused DSO does not exist: {dso}")
    torch.ops.load_library(str(dso))


@dataclass
class MultiInputs:
    base: object
    x_cpu: torch.Tensor
    x: torch.Tensor
    w13_dequant: torch.Tensor
    w2_dequant: torch.Tensor

    def args(
        self,
        n_tokens: int,
        logits: torch.Tensor,
        output: torch.Tensor | None = None,
    ) -> list[object]:
        x = self.x[:n_tokens]
        return [
            x,
            logits,
            self.base.w13_qweight_s4,
            self.base.w13_scales,
            self.base.w2_qweight_s4,
            self.base.w2_scales,
            self.base.shared_gate_up_weight,
            self.base.shared_down_weight,
            self.base.shared_expert_gate_weight,
            torch.empty_like(x) if output is None else output,
            TOP_K,
            NUM_SHARED_EXPERTS,
            NUM_EXPERTS,
        ]


def build_multi_inputs() -> MultiInputs:
    base = build_inputs()
    generator = torch.Generator().manual_seed(20260902)
    x_cpu = (
        torch.randn(MAX_TOKENS, HIDDEN_SIZE, generator=generator) * 0.05
    ).half()
    result = MultiInputs(
        base=base,
        x_cpu=x_cpu,
        x=x_cpu.to("xpu"),
        w13_dequant=_dequant_selected_w13(base, SELECTED_EXPERT_IDS),
        w2_dequant=_dequant_selected_w2(base, SELECTED_EXPERT_IDS),
    )
    torch.xpu.synchronize()
    return result


@pytest.fixture(scope="module")
def inputs() -> MultiInputs:
    return build_multi_inputs()


def make_logits(n_tokens: int, variant: int = 0) -> torch.Tensor:
    rows = []
    for token in range(n_tokens):
        if (token + variant) % 2:
            rows.append(torch.zeros(NUM_EXPERTS, dtype=torch.float16))
            continue
        row = torch.full((NUM_EXPERTS,), -4.0, dtype=torch.float16)
        values = torch.arange(TOP_K, 0, -1, dtype=torch.float16)
        shift = (token + variant) % TOP_K
        expert_ids = UNIQUE_EXPERT_IDS[shift:] + UNIQUE_EXPERT_IDS[:shift]
        row[torch.tensor(expert_ids)] = values
        rows.append(row)
    return torch.stack(rows)


def _reference_components(
    data: MultiInputs,
    n_tokens: int,
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    topk_weight, topk_idx = _topk_reference(logits)
    x = data.x_cpu[:n_tokens].float()
    routed = torch.zeros(n_tokens, HIDDEN_SIZE, dtype=torch.float32)

    for expert_id in SELECTED_EXPERT_IDS:
        positions = torch.nonzero(topk_idx == expert_id, as_tuple=False)
        if positions.numel() == 0:
            continue
        token_idx = positions[:, 0]
        route_idx = positions[:, 1]
        slot = EXPERT_SLOT_BY_ID[expert_id]
        projected = F.linear(x[token_idx], data.w13_dequant[slot])
        intermediate = (
            F.silu(projected[:, :ROUTED_SIZE])
            * projected[:, ROUTED_SIZE:]
        ).half()
        down = F.linear(intermediate.float(), data.w2_dequant[slot])
        weighted = down * topk_weight[token_idx, route_idx].float().unsqueeze(1)
        routed.index_add_(0, token_idx, weighted)

    shared_projected = F.linear(x, data.base.shared_gate_up_cpu.float())
    shared_intermediate = (
        F.silu(shared_projected[:, :SHARED_SIZE])
        * shared_projected[:, SHARED_SIZE:]
    ).half()
    shared = F.linear(
        shared_intermediate.float(), data.base.shared_down_cpu.float()
    )
    shared_gate = torch.sigmoid(
        F.linear(x, data.base.shared_gate_cpu.float())
    )
    return routed, shared * shared_gate


def _reference(
    data: MultiInputs,
    n_tokens: int,
    logits: torch.Tensor,
) -> torch.Tensor:
    routed, shared = _reference_components(data, n_tokens, logits)
    return (routed + shared).half()


def _op(*args: object) -> torch.Tensor:
    return torch.ops.moe_int4_ops.moe_forward_multi_m_cutlass_nmajor_int4_fp16_shared_asymmetric_out_v1(
        *args
    )


def test_00_preflight_rejects_before_first_valid_submit(
    inputs: MultiInputs,
) -> None:
    logits = make_logits(2).to("xpu")
    valid = inputs.args(2, logits)
    independent_storage_alias = torch.utils.dlpack.from_dlpack(
        torch.utils.dlpack.to_dlpack(inputs.x[:2])
    )
    assert independent_storage_alias.data_ptr() == inputs.x.data_ptr()
    assert not torch._C._is_alias_of(independent_storage_alias, inputs.x)
    invalid_cases: list[tuple[int, object, str]] = [
        (
            0,
            torch.zeros(1, HIDDEN_SIZE, dtype=torch.float16, device="xpu"),
            "multi-m asymmetric out v1 requires 2..32 tokens",
        ),
        (
            0,
            torch.zeros(33, HIDDEN_SIZE, dtype=torch.float16, device="xpu"),
            "multi-m asymmetric out v1 requires 2..32 tokens",
        ),
        (
            0,
            torch.zeros(HIDDEN_SIZE, dtype=torch.float16, device="xpu"),
            "multi-m asymmetric out v1 requires rank-2 x",
        ),
        (
            7,
            inputs.base.shared_down_weight[:, :-1],
            "shared_down_weight has an unsupported shape",
        ),
        (9, inputs.x[:2], "output must not alias any input tensor"),
        (
            9,
            independent_storage_alias,
            "output must not alias any input tensor",
        ),
        (
            1,
            torch.zeros(2, NUM_EXPERTS, dtype=torch.float32, device="xpu"),
            "logits has an unsupported dtype",
        ),
        (1, make_logits(2), "logits must be on XPU"),
        (
            0,
            torch.empty(
                2, 2 * HIDDEN_SIZE, dtype=torch.float16, device="xpu"
            )[:, ::2],
            "x must be contiguous",
        ),
        (
            1,
            torch.empty(
                2 * NUM_EXPERTS + 1, dtype=torch.float16, device="xpu"
            )[1:].view(2, NUM_EXPERTS),
            "logits data pointer must be 16-byte aligned",
        ),
    ]
    for index, replacement, expected_error in invalid_cases:
        args = valid.copy()
        args[index] = replacement
        with pytest.raises(RuntimeError, match=re.escape(expected_error)):
            _op(*args)

    scalar_cases = (
        (10, 9, "multi-m asymmetric out v1 requires top_k=10"),
        (11, 2, "multi-m asymmetric out v1 requires one shared expert"),
        (12, 256, "multi-m asymmetric out v1 requires 512 routed experts"),
    )
    for scalar_index, invalid_value, expected_error in scalar_cases:
        args = valid.copy()
        args[scalar_index] = invalid_value
        with pytest.raises(RuntimeError, match=re.escape(expected_error)):
            _op(*args)


def test_source_contract_and_tail_are_transactional() -> None:
    source = (
        Path(__file__).parents[1] / "csrc/moe_batch/moe_int4.sycl"
    ).read_text()
    start = source.index(
        "torch::Tensor "
        "moe_forward_multi_m_cutlass_nmajor_int4_fp16_shared_asymmetric_out_v1("
    )
    end = source.index("torch::Tensor moe_tiny_fp16_shared_up(", start)
    body = source[start:end]
    alias_guard = body.index("output must not alias any input tensor")
    allocation = body.index(
        "ensure_moe_multi_m_asymmetric_v1_buffers(device, stream)"
    )
    topk_submit = body.index("moe_topk_v2_host<N_ROUTED_EXPERTS, TOP_K>")
    up_submit = body.index(
        "moe_ws_up_cutlass_int4_with_shared_fp16_kernel<int32_t>"
    )
    down_submit = body.index(
        "moe_ws_down_cutlass_int4_with_shared_fp16_kernel<int32_t, 4>"
    )
    assert alias_guard < allocation < topk_submit < up_submit < down_submit

    cache_start = source.index(
        "static MoeMultiMAsymmetricV1Buffers& "
        "ensure_moe_multi_m_asymmetric_v1_buffers("
    )
    cache_end = source.index(
        "torch::Tensor "
        "moe_forward_multi_m_cutlass_nmajor_int4_fp16_shared_asymmetric_out_v1(",
        cache_start,
    )
    cache = source[cache_start:cache_end]
    assert "MoeMultiMAsymmetricV1Buffers candidate" in cache
    assert cache.rindex("torch::empty") < cache.index(
        "s_moe_multi_m_asymmetric_v1_buffers_by_stream.emplace"
    )
    assert "s_moe_multi_m_asymmetric_v1_buffers_by_stream[" not in cache
    assert "MAX_N_TOKENS = 32" in cache

    down_start = source.index(
        "void moe_ws_down_cutlass_int4_with_shared_fp16_kernel("
    )
    down_end = source.index(
        "template <typename IndexT>\nvoid moe_tiny_m_down_cutlass_int4_kernel(",
        down_start,
    )
    down = source[down_start:down_end]
    assert "k + VL <= shared_inter_size" in down
    assert "for (; k < shared_inter_size; ++k)" in down
    assert "for (int k = 0; k < shared_inter_size; k += VL)" not in down
    assert "Tensor(a!) output" in source
    assert "-> Tensor(a!)" in source

    package_init = (
        Path(__file__).parents[1]
        / "python/custom_esimd_kernels_vllm/__init__.py"
    ).read_text()
    assert (
        "moe_forward_multi_m_cutlass_nmajor_int4_fp16_shared_"
        "asymmetric_out_v1," in package_init
    )


@pytest.mark.parametrize("n_tokens", TOKEN_COUNTS)
def test_multi_m_matches_full_reference(
    inputs: MultiInputs,
    n_tokens: int,
) -> None:
    logits_cpu = make_logits(n_tokens)
    logits = logits_cpu.to("xpu")
    topk_weight, topk_idx = torch.ops.moe_int4_ops.moe_topk_int4(
        logits, TOP_K, NUM_EXPERTS, True
    )
    expected_weight, expected_idx = _topk_reference(logits_cpu)
    torch.testing.assert_close(
        topk_idx.cpu(), expected_idx.to(torch.int32), rtol=0, atol=0
    )
    torch.testing.assert_close(
        topk_weight.cpu(), expected_weight, rtol=0, atol=0
    )

    reference = _reference(inputs, n_tokens, logits_cpu)
    output_buffer = torch.empty_like(inputs.x[:n_tokens])
    output = _op(*inputs.args(n_tokens, logits, output_buffer))
    assert output.data_ptr() == output_buffer.data_ptr()
    assert output.shape == (n_tokens, HIDDEN_SIZE)
    assert output.dtype == torch.float16
    assert torch.count_nonzero(
        inputs.base.w13_selected[:, SHARED_SIZE:ROUTED_SIZE]
    ) > 0
    assert torch.count_nonzero(
        inputs.base.w13_selected[:, ROUTED_SIZE + SHARED_SIZE :]
    ) > 0
    assert torch.count_nonzero(
        inputs.base.w2_selected[..., SHARED_SIZE // 2 :]
    ) == 0
    torch.testing.assert_close(
        output.cpu(), reference, rtol=0, atol=OUTPUT_ATOL
    )


def test_public_python_wrapper_preserves_caller_output(
    inputs: MultiInputs,
) -> None:
    ops_path = (
        Path(__file__).parents[1]
        / "python/custom_esimd_kernels_vllm/ops.py"
    )
    wrapper = runpy.run_path(str(ops_path))[
        "moe_forward_multi_m_cutlass_nmajor_int4_fp16_shared_asymmetric_out_v1"
    ]

    n_tokens = 2
    logits = make_logits(n_tokens).to("xpu")
    output_buffer = torch.empty_like(inputs.x[:n_tokens])
    output = wrapper(*inputs.args(n_tokens, logits, output_buffer))
    assert output.data_ptr() == output_buffer.data_ptr()
    torch.testing.assert_close(
        output,
        _op(*inputs.args(n_tokens, logits)),
        rtol=0,
        atol=0,
    )


def test_shared_tail_and_routed_components_are_independent(
    inputs: MultiInputs,
) -> None:
    n_tokens = 4
    logits_cpu = make_logits(n_tokens)
    logits = logits_cpu.to("xpu")
    routed_reference, shared_reference = _reference_components(
        inputs, n_tokens, logits_cpu
    )
    reference = (routed_reference + shared_reference).half()

    shared_only_args = inputs.args(n_tokens, logits)
    shared_only_args[3] = torch.zeros_like(inputs.base.w13_scales)
    shared_only = _op(*shared_only_args).cpu()
    routed_only_args = inputs.args(n_tokens, logits)
    routed_only_args[6] = torch.zeros_like(inputs.base.shared_gate_up_weight)
    routed_only = _op(*routed_only_args).cpu()

    assert shared_only.abs().max() > 10 * OUTPUT_ATOL
    assert routed_only.abs().max() > 10 * OUTPUT_ATOL
    torch.testing.assert_close(
        shared_only, shared_reference.half(), rtol=0, atol=OUTPUT_ATOL
    )
    torch.testing.assert_close(
        routed_only, routed_reference.half(), rtol=0, atol=OUTPUT_ATOL
    )
    torch.testing.assert_close(
        (shared_only.float() + routed_only.float()).half(),
        reference,
        rtol=0,
        atol=OUTPUT_ATOL,
    )


def test_m_gt1_equal_width_shared_down_regression(
    inputs: MultiInputs,
) -> None:
    n_tokens = 4
    logits_cpu = make_logits(n_tokens)
    logits = logits_cpu.to("xpu")
    shared_gate_up_cpu = torch.zeros(
        2 * ROUTED_SIZE, HIDDEN_SIZE, dtype=torch.float16
    )
    shared_gate_up_cpu[:SHARED_SIZE] = inputs.base.shared_gate_up_cpu[
        :SHARED_SIZE
    ]
    shared_gate_up_cpu[
        ROUTED_SIZE : ROUTED_SIZE + SHARED_SIZE
    ] = inputs.base.shared_gate_up_cpu[SHARED_SIZE:]
    shared_down_cpu = torch.zeros(
        HIDDEN_SIZE, ROUTED_SIZE, dtype=torch.float16
    )
    shared_down_cpu[:, :SHARED_SIZE] = inputs.base.shared_down_cpu

    output = torch.ops.moe_int4_ops.moe_forward_cutlass_nmajor_int4_full(
        inputs.x[:n_tokens],
        logits,
        inputs.base.w13_qweight_s4,
        inputs.base.w13_scales,
        inputs.base.w2_qweight_s4,
        inputs.base.w2_scales,
        shared_gate_up_cpu.to("xpu"),
        shared_down_cpu.to("xpu"),
        inputs.base.shared_expert_gate_weight,
        TOP_K,
        NUM_SHARED_EXPERTS,
        NUM_EXPERTS,
    )
    torch.testing.assert_close(
        output.cpu(),
        _reference(inputs, n_tokens, logits_cpu),
        rtol=0,
        atol=OUTPUT_ATOL,
    )


def test_caller_owned_outputs_survive_later_calls(inputs: MultiInputs) -> None:
    n_tokens = 4
    logits = make_logits(n_tokens).to("xpu")
    output_a = torch.empty_like(inputs.x[:n_tokens])
    output_b = torch.empty_like(inputs.x[:n_tokens])
    returned_a = _op(*inputs.args(n_tokens, logits, output_a))
    first_result = returned_a.clone()
    returned_b = _op(*inputs.args(n_tokens, logits, output_b))
    torch.xpu.synchronize()
    assert returned_a.data_ptr() == output_a.data_ptr()
    assert returned_b.data_ptr() == output_b.data_ptr()
    assert output_a.data_ptr() != output_b.data_ptr()
    torch.testing.assert_close(output_a, first_result, rtol=0, atol=0)


def test_two_streams_use_independent_scratch(inputs: MultiInputs) -> None:
    n_tokens = 4
    logits_a_cpu = make_logits(n_tokens, variant=0)
    logits_b_cpu = make_logits(n_tokens, variant=1)
    logits_a = logits_a_cpu.to("xpu")
    logits_b = logits_b_cpu.to("xpu")
    output_a = torch.empty_like(inputs.x[:n_tokens])
    output_b = torch.empty_like(inputs.x[:n_tokens])
    stream_a = torch.xpu.Stream()
    stream_b = torch.xpu.Stream()
    with torch.xpu.stream(stream_a):
        _op(*inputs.args(n_tokens, logits_a, output_a))
    with torch.xpu.stream(stream_b):
        _op(*inputs.args(n_tokens, logits_b, output_b))
    torch.xpu.synchronize()

    reference_a = _reference(inputs, n_tokens, logits_a_cpu)
    reference_b = _reference(inputs, n_tokens, logits_b_cpu)
    torch.testing.assert_close(
        output_a.cpu(), reference_a, rtol=0, atol=OUTPUT_ATOL
    )
    torch.testing.assert_close(
        output_b.cpu(), reference_b, rtol=0, atol=OUTPUT_ATOL
    )
