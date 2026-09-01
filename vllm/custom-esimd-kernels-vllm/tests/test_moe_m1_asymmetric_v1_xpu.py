import gc
import os
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

HIDDEN_SIZE = 2560
NUM_EXPERTS = 512
TOP_K = 10
ROUTED_SIZE = 128
SHARED_SIZE = 80
NUM_SHARED_EXPERTS = 1
TIE_EXPERT_IDS = tuple(range(TOP_K))
UNIQUE_EXPERT_IDS = (32, 63, 64, 127, 128, 255, 256, 383, 510, 511)
SELECTED_EXPERT_IDS = TIE_EXPERT_IDS + UNIQUE_EXPERT_IDS
SELECTED_EXPERTS = len(SELECTED_EXPERT_IDS)
EXPERT_SLOT_BY_ID = {
    expert_id: slot for slot, expert_id in enumerate(SELECTED_EXPERT_IDS)
}
GROUP_SIZE = 128
SHARED_WEIGHT_SCALE = 0.2
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
    dso = _dso_path()
    if not dso.is_file():
        raise RuntimeError(f"focused DSO does not exist: {dso}")
    torch.ops.load_library(str(dso))


@dataclass
class Inputs:
    x_cpu: torch.Tensor
    x: torch.Tensor
    w13_selected: torch.Tensor
    w13_selected_scales_cpu: torch.Tensor
    w13_qweight_s4: torch.Tensor
    w13_scales: torch.Tensor
    w2_selected: torch.Tensor
    w2_selected_scales_cpu: torch.Tensor
    w2_qweight_s4: torch.Tensor
    w2_scales: torch.Tensor
    shared_gate_up_cpu: torch.Tensor
    shared_gate_up_weight: torch.Tensor
    shared_down_cpu: torch.Tensor
    shared_down_weight: torch.Tensor
    shared_gate_cpu: torch.Tensor
    shared_expert_gate_weight: torch.Tensor
    output: torch.Tensor

    def args(
        self,
        logits: torch.Tensor,
        output: torch.Tensor | None = None,
    ) -> list[object]:
        return [
            self.x,
            logits,
            self.w13_qweight_s4,
            self.w13_scales,
            self.w2_qweight_s4,
            self.w2_scales,
            self.shared_gate_up_weight,
            self.shared_down_weight,
            self.shared_expert_gate_weight,
            self.output if output is None else output,
            TOP_K,
            NUM_SHARED_EXPERTS,
            NUM_EXPERTS,
        ]


def build_inputs() -> Inputs:
    generator = torch.Generator().manual_seed(20260901)
    x_cpu = (torch.randn(1, HIDDEN_SIZE, generator=generator) * 0.05).half()

    w13_selected = torch.randint(
        0,
        256,
        (SELECTED_EXPERTS, 2 * ROUTED_SIZE, HIDDEN_SIZE // 2),
        dtype=torch.uint8,
        generator=generator,
    )
    selected_expert_indices = torch.tensor(
        SELECTED_EXPERT_IDS, dtype=torch.int64, device="xpu"
    )
    w13_qweight_s4 = torch.zeros(
        NUM_EXPERTS,
        2 * ROUTED_SIZE,
        HIDDEN_SIZE // 2,
        dtype=torch.uint8,
        device="xpu",
    )
    w13_qweight_s4.index_copy_(
        0, selected_expert_indices, w13_selected.to("xpu")
    )
    scale_expert = torch.arange(SELECTED_EXPERTS).view(-1, 1, 1)
    scale_row = torch.arange(2 * ROUTED_SIZE).view(1, -1, 1)
    scale_group = torch.arange(HIDDEN_SIZE // GROUP_SIZE).view(1, 1, -1)
    w13_selected_scales_cpu = (
        0.012
        + 0.0004 * scale_expert
        + 0.0002 * (scale_row % 7)
        + 0.00007 * scale_group
    ).half()
    w13_scales = torch.zeros(
        (NUM_EXPERTS, 2 * ROUTED_SIZE, HIDDEN_SIZE // GROUP_SIZE),
        dtype=torch.float16,
        device="xpu",
    )
    w13_scales.index_copy_(
        0, selected_expert_indices, w13_selected_scales_cpu.to("xpu")
    )

    w2_selected = torch.randint(
        0,
        256,
        (SELECTED_EXPERTS, HIDDEN_SIZE, ROUTED_SIZE // 2),
        dtype=torch.uint8,
        generator=generator,
    )
    # Physical routed channels [80,128) are padding. Keep their up weights
    # non-zero but force their down weights to exact signed-S4 zero.
    w2_selected[..., SHARED_SIZE // 2 :] = 0
    w2_qweight_s4 = torch.zeros(
        NUM_EXPERTS,
        HIDDEN_SIZE,
        ROUTED_SIZE // 2,
        dtype=torch.uint8,
        device="xpu",
    )
    w2_qweight_s4.index_copy_(
        0, selected_expert_indices, w2_selected.to("xpu")
    )
    w2_scale_expert = torch.arange(SELECTED_EXPERTS).view(-1, 1, 1)
    w2_scale_row = torch.arange(HIDDEN_SIZE).view(1, -1, 1)
    w2_selected_scales_cpu = (
        0.07
        + 0.002 * w2_scale_expert
        + 0.0003 * (w2_scale_row % 17)
    ).half()
    w2_scales = torch.zeros(
        (NUM_EXPERTS, HIDDEN_SIZE, ROUTED_SIZE // GROUP_SIZE),
        dtype=torch.float16,
        device="xpu",
    )
    w2_scales.index_copy_(
        0, selected_expert_indices, w2_selected_scales_cpu.to("xpu")
    )

    shared_gate_up_cpu = (
        torch.randn(2 * SHARED_SIZE, HIDDEN_SIZE, generator=generator)
        * SHARED_WEIGHT_SCALE
    ).half()
    shared_down_cpu = (
        torch.randn(HIDDEN_SIZE, SHARED_SIZE, generator=generator)
        * SHARED_WEIGHT_SCALE
    ).half()
    shared_gate_cpu = (
        torch.randn(NUM_SHARED_EXPERTS, HIDDEN_SIZE, generator=generator) * 0.04
    ).half()

    result = Inputs(
        x_cpu=x_cpu,
        x=x_cpu.to("xpu"),
        w13_selected=w13_selected,
        w13_selected_scales_cpu=w13_selected_scales_cpu,
        w13_qweight_s4=w13_qweight_s4,
        w13_scales=w13_scales,
        w2_selected=w2_selected,
        w2_selected_scales_cpu=w2_selected_scales_cpu,
        w2_qweight_s4=w2_qweight_s4,
        w2_scales=w2_scales,
        shared_gate_up_cpu=shared_gate_up_cpu,
        shared_gate_up_weight=shared_gate_up_cpu.to("xpu"),
        shared_down_cpu=shared_down_cpu,
        shared_down_weight=shared_down_cpu.to("xpu"),
        shared_gate_cpu=shared_gate_cpu,
        shared_expert_gate_weight=shared_gate_cpu.to("xpu"),
        output=torch.empty(1, HIDDEN_SIZE, dtype=torch.float16, device="xpu"),
    )
    torch.xpu.synchronize()
    return result


@pytest.fixture(scope="module")
def inputs() -> Inputs:
    return build_inputs()


def make_logits(case: str) -> torch.Tensor:
    if case == "unique":
        logits = torch.full((1, NUM_EXPERTS), -4.0, dtype=torch.float16)
        logits[0, torch.tensor(UNIQUE_EXPERT_IDS)] = torch.arange(
            TOP_K, 0, -1, dtype=torch.float16
        )
        return logits
    if case == "tie":
        return torch.zeros(1, NUM_EXPERTS, dtype=torch.float16)
    raise AssertionError(f"unknown logits case: {case}")


def _unpack_signed_s4(packed: torch.Tensor) -> torch.Tensor:
    low = (packed.to(torch.int16) & 0xF).to(torch.float32)
    high = ((packed.to(torch.int16) >> 4) & 0xF).to(torch.float32)
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)
    unpacked = torch.empty(*packed.shape[:-1], packed.shape[-1] * 2)
    unpacked[..., 0::2] = low
    unpacked[..., 1::2] = high
    return unpacked


def _dequant_selected_w13(
    data: Inputs, expert_ids: tuple[int, ...]
) -> torch.Tensor:
    slots = [EXPERT_SLOT_BY_ID[expert_id] for expert_id in expert_ids]
    unpacked = _unpack_signed_s4(data.w13_selected[slots])
    scales = data.w13_selected_scales_cpu[slots].float().repeat_interleave(
        GROUP_SIZE, dim=-1
    )
    return unpacked * scales


def _dequant_selected_w2(
    data: Inputs, expert_ids: tuple[int, ...]
) -> torch.Tensor:
    slots = [EXPERT_SLOT_BY_ID[expert_id] for expert_id in expert_ids]
    unpacked = _unpack_signed_s4(data.w2_selected[slots])
    return unpacked * data.w2_selected_scales_cpu[slots].float()


def _topk_reference(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities = torch.softmax(logits.float(), dim=-1)
    topk_idx = torch.argsort(
        probabilities, dim=-1, descending=True, stable=True
    )[:, :TOP_K]
    topk_weight = probabilities.gather(1, topk_idx)
    topk_weight = (
        topk_weight / topk_weight.sum(dim=-1, keepdim=True)
    ).half()
    return topk_weight, topk_idx


def _reference(
    data: Inputs,
    topk_weight: torch.Tensor,
    topk_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = data.x_cpu[0].float()
    routed = torch.zeros(HIDDEN_SIZE, dtype=torch.float32)

    for route in range(TOP_K):
        expert = int(topk_idx[0, route])
        w13 = _dequant_selected_w13(data, (expert,))[0]
        projected = torch.mv(w13, x)
        intermediate = (
            F.silu(projected[:ROUTED_SIZE]) * projected[ROUTED_SIZE:]
        ).half()

        w2 = _dequant_selected_w2(data, (expert,))[0]
        routed += float(topk_weight[0, route]) * torch.mv(
            w2, intermediate.float()
        )

    shared_projected = torch.mv(data.shared_gate_up_cpu.float(), x)
    shared_intermediate = (
        F.silu(shared_projected[:SHARED_SIZE])
        * shared_projected[SHARED_SIZE:]
    ).half()
    shared_gate = torch.sigmoid(torch.mv(data.shared_gate_cpu.float(), x))[0]
    shared = shared_gate * torch.mv(
        data.shared_down_cpu.float(), shared_intermediate.float()
    )
    return (routed + shared).half().view(1, -1), routed, shared


def _op(*args: object) -> torch.Tensor:
    return torch.ops.moe_int4_ops.moe_forward_m1_cutlass_nmajor_int4_fp16_shared_asymmetric_out_v1(
        *args
    )


def test_00_preflight_rejects_before_first_valid_submit(inputs: Inputs) -> None:
    logits = make_logits("unique").to("xpu")
    valid = inputs.args(logits)

    invalid_cases: list[tuple[int, object, str]] = [
        (
            0,
            torch.zeros(2, HIDDEN_SIZE, dtype=torch.float16, device="xpu"),
            "x has an unsupported shape",
        ),
        (
            7,
            inputs.shared_down_weight[:, :-1],
            "shared_down_weight has an unsupported shape",
        ),
        (9, inputs.x, "output must not alias any input tensor"),
        (
            1,
            torch.zeros(1, NUM_EXPERTS, dtype=torch.float32, device="xpu"),
            "logits has an unsupported dtype",
        ),
        (1, make_logits("unique"), "logits must be on XPU"),
        (
            0,
            torch.empty(1, 2 * HIDDEN_SIZE, dtype=torch.float16, device="xpu")[
                :, ::2
            ],
            "x must be contiguous",
        ),
        (
            1,
            torch.empty(
                NUM_EXPERTS + 1, dtype=torch.float16, device="xpu"
            )[1:].view(1, NUM_EXPERTS),
            "logits data pointer must be 16-byte aligned",
        ),
    ]
    for index, replacement, expected_error in invalid_cases:
        args = valid.copy()
        args[index] = replacement
        with pytest.raises(RuntimeError, match=re.escape(expected_error)):
            _op(*args)

    scalar_cases = (
        (10, 9, "asymmetric out v1 requires top_k=10"),
        (11, 2, "asymmetric out v1 requires one shared expert"),
        (12, 256, "asymmetric out v1 requires 512 routed experts"),
    )
    for scalar_index, invalid_value, expected_error in scalar_cases:
        args = valid.copy()
        args[scalar_index] = invalid_value
        with pytest.raises(RuntimeError, match=re.escape(expected_error)):
            _op(*args)


def test_source_preflight_precedes_allocation_and_submit() -> None:
    source = (
        Path(__file__).parents[1] / "csrc/moe_batch/moe_int4.sycl"
    ).read_text()
    start = source.index(
        "torch::Tensor "
        "moe_forward_m1_cutlass_nmajor_int4_fp16_shared_asymmetric_out_v1("
    )
    end = source.index("torch::Tensor moe_tiny_fp16_shared_up(", start)
    body = source[start:end]
    last_preflight = body.index(
        'shared_expert_gate_weight, "shared_expert_gate_weight"'
    )
    allocation = body.index(
        "ensure_moe_m1_asymmetric_v1_buffers(device, stream)"
    )
    topk_submit = body.index("moe_topk_v2_host<N_ROUTED_EXPERTS, TOP_K>")
    up_submit = body.index(
        "moe_tiny_m_up_cutlass_int4_with_shared_fp16_kernel<int32_t>"
    )
    down_submit = body.index(
        "moe_tiny_m_down_cutlass_int4_with_shared_fp16_htile_kernel<int32_t>"
    )
    assert last_preflight < allocation < topk_submit < up_submit < down_submit

    cache_start = source.index(
        "static MoeM1AsymmetricV1Buffers& "
        "ensure_moe_m1_asymmetric_v1_buffers("
    )
    cache_end = source.index(
        "torch::Tensor "
        "moe_forward_m1_cutlass_nmajor_int4_fp16_shared_asymmetric_out_v1(",
        cache_start,
    )
    cache_body = source[cache_start:cache_end]
    candidate = cache_body.index("MoeM1AsymmetricV1Buffers candidate")
    last_allocation = cache_body.rindex("torch::empty")
    commit = cache_body.index(
        "s_moe_m1_asymmetric_v1_buffers_by_stream.emplace"
    )
    assert "s_moe_m1_asymmetric_v1_buffers_by_stream[" not in cache_body
    assert "c10::xpu::XPUStream" in cache_body
    assert candidate < last_allocation < commit
    assert "final_outputs" not in cache_body
    assert 'output, "output"' in body
    assert body.index('output, "output"') < allocation
    assert body.index("output must not alias any input tensor") < allocation
    assert "Tensor(a!) output" in source
    assert "-> Tensor(a!)" in source


def test_legacy_cache_key_is_stream_and_geometry_transactional() -> None:
    source = (
        Path(__file__).parents[1] / "csrc/moe_batch/moe_int4.sycl"
    ).read_text()
    start = source.index("struct Int4MoeBufferKey {")
    end = source.index("std::tuple<torch::Tensor, torch::Tensor> moe_topk_int4(")
    cache = source[start:end]

    for field in (
        "c10::xpu::XPUStream stream",
        "int capacity",
        "int top_k",
        "int num_shared_experts",
        "int hidden_size",
        "int intermediate_size",
    ):
        assert field in cache
    assert "capture_instance" not in cache
    assert "static thread_local std::unordered_map" not in cache
    assert "while (capacity < n_tokens)" in cache
    assert "capacity *= 2" in cache
    assert "std::lock_guard<std::mutex> lock" in cache
    assert "s_int4_moe_buffers.find(key)" in cache
    assert "std::make_unique<Int4MoeBuffers>()" in cache
    ensure = cache[
        cache.index("static Int4MoeBuffers& ensure_int4_moe_buffers(") :
        cache.index("static torch::Tensor take_int4_final_output(")
    ]
    assert ensure.rindex("torch::empty") < ensure.index(
        "s_int4_moe_buffers.emplace(key, std::move(candidate))"
    )
    assert "std::recursive_mutex submit_mutex" in cache
    assert "s_int4_moe_buffers[key]" not in cache

    capture = cache[
        cache.index("static Int4MoeRawBuffers allocate_int4_moe_capture_workspace(") :
        cache.index("static Int4MoeRawBuffers view_int4_moe_eager_buffers(")
    ]
    assert capture.count("torch::empty(") == 1
    assert "final_output_offset = reserve_int4_moe_capture_span" in capture
    assert capture.index("final_output_offset =") < capture.index(
        "topk_idx_offset ="
    )
    assert ".narrow(" in capture and ".view({n_tokens, hidden_size})" in capture

    acquire = cache[
        cache.index("static Int4MoeWorkspaceLease acquire_int4_moe_workspace(") :
    ]
    capture_branch = acquire[
        acquire.index("if (stream.is_capturing())") :
        acquire.index("auto& cached = ensure_int4_moe_buffers(")
    ]
    assert "allocate_int4_moe_capture_workspace(" in capture_branch
    assert "ensure_int4_moe_buffers(" not in capture_branch
    assert "std::unique_lock<std::recursive_mutex> submit_lock" in acquire

    callers = source[end : source.index("TORCH_LIBRARY_FRAGMENT", end)]
    assert callers.count("acquire_int4_moe_workspace(") == 3
    from_logits = callers[
        callers.index(
            "torch::Tensor "
            "moe_forward_tiny_cutlass_nmajor_int4_full_fp16_shared_from_logits("
        ) : callers.index("static inline void check_moe_asymmetric_v1_tensor(")
    ]
    assert from_logits.count("acquire_int4_moe_workspace(") == 1
    assert "return moe_forward_tiny_cutlass_nmajor_int4_full_fp16_shared(" not in (
        from_logits
    )
    assert (
        from_logits.index("check_moe_tiny_full_fp16_shared_inputs(")
        < from_logits.index(
            "n_routed_experts == geometry.n_routed_experts"
        )
        < from_logits.index("acquire_int4_moe_workspace(")
        < from_logits.index("try_moe_topk_v2_int4(")
    )
    direct = callers[
        callers.index(
            "torch::Tensor "
            "moe_forward_tiny_cutlass_nmajor_int4_full_fp16_shared("
        ) : callers.index(
            "torch::Tensor "
            "moe_forward_tiny_cutlass_nmajor_int4_full_fp16_shared_from_logits("
        )
    ]
    assert ".to(torch::kInt)" not in direct
    assert "submit_moe_topk_index_int64_to_int32(" in direct
    assert direct.index("acquire_int4_moe_workspace(") < direct.index(
        "submit_moe_topk_index_int64_to_int32("
    )
    preflight = callers[
        callers.index("check_moe_tiny_full_fp16_shared_inputs(") :
        callers.index("static void submit_moe_tiny_full_fp16_shared(")
    ]
    assert "S_INT4_NMAJOR_GROUP_SIZE" in preflight
    assert "w13_scales.size(0) == n_routed_experts_i64" in preflight
    assert "w2_scales.size(0) == n_routed_experts_i64" in preflight


def _make_legacy_generic_case(n_tokens: int = 64) -> tuple[
    Callable[[torch.Tensor], torch.Tensor], tuple[torch.Tensor, ...], torch.xpu.Stream
]:
    hidden_size, intermediate_size = 256, 128
    num_experts = 1
    gate_up_qweight = torch.zeros(
        (num_experts, 2 * intermediate_size, hidden_size // 8),
        dtype=torch.int32,
        device="xpu",
    )
    gate_up_scales = torch.zeros(
        (num_experts, 2 * intermediate_size, hidden_size // GROUP_SIZE),
        dtype=torch.float16,
        device="xpu",
    )
    down_qweight = torch.zeros(
        (num_experts, hidden_size, intermediate_size // 8),
        dtype=torch.int32,
        device="xpu",
    )
    down_scales = torch.zeros(
        (num_experts, hidden_size, intermediate_size // GROUP_SIZE),
        dtype=torch.float16,
        device="xpu",
    )
    shared_gate_up = torch.zeros(
        (2 * intermediate_size, hidden_size),
        dtype=torch.float16,
        device="xpu",
    )
    eye = torch.eye(intermediate_size, dtype=torch.float16, device="xpu")
    shared_gate_up[:intermediate_size, :intermediate_size].copy_(eye)
    shared_gate_up[intermediate_size:, :intermediate_size].copy_(eye)
    shared_down = torch.zeros(
        (hidden_size, intermediate_size), dtype=torch.float16, device="xpu"
    )
    shared_down[:intermediate_size].copy_(eye)
    shared_gate = torch.zeros(
        (1, hidden_size), dtype=torch.float16, device="xpu"
    )
    unused_scale = torch.empty(0, dtype=torch.float16, device="xpu")
    logits = torch.zeros(
        (n_tokens, num_experts), dtype=torch.float16, device="xpu"
    )
    inputs = (
        torch.ones((n_tokens, hidden_size), dtype=torch.float16, device="xpu"),
        torch.full(
            (n_tokens, hidden_size), 2, dtype=torch.float16, device="xpu"
        ),
    )
    stream = torch.xpu.Stream()
    torch.xpu.synchronize()

    def call(x: torch.Tensor) -> torch.Tensor:
        return torch.ops.moe_int4_ops.moe_forward_full_int4(
            x,
            logits,
            gate_up_qweight,
            gate_up_scales,
            shared_gate_up,
            unused_scale,
            down_qweight,
            down_scales,
            shared_down,
            unused_scale,
            shared_gate,
            1,
            1,
            num_experts,
            True,
        )

    return call, inputs, stream


def _make_legacy_nmajor_case(
    *, intermediate_size: int = 128, num_experts: int = 2
) -> dict[str, torch.Tensor]:
    hidden_size = 256
    x = torch.ones((1, hidden_size), dtype=torch.float16, device="xpu")
    w13 = torch.zeros(
        (num_experts, 2 * intermediate_size, hidden_size // 2),
        dtype=torch.uint8,
        device="xpu",
    )
    w13_scales = torch.full(
        (num_experts, 2 * intermediate_size, hidden_size // GROUP_SIZE),
        0.01,
        dtype=torch.float16,
        device="xpu",
    )
    w2 = torch.zeros(
        (num_experts, hidden_size, intermediate_size // 2),
        dtype=torch.uint8,
        device="xpu",
    )
    w2_scales = torch.full(
        (num_experts, hidden_size, intermediate_size // GROUP_SIZE),
        0.01,
        dtype=torch.float16,
        device="xpu",
    )
    if num_experts > 1:
        w13[1].fill_(0x11)
        w2[1].fill_(0x11)
    return {
        "x": x,
        "logits": torch.zeros(
            (1, num_experts), dtype=torch.float16, device="xpu"
        ),
        "w13": w13,
        "w13_scales": w13_scales,
        "w2": w2,
        "w2_scales": w2_scales,
        "shared_gate_up": torch.zeros(
            (2 * intermediate_size, hidden_size),
            dtype=torch.float16,
            device="xpu",
        ),
        "shared_down": torch.zeros(
            (hidden_size, intermediate_size),
            dtype=torch.float16,
            device="xpu",
        ),
        "shared_gate": torch.zeros(
            (1, hidden_size), dtype=torch.float16, device="xpu"
        ),
    }


def _legacy_direct_topk(
    case: dict[str, torch.Tensor], topk_idx: torch.Tensor
) -> torch.Tensor:
    return torch.ops.moe_int4_ops.moe_forward_tiny_cutlass_nmajor_int4_full_fp16_shared(
        case["x"],
        case["w13"],
        case["w13_scales"],
        case["w2"],
        case["w2_scales"],
        torch.ones_like(topk_idx, dtype=torch.float16),
        topk_idx,
        case["shared_gate_up"],
        case["shared_down"],
        case["shared_gate"],
        1,
    )


def _legacy_from_logits(
    case: dict[str, torch.Tensor],
    *,
    logits: torch.Tensor | None = None,
    w13_scales: torch.Tensor | None = None,
    w2_scales: torch.Tensor | None = None,
    n_routed_experts: int | None = None,
) -> torch.Tensor:
    routed_experts = case["w13"].size(0)
    return torch.ops.moe_int4_ops.moe_forward_tiny_cutlass_nmajor_int4_full_fp16_shared_from_logits(
        case["x"],
        case["logits"] if logits is None else logits,
        case["w13"],
        case["w13_scales"] if w13_scales is None else w13_scales,
        case["w2"],
        case["w2_scales"] if w2_scales is None else w2_scales,
        case["shared_gate_up"],
        case["shared_down"],
        case["shared_gate"],
        1,
        1,
        routed_experts if n_routed_experts is None else n_routed_experts,
    )


def test_legacy_cache_same_stream_two_host_threads_are_isolated() -> None:
    call, inputs, stream = _make_legacy_generic_case()
    with torch.xpu.stream(stream):
        references = tuple(call(x).cpu().clone() for x in inputs)
    assert not torch.equal(references[0], references[1])

    barrier = threading.Barrier(2)
    failures: list[object] = []

    def worker(worker_id: int) -> None:
        try:
            with torch.xpu.stream(stream):
                for iteration in range(50):
                    barrier.wait(timeout=30)
                    output = call(inputs[worker_id]).cpu()
                    if not torch.equal(output, references[worker_id]):
                        other = torch.equal(output, references[1 - worker_id])
                        failures.append(
                            (
                                worker_id,
                                iteration,
                                "other" if other else "mixed",
                                float(
                                    (output - references[worker_id]).abs().max()
                                ),
                            )
                        )
        except BaseException as error:  # noqa: BLE001
            failures.append((worker_id, "exception", repr(error)))

    threads = tuple(
        threading.Thread(target=worker, args=(worker_id,))
        for worker_id in range(2)
    )
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=180)
    assert not any(thread.is_alive() for thread in threads)
    assert failures == []


@pytest.mark.parametrize("n_tokens", (1, 64))
def test_legacy_capture_workspace_is_output_owned_and_reclaimed(
    n_tokens: int,
) -> None:
    call, inputs, stream = _make_legacy_generic_case(n_tokens)
    with torch.xpu.stream(stream):
        eager_output = call(inputs[0])
    stream.synchronize()
    reference = eager_output.cpu().clone()
    del eager_output
    gc.collect()
    baseline = torch.xpu.memory_allocated()

    graphs: list[torch.xpu.XPUGraph] = []
    outputs: list[torch.Tensor] = []
    for _ in range(4):
        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph, stream=stream):
            output = call(inputs[0])
        stream.synchronize()
        assert output.storage_offset() == 0
        assert output.untyped_storage().nbytes() > (
            output.numel() * output.element_size()
        )
        graphs.append(graph)
        outputs.append(output)

    assert len({output.data_ptr() for output in outputs}) == len(outputs)
    for graph, output in zip(graphs, outputs, strict=True):
        graph.replay()
        stream.synchronize()
        assert torch.equal(output.cpu(), reference)

    for graph in graphs:
        graph.reset()
    graphs.clear()
    outputs.clear()
    del graph, output
    gc.collect()
    torch.xpu.synchronize()
    assert torch.xpu.memory_allocated() <= baseline + 4096


def test_legacy_capture_int64_topk_replays_with_private_workspace() -> None:
    case = _make_legacy_nmajor_case()
    stream = torch.xpu.Stream()
    topk_idx_a = torch.zeros((1, 1), dtype=torch.int64, device="xpu")
    topk_idx_b = torch.ones((1, 1), dtype=torch.int64, device="xpu")

    graph_a = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph_a, stream=stream):
        output_a = _legacy_direct_topk(case, topk_idx_a)
    stream.synchronize()

    graph_b = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph_b, stream=stream):
        output_b = _legacy_direct_topk(case, topk_idx_b)
    stream.synchronize()

    assert output_a.data_ptr() != output_b.data_ptr()
    for graph, output, expect_nonzero in (
        (graph_b, output_b, True),
        (graph_a, output_a, False),
        (graph_b, output_b, True),
    ):
        graph.replay()
        stream.synchronize()
        nonzero = torch.count_nonzero(output).cpu().item()
        assert (nonzero > 0) is expect_nonzero

    graph_a.reset()
    graph_b.reset()


def test_legacy_tiny_full_quant_contract_is_preflighted() -> None:
    case = _make_legacy_nmajor_case()
    bad_w13_scales = torch.zeros(
        (2, 256, 1), dtype=torch.float16, device="xpu"
    )
    with pytest.raises(RuntimeError, match="w13_scales has an unsupported shape"):
        _legacy_from_logits(case, w13_scales=bad_w13_scales)

    bad_w2_scales = torch.zeros(
        (2, 256, 2), dtype=torch.float16, device="xpu"
    )
    with pytest.raises(RuntimeError, match="w2_scales has an unsupported shape"):
        _legacy_from_logits(case, w2_scales=bad_w2_scales)

    mismatched_logits = torch.zeros(
        (1, 3), dtype=torch.float16, device="xpu"
    )
    with pytest.raises(
        RuntimeError,
        match="n_routed_experts must match the routed weight expert dimension",
    ):
        _legacy_from_logits(
            case, logits=mismatched_logits, n_routed_experts=3
        )

    bad_group_case = _make_legacy_nmajor_case(intermediate_size=64)
    with pytest.raises(
        RuntimeError, match="intermediate_size divisible by 128"
    ):
        _legacy_from_logits(bad_group_case)


@pytest.mark.parametrize("n_tokens", (1, 64))
def test_legacy_from_logits_failed_capture_does_not_publish_workspace(
    n_tokens: int,
) -> None:
    hidden_size, intermediate_size = 256, 128
    num_experts = 1
    x = torch.zeros(
        (n_tokens, hidden_size), dtype=torch.float16, device="xpu"
    )
    logits = torch.zeros(
        (n_tokens, num_experts), dtype=torch.float16, device="xpu"
    )
    w13 = torch.zeros(
        (num_experts, 2 * intermediate_size, hidden_size // 2),
        dtype=torch.uint8,
        device="xpu",
    )
    w13_scales = torch.zeros(
        (num_experts, 2 * intermediate_size, hidden_size // GROUP_SIZE),
        dtype=torch.float16,
        device="xpu",
    )
    w2 = torch.zeros(
        (num_experts, hidden_size, intermediate_size // 2),
        dtype=torch.uint8,
        device="xpu",
    )
    w2_scales = torch.zeros(
        (num_experts, hidden_size, intermediate_size // GROUP_SIZE),
        dtype=torch.float16,
        device="xpu",
    )
    bad_shared_gate_up = torch.empty(
        (0, hidden_size), dtype=torch.float16, device="xpu"
    )
    shared_down = torch.zeros(
        (hidden_size, intermediate_size), dtype=torch.float16, device="xpu"
    )
    shared_gate = torch.zeros(
        (1, hidden_size), dtype=torch.float16, device="xpu"
    )
    stream = torch.xpu.Stream()
    torch.xpu.synchronize()
    gc.collect()
    baseline = torch.xpu.memory_allocated()

    for _ in range(3):
        graph = torch.xpu.XPUGraph()
        with (
            pytest.raises(RuntimeError, match="shared_gate_up_weight"),
            pytest.warns(UserWarning, match="XPU Graph is empty"),
            torch.xpu.graph(graph, stream=stream),
        ):
            torch.ops.moe_int4_ops.moe_forward_tiny_cutlass_nmajor_int4_full_fp16_shared_from_logits(
                    x,
                    logits,
                    w13,
                    w13_scales,
                    w2,
                    w2_scales,
                    bad_shared_gate_up,
                    shared_down,
                    shared_gate,
                    1,
                    1,
                    num_experts,
                )
        graph.reset()
        del graph
        gc.collect()
        torch.xpu.synchronize()
        assert torch.xpu.memory_allocated() <= baseline + 4096


def test_asymmetric_cache_does_not_poison_legacy_geometry(
    inputs: Inputs,
) -> None:
    _op(*inputs.args(make_logits("unique").to("xpu")))

    hidden_size = 3072
    intermediate_size = 128
    num_experts = 1
    x = torch.zeros(1, hidden_size, dtype=torch.float16, device="xpu")
    logits = torch.zeros(1, num_experts, dtype=torch.float16, device="xpu")
    w13 = torch.zeros(
        num_experts,
        2 * intermediate_size,
        hidden_size // 2,
        dtype=torch.uint8,
        device="xpu",
    )
    w13_scales = torch.zeros(
        num_experts,
        2 * intermediate_size,
        hidden_size // GROUP_SIZE,
        dtype=torch.float16,
        device="xpu",
    )
    w2 = torch.zeros(
        num_experts,
        hidden_size,
        intermediate_size // 2,
        dtype=torch.uint8,
        device="xpu",
    )
    w2_scales = torch.zeros(
        num_experts,
        hidden_size,
        intermediate_size // GROUP_SIZE,
        dtype=torch.float16,
        device="xpu",
    )
    shared_gate_up = torch.zeros(
        2 * intermediate_size,
        hidden_size,
        dtype=torch.float16,
        device="xpu",
    )
    shared_down = torch.zeros(
        hidden_size,
        intermediate_size,
        dtype=torch.float16,
        device="xpu",
    )
    shared_gate = torch.zeros(
        1, hidden_size, dtype=torch.float16, device="xpu"
    )
    output = torch.ops.moe_int4_ops.moe_forward_tiny_cutlass_nmajor_int4_full_fp16_shared_from_logits(
        x,
        logits,
        w13,
        w13_scales,
        w2,
        w2_scales,
        shared_gate_up,
        shared_down,
        shared_gate,
        1,
        1,
        num_experts,
    )
    assert output.shape == (1, hidden_size)
    assert torch.count_nonzero(output).cpu().item() == 0

    target_shared_gate_up = torch.zeros(
        2 * ROUTED_SIZE, HIDDEN_SIZE, dtype=torch.float16, device="xpu"
    )
    target_shared_gate_up[:SHARED_SIZE].copy_(
        inputs.shared_gate_up_weight[:SHARED_SIZE]
    )
    target_shared_gate_up[
        ROUTED_SIZE : ROUTED_SIZE + SHARED_SIZE
    ].copy_(inputs.shared_gate_up_weight[SHARED_SIZE:])
    target_shared_down = torch.zeros(
        HIDDEN_SIZE, ROUTED_SIZE, dtype=torch.float16, device="xpu"
    )
    target_shared_down[:, :SHARED_SIZE].copy_(inputs.shared_down_weight)
    target_output = torch.ops.moe_int4_ops.moe_forward_tiny_cutlass_nmajor_int4_full_fp16_shared_from_logits(
        inputs.x,
        make_logits("unique").to("xpu"),
        inputs.w13_qweight_s4,
        inputs.w13_scales,
        inputs.w2_qweight_s4,
        inputs.w2_scales,
        target_shared_gate_up,
        target_shared_down,
        inputs.shared_expert_gate_weight,
        TOP_K,
        NUM_SHARED_EXPERTS,
        NUM_EXPERTS,
    )
    assert target_output.shape == (1, HIDDEN_SIZE)


@pytest.mark.parametrize("case", ("unique", "tie"))
def test_m1_asymmetric_v1_matches_full_reference(
    inputs: Inputs, case: str
) -> None:
    logits_cpu = make_logits(case)
    logits = logits_cpu.to("xpu")
    topk_weight, topk_idx = torch.ops.moe_int4_ops.moe_topk_int4(
        logits, TOP_K, NUM_EXPERTS, True
    )
    topk_weight = topk_weight.cpu()
    topk_idx = topk_idx.cpu()

    probabilities = torch.softmax(logits_cpu.float(), dim=-1)
    expected_idx = torch.argsort(
        probabilities, dim=-1, descending=True, stable=True
    )[:, :TOP_K]
    torch.testing.assert_close(
        topk_idx,
        expected_idx.to(torch.int32),
        rtol=0,
        atol=0,
    )

    reference, routed, shared = _reference(inputs, topk_weight, topk_idx)
    output = _op(*inputs.args(logits)).cpu()

    assert routed.abs().max() > 1e-4
    assert shared.abs().max() > 1e-4
    assert torch.count_nonzero(
        inputs.w13_selected[:, SHARED_SIZE:ROUTED_SIZE]
    ) > 0
    assert torch.count_nonzero(
        inputs.w13_selected[:, ROUTED_SIZE + SHARED_SIZE :]
    ) > 0
    assert torch.count_nonzero(
        inputs.w2_selected[..., SHARED_SIZE // 2 :]
    ) == 0
    assert output.shape == (1, HIDDEN_SIZE)
    assert output.dtype == torch.float16
    if case == "tie":
        # Every expected low-ID tie expert and each nontrivial high-ID
        # unique expert has independent packed weights and scales. TopK exactness
        # is checked above; the contribution bound makes an omitted tie route
        # observable in the fused output.
        for route in range(TOP_K):
            expert = int(topk_idx[0, route])
            w13 = _dequant_selected_w13(inputs, (expert,))[0]
            projected = torch.mv(w13, inputs.x_cpu[0].float())
            intermediate = (
                F.silu(projected[:ROUTED_SIZE])
                * projected[ROUTED_SIZE:]
            ).half()
            w2 = _dequant_selected_w2(inputs, (expert,))[0]
            contribution = float(topk_weight[0, route]) * torch.mv(
                w2, intermediate.float()
            )
            assert contribution.abs().max() > 2 * OUTPUT_ATOL
    assert torch.unique(inputs.w13_selected_scales_cpu).numel() > 32
    assert torch.unique(inputs.w2_selected_scales_cpu).numel() > 32
    torch.testing.assert_close(output, reference, rtol=0, atol=OUTPUT_ATOL)


def test_caller_owned_outputs_do_not_alias_each_other(inputs: Inputs) -> None:
    logits = make_logits("unique").to("xpu")
    output_a = torch.empty_like(inputs.output)
    output_b = torch.empty_like(inputs.output)
    returned_a = _op(*inputs.args(logits, output_a))
    first_result = returned_a.clone()
    returned_b = _op(*inputs.args(logits, output_b))
    torch.xpu.synchronize()
    assert returned_a.data_ptr() == output_a.data_ptr()
    assert returned_b.data_ptr() == output_b.data_ptr()
    assert output_a.data_ptr() != output_b.data_ptr()
    torch.testing.assert_close(output_a, first_result, rtol=0, atol=0)


def test_two_streams_use_independent_scratch(inputs: Inputs) -> None:
    logits_a_cpu = make_logits("unique")
    logits_b_cpu = make_logits("tie")
    logits_a = logits_a_cpu.to("xpu")
    logits_b = logits_b_cpu.to("xpu")
    output_a = torch.empty_like(inputs.output)
    output_b = torch.empty_like(inputs.output)
    stream_a = torch.xpu.Stream()
    stream_b = torch.xpu.Stream()
    with torch.xpu.stream(stream_a):
        _op(*inputs.args(logits_a, output_a))
    with torch.xpu.stream(stream_b):
        _op(*inputs.args(logits_b, output_b))
    torch.xpu.synchronize()

    weight_a, idx_a = _topk_reference(logits_a_cpu)
    weight_b, idx_b = _topk_reference(logits_b_cpu)
    reference_a, _, _ = _reference(inputs, weight_a, idx_a)
    reference_b, _, _ = _reference(inputs, weight_b, idx_b)
    torch.testing.assert_close(output_a.cpu(), reference_a, rtol=0, atol=OUTPUT_ATOL)
    torch.testing.assert_close(output_b.cpu(), reference_b, rtol=0, atol=OUTPUT_ATOL)


def test_routed_and_shared_components_are_independently_observable(
    inputs: Inputs,
) -> None:
    logits = make_logits("unique").to("xpu")
    topk_weight, topk_idx = torch.ops.moe_int4_ops.moe_topk_int4(
        logits, TOP_K, NUM_EXPERTS, True
    )
    _, routed, shared = _reference(
        inputs, topk_weight.cpu(), topk_idx.cpu()
    )
    assert routed.abs().max() > 10 * OUTPUT_ATOL
    assert shared.abs().max() > 10 * OUTPUT_ATOL

    shared_only_args = inputs.args(logits)
    shared_only_args[3] = torch.zeros_like(inputs.w13_scales)
    shared_only = _op(*shared_only_args).cpu()
    torch.testing.assert_close(
        shared_only, shared.half().view(1, -1), rtol=0, atol=OUTPUT_ATOL
    )

    routed_only_args = inputs.args(logits)
    routed_only_args[6] = torch.zeros_like(inputs.shared_gate_up_weight)
    routed_only = _op(*routed_only_args).cpu()
    torch.testing.assert_close(
        routed_only, routed.half().view(1, -1), rtol=0, atol=OUTPUT_ATOL
    )
