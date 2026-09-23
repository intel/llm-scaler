import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_moe_m1_asymmetric_v1_xpu import (
    HIDDEN_SIZE,
    NUM_EXPERTS,
    NUM_SHARED_EXPERTS,
    OUTPUT_ATOL,
    ROUTED_SIZE,
    SHARED_SIZE,
    TOP_K,
    _dequant_selected_w2,
    _dequant_selected_w13,
    _topk_reference,
    build_inputs,
    make_logits,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dso", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup-iterations", type=int, default=20)
    parser.add_argument("--formal-iterations", type=int, default=200)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _measure(fn, warmup_iterations: int, formal_iterations: int) -> dict[str, float]:
    # One warmup phase.
    for _ in range(warmup_iterations):
        result = fn()
    torch.xpu.synchronize()

    # One formal phase.
    start_event = torch.xpu.Event(enable_timing=True)
    end_event = torch.xpu.Event(enable_timing=True)
    wall_start_ns = time.perf_counter_ns()
    start_event.record()
    for _ in range(formal_iterations):
        result = fn()
    end_event.record()
    torch.xpu.synchronize()
    wall_end_ns = time.perf_counter_ns()
    return {
        "event_us_per_call": (
            start_event.elapsed_time(end_event) * 1000.0 / formal_iterations
        ),
        "wall_us_per_call": (
            (wall_end_ns - wall_start_ns) / 1000.0 / formal_iterations
        ),
        "output_abs_max": float(result.detach().abs().max().cpu()),
    }


def main() -> None:
    args = _parse_args()
    if args.warmup_iterations <= 0 or args.formal_iterations <= 0:
        raise ValueError("iteration counts must be positive")
    torch.ops.load_library(str(args.dso))

    data = build_inputs()
    logits = make_logits("unique").to("xpu")
    _, topk_idx_cpu = _topk_reference(make_logits("unique"))
    selected_expert_ids = tuple(int(value) for value in topk_idx_cpu[0])
    selected_w13 = _dequant_selected_w13(
        data, selected_expert_ids
    ).to("xpu")
    selected_w2 = _dequant_selected_w2(
        data, selected_expert_ids
    ).to("xpu")
    torch.xpu.synchronize()

    def native_with(
        w13_scales: torch.Tensor,
        shared_gate_up_weight: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ops.moe_int4_ops.moe_forward_m1_cutlass_nmajor_int4_fp16_shared_asymmetric_out_v1(
            data.x,
            logits,
            data.w13_qweight_s4,
            w13_scales,
            data.w2_qweight_s4,
            data.w2_scales,
            shared_gate_up_weight,
            data.shared_down_weight,
            data.shared_expert_gate_weight,
            output,
            TOP_K,
            NUM_SHARED_EXPERTS,
            NUM_EXPERTS,
        )

    def native() -> torch.Tensor:
        return native_with(
            data.w13_scales, data.shared_gate_up_weight, data.output
        )

    def torch_routed_reference() -> torch.Tensor:
        probabilities = torch.softmax(logits.float(), dim=-1)
        topk_weight, _ = torch.topk(probabilities, TOP_K, dim=-1)
        topk_weight = (
            topk_weight / topk_weight.sum(dim=-1, keepdim=True)
        ).half()
        expanded_x = data.x.float().expand(TOP_K, -1).unsqueeze(-1)
        projected = torch.bmm(selected_w13, expanded_x).squeeze(-1)
        routed_intermediate = (
            F.silu(projected[:, :ROUTED_SIZE].float())
            * projected[:, ROUTED_SIZE:].float()
        ).half()
        routed_per_expert = torch.bmm(
            selected_w2, routed_intermediate.float().unsqueeze(-1)
        ).squeeze(-1)
        return (
            routed_per_expert.float()
            * topk_weight[0].float().unsqueeze(-1)
        ).sum(dim=0, keepdim=True)

    def torch_shared_reference() -> torch.Tensor:
        shared_projected = F.linear(data.x, data.shared_gate_up_weight)
        shared_intermediate = (
            F.silu(shared_projected[:, :SHARED_SIZE].float())
            * shared_projected[:, SHARED_SIZE:].float()
        ).half()
        shared = F.linear(shared_intermediate, data.shared_down_weight)
        shared_gate = torch.sigmoid(
            F.linear(data.x, data.shared_expert_gate_weight).float()
        )
        return shared.float() * shared_gate

    def torch_reference() -> torch.Tensor:
        return (torch_routed_reference() + torch_shared_reference()).half()

    zero_w13_scales = torch.zeros_like(data.w13_scales)
    zero_shared_gate_up = torch.zeros_like(data.shared_gate_up_weight)
    native_probe = native_with(
        data.w13_scales,
        data.shared_gate_up_weight,
        torch.empty_like(data.output),
    )
    reference_probe = torch_reference()
    shared_only_native = native_with(
        zero_w13_scales,
        data.shared_gate_up_weight,
        torch.empty_like(data.output),
    )
    shared_only_reference = torch_shared_reference().half()
    routed_only_native = native_with(
        data.w13_scales,
        zero_shared_gate_up,
        torch.empty_like(data.output),
    )
    routed_only_reference = torch_routed_reference().half()
    torch.xpu.synchronize()

    max_abs_error = float(
        (native_probe.float() - reference_probe.float()).abs().max().cpu()
    )
    shared_max_abs_error = float(
        (shared_only_native.float() - shared_only_reference.float())
        .abs()
        .max()
        .cpu()
    )
    routed_max_abs_error = float(
        (routed_only_native.float() - routed_only_reference.float())
        .abs()
        .max()
        .cpu()
    )
    assert float(shared_only_reference.abs().max().cpu()) > 10 * OUTPUT_ATOL
    assert float(routed_only_reference.abs().max().cpu()) > 10 * OUTPUT_ATOL
    torch.testing.assert_close(
        native_probe.cpu(), reference_probe.cpu(), rtol=0, atol=OUTPUT_ATOL
    )
    torch.testing.assert_close(
        shared_only_native.cpu(),
        shared_only_reference.cpu(),
        rtol=0,
        atol=OUTPUT_ATOL,
    )
    torch.testing.assert_close(
        routed_only_native.cpu(),
        routed_only_reference.cpu(),
        rtol=0,
        atol=OUTPUT_ATOL,
    )

    reference_timing = _measure(
        torch_reference, args.warmup_iterations, args.formal_iterations
    )
    native_timing = _measure(
        native, args.warmup_iterations, args.formal_iterations
    )
    event_speedup = (
        reference_timing["event_us_per_call"]
        / native_timing["event_us_per_call"]
    )
    wall_speedup = (
        reference_timing["wall_us_per_call"]
        / native_timing["wall_us_per_call"]
    )

    report = {
        "dso": str(args.dso),
        "dso_sha256": _sha256(args.dso),
        "shape": {
            "x": [1, HIDDEN_SIZE],
            "logits": [1, NUM_EXPERTS],
            "top_k": TOP_K,
            "routed_physical": ROUTED_SIZE,
            "shared_physical": SHARED_SIZE,
        },
        "dtype": "float16 activation/scales/shared, signed-s4 uint8 routed",
        "accuracy_probe_max_abs_error": max_abs_error,
        "accuracy_probe_shared_only_max_abs_error": shared_max_abs_error,
        "accuracy_probe_routed_only_max_abs_error": routed_max_abs_error,
        "warmup": {
            "phases_per_implementation": 1,
            "iterations": args.warmup_iterations,
        },
        "formal": {
            "phases_per_implementation": 1,
            "iterations": args.formal_iterations,
            "torch_reference": reference_timing,
            "native_asymmetric_out_v1": native_timing,
            "event_speedup": event_speedup,
            "wall_speedup": wall_speedup,
        },
        "baseline_scope": (
            "Allocation-inclusive XPU Torch operator chain for the same fixed "
            "E512/K10/P128/S80 math versus the fused caller-owned-output "
            "native operator; this is not a pure kernel-to-kernel comparison "
            "or a TP8 production-server fallback measurement"
        ),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if event_speedup <= 1.0 or wall_speedup <= 1.0:
        raise RuntimeError("asymmetric v1 did not beat the XPU Torch reference")


if __name__ == "__main__":
    main()
