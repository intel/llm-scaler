import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_moe_multi_m_asymmetric_v1_xpu import (
    HIDDEN_SIZE,
    NUM_EXPERTS,
    NUM_SHARED_EXPERTS,
    OUTPUT_ATOL,
    ROUTED_SIZE,
    SHARED_SIZE,
    TOP_K,
    _reference,
    build_multi_inputs,
    make_logits,
)

BENCHMARK_TOKEN_COUNTS = (2, 4, 16, 32)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dso", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup-iterations", type=int, default=10)
    parser.add_argument("--formal-iterations", type=int, default=50)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _measure(fn, warmup_iterations: int, formal_iterations: int) -> dict[str, float]:
    # Exactly one warmup phase for this implementation and shape.
    for _ in range(warmup_iterations):
        result = fn()
    torch.xpu.synchronize()

    # Exactly one formal phase for this implementation and shape.
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
    data = build_multi_inputs()
    results = {}

    for n_tokens in BENCHMARK_TOKEN_COUNTS:
        logits_cpu = make_logits(n_tokens)
        logits = logits_cpu.to("xpu")
        x = data.x[:n_tokens]
        output = torch.empty_like(x)

        def generic(
            x: torch.Tensor = x,
            logits: torch.Tensor = logits,
        ) -> torch.Tensor:
            return torch.ops.moe_int4_ops.moe_forward_cutlass_nmajor_int4_full(
                x,
                logits,
                data.base.w13_qweight_s4,
                data.base.w13_scales,
                data.base.w2_qweight_s4,
                data.base.w2_scales,
                data.base.shared_gate_up_weight,
                data.base.shared_down_weight,
                data.base.shared_expert_gate_weight,
                TOP_K,
                NUM_SHARED_EXPERTS,
                NUM_EXPERTS,
            )

        def candidate(
            x: torch.Tensor = x,
            logits: torch.Tensor = logits,
            output: torch.Tensor = output,
        ) -> torch.Tensor:
            return torch.ops.moe_int4_ops.moe_forward_multi_m_cutlass_nmajor_int4_fp16_shared_asymmetric_out_v1(
                x,
                logits,
                data.base.w13_qweight_s4,
                data.base.w13_scales,
                data.base.w2_qweight_s4,
                data.base.w2_scales,
                data.base.shared_gate_up_weight,
                data.base.shared_down_weight,
                data.base.shared_expert_gate_weight,
                output,
                TOP_K,
                NUM_SHARED_EXPERTS,
                NUM_EXPERTS,
            )

        reference_probe = _reference(data, n_tokens, logits_cpu)
        generic_probe = generic().cpu()
        candidate_probe = candidate().cpu()
        generic_max_abs = float(
            (generic_probe.float() - reference_probe.float()).abs().max()
        )
        candidate_max_abs = float(
            (candidate_probe.float() - reference_probe.float()).abs().max()
        )
        torch.testing.assert_close(
            generic_probe, reference_probe, rtol=0, atol=OUTPUT_ATOL
        )
        torch.testing.assert_close(
            candidate_probe, reference_probe, rtol=0, atol=OUTPUT_ATOL
        )

        generic_timing = _measure(
            generic, args.warmup_iterations, args.formal_iterations
        )
        candidate_timing = _measure(
            candidate, args.warmup_iterations, args.formal_iterations
        )
        event_speedup = (
            generic_timing["event_us_per_call"]
            / candidate_timing["event_us_per_call"]
        )
        wall_speedup = (
            generic_timing["wall_us_per_call"]
            / candidate_timing["wall_us_per_call"]
        )
        if event_speedup <= 1.0 or wall_speedup <= 1.0:
            raise RuntimeError(
                f"M={n_tokens} candidate did not beat generic baseline: "
                f"event={event_speedup}, wall={wall_speedup}"
            )
        results[str(n_tokens)] = {
            "accuracy_probe_generic_max_abs_error": generic_max_abs,
            "accuracy_probe_candidate_max_abs_error": candidate_max_abs,
            "generic_allocation_inclusive": generic_timing,
            "multi_m_asymmetric_out_v1": candidate_timing,
            "event_speedup": event_speedup,
            "wall_speedup": wall_speedup,
        }

    report = {
        "dso": str(args.dso),
        "dso_sha256": _sha256(args.dso),
        "shape": {
            "tokens": list(BENCHMARK_TOKEN_COUNTS),
            "hidden": HIDDEN_SIZE,
            "experts": NUM_EXPERTS,
            "top_k": TOP_K,
            "routed_physical": ROUTED_SIZE,
            "shared_physical": SHARED_SIZE,
        },
        "dtype": "float16 activation/scales/shared, signed-s4 uint8 routed",
        "warmup": {
            "phases_per_implementation_and_shape": 1,
            "iterations": args.warmup_iterations,
        },
        "formal": {
            "phases_per_implementation_and_shape": 1,
            "iterations": args.formal_iterations,
            "results": results,
        },
        "baseline_scope": (
            "Existing allocation-inclusive generic asymmetric full operator "
            "versus the same-DSO fixed E512/K10/P128/S80 WS kernels with "
            "per-XPU-stream scratch and caller-owned output; this is not a "
            "TP8 production-server fallback or end-to-end measurement"
        ),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
