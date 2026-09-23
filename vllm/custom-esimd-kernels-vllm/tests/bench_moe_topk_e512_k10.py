import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

NUM_EXPERTS = 512
TOP_K = 10


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dso", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup-iterations", type=int, default=100)
    parser.add_argument("--formal-iterations", type=int, default=2000)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _call(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.moe_int4_ops.moe_topk_int4(
        logits, TOP_K, NUM_EXPERTS, True
    )


def main() -> None:
    args = _parse_args()
    if args.warmup_iterations <= 0 or args.formal_iterations <= 0:
        raise ValueError("iteration counts must be positive")
    torch.ops.load_library(str(args.dso))

    generator = torch.Generator().manual_seed(20260901)
    logits = torch.randn(1, NUM_EXPERTS, generator=generator).half().to("xpu")

    # One warmup phase.
    for _ in range(args.warmup_iterations):
        result = _call(logits)
    torch.xpu.synchronize()

    # One formal phase. Event time measures the submitted XPU work; wall time
    # also includes dispatcher and caller-owned output allocation overhead.
    start_event = torch.xpu.Event(enable_timing=True)
    end_event = torch.xpu.Event(enable_timing=True)
    wall_start_ns = time.perf_counter_ns()
    start_event.record()
    for _ in range(args.formal_iterations):
        result = _call(logits)
    end_event.record()
    torch.xpu.synchronize()
    wall_end_ns = time.perf_counter_ns()

    report = {
        "label": args.label,
        "dso": str(args.dso),
        "dso_sha256": _sha256(args.dso),
        "shape": [1, NUM_EXPERTS],
        "dtype": "float16",
        "top_k": TOP_K,
        "warmup": {
            "phases": 1,
            "iterations": args.warmup_iterations,
        },
        "formal": {
            "phases": 1,
            "iterations": args.formal_iterations,
            "event_us_per_call": (
                start_event.elapsed_time(end_event)
                * 1000.0
                / args.formal_iterations
            ),
            "wall_us_per_call": (
                (wall_end_ns - wall_start_ns)
                / 1000.0
                / args.formal_iterations
            ),
        },
        "last_output_shapes": [list(tensor.shape) for tensor in result],
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
