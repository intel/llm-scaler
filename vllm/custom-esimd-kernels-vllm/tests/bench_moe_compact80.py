import argparse
import json
import statistics
from pathlib import Path

import torch
from test_moe_multi_m_asymmetric_v1_xpu import build_multi_inputs, make_logits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dso", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--real-weights", action="store_true")
    p.add_argument("--input-multiplier", type=float, default=1.0)
    p.add_argument("--all-m-sizes", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.ops.load_library(args.dso)
    data = build_multi_inputs()
    data.x = (data.x * args.input_multiplier).contiguous()
    b = data.base
    if args.real_weights:
        from compact80_real_weights import populate_real_experts

        populate_real_experts(b)
    cw13 = torch.cat(
        (b.w13_qweight_s4[:, :80], b.w13_qweight_s4[:, 128:208]), 1
    ).contiguous()
    cs13 = torch.cat((b.w13_scales[:, :80], b.w13_scales[:, 128:208]), 1).contiguous()
    cw2 = b.w2_qweight_s4[..., :40].contiguous()
    result = {}
    for m in range(1, 33) if args.all_m_sizes else (1, 2, 4, 8, 16, 32):
        x = data.x[:m]
        logits = make_logits(m).to("xpu")
        oldout = torch.empty_like(x)
        newout = torch.empty_like(x)
        mode = "m1" if m == 1 else "multi_m"
        old = getattr(
            torch.ops.moe_int4_ops,
            f"moe_forward_{mode}_cutlass_nmajor_int4_fp16_shared_asymmetric_out_v1",
        )
        new = getattr(
            torch.ops.moe_int4_ops,
            f"moe_forward_{mode}_cutlass_nmajor_int4_fp16_shared_compact80_out_v1",
        )
        common = [
            b.shared_gate_up_weight,
            b.shared_down_weight,
            b.shared_expert_gate_weight,
        ]
        oa = [
            x,
            logits,
            b.w13_qweight_s4,
            b.w13_scales,
            b.w2_qweight_s4,
            b.w2_scales,
            *common,
            oldout,
            10,
            1,
            512,
        ]
        na = [x, logits, cw13, cs13, cw2, b.w2_scales, *common, newout, 10, 1, 512]
        old(*oa)
        new(*na)
        torch.xpu.synchronize()
        diff = (oldout.float() - newout.float()).abs()
        equal = torch.equal(oldout, newout)
        item = {"equal": equal, "max_abs": diff.max().item()}
        print(m, item, flush=True)
        assert equal, item
        # Same-stream async reuse; do not host-sync between producer and consumer.
        stream = torch.xpu.Stream()
        stream.wait_stream(torch.xpu.current_stream())
        with torch.xpu.stream(stream):
            for _ in range(20):
                new(*na)
        torch.xpu.current_stream().wait_stream(stream)
        torch.xpu.synchronize()
        assert torch.equal(oldout, newout)
        for op, a in ((old, oa), (new, na)):
            for _ in range(10):
                op(*a)
        torch.xpu.synchronize()
        times = {"old": [], "compact": []}
        for rnd in range(6):
            seq = ((old, oa, "old"), (new, na, "compact"))
            if rnd % 2:
                seq = seq[::-1]
            for op, a, key in seq:
                start = torch.xpu.Event(enable_timing=True)
                end = torch.xpu.Event(enable_timing=True)
                start.record()
                for _ in range(50):
                    op(*a)
                end.record()
                end.synchronize()
                times[key].append(start.elapsed_time(end) * 1000 / 50)
        item.update({k + "_us": statistics.median(v) for k, v in times.items()})
        item["ratio"] = item["compact_us"] / item["old_us"]
        item["rounds"] = times
        result[m] = item
        print(m, item, flush=True)
    Path(args.output).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
