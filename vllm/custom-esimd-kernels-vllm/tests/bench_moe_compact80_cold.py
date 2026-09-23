"""Rotating layer weights to distinguish warm microbench from cold decode."""

import argparse
import json
import statistics
from pathlib import Path

import torch
from compact80_real_weights import populate_real_experts
from test_moe_multi_m_asymmetric_v1_xpu import build_multi_inputs, make_logits

p = argparse.ArgumentParser()
p.add_argument("--dso", required=True)
p.add_argument("--output", required=True)
p.add_argument("--layers", type=int, default=16)
a = p.parse_args()
torch.set_num_threads(4)
torch.ops.load_library(a.dso)
data = build_multi_inputs()
b = data.base
populate_real_experts(b)
old_layers = []
new_layers = []
for _ in range(a.layers):
    old_layers.append(
        (
            b.w13_qweight_s4.clone(),
            b.w13_scales.clone(),
            b.w2_qweight_s4.clone(),
            b.w2_scales.clone(),
        )
    )
    new_layers.append(
        (
            torch.cat((b.w13_qweight_s4[:, :80], b.w13_qweight_s4[:, 128:208]), 1),
            torch.cat((b.w13_scales[:, :80], b.w13_scales[:, 128:208]), 1),
            b.w2_qweight_s4[..., :40].contiguous(),
            b.w2_scales.clone(),
        )
    )
torch.xpu.synchronize()
result = {}
for m in (1, 4):
    x = data.x[:m]
    logits = make_logits(m).to("xpu")
    out = torch.empty_like(x)
    mode = "m1" if m == 1 else "multi_m"
    old = getattr(
        torch.ops.moe_int4_ops,
        f"moe_forward_{mode}_cutlass_nmajor_int4_fp16_shared_asymmetric_out_v1",
    )
    new = getattr(
        torch.ops.moe_int4_ops,
        f"moe_forward_{mode}_cutlass_nmajor_int4_fp16_shared_compact80_out_v1",
    )
    common = (
        b.shared_gate_up_weight,
        b.shared_down_weight,
        b.shared_expert_gate_weight,
    )

    def run(op, layers, x=x, logits=logits, common=common, out=out):
        for w13, s13, w2, s2 in layers:
            op(x, logits, w13, s13, w2, s2, *common, out, 10, 1, 512)

    run(old, old_layers)
    run(new, new_layers)
    torch.xpu.synchronize()
    times = {"old": [], "compact": []}
    for rnd in range(6):
        items = [(old, old_layers, "old"), (new, new_layers, "compact")]
        if rnd % 2:
            items.reverse()
        for op, layers, key in items:
            start = torch.xpu.Event(enable_timing=True)
            end = torch.xpu.Event(enable_timing=True)
            start.record()
            for _ in range(5):
                run(op, layers)
            end.record()
            end.synchronize()
            times[key].append(start.elapsed_time(end) * 1000 / (5 * a.layers))
    row = {k: statistics.median(v) for k, v in times.items()}
    row["ratio"] = row["compact"] / row["old"]
    row["rounds"] = times
    result[m] = row
    print(m, row, flush=True)
Path(a.output).write_text(json.dumps(result, indent=2))
