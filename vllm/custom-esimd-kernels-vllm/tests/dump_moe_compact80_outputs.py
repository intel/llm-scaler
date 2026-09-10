"""Compare installed-vs-build-only MoE outputs in separate processes."""

import argparse

import torch
from compact80_real_weights import populate_real_experts
from test_moe_multi_m_asymmetric_v1_xpu import build_multi_inputs, make_logits

p = argparse.ArgumentParser()
p.add_argument("--dso", required=True)
p.add_argument("--output", required=True)
p.add_argument("--compact", action="store_true")
a = p.parse_args()
torch.set_num_threads(4)
torch.ops.load_library(a.dso)
data = build_multi_inputs()
b = data.base
populate_real_experts(b)
w13, s13, w2 = b.w13_qweight_s4, b.w13_scales, b.w2_qweight_s4
if a.compact:
    w13 = torch.cat((w13[:, :80], w13[:, 128:208]), 1).contiguous()
    s13 = torch.cat((s13[:, :80], s13[:, 128:208]), 1).contiguous()
    w2 = w2[..., :40].contiguous()
out = {}
for multiplier in (1, 20, 60):
    for m in range(1, 33):
        x = (data.x[:m] * multiplier).contiguous()
        logits = make_logits(m).to("xpu")
        y = torch.empty_like(x)
        mode = "m1" if m == 1 else "multi_m"
        variant = "compact80" if a.compact else "asymmetric"
        op = getattr(
            torch.ops.moe_int4_ops,
            f"moe_forward_{mode}_cutlass_nmajor_int4_fp16_shared_{variant}_out_v1",
        )
        op(
            x,
            logits,
            w13,
            s13,
            w2,
            b.w2_scales,
            b.shared_gate_up_weight,
            b.shared_down_weight,
            b.shared_expert_gate_weight,
            y,
            10,
            1,
            512,
        )
        out[(multiplier, m)] = y.cpu()
torch.save(out, a.output)
print("Saved", len(out), "outputs", flush=True)
