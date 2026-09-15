import argparse
import json
import statistics
from pathlib import Path

import torch
from test_moe_multi_m_asymmetric_v1_xpu import build_multi_inputs
from vllm.model_executor.layers.quantization._qwen38_compact_moe import (
    guard_compact80_scales,
    make_xpu_fused_moe,
)
from vllm_xpu_kernels.fused_moe_interface import XpuFusedMoe


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dso", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--real-weights", action="store_true")
    p.add_argument("--uniform-routing", action="store_true")
    p.add_argument("--input-multiplier", type=float, default=1.0)
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.ops.load_library(args.dso)
    b = build_multi_inputs().base
    if args.real_weights:
        from compact80_real_weights import populate_real_experts

        populate_real_experts(b)
    # Match production's zero gate/up padding as well as zero down padding.
    b.w13_qweight_s4[:, 80:128] = 0
    b.w13_qweight_s4[:, 208:] = 0
    b.w13_scales[:, 80:128] = 0
    b.w13_scales[:, 208:] = 0
    raw13 = b.w13_qweight_s4 ^ 0x88
    raw2 = b.w2_qweight_s4 ^ 0x88
    compact13 = torch.cat((raw13[:, :80], raw13[:, 128:208]), 1).contiguous()
    compact13._qwen38_compact80 = True
    compact2 = raw2[..., :40].contiguous()
    cs13 = torch.cat((b.w13_scales[:, :80], b.w13_scales[:, 128:208]), 1).contiguous()
    cs13 = guard_compact80_scales(cs13)
    common = {
        "w13_bias": None,
        "w2_bias": None,
        "w2_scales": b.w2_scales,
        "n_experts_per_token": 10,
        "activation": "silu",
        "num_experts": 512,
    }
    old = XpuFusedMoe(w13=raw13, w13_scales=b.w13_scales, w2=raw2, **common)
    new = make_xpu_fused_moe(w13=compact13, w13_scales=cs13, w2=compact2, **common)
    pointers = (new.w13.data_ptr(), new.w2.data_ptr(), cs13.data_ptr())
    second = make_xpu_fused_moe(w13=compact13, w13_scales=cs13, w2=compact2, **common)
    assert second.recipe == "int4"
    assert pointers == (second.w13.data_ptr(), second.w2.data_ptr(), cs13.data_ptr()), (
        "Creating a second consumer must not convert or copy canonical weights"
    )
    results = {}
    generator = torch.Generator().manual_seed(1987)
    for m in (1, 33, 64, 128, 129, 1024, 2048, 4096):
        x = (
            (torch.randn(m, 2560, generator=generator) * 0.05 * args.input_multiplier)
            .half()
            .to("xpu")
        )
        # Use selected experts actually populated by the shared test asset.
        ids = (
            torch.tensor([32, 63, 64, 127, 128, 255, 256, 383, 510, 511], device="xpu")
            .expand(m, 10)
            .contiguous()
        )
        if args.uniform_routing:
            ids = (
                torch.rand(m, 512, generator=generator)
                .topk(10, dim=1)
                .indices.to("xpu")
            )
        weights = torch.softmax(torch.randn(m, 10, generator=generator), dim=1).to(
            "xpu"
        )
        a = torch.empty_like(x)
        b_out = torch.empty_like(x)

        def baseline(a=a, x=x, weights=weights, ids=ids):
            old.apply(a, x, weights, ids)

        def candidate(b_out=b_out, x=x, weights=weights, ids=ids):
            new.apply(b_out, x, weights, ids)

        baseline()
        candidate()
        torch.xpu.synchronize()
        equal = torch.equal(a, b_out)
        result = {
            "equal": equal,
            "max_abs": (a.float() - b_out.float()).abs().max().item(),
        }
        print(m, result, flush=True)
        assert equal, result
        for _ in range(3):
            baseline()
            candidate()
        torch.xpu.synchronize()
        times = {"old": [], "compact": []}
        for rnd in range(5):
            seq = ((baseline, "old"), (candidate, "compact"))
            if rnd % 2:
                seq = seq[::-1]
            for fn, key in seq:
                s = torch.xpu.Event(enable_timing=True)
                e = torch.xpu.Event(enable_timing=True)
                s.record()
                for _ in range(10):
                    fn()
                e.record()
                e.synchronize()
                times[key].append(s.elapsed_time(e) * 1000 / 10)
        result.update({k + "_us": statistics.median(v) for k, v in times.items()})
        result["ratio"] = result["compact_us"] / result["old_us"]
        result["rounds"] = times
        results[m] = result
        print(m, result, flush=True)
    Path(args.output).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
