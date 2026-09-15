"""Real checkpoint slices, quantized by the unchanged vLLM group128 code."""

import json
from pathlib import Path

import torch
from safetensors import safe_open


def populate_real_experts(base):
    from vllm.model_executor.layers.quantization.sym_int4 import (
        _quantize_moe_w2,
        _quantize_moe_w13,
    )

    root = Path("/llm/models/Qwen3.8-Flash-Next")
    index = json.loads((root / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]

    def read(key, expert, rows, cols):
        with safe_open(str(root / index[key]), framework="pt", device="cpu") as f:
            return f.get_slice(key)[expert : expert + 1, rows, cols].half()

    quantized = []
    for layer, expert, rank in ((0, 0, 0), (23, 255, 7), (47, 511, 3)):
        prefix = f"model.language_model.layers.{layer}.mlp.experts."
        a = rank * 80
        w13 = torch.cat(
            (
                read(prefix + "gate_up_proj", expert, slice(a, a + 80), slice(None)),
                read(
                    prefix + "gate_up_proj",
                    expert,
                    slice(640 + a, 720 + a),
                    slice(None),
                ),
            ),
            dim=1,
        )
        w2 = read(prefix + "down_proj", expert, slice(None), slice(a, a + 80))
        q13, s13, _ = _quantize_moe_w13(w13)
        q2, s2, _ = _quantize_moe_w2(w2)
        quantized.append((q13 ^ 0x88, s13, q2 ^ 0x88, s2))
    slots = (torch.arange(512, device="xpu") % 3).long()
    for i, target in enumerate(
        (base.w13_qweight_s4, base.w13_scales, base.w2_qweight_s4, base.w2_scales)
    ):
        values = torch.cat([entry[i] for entry in quantized]).to("xpu")
        target.copy_(values.index_select(0, slots))
    torch.xpu.synchronize()
    print(
        "Real weights: 3 checkpoint layer/expert/TP slices repeated across 512 slots",
        flush=True,
    )
