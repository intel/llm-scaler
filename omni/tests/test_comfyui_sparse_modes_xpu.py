"""Real ComfyUI sparse-node integration on XPU, with bounded weight-free fixtures.

These fixtures validate node/kernel composition, not a reduced H3 workflow or
trained SLA/FastH3 output quality. Full workflow admission lives in the tuning
repository and requires the matching owner-supplied weights.
"""
import asyncio
import importlib.util
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture(scope="module")
def runtime():
    assert sys.executable == "/opt/venv/bin/python"
    assert torch.xpu.device_count() == 1
    assert torch.xpu.get_device_properties(0).device_id == 0xE223
    sys.path.insert(0, "/llm/ComfyUI")
    runpy.run_path("/llm/ComfyUI/custom_nodes/ComfyUI-OmniXPU/prestartup_script.py")
    import nodes
    import comfy_kitchen as ck
    assert asyncio.run(nodes.load_custom_node(
        "/llm/ComfyUI/comfy_extras/nodes_sparse_attention.py", module_parent="comfy_extras"))
    cls = nodes.NODE_CLASS_MAPPINGS["BlockSparseAttention"]
    node = sys.modules[cls.__module__]
    path = Path("/llm/ComfyUI/custom_nodes/ComfyUI-OmniXPU/__init__.py")
    spec = importlib.util.spec_from_file_location("ComfyUI-OmniXPU", path,
                                                submodule_search_locations=[str(path.parent)])
    omni = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = omni
    spec.loader.exec_module(omni)
    assert "OmniXPUStatus" in omni.NODE_CLASS_MAPPINGS
    assert hasattr(node.h3_eligible, "__omnixpu_sparse_eligibility_original__")
    assert ck.sol_attn_is_available(torch.device("xpu:0"))
    assert "comfy_kitchen_xpu_runtime/_vendor" in ck.__file__
    return ck, node


def new_patch(node, mode, ratio=0.1, extra_tokens=0):
    return node.SparseAttnPatch(tau=1.3, topk_ratio=ratio, vsa=mode == "vsa",
        sigma_start=1.0, sigma_end=0.0, min_tokens=0, dense_blocks=set(),
        sink_conditioning="exact_kv_and_rows", extra_tokens=extra_tokens, verbose=False)


def reference_plan(layout):
    # Integer coordinate enumeration independent of the tensor reshape/permute
    # implementation. All subsequent numerical reference operations run on XPU.
    _, nt, nh, nw, _ = layout.signature
    nh, nw = nh // 2, nw // 2
    rows, lengths, prefix = [], [], 0
    for start, stop, kind in layout.segments:
        if kind != "video":
            tiles = [list(range(a, min(a + 64, stop))) for a in range(start, stop, 64)]
            prefix += len(tiles)
        else:
            tiles = []
            for t in range(0, nt, 4):
                for h in range(0, nh, 4):
                    for w in range(0, nw, 4):
                        tiles.append([start + (tt * nh + hh) * nw + ww
                            for tt in range(t, min(t + 4, nt))
                            for hh in range(h, min(h + 4, nh))
                            for ww in range(w, min(w + 4, nw))])
        for tile in tiles:
            lengths.append(len(tile))
            rows.extend(tile + [-1] * (64 - len(tile)))
    src = torch.tensor(rows, device="xpu", dtype=torch.int64)
    inv = torch.empty(layout.seq_len, device="xpu", dtype=torch.int64)
    for padded, original in enumerate(rows):
        if original >= 0:
            inv[original] = padded
    return src, inv, torch.tensor(lengths, device="xpu", dtype=torch.int32), prefix


@pytest.mark.parametrize("text,height", [(5, 10), (67, 14)])
def test_vsa_real_h3_layout_padding_inverse_rope_and_cache(runtime, text, height):
    _, node = runtime
    from comfy.ldm.minimax.model import PackedLayout
    layout = PackedLayout(text, 5, height, 10, 3)
    patch = new_patch(node, "vsa")
    plan = patch.vsa_plan(layout, torch.device("xpu:0"))
    src, inv, lengths, prefix = reference_plan(layout)
    for name, expected in (("src", src), ("inv", inv), ("block_len", lengths)):
        torch.testing.assert_close(plan[name], expected, rtol=0, atol=0)
    assert plan["n_prefix"] == prefix and plan["n"] == len(src)
    assert patch.vsa_plan(layout, torch.device("xpu:0")) is plan
    rope = torch.arange(layout.seq_len * 4, device="xpu", dtype=torch.float32).view(1, -1, 1, 1, 2, 2)
    padded = patch.vsa_rope_freqs(rope, plan)
    torch.testing.assert_close(padded[:, inv], rope, rtol=0, atol=0)
    assert int(torch.count_nonzero(padded[:, src < 0])) == 0
    assert patch.vsa_rope_freqs(rope, plan) is padded
    patch.reset()
    assert not patch.vsa_plans and patch.vsa_rope is None and not patch.pooled


@pytest.mark.parametrize("mode", ["sol-attn", "sla", "vsa"])
def test_registered_lowercase_selection_applies_real_patch_and_cleanup(runtime, mode):
    _, node = runtime
    from comfy.ldm.minimax.model import MiniMaxH3Model
    from comfy.patcher_extension import CallbacksMP
    schema = node.BlockSparseAttention.define_schema()
    options = next(item for item in schema.inputs if item.id == "selection").options
    assert [option.key for option in options] == ["sol-attn", "sla", "vsa"]
    assert [option.inputs[0].default for option in options] == [1.3, 10.0, 10.0]
    model = MiniMaxH3Model.__new__(MiniMaxH3Model)
    torch.nn.Module.__init__(model)
    model.blocks = [SimpleNamespace(attn=SimpleNamespace(to_gate_compress=object()))]

    class Patcher:
        def __init__(self):
            self.model_options = {"transformer_options": {}}
            self.callbacks, self.replacements = {}, []
        def get_model_object(self, name):
            return model if name == "diffusion_model" else SimpleNamespace(percent_to_sigma=lambda x: 1 - x)
        def clone(self):
            return Patcher()
        def add_callback_with_key(self, kind, key, callback):
            self.callbacks[kind] = callback
        def set_model_patch_replace(self, callback, *key):
            self.replacements.append((callback, key))

    original = Patcher()
    result = node.BlockSparseAttention.execute(original,
        {"selection": mode, "tau": 1.3, "keep_percent": 10.0}, 0.2, 1.0,
        min_tokens=0, extra_tokens=256).result[0]
    assert result is not original and len(result.replacements) == 1
    override = result.model_options["transformer_options"]["optimized_attention_override"]
    patch = next(cell.cell_contents for cell in override.__closure__
                 if isinstance(cell.cell_contents, node.SparseAttnPatch))
    assert patch.vsa == (mode == "vsa")
    assert patch.topk_ratio == (0 if mode == "sol-attn" else 0.1)
    assert patch.extra_tokens == (0 if mode == "vsa" else 256)
    patch.pooled["probe"] = object()
    result.callbacks[CallbacksMP.ON_CLEANUP](result)
    assert not patch.pooled
    result.callbacks[CallbacksMP.ON_PREPARE_STATE](result, None, result.model_options)
    assert result.model_options["transformer_options"]["optimized_attention_override"] is override


@pytest.mark.parametrize("mode,extra_tokens", [("sla", 0), ("sla", 256), ("vsa", 0)])
@pytest.mark.parametrize("ratio", [0.1, 0.4])
def test_real_h3_sparse_producer_bootstrap_replay_and_projection(runtime, monkeypatch, mode, extra_tokens, ratio):
    ck, node = runtime
    from comfy.ldm.minimax.model import Attention, PackedLayout
    from comfy_kitchen.backends.eager.sol_attn import sol_attn as eager
    # Reuse the independently expressed FP32 RMS/split-half RoPE oracle.
    test_root = Path(__file__).parents[1] / "omni_xpu_kernel/tests"
    sys.path.insert(0, str(test_root))
    from test_cute_sol_chunked_correctness import norm_rope_reference
    from test_cute_sparse_modes_xpu import close_on_live_rows
    from omni_xpu_kernel.cute import sol_attn_v2
    torch.manual_seed(40419)
    layout = PackedLayout(5, 5, 10, 10, 3)
    n, heads = layout.seq_len, 2
    attn = Attention(256, heads, 128, 1e-6, gate_compress=mode == "vsa",
                     dtype=torch.bfloat16, device="xpu", operations=torch.nn)
    x = torch.randn(n, 256, device="xpu", dtype=torch.bfloat16) * 0.5
    before = x.clone()
    angles = torch.randn(1, n, 1, 48, device="xpu")
    co, si = angles.cos(), angles.sin()
    rope = torch.stack((co, -si, si, co), -1).reshape(1, n, 1, 48, 2, 2).to(torch.bfloat16)
    patch = new_patch(node, mode, ratio, extra_tokens)
    options = {"sigmas": [0.5], "minimax_h3_layout": layout, "uuids": ["fixture"]}
    assert node.h3_eligible(attn, x, rope, options, patch, 0)
    calls = []
    original = ck.sol_attn_chunked

    def observed(*args, **kwargs):
        calls.append({"bootstrap": kwargs["kmean"] is None, **kwargs})
        return original(*args, **kwargs)

    monkeypatch.setattr(ck, "sol_attn_chunked", observed)
    with torch.no_grad():
        actual = [node.h3_sparse_attention(attn, x, rope, options, patch, 0) for _ in range(2)]
        projected_x, freqs = x, rope
        controls = dict(topk_ratio=ratio, token_aug=extra_tokens)
        if mode == "vsa":
            src, inv, lengths, prefix = reference_plan(layout)
            projected_x = x[src.clamp_min(0)] * (src >= 0).unsqueeze(1)
            freqs = rope.new_zeros((1, len(src), *rope.shape[2:]))
            freqs[:, inv] = rope
            controls.update(tail=False, block_len=lengths,
                coarse_gate=attn.to_gate_compress(projected_x).view(1, -1, heads, 128),
                sink_blocks=[0, prefix], sink_q=[0, prefix])
        else:
            sink, sink_q = patch.sinks(options, n)
            controls.update(sink_blocks=list(sink), sink_q=list(sink_q))
        q, k, v = attn.qkv_proj(projected_x).view(1, -1, 3, heads, 128).unbind(2)
        weights = (attn.q_norm.weight, attn.k_norm.weight)
        if extra_tokens:
            # Augmentation is absent from eager math. This separate composition
            # case compares chunked projection against the full native API.
            from omni_xpu_kernel import rotary
            q, k = q.clone(), k.clone()
            rotary.rms_kitchen_rope_split_half_(q, k, freqs, *weights, rot_dim=96)
            expected = sol_attn_v2.sol_attn(q, k, v, **controls)
        else:
            q, k = norm_rope_reference(q, k, freqs, weights, 96)
            expected = eager(q, k, v, **controls)
        expected = expected.view(-1, heads * 128)
        if mode == "vsa":
            expected = expected[inv]
        expected = attn.out_proj(expected)
    for value in actual:
        close_on_live_rows(value.unsqueeze(0), expected.unsqueeze(0), torch.ones(n, device="xpu", dtype=torch.bool))
    assert [call["bootstrap"] for call in calls] == [True, False]
    for call in calls:
        assert call["topk_ratio"] == ratio and call["token_aug"] == extra_tokens
        if mode == "vsa":
            assert call["tail"] is False
            assert call["block_len"].device.type == call["coarse_gate"].device.type == "xpu"
    torch.testing.assert_close(x, before, rtol=0, atol=0)
    patch.reset()
    assert not patch.pooled
    if mode == "vsa":
        assert not node.h3_eligible(attn, x, rope, {"sigmas": [0.5]}, patch, 0)
