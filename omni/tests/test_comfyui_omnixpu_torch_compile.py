"""The real native ComfyUI compile node around real attention and INT8 FFN APIs.

The small module fixtures exercise interfaces, not workflow performance. Run in
an admitted XPU container with installed ComfyUI and matching Omni native DSOs.
A child process keeps ComfyUI/provider initialization out of other unit tests.
"""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


COMFYUI_ROOT = Path(os.environ.get("COMFYUI_ROOT", "/llm/ComfyUI"))
OMNI_ROOT = Path(__file__).resolve().parents[1]


def test_native_compile_node_keeps_attention_and_ffn_graphs_stable(tmp_path):
    import torch

    if not COMFYUI_ROOT.joinpath("comfy_extras/nodes_torch_compile.py").is_file():
        pytest.skip("real ComfyUI native compile node required")
    if not torch.xpu.is_available():
        pytest.skip("installed XPU runtime required")
    output = tmp_path / "native-compile.json"
    completed = subprocess.run(
        [sys.executable, "-B", str(Path(__file__).resolve()), "--probe", str(output)],
        capture_output=True, text=True, timeout=300,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(output.read_text())
    for case in result.values():
        assert case["unique_graphs"] == [1] * 12
        assert case["graph_breaks"] == {}
        assert case["outputs_equal"] is True
        assert case["compiled_counter_delta"] == 0


def _probe(output):
    import importlib
    import importlib.util
    import runpy
    import types

    sys.path.insert(0, str(COMFYUI_ROOT))
    runpy.run_path(str(OMNI_ROOT / "ComfyUI-OmniXPU/prestartup_script.py"))
    # Test candidate Python against the installed native DSOs without copying or
    # rebuilding a wheel. This is explicitly source-overlay interface evidence.
    installed = Path(importlib.util.find_spec("omni_xpu_kernel").origin).parent
    extensions = sorted((installed / "cute").glob("cute_fmha_torch*.so"))
    extensions += sorted((installed / "cute").glob("cute_fmha_torch*.pyd"))
    if not extensions:
        raise RuntimeError("installed CUTE extension required")
    os.environ["OMNI_CUTE_FMHA_SO"] = str(extensions[0])
    sys.path.insert(0, str(OMNI_ROOT / "omni_xpu_kernel"))
    import torch
    import omni_xpu_kernel as omni

    omni.__path__.append(str(installed))
    omni._load_extension()
    package = types.ModuleType("omnixpu_compile_test")
    package.__path__ = [str(OMNI_ROOT / "ComfyUI-OmniXPU")]
    sys.modules[package.__name__] = package
    attention = importlib.import_module(package.__name__ + ".adapters.attention")
    ffn_adapter = importlib.import_module(package.__name__ + ".adapters.int8_ffn")
    import comfy.ops
    import comfy.ldm.modules.attention as comfy_attention
    from comfy.ldm.lumina.model import FeedForward
    from comfy.model_patcher import ModelPatcher
    from comfy.patcher_extension import WrapperExecutor, WrappersMP
    from comfy_extras.nodes_torch_compile import TorchCompileModel
    from comfy_kitchen.tensor import QuantizedTensor
    from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout
    from torch._dynamo.utils import counters

    assert attention.apply()[0]
    assert ffn_adapter.apply()[0]
    torch.manual_seed(123)

    class Attention(torch.nn.Module):
        def forward(self, x):
            return comfy_attention.optimized_attention(
                x, x, x, 30, skip_reshape=True, transformer_options={}
            )

    ops = comfy.ops.mixed_precision_ops(compute_dtype=torch.bfloat16)
    ffn = FeedForward(128, 256, 64, None, operation_settings={
        "operations": ops, "device": torch.device("xpu"), "dtype": torch.bfloat16,
    })
    for name in ("w1", "w2", "w3"):
        layer = getattr(ffn, name)
        shape = (layer.out_features, layer.in_features)
        qdata = torch.randint(-127, 128, shape, device="xpu", dtype=torch.int8)
        params = TensorWiseINT8Layout.Params(
            scale=torch.full((shape[0], 1), 0.001, device="xpu"),
            orig_dtype=torch.bfloat16, orig_shape=shape,
            convrot=True, convrot_groupsize=16,
        )
        layer.weight = torch.nn.Parameter(
            QuantizedTensor(qdata, "TensorWiseINT8Layout", params), requires_grad=False,
        )
        layer.quant_format = "int8_tensorwise"
        layer.layout_type = "TensorWiseINT8Layout"

    class Model(torch.nn.Module):
        def __init__(self, diffusion):
            super().__init__()
            self.diffusion_model = diffusion

        def apply_model(self, x):
            return self.diffusion_model(x)

    results = {}
    fixtures = (
        ("attention", Attention(), torch.randn(1, 32, 30, 128, device="xpu",
                                              dtype=torch.bfloat16).transpose(1, 2)),
        ("ffn", ffn, torch.randn(1, 32, 128, device="xpu", dtype=torch.bfloat16)),
    )
    with torch.inference_mode():
        for name, module, input in fixtures:
            torch._dynamo.reset()
            counters.clear()
            parent = Model(module)
            patcher = ModelPatcher(parent, torch.device("xpu"), torch.device("cpu"))
            compiled_patcher = TorchCompileModel.execute(patcher, "inductor")[0]
            config = compiled_patcher.model_options["torch_compile_kwargs"]
            assert config["backend"] == "inductor" and config["fullgraph"] is False
            assert config["dynamic"] is None and config["mode"] is None
            wrappers = compiled_patcher.get_wrappers(WrappersMP.APPLY_MODEL, "torch.compile")
            executor = WrapperExecutor.new_class_executor(parent.apply_model, parent, wrappers)
            counts = []
            before = attention.get_stats()["cute"] if name == "attention" else ffn_adapter.get_stats()["routed"]
            for index in range(12):
                x = input + index * 0.01
                actual = executor.execute(x)
                # Eager calls deliberately advance the same diagnostic counters
                # between compiled calls, so a guard on those counters fails.
                reference = module(x)
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                counts.append(counters["stats"]["unique_graphs"])
            after = attention.get_stats()["cute"] if name == "attention" else ffn_adapter.get_stats()["routed"]
            results[name] = {"unique_graphs": counts, "graph_breaks": dict(counters["graph_break"]),
                             "outputs_equal": True, "compiled_counter_delta": after - before - 12}
    output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    output = Path(sys.argv[2])
    sys.argv = [sys.argv[0]]
    _probe(output)
