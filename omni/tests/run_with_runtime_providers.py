"""Activate installed providers before pytest can import Torch on Linux.

Use the standard entrypoint's verified AIMDO preload, then pass pytest's args
unchanged. This preserves the ordinary allocator takeover ordering.
"""
import importlib.util
from pathlib import Path
import sys

from comfy_aimdo import control  # Official startup entry; no Torch import.

path = Path(__file__).parents[1] / "ComfyUI-OmniXPU" / "runtime_bootstrap.py"
name = "_comfyui_omnixpu_runtime_bootstrap"
spec = importlib.util.spec_from_file_location(name, path)
runtime = importlib.util.module_from_spec(spec)
sys.modules[name] = runtime
spec.loader.exec_module(runtime)
state = runtime.bootstrap(dynamic_vram_override=True)
assert all(state["providers"][name]["status"] == "active"
           for name in ("comfy_kitchen.xpu", "comfy_aimdo.xpu")), state

import pytest

raise SystemExit(pytest.main(sys.argv[1:]))
