"""Portable contracts for the physical-B580 H3 RMS modulation adapter."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch


_PLUGIN = Path(__file__).parents[1] / "ComfyUI-OmniXPU"
_PATCHES = _PLUGIN / "patches"
_ADAPTERS = _PLUGIN / "adapters"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_adapter(monkeypatch):
    package_name = "omnixpu_h3_rms_modulation_test"
    package = types.ModuleType(package_name)
    package.__path__ = [str(_PLUGIN)]
    patches = types.ModuleType(f"{package_name}.patches")
    patches.__path__ = [str(_PATCHES)]
    adapters = types.ModuleType(f"{package_name}.adapters")
    adapters.__path__ = [str(_ADAPTERS)]
    monkeypatch.setitem(sys.modules, package_name, package)
    monkeypatch.setitem(sys.modules, patches.__name__, patches)
    monkeypatch.setitem(sys.modules, adapters.__name__, adapters)
    _load_module(f"{patches.__name__}.debug", _PATCHES / "debug.py")
    return _load_module(
        f"{adapters.__name__}.h3_rms_modulation",
        _ADAPTERS / "h3_rms_modulation.py",
    )


def _packed_modulation():
    packed = torch.randn(3, 6 * 5376, dtype=torch.bfloat16)
    return packed.chunk(6, dim=-1)


def test_segment_contract_is_structural_not_sequence_specific(monkeypatch):
    adapter = _load_adapter(monkeypatch)

    assert adapter._segment_reason(
        [(0, 388, 0), (388, 802, 1), (802, 15787, 2)],
        15787,
        3,
    ) == ""
    assert adapter._segment_reason(
        [(0, 388, 0), (388, 802, 1), (802, 44929, 2)],
        44929,
        3,
    ) == ""
    assert adapter._segment_reason([(0, 17, torch.tensor(0))], 17, 3) == (
        "segment_value_type"
    )
    assert adapter._segment_reason([(1, 17, 0)], 17, 3) == (
        "segment_coverage"
    )
    assert adapter._segment_reason([(0, 17, 3)], 17, 3) == (
        "segment_modulation_row"
    )


def test_modulation_requires_the_real_packed_six_chunk_layout(monkeypatch):
    adapter = _load_adapter(monkeypatch)
    x = torch.randn(17, 5376, dtype=torch.bfloat16)
    segments = [(0, 3, 0), (3, 8, 1), (8, 17, 2)]
    values = _packed_modulation()

    assert values[0].stride() == (6 * 5376, 1)
    assert adapter._modulation_reason(values, x, segments) == ""
    contiguous = tuple(value.contiguous() for value in values)
    assert adapter._modulation_reason(contiguous, x, segments) == (
        "modulation_layout"
    )


def test_adapter_is_source_guarded_and_registered_after_generic_norm(
    monkeypatch,
):
    adapter = _load_adapter(monkeypatch)
    source = (_ADAPTERS / "h3_rms_modulation.py").read_text(encoding="utf-8")
    registry = (_PATCHES / "__init__.py").read_text(encoding="utf-8")

    assert len(adapter._EXPECTED_FORWARD_SHA256) == 64
    assert "inspect.getsource(original)" in source
    assert "physical_bmg_sku" in source
    assert 'observed.get("sku_forced") is not False' in source
    assert registry.index('"norm_adapter"') < registry.index(
        '"h3_rms_modulation_adapter"'
    )


def test_adapter_has_strict_training_weight_and_failure_boundaries(monkeypatch):
    adapter = _load_adapter(monkeypatch)
    source = (_ADAPTERS / "h3_rms_modulation.py").read_text(encoding="utf-8")

    assert adapter._MAX_SEGMENTS == 8
    assert "comfy.model_management.in_training" in source
    assert "CastBiasWeightContext" in source
    assert "run_every_op()" in source
    assert "contract changed after routing" in source
    assert "sequence ==" not in source
    assert "sequence_length ==" not in source
