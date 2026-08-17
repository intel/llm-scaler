"""Installed-ComfyUI tests for bounded large-video preprocessing."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


_ADAPTER = (
    Path(__file__).parents[1]
    / "ComfyUI-OmniXPU"
    / "adapters"
    / "large_video_preprocess.py"
)


def _load_adapter():
    name = "omnixpu_large_video_preprocess_test"
    spec = importlib.util.spec_from_file_location(name, _ADAPTER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _reference_seedvr_pad(images):
    if images.shape[-1] > 3:
        images = images[..., :3]
    if images.dim() == 4:
        images = images.unsqueeze(0)
    images = images.permute(0, 1, 4, 2, 3)
    batch, frames, channels, height, width = images.shape
    images = images.reshape(batch * frames, channels, height, width)
    images = images.clamp(0.0, 1.0)
    pad_height = (16 - height % 16) % 16
    pad_width = (16 - width % 16) % 16
    images = torch.nn.functional.pad(images, (0, pad_width, 0, pad_height))
    images = images.reshape(
        batch, frames, channels, height + pad_height, width + pad_width
    )
    if frames > 1 and not (frames > 4 and (frames - 1) % 4 == 0):
        padded_frames = 5 if frames <= 4 else frames + 4 - ((frames - 1) % 4)
        padding = images[:, -1:].repeat(1, padded_frames - frames, 1, 1, 1)
        images = torch.cat([images, padding], dim=1)
    return images.permute(0, 1, 3, 4, 2).contiguous()


@pytest.mark.parametrize("channels", [1, 3])
def test_bounded_lanczos_matches_installed_upstream(channels):
    comfy_utils = pytest.importorskip("comfy.utils")
    adapter = _load_adapter()
    generator = torch.Generator(device="cpu").manual_seed(71 + channels)
    samples = torch.randn(3, channels, 7, 9, generator=generator) * 0.7 + 0.5

    expected = comfy_utils.lanczos(samples, 13, 11)
    actual = adapter._bounded_lanczos(comfy_utils, samples, 13, 11)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 7, 9, 3),
        (2, 7, 9, 4),
        (4, 7, 9, 3),
        (5, 7, 9, 3),
        (6, 7, 9, 3),
        (2, 6, 7, 9, 3),
    ],
)
def test_bounded_seedvr_pad_matches_upstream_layout(shape, monkeypatch):
    adapter = _load_adapter()
    monkeypatch.setattr(adapter, "_COPY_GROUP_BYTES", 7 * 9 * 3 * 4)
    generator = torch.Generator(device="cpu").manual_seed(sum(shape))
    images = torch.randn(shape, generator=generator) * 0.7 + 0.5

    expected = _reference_seedvr_pad(images)
    actual = adapter._bounded_seedvr_pad_tensor(images)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_routes_use_output_bytes_not_a_model_name():
    adapter = _load_adapter()
    samples = torch.empty(124, 3, 1088, 1920, device="meta")
    images = torch.empty(124, 2176, 3840, 3, device="meta")

    assert adapter._lanczos_output_bytes(samples, 3840, 2176) == 12_433_489_920
    assert adapter._seedvr_output_bytes(images) == 12_533_760_000
    assert adapter._seedvr_output_shape(images) == (1, 125, 2176, 3840, 3)


def test_bounded_vae_stage_matches_whole_input_conversion(monkeypatch):
    adapter = _load_adapter()
    monkeypatch.setattr(adapter, "_COPY_GROUP_BYTES", 1)

    class FirstStage:
        def __init__(self):
            self.kwargs = None

        def encode_tiled(self, value, **kwargs):
            self.kwargs = kwargs
            return value.square()

    class VAE:
        device = torch.device("cpu")
        output_device = torch.device("cpu")
        vae_dtype = torch.float16

        def __init__(self):
            self.first_stage_model = FirstStage()

        @staticmethod
        def process_input(value):
            return value * 2.0 - 1.0

        @staticmethod
        def vae_output_dtype():
            return torch.float32

    generator = torch.Generator(device="cpu").manual_seed(83)
    pixels = torch.rand(1, 3, 5, 7, 9, generator=generator).movedim(-1, -2)
    vae = VAE()
    kwargs = {"tile_x": 4, "overlap": 1}
    expected = vae.first_stage_model.encode_tiled(
        vae.process_input(pixels).to(vae.vae_dtype).to(vae.device), **kwargs
    ).to(device=vae.output_device, dtype=vae.vae_output_dtype())

    actual = adapter._bounded_vae_stage(vae, pixels, kwargs)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not pixels.is_contiguous()
    assert vae.first_stage_model.kwargs == kwargs


def test_vae_stage_route_is_structural(monkeypatch):
    adapter = _load_adapter()
    monkeypatch.setattr(adapter, "_VAE_STAGING_THRESHOLD_BYTES", 1024)

    class VAE:
        device = torch.device("xpu")

    large = torch.empty(1, 3, 5, 16, 16)
    small = torch.empty(1, 3, 1, 4, 4)

    assert adapter._should_bound_vae_stage(VAE(), large)
    assert not adapter._should_bound_vae_stage(VAE(), small)


def test_apply_wraps_the_registered_v3_node_contract_once(monkeypatch):
    comfy_utils = pytest.importorskip("comfy.utils")
    cli_args = pytest.importorskip("comfy.cli_args")
    cli_args.args.cpu = True
    seedvr_nodes = pytest.importorskip("comfy_extras.nodes_seedvr")
    registered_nodes = types.ModuleType("nodes")
    registered_nodes.NODE_CLASS_MAPPINGS = {
        "SeedVR2Preprocess": seedvr_nodes.SeedVR2Preprocess
    }
    monkeypatch.setitem(sys.modules, "nodes", registered_nodes)
    adapter = _load_adapter()

    assert adapter.apply() == (True, None)
    assert getattr(comfy_utils.lanczos, adapter._PATCH_MARKER)
    comfy_sd = pytest.importorskip("comfy.sd")
    assert getattr(comfy_sd.VAE._encode_tiled_owned, adapter._PATCH_MARKER)
    execute = seedvr_nodes.SeedVR2Preprocess.__dict__["execute"]
    assert getattr(execute.__func__, adapter._PATCH_MARKER)
    assert adapter.apply() == (
        False,
        "large-video preprocessing adapter is already applied",
    )
