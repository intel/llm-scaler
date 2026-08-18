"""Bound CPU materialization for large video preprocessing operations.

The adapter preserves the upstream PIL Lanczos result and SeedVR tensor
layout.  It changes only allocation lifetime: frames are converted in bounded
groups and written directly into their final output tensors.
"""

from __future__ import annotations

import functools
import inspect
import logging

import torch


log = logging.getLogger("ComfyUI-OmniXPU")


_LANCZOS_THRESHOLD_BYTES = 1024**3
_SEEDVR_PAD_THRESHOLD_BYTES = 1024**3
_VAE_STAGING_THRESHOLD_BYTES = 1024**3
_COPY_GROUP_BYTES = 256 * 1024**2
_PATCH_MARKER = "_omnixpu_large_video_preprocess_patched"
_lanczos_logged = False
_seedvr_pad_logged = False
_vae_stage_logged = False

_LANCZOS_SOURCE_CONTRACT = (
    "Image.fromarray(np.clip(255. * image.cpu().numpy(), 0, 255)",
    "image.resize((width, height), resample=Image.Resampling.LANCZOS)",
    "result = torch.stack(images)",
    "return result.to(samples.device, samples.dtype)",
)
_SEEDVR_PAD_SOURCE_CONTRACT = (
    "images = torch.clamp(images, 0.0, 1.0)",
    "images = div_pad(images, (16, 16))",
    "images = cut_videos(images)",
    "images.permute(0, 1, 3, 4, 2).contiguous()",
)
_SEEDVR_EXECUTE_SOURCE_CONTRACT = (
    '_seedvr2_input_shorter_edge(resized_images, "SeedVR2Preprocess")',
    "return _seedvr2_pad(",
)
_VAE_STAGING_SOURCE_CONTRACT = (
    "x = self.process_input(pixel_samples).to(self.vae_dtype).to(self.device)",
    "self.first_stage_model.encode_tiled(x, **kwargs)",
    "out.to(device=self.output_device, dtype=self.vae_output_dtype())",
)


def _source_has(function, snippets):
    try:
        source = inspect.getsource(function)
    except (OSError, TypeError):
        return False
    return all(snippet in source for snippet in snippets)


def _lanczos_output_bytes(samples, width, height):
    channels = 1
    if samples.ndim == 4 and samples.shape[1] != 1:
        channels = samples.shape[1]
    return int(samples.shape[0]) * int(width) * int(height) * channels * 4


def _resize_lanczos_frame(comfy_utils, image, width, height):
    image = comfy_utils.Image.fromarray(
        comfy_utils.np.clip(
            255.0 * image.cpu().numpy(), 0, 255
        ).astype(comfy_utils.np.uint8)
    )
    image = image.resize(
        (width, height), resample=comfy_utils.Image.Resampling.LANCZOS
    )
    array = comfy_utils.np.array(image).astype(comfy_utils.np.float32) / 255.0
    result = torch.from_numpy(array)
    return result.movedim(-1, 0) if array.ndim == 3 else result


def _bounded_lanczos(comfy_utils, samples, width, height):
    if samples.ndim == 4:
        samples = (
            samples.squeeze(1)
            if samples.shape[1] == 1
            else samples.movedim(1, -1)
        )

    result = None
    for index, image in enumerate(samples):
        frame = _resize_lanczos_frame(comfy_utils, image, width, height)
        if result is None:
            result = frame.new_empty((samples.shape[0], *frame.shape))
        result[index].copy_(frame)

    # The guarded route excludes an empty leading dimension, matching the
    # upstream torch.stack failure for that input.
    return result.to(samples.device, samples.dtype)


def _padded_time(frames):
    if frames == 1 or (frames > 4 and (frames - 1) % 4 == 0):
        return frames
    if frames <= 4:
        return 5
    return frames + 4 - ((frames - 1) % 4)


def _seedvr_output_shape(images):
    if images.dim() == 4:
        batch, frames, height, width = 1, *images.shape[:3]
    elif images.dim() == 5:
        batch, frames, height, width = images.shape[:4]
    else:
        return None
    channels = min(images.shape[-1], 3)
    padded_height = (height + 15) // 16 * 16
    padded_width = (width + 15) // 16 * 16
    return (
        int(batch),
        _padded_time(int(frames)),
        int(padded_height),
        int(padded_width),
        int(channels),
    )


def _seedvr_output_bytes(images):
    shape = _seedvr_output_shape(images)
    if shape is None or images.shape[-1] == 0:
        return 0
    elements = 1
    for dimension in shape:
        elements *= dimension
    return elements * images.element_size()


def _bounded_seedvr_pad_tensor(images):
    if images.shape[-1] > 3:
        images = images[..., :3]
    if images.dim() == 4:
        images = images.unsqueeze(0)
    elif images.dim() != 5:
        raise ValueError(
            "SeedVR2 preprocessing expected a 4-D or 5-D IMAGE tensor, "
            f"got shape {tuple(images.shape)}"
        )

    batch, frames, height, width, channels = images.shape
    if frames < 1:
        raise ValueError("SeedVR2Preprocess expected at least one frame.")
    output_shape = _seedvr_output_shape(images)
    output = images.new_zeros(output_shape)

    frame_bytes = max(
        1, int(height) * int(width) * int(channels) * images.element_size()
    )
    frames_per_group = max(1, _COPY_GROUP_BYTES // frame_bytes)
    for batch_index in range(batch):
        for start in range(0, frames, frames_per_group):
            end = min(frames, start + frames_per_group)
            output[batch_index, start:end, :height, :width].copy_(
                images[batch_index, start:end].clamp(0.0, 1.0)
            )

    padded_frames = output.shape[1]
    if padded_frames > frames:
        output[:, frames:padded_frames].copy_(
            output[:, frames - 1 : frames].expand(
                -1, padded_frames - frames, -1, -1, -1
            )
        )
    return output


def _should_bound_vae_stage(vae, pixel_samples):
    return bool(
        pixel_samples.ndim == 5
        and pixel_samples.shape[2] > 0
        and pixel_samples.device.type == "cpu"
        and getattr(vae.device, "type", None) == "xpu"
        and not pixel_samples.requires_grad
        and pixel_samples.numel() * pixel_samples.element_size()
        > _VAE_STAGING_THRESHOLD_BYTES
    )


def _bounded_vae_stage(vae, pixel_samples, kwargs):
    staged = torch.empty_like(
        pixel_samples,
        device=vae.device,
        dtype=vae.vae_dtype,
        memory_format=torch.preserve_format,
    )
    frames = pixel_samples.shape[2]
    elements_per_frame = pixel_samples.numel() // frames
    live_bytes_per_frame = elements_per_frame * (
        pixel_samples.element_size() + 4 + staged.element_size()
    )
    frames_per_group = max(1, _COPY_GROUP_BYTES // live_bytes_per_frame)
    for start in range(0, frames, frames_per_group):
        length = min(frames_per_group, frames - start)
        source = pixel_samples.narrow(2, start, length)
        processed = vae.process_input(source).to(vae.vae_dtype)
        staged.narrow(2, start, length).copy_(processed)

    output = vae.first_stage_model.encode_tiled(staged, **kwargs)
    return output.to(
        device=vae.output_device,
        dtype=vae.vae_output_dtype(),
    )


def apply():
    try:
        import comfy.sd as comfy_sd
        import comfy.utils as comfy_utils
        import nodes as comfy_nodes
    except ModuleNotFoundError as exc:
        if exc.name in {"comfy", "comfy.sd", "comfy.utils", "nodes"}:
            return False, "required ComfyUI preprocessing modules are unavailable"
        raise

    original_lanczos = getattr(comfy_utils, "lanczos", None)
    vae = getattr(comfy_sd, "VAE", None)
    original_vae_stage = (
        getattr(vae, "_encode_tiled_owned", None) if vae is not None else None
    )
    preprocess = comfy_nodes.NODE_CLASS_MAPPINGS.get("SeedVR2Preprocess")
    execute_descriptor = (
        preprocess.__dict__.get("execute") if preprocess is not None else None
    )
    original_execute = (
        execute_descriptor.__func__
        if isinstance(execute_descriptor, classmethod)
        else None
    )
    if (
        getattr(original_lanczos, _PATCH_MARKER, False)
        or getattr(original_execute, _PATCH_MARKER, False)
        or getattr(original_vae_stage, _PATCH_MARKER, False)
    ):
        return False, "large-video preprocessing adapter is already applied"
    execute_globals = original_execute.__globals__ if original_execute else {}
    original_seedvr_pad = execute_globals.get("_seedvr2_pad")
    input_shorter_edge = execute_globals.get("_seedvr2_input_shorter_edge")
    node_io = (
        original_seedvr_pad.__globals__.get("io")
        if original_seedvr_pad is not None
        else None
    )
    if (
        original_lanczos is None
        or original_execute is None
        or original_vae_stage is None
        or original_seedvr_pad is None
        or input_shorter_edge is None
        or node_io is None
    ):
        return False, "required ComfyUI preprocessing functions are unavailable"
    if not _source_has(original_lanczos, _LANCZOS_SOURCE_CONTRACT):
        return False, "unsupported comfy.utils.lanczos contract"
    if not _source_has(original_seedvr_pad, _SEEDVR_PAD_SOURCE_CONTRACT):
        return False, "unsupported SeedVR2 preprocessing contract"
    if not _source_has(original_execute, _SEEDVR_EXECUTE_SOURCE_CONTRACT):
        return False, "unsupported SeedVR2Preprocess.execute contract"
    if not _source_has(original_vae_stage, _VAE_STAGING_SOURCE_CONTRACT):
        return False, "unsupported VAE._encode_tiled_owned contract"

    @functools.wraps(original_lanczos)
    def lanczos(samples, width, height):
        global _lanczos_logged
        output_bytes = _lanczos_output_bytes(samples, width, height)
        if samples.shape[0] == 0 or output_bytes <= _LANCZOS_THRESHOLD_BYTES:
            return original_lanczos(samples, width, height)
        if not _lanczos_logged:
            log.info(
                "[OmniXPU] bounded Lanczos preprocessing: "
                "frames=%d output=%dx%d float_bytes=%d",
                samples.shape[0],
                width,
                height,
                output_bytes,
            )
            _lanczos_logged = True
        return _bounded_lanczos(comfy_utils, samples, width, height)

    @classmethod
    @functools.wraps(original_execute)
    def execute(cls, resized_images):
        global _seedvr_pad_logged
        output_bytes = _seedvr_output_bytes(resized_images)
        if (
            resized_images.dim() not in {4, 5}
            or resized_images.shape[-1] == 0
            or output_bytes <= _SEEDVR_PAD_THRESHOLD_BYTES
        ):
            return original_execute(cls, resized_images)
        upscaled_shorter_edge = input_shorter_edge(
            resized_images, "SeedVR2Preprocess"
        )
        if upscaled_shorter_edge < 2:
            return original_execute(cls, resized_images)
        if not _seedvr_pad_logged:
            log.info(
                "[OmniXPU] bounded SeedVR preprocessing: "
                "input_shape=%s output_shape=%s output_bytes=%d",
                tuple(resized_images.shape),
                _seedvr_output_shape(resized_images),
                output_bytes,
            )
            _seedvr_pad_logged = True
        return node_io.NodeOutput(_bounded_seedvr_pad_tensor(resized_images))

    @functools.wraps(original_vae_stage)
    def vae_stage(self, pixel_samples, **kwargs):
        global _vae_stage_logged
        if not _should_bound_vae_stage(self, pixel_samples):
            return original_vae_stage(self, pixel_samples, **kwargs)
        if not _vae_stage_logged:
            log.info(
                "[OmniXPU] bounded VAE input staging: "
                "shape=%s source_bytes=%d target_dtype=%s",
                tuple(pixel_samples.shape),
                pixel_samples.numel() * pixel_samples.element_size(),
                self.vae_dtype,
            )
            _vae_stage_logged = True
        return _bounded_vae_stage(self, pixel_samples, kwargs)

    setattr(lanczos, _PATCH_MARKER, True)
    setattr(execute.__func__, _PATCH_MARKER, True)
    setattr(vae_stage, _PATCH_MARKER, True)
    comfy_utils.lanczos = lanczos
    preprocess.execute = execute
    vae._encode_tiled_owned = vae_stage
    return True, None
