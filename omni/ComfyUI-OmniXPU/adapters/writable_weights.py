"""Bridge ComfyUI writable backing and its stream boundary to XPU."""
from __future__ import annotations

import functools
import inspect
import textwrap

import torch

_MARKER = "__omnixpu_writable_weight_original__"


def _current_stream(device):
    return (torch.xpu.current_stream(device) if device.type == "xpu"
            else torch.cuda.current_stream(device))


def _stream_pointer(stream, device):
    if stream is None:
        return 0
    return int(stream.sycl_queue if device is not None and device.type == "xpu"
               else stream.cuda_stream)


def _rewrite(function, replacements):
    if function.__closure__:
        raise ValueError("writable method unexpectedly captures closure state")
    source = textwrap.dedent(inspect.getsource(function))
    for before, after, count in replacements:
        if source.count(before) != count:
            raise ValueError(f"unsupported {function.__name__} writable contract")
        source = source.replace(before, after)
    namespace = {**function.__globals__, "_omnixpu_current_stream": _current_stream,
                 "_omnixpu_stream_pointer": _stream_pointer}
    exec(compile(source, inspect.getfile(function), "exec"), namespace)
    return functools.update_wrapper(namespace[function.__name__], function)


def apply():
    import comfy.memory_management as memory
    import comfy.model_management as management
    import comfy.ops as ops
    from comfy_aimdo import host_buffer

    if management.get_torch_device().type != "xpu":
        return False, "writable weight adapter requires XPU"
    if not hasattr(ops, "WritableWeight"):
        return False, "ComfyUI writable weights are unavailable"
    originals = (memory.write_tensor_file_slice, memory.read_tensor_file_slice_into,
                 ops.WritableWeight.writeback)
    if all(hasattr(f, _MARKER) for f in originals):
        return True, "already patched"
    if any(hasattr(f, _MARKER) for f in originals):
        return False, "writable weight adapter is partially applied"
    if not callable(getattr(host_buffer, "write_device_to_file", None)):
        return False, "AIMDO device-to-file writeback is unavailable"
    write, read, writeback = originals
    try:
        if tuple(inspect.signature(write).parameters) != ("tensor", "source", "stream"):
            raise ValueError("unsupported write_tensor_file_slice signature")
        if tuple(inspect.signature(read).parameters) != ("tensor", "destination", "stream", "destination2"):
            raise ValueError("unsupported read_tensor_file_slice_into signature")
        if tuple(inspect.signature(writeback).parameters) != ("self", "value"):
            raise ValueError("unsupported WritableWeight.writeback signature")
        write_xpu = _rewrite(write, (
            ('if source.device.type == "cuda":', 'if source.device.type in ("cuda", "xpu"):', 1),
            ('torch.cuda.current_stream(source.device)', '_omnixpu_current_stream(source.device)', 1),
            ('stream.cuda_stream,', '_omnixpu_stream_pointer(stream, source.device),', 1),
            ('CPU or CUDA source', 'CPU, CUDA or XPU source', 1),
        ))
        read_xpu = _rewrite(read, ((
            'getattr(stream, "cuda_stream", 0) if stream is not None else 0',
            '_omnixpu_stream_pointer(stream, destination2.device if destination2 is not None else None)', 2,
        ),))
        writeback_xpu = _rewrite(writeback, ((
            'torch.cuda.current_stream(value.device).synchronize()',
            '_omnixpu_current_stream(value.device).synchronize()', 1,
        ),))
    except (AttributeError, OSError, TypeError, ValueError, SyntaxError) as exc:
        return False, str(exc)

    @functools.wraps(write)
    def write_file(tensor, source, stream=None):
        fn = write_xpu if source.device.type == "xpu" else write
        return fn(tensor, source, stream=stream)

    @functools.wraps(read)
    def read_file(tensor, destination, stream=None, destination2=None):
        xpu = any(t is not None and t.device.type == "xpu" for t in (destination, destination2))
        return (read_xpu if xpu else read)(tensor, destination, stream=stream, destination2=destination2)

    @functools.wraps(writeback)
    def save(self, value):
        return (writeback_xpu if value.device.type == "xpu" else writeback)(self, value)

    for replacement, original in zip((write_file, read_file, save), originals):
        setattr(replacement, _MARKER, original)
    read_xpu.__globals__["read_tensor_file_slice_into"] = read_file
    memory.write_tensor_file_slice, memory.read_tensor_file_slice_into = write_file, read_file
    ops.WritableWeight.writeback = save
    return True, None
