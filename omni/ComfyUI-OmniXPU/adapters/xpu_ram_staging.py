"""Enable ComfyUI's host RAM weight staging on a discrete Intel XPU.

ComfyUI keeps DynamicVRAM weights staged in host RAM ("pinned memory") only on
NVIDIA and AMD, and registers those host buffers through
``torch.cuda.cudart()``. On XPU the staging budget stays disabled, so every
VBAR fault reads its weights from the model file again (intel/llm-scaler#761).

This adapter gives a discrete XPU the budget ComfyUI computes for CUDA and
answers ComfyUI's two host-registration calls with success. The staged buffers
remain ordinary pageable RAM, which the XPU copy path already accepts; nothing
is page-locked. Integrated XPUs are left alone because their device memory is
system RAM.
"""

from __future__ import annotations

import functools
import inspect
import logging
import re

import torch


log = logging.getLogger("ComfyUI-OmniXPU")

_PATCH_MARKER = "_omnixpu_xpu_ram_staging"
_REGISTRATION_CALLS = frozenset(("cudaHostRegister", "cudaHostUnregister"))
_CUDART_CALL = re.compile(r"torch\.cuda\.cudart\(\)\.(\w+)")
_BUDGET_CONTRACT = (
    "if is_nvidia() or is_amd():",
    "MAX_PINNED_MEMORY = ram * 0.40",
)


class _XPUHostRegistration:
    """Stand-in for ``cudart()`` that only answers host registration."""

    def cudaHostRegister(self, ptr, size, flags):
        return 0

    def cudaHostUnregister(self, ptr):
        return 0

    def __getattr__(self, name):
        raise AssertionError("Torch not compiled with CUDA enabled")


def _contract_violation(model_management_source, registration_sources):
    """Return why the installed ComfyUI does not match, or None."""
    if not all(fragment in model_management_source for fragment in _BUDGET_CONTRACT):
        return "unsupported ComfyUI pinned-memory budget contract"
    calls = set()
    for source in registration_sources:
        calls.update(_CUDART_CALL.findall(source))
    if not calls:
        return "ComfyUI host registration no longer uses torch.cuda.cudart()"
    unexpected = calls - _REGISTRATION_CALLS
    if unexpected:
        return "unsupported cudart calls: " + ", ".join(sorted(unexpected))
    return None


def _budget(model_management, torch_module):
    """ComfyUI's own CUDA staging budget for this host."""
    ram = model_management.get_total_memory(torch_module.device("cpu"))
    if model_management.WINDOWS:
        return ram * 0.40
    system_memory = model_management.comfy.system_memory
    swap = (
        0
        if system_memory.cgroup_memory_limit() is not None
        else model_management.get_disk_swap_total()
    )
    return max(ram * 0.40, min(ram * 0.90, ram - 4 * 1024**3, ram + swap - 16 * 1024**3))


def _install_registration(torch_module):
    original = torch_module.cuda.cudart
    if getattr(original, _PATCH_MARKER, False):
        return
    registration = _XPUHostRegistration()

    @functools.wraps(original)
    def cudart():
        if torch_module.cuda.is_available():
            return original()
        return registration

    setattr(cudart, _PATCH_MARKER, True)
    torch_module.cuda.cudart = cudart


def _enable(model_management, sources, torch_module):
    if getattr(model_management, _PATCH_MARKER, False):
        return False, "XPU RAM staging is already enabled"
    if not model_management.is_intel_xpu():
        return False, "not running on an Intel XPU"
    if torch_module.cuda.is_available():
        return False, "a CUDA runtime is present"
    if model_management.DISABLE_PINNED_MEMORY:
        return False, "pinned memory disabled by --disable-pinned-memory"
    if model_management.MAX_PINNED_MEMORY > 0:
        return False, "ComfyUI already stages weights in RAM on this backend"

    violation = _contract_violation(*sources)
    if violation is not None:
        return False, violation

    device = model_management.get_torch_device()
    integrated = getattr(
        torch_module.xpu.get_device_properties(device), "is_integrated_gpu", None
    )
    if integrated is None:
        return False, "Torch does not report whether the XPU is integrated"
    if integrated:
        return False, "integrated XPU memory is system RAM"

    _install_registration(torch_module)
    model_management.MAX_PINNED_MEMORY = _budget(model_management, torch_module)
    setattr(model_management, _PATCH_MARKER, True)
    log.info(
        "[OmniXPU] XPU RAM weight staging enabled: budget %d MiB",
        model_management.MAX_PINNED_MEMORY // (1024 * 1024),
    )
    return True, None


def apply():
    try:
        import comfy.model_management as model_management
        import comfy.model_patcher as model_patcher
        import comfy.pinned_memory as pinned_memory
    except ModuleNotFoundError as exc:
        return False, f"ComfyUI module unavailable: {exc.name}"

    sources = (
        inspect.getsource(model_management),
        [inspect.getsource(module) for module in (model_management, pinned_memory, model_patcher)],
    )
    return _enable(model_management, sources, torch)
