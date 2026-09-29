"""Persistent ConvRot cache allocation stays in the caller's owner context."""

import contextlib

import pytest
import torch

from omni_xpu_kernel import int8


class NativeConvRot:
    def __init__(self, events):
        self.events = events

    def prepare_convrot_hadamard(self, exemplar, group_size, fp32):
        self.events.append(('prepare', group_size, fp32))

    def clear_convrot_hadamard_cache(self, device_index):
        self.events.append(('clear', device_index))
        return 1

    def rotate_convrot(self, value, group_size):
        self.events.append(('rotate', group_size))
        return value

    def quantize_int8_convrot_weight(self, weight, group_size, stochastic_rounding):
        self.events.append(('quantize', group_size))
        return weight, weight

    def dequantize_int8_convrot_weight(self, q, scale, group_size):
        self.events.append(('dequantize', group_size))
        return q


def test_convrot_persistent_cache_precedes_tensor_route(monkeypatch):
    events = []

    @contextlib.contextmanager
    def native_owner_context():
        events.append(('enter',))
        yield
        events.append(('exit',))

    monkeypatch.setattr(int8, '_get_native', lambda: NativeConvRot(events))
    int8.set_allocation_context_factory(native_owner_context)
    try:
        x = torch.ones((1, 256), dtype=torch.float32)
        assert int8.rotate_convrot(x, 256) is x
        assert events == [('enter',), ('prepare', 256, False),
                          ('exit',), ('rotate', 256)]

        events.clear()
        assert int8.prepare_convrot_hadamard(x, 64, fp32=True)
        assert events == [('enter',), ('prepare', 64, True), ('exit',)]

        events.clear()
        assert int8.quantize_int8_convrot_weight(x, 256)[0] is x
        assert events == [('enter',), ('prepare', 256, False),
                          ('exit',), ('quantize', 256)]

        events.clear()
        q = torch.ones((1, 256), dtype=torch.int8)
        assert int8.dequantize_int8_convrot_weight(q, x[:, :1], 256) is q
        assert events == [('enter',), ('prepare', 256, True),
                          ('exit',), ('dequantize', 256)]
    finally:
        int8.set_allocation_context_factory(contextlib.nullcontext)


def test_convrot_allocation_context_requires_factory():
    with pytest.raises(TypeError, match='allocation context factory'):
        int8.set_allocation_context_factory(None)


def test_convrot_cache_release_waits_for_xpu_before_dropping_owner(monkeypatch):
    events = []
    monkeypatch.setattr(int8, '_get_native', lambda: NativeConvRot(events))
    monkeypatch.setattr(torch.xpu, 'synchronize',
                        lambda device: events.append(('synchronize', device)))
    assert int8.clear_convrot_hadamard_cache(0) == 1
    assert events == [('synchronize', 0), ('clear', 0)]
    with pytest.raises(ValueError, match='device index'):
        int8.clear_convrot_hadamard_cache(-1)
    assert events == [('synchronize', 0), ('clear', 0)]
