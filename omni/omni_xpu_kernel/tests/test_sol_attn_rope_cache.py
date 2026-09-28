"""Persistent RoPE layout copies follow the caller's allocation context."""

import torch

from omni_xpu_kernel.cute import sol_attn_v2


class AllocationContext:
    def __init__(self):
        self.entries = 0

    def __enter__(self):
        self.entries += 1

    def __exit__(self, *_exc):
        return False


def test_noncontiguous_rope_copy_is_cached_and_tracks_source_version():
    t, rot = 3, 8
    source = torch.arange(t * rot * 2, dtype=torch.float32)
    freqs = source.reshape(t, rot // 2, 2, 2).transpose(0, 1)
    assert not freqs.is_contiguous()
    first_context = AllocationContext()
    second_context = AllocationContext()
    try:
        sol_attn_v2.set_allocation_context_factory(lambda: first_context)
        first = sol_attn_v2._cached_rope_freqs(freqs, t, rot)
        second = sol_attn_v2._cached_rope_freqs(freqs, t, rot)
        assert first is second
        assert first_context.entries == 1
        torch.testing.assert_close(first, freqs.reshape(1, t, 1, rot // 2, 2, 2))

        freqs.add_(1)
        updated = sol_attn_v2._cached_rope_freqs(freqs, t, rot)
        assert updated is not first
        assert first_context.entries == 2
        torch.testing.assert_close(updated, freqs.reshape(1, t, 1, rot // 2, 2, 2))

        sol_attn_v2.set_allocation_context_factory(lambda: second_context)
        replacement = sol_attn_v2._cached_rope_freqs(freqs, t, rot)
        assert replacement is not updated
        assert second_context.entries == 1

        contiguous = torch.arange(t * rot * 2).reshape(1, t, 1, rot // 2, 2, 2)
        direct = sol_attn_v2._cached_rope_freqs(contiguous, t, rot)
        assert direct.data_ptr() == contiguous.data_ptr()
        assert second_context.entries == 1
    finally:
        sol_attn_v2.set_allocation_context_factory(
            lambda: sol_attn_v2._NULL_ALLOCATION_CONTEXT
        )
