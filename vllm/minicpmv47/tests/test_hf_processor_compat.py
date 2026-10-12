# SPDX-License-Identifier: Apache-2.0
"""Checkpoint documentation compatibility must leave processor behavior intact."""

from hf_processor_compat import install
from transformers import video_processing_utils


def test_restores_removed_checkpoint_documentation_symbol(monkeypatch):
    monkeypatch.delattr(
        video_processing_utils, "BASE_VIDEO_PROCESSOR_DOCSTRING", raising=False
    )
    processor = video_processing_utils.BaseVideoProcessor
    install()
    assert video_processing_utils.BASE_VIDEO_PROCESSOR_DOCSTRING == ""
    assert video_processing_utils.BaseVideoProcessor is processor


def test_preserves_existing_documentation_symbol(monkeypatch):
    monkeypatch.setattr(
        video_processing_utils,
        "BASE_VIDEO_PROCESSOR_DOCSTRING",
        "Existing docs",
        raising=False,
    )
    install()
    install()
    assert video_processing_utils.BASE_VIDEO_PROCESSOR_DOCSTRING == "Existing docs"
