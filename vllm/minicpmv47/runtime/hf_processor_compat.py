# SPDX-License-Identifier: Apache-2.0
"""Support checkpoint processors generated before Transformers 5.17."""


def install() -> None:
    from transformers import video_processing_utils

    # Older MiniCPM-V video processors use this only in docstring decorators.
    if not hasattr(video_processing_utils, "BASE_VIDEO_PROCESSOR_DOCSTRING"):
        video_processing_utils.BASE_VIDEO_PROCESSOR_DOCSTRING = ""
