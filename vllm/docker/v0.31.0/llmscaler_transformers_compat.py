"""Keep checkpoint video processor docstring imports compatible with 5.18."""
from importlib.metadata import version

if version("transformers") == "5.18.0":
    import transformers.video_processing_utils as video_processing_utils

    # MiniCPM-V 4.7 checkpoint code imports this solely for documentation.
    # Its removal upstream must not force a change to read-only model files.
    if not hasattr(video_processing_utils, "BASE_VIDEO_PROCESSOR_DOCSTRING"):
        video_processing_utils.BASE_VIDEO_PROCESSOR_DOCSTRING = ""
