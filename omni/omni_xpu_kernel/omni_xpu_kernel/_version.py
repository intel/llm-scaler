"""Single source of truth for the Kernel wheel and Omni image versions."""

# Docker tags cannot contain the PEP 440 ``+`` local-version separator, so the
# image keeps the shared base version while Python artifacts add the Torch ABI
# they were compiled against.
__image_version__ = "0.1.0-b8-dev"
__torch_version__ = "2.11.0"
__version__ = "0.1.0-b8-dev+torch211"
