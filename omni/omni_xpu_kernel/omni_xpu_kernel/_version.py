"""Version helpers shared by source builds and the installed package."""

from importlib.metadata import PackageNotFoundError, version as distribution_version

# Docker tags cannot contain the PEP 440 ``+`` local-version separator, so the
# image keeps the shared base version while Python artifacts add the Torch ABI
# they were compiled against.
__image_version__ = "0.1.0-b8-dev"
__base_version__ = "0.1.0b8.dev0"
__supported_torch_minors__ = ("2.10", "2.11", "2.12")


def get_public_torch_version(torch_version):
    """Remove a wheel local-version suffix such as ``+xpu``."""
    public_version = str(torch_version).split("+", 1)[0]
    components = public_version.split(".")
    if len(components) < 2 or not all(component.isdigit() for component in components[:2]):
        raise RuntimeError(f"Unable to determine Torch minor from version {torch_version!r}")
    return public_version


def get_torch_minor(torch_version):
    """Return and validate the ABI-relevant ``major.minor`` pair."""
    public_version = get_public_torch_version(torch_version)
    torch_minor = ".".join(public_version.split(".")[:2])
    if torch_minor not in __supported_torch_minors__:
        supported = ", ".join(__supported_torch_minors__)
        raise RuntimeError(
            f"Unsupported Torch version {torch_version!r}; supported XPU minors: {supported}"
        )
    return torch_minor


def get_torch_tag(torch_version):
    """Return the PEP 440 local tag for a supported Torch build."""
    return "torch" + get_torch_minor(torch_version).replace(".", "")


def get_package_version(torch_version):
    """Return the native wheel version for the selected Torch ABI."""
    return f"{__base_version__}+{get_torch_tag(torch_version)}"


def get_installed_torch_version():
    """Read the active build environment without importing the Torch runtime."""
    try:
        torch_version = distribution_version("torch")
    except PackageNotFoundError as error:
        raise RuntimeError(
            "PyTorch must be installed before building omni_xpu_kernel; "
            "use --no-build-isolation"
        ) from error
    return get_public_torch_version(torch_version)


# Runtime values match the active Torch environment. Wheel metadata pins that
# exact public version, while the local version tag records its native ABI minor.
__torch_version__ = get_installed_torch_version()
__version__ = get_package_version(__torch_version__)
