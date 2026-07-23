"""Version helpers shared by source builds and the installed package."""

import os
import re
from importlib.metadata import (
    PackageNotFoundError,
    distribution,
    version as distribution_version,
)
from pathlib import Path

# Docker tags cannot contain the PEP 440 ``+`` local-version separator, so the
# image keeps the shared base version while Python artifacts add the Torch ABI
# they were compiled against.
__image_version__ = "0.1.0-b9-dev"
__base_version__ = "0.1.0b9.dev0"
__supported_torch_minors__ = ("2.10", "2.11", "2.12")
__supported_xpu_targets__ = ("bmg", "ptl-h")


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


def normalize_xpu_target(xpu_target):
    """Return the canonical build target used by the Intel GPU compiler."""
    target = str(xpu_target).strip().lower()
    if target not in __supported_xpu_targets__:
        supported = ", ".join(__supported_xpu_targets__)
        raise RuntimeError(
            f"Unsupported OMNI_XPU_DEVICE {xpu_target!r}; supported targets: {supported}"
        )
    return target


def get_build_xpu_target():
    """Read and validate the AOT target selected for a source build."""
    return normalize_xpu_target(os.environ.get("OMNI_XPU_DEVICE", "bmg"))


def get_xpu_target_tag(xpu_target):
    """Return a PEP 440 local-version component for an AOT target."""
    return normalize_xpu_target(xpu_target).replace("-", "")


def get_xpu_target_from_package_version(package_version):
    """Recover the AOT target from a Torch- and GPU-tagged wheel version."""
    match = re.search(r"\+torch\d+\.(bmg|ptlh)$", str(package_version), re.IGNORECASE)
    if not match:
        raise RuntimeError(
            "omni_xpu_kernel wheel version has no supported GPU target tag: "
            f"{package_version}"
        )
    target_tag = match.group(1).lower()
    return "ptl-h" if target_tag == "ptlh" else target_tag


def get_package_version(torch_version, xpu_target=None):
    """Return the native wheel version for the selected Torch ABI and GPU."""
    target = get_build_xpu_target() if xpu_target is None else normalize_xpu_target(xpu_target)
    return (
        f"{__base_version__}+{get_torch_tag(torch_version)}."
        f"{get_xpu_target_tag(target)}"
    )


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


def get_required_torch_version(requirements):
    """Return the exact Torch version pinned by wheel metadata."""
    for requirement in requirements or ():
        match = re.match(r"^\s*torch\s*==(?!=)\s*([^;,\s]+)", requirement, re.IGNORECASE)
        if match:
            return get_public_torch_version(match.group(1))
    raise RuntimeError("omni_xpu_kernel wheel metadata has no exact torch requirement")


def get_packaged_build_info(version_file=None):
    """Read build identity from this installed wheel, not the active Torch.

    A source checkout may coexist with an older installed wheel. Compare the
    distribution-owned ``_version.py`` path before trusting its dist-info so a
    source build still selects the Torch version from its build environment.
    """
    current_file = Path(version_file or __file__).resolve()
    try:
        package_distribution = distribution("omni-xpu-kernel")
    except PackageNotFoundError:
        return None

    distribution_files = package_distribution.files or ()
    if not any(str(path).endswith(".dist-info/RECORD") for path in distribution_files):
        # Source-tree egg-info is metadata for a prospective build, not an
        # immutable installed wheel identity.
        return None

    packaged_file = Path(
        package_distribution.locate_file(Path("omni_xpu_kernel") / "_version.py")
    ).resolve()
    if packaged_file != current_file:
        return None

    package_version = str(package_distribution.version)
    torch_version = get_required_torch_version(package_distribution.requires)
    xpu_target = get_xpu_target_from_package_version(package_version)
    expected_version = get_package_version(torch_version, xpu_target)
    if package_version != expected_version:
        raise RuntimeError(
            "omni_xpu_kernel wheel metadata is inconsistent: "
            f"version is {package_version}, torch requirement selects {expected_version}"
        )
    return package_version, torch_version, xpu_target


# A source tree derives the prospective wheel identity from its active build
# environment. An installed wheel instead reports its immutable dist-info so a
# later Torch replacement cannot make one native artifact impersonate another.
_packaged_build_info = get_packaged_build_info()
if _packaged_build_info is None:
    __torch_version__ = get_installed_torch_version()
    __xpu_target__ = get_build_xpu_target()
    __version__ = get_package_version(__torch_version__, __xpu_target__)
else:
    __version__, __torch_version__, __xpu_target__ = _packaged_build_info
