"""
omni_xpu_kernel - High-performance Intel XPU kernels

Build and install using pip:
    pip install . --no-build-isolation

For development:
    pip install -e . --no-build-isolation

Note: --no-build-isolation is required because the build depends on the
installed PyTorch version for finding headers and libraries.

Supported platforms:
    - Linux (with Intel oneAPI)
    - Windows (with Intel oneAPI)
"""

import os
import ctypes
import re
import sys
import subprocess
import shutil
import platform
import sysconfig
from runpy import run_path
from pathlib import Path
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

IS_WINDOWS = platform.system() == "Windows"
VALIDATED_ONEDNN_VERSION = (3, 9, 1)


VERSION_NAMESPACE = run_path(str(Path(__file__).parent / "omni_xpu_kernel" / "_version.py"))
BUILD_TORCH_VERSION = VERSION_NAMESPACE["get_installed_torch_version"]()
BUILD_XPU_TARGET = VERSION_NAMESPACE["get_build_xpu_target"]()
PACKAGE_VERSION = VERSION_NAMESPACE["get_package_version"](
    BUILD_TORCH_VERSION, BUILD_XPU_TARGET
)
XPU_ARCH_MACROS = {
    "bmg": "OMNI_XPU_ARCH_BMG",
    "ptl-h": "OMNI_XPU_ARCH_PTL_H",
}
XPU_ARCH_MACRO = XPU_ARCH_MACROS[BUILD_XPU_TARGET]

BMG_CUTE_REMAINDER_MASK_ORIGINAL = """\
          FragSRow k_rem_mask;
          int k_val = get<0>(tKgK_cur(0,0,0,k_idx,0)) + kblocks_cache * get<1>(TileShapeQK{});
          int k = k_val + get_sub_group().get_local_id()[0];
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < k_rem_mask.size(); i++, k += intel::sg_size) {
            k_rem_mask(i) = (k < seq_len) ? ElementS(sycl::nan(0u)) : ElementS(-INFINITY);
          }
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < tSrS.size(); i++) {
            tSrS(i) = sycl::fmin(tSrS(i), broadcast<1>(k_rem_mask, tSrS, i));
          }
"""

BMG_CUTE_REMAINDER_MASK_REPLACEMENT = """\
          Tensor cPgP = make_identity_tensor(make_shape(seq_len, seq_len));
          Tensor gP = local_tile(
              cPgP, take<0,2>(TileShapeQK{}),
              make_coord(get<0>(blk_qv), K));
          auto cS_thread = thr_mma_qk.partition_C(gP);
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < tSrS.size(); ++i) {
            int col_idx = get<1>(cS_thread(i));
            if (col_idx >= seq_len) {
              tSrS(i) = ElementS(-INFINITY);
            }
          }
"""


def get_core_aot_compile_args(xpu_target):
    """Return the target-specific flags shared by every core AOT build."""
    if xpu_target not in XPU_ARCH_MACROS:
        supported = ", ".join(XPU_ARCH_MACROS)
        raise RuntimeError(
            f"Unsupported core AOT target {xpu_target!r}; supported targets: {supported}"
        )
    return [
        "-fsycl-targets=spir64_gen",
        "-Xsycl-target-backend",
        f"-device {xpu_target}",
        "-DOMNI_XPU_CORE_AOT=1",
    ]


def prepare_bmg_cute_include_overlay(cutlass_root, build_temp):
    """Create a private BMG mainloop overlay without modifying sycl-tla.

    The pinned sycl-tla remainder mask broadcasts a row fragment using a
    mapping that is incorrect for the BMG KV32/SG16 configuration.  Generate a
    build-local copy that derives each score element's key coordinate from the
    MMA partition.  PTL-H continues to compile the pinned upstream header.
    """
    applications = Path(cutlass_root) / "applications"
    relative_dir = Path("flash_attention_v2") / "collective"
    source_dir = applications / relative_dir
    mainloop_source = source_dir / "xe_fmha_fwd_mainloop.hpp"
    fusion_source = source_dir / "fmha_fusion.hpp"
    if not mainloop_source.is_file() or not fusion_source.is_file():
        raise RuntimeError(
            "BMG CUTE overlay needs the pinned sycl-tla collective headers: "
            f"{mainloop_source} and {fusion_source}"
        )

    text = mainloop_source.read_text(encoding="utf-8")
    matches = text.count(BMG_CUTE_REMAINDER_MASK_ORIGINAL)
    if matches != 1:
        raise RuntimeError(
            "Pinned sycl-tla BMG remainder-mask source no longer matches the "
            f"validated overlay contract (matches={matches})"
        )

    overlay_root = Path(build_temp) / "cute_bmg_include_overlay"
    overlay_dir = overlay_root / relative_dir
    overlay_dir.mkdir(parents=True, exist_ok=True)
    (overlay_dir / mainloop_source.name).write_text(
        text.replace(
            BMG_CUTE_REMAINDER_MASK_ORIGINAL,
            BMG_CUTE_REMAINDER_MASK_REPLACEMENT,
            1,
        ),
        encoding="utf-8",
    )
    shutil.copyfile(fusion_source, overlay_dir / fusion_source.name)
    return overlay_root


def get_icpx_path():
    """Find Intel icpx compiler."""
    # On Windows, the compiler is icx.exe (for C++) or icpx is a symlink
    compiler_name = "icx" if IS_WINDOWS else "icpx"
    compiler_exe = compiler_name + (".exe" if IS_WINDOWS else "")
    
    icpx = shutil.which(compiler_exe)
    if icpx:
        return icpx
    
    if IS_WINDOWS:
        # Try common oneAPI installation paths on Windows
        program_files = os.environ.get("ProgramFiles", "C:\\Program Files")
        program_files_x86 = os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)")
        candidates = [
            os.path.join(program_files, "Intel", "oneAPI", "compiler", "latest", "bin", "icx.exe"),
            os.path.join(program_files, "Intel", "oneAPI", "compiler", "2025.1", "bin", "icx.exe"),
            os.path.join(program_files, "Intel", "oneAPI", "compiler", "2024.2", "bin", "icx.exe"),
            os.path.join(program_files_x86, "Intel", "oneAPI", "compiler", "latest", "bin", "icx.exe"),
        ]
    else:
        # Try common oneAPI installation paths on Linux
        candidates = [
            "/opt/intel/oneapi/compiler/latest/bin/icpx",
            "/opt/intel/oneapi/compiler/2025.1/bin/icpx",
            "/opt/intel/oneapi/compiler/2024.2/bin/icpx",
            os.path.expanduser("~/intel/oneapi/compiler/latest/bin/icpx"),
        ]
    
    for path in candidates:
        if os.path.exists(path):
            return path
    
    return None


def get_compile_env(onednn_include=""):
    """Keep the explicitly selected oneDNN include directory authoritative."""
    env = os.environ.copy()
    if not onednn_include:
        return env

    selected = os.path.normcase(os.path.realpath(onednn_include))
    for name in ("CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH"):
        value = env.get(name)
        if not value:
            continue
        paths = value.split(os.pathsep)
        paths = [
            path for path in paths
            if not path or os.path.normcase(os.path.realpath(path)) != selected
        ]
        if paths:
            env[name] = os.pathsep.join(paths)
        else:
            env.pop(name, None)
    return env


def get_origin_rpath(extension_name, target_dir):
    """Return an install-prefix-relative ELF search path for an extension."""
    platlib = Path(sysconfig.get_path("platlib")).resolve()
    extension_dir = platlib.joinpath(*extension_name.split(".")[:-1])
    target_dir = Path(target_dir).resolve()
    install_roots = {
        Path(sys.prefix).resolve(),
        Path(sysconfig.get_path("data")).resolve(),
    }
    if not any(
        target_dir == root or root in target_dir.parents
        for root in install_roots
    ):
        # Explicit system-library overrides remain visibly non-relocatable.
        return target_dir.as_posix()
    relative = os.path.relpath(target_dir, extension_dir)
    if relative == ".":
        return "$ORIGIN"
    return "$ORIGIN/" + Path(relative).as_posix()


def get_runtime_library_dir():
    """Directory populated by Intel runtime wheels pulled in by torch XPU."""
    if IS_WINDOWS:
        return Path(sys.prefix) / "Library" / "bin"
    # Debian's system Python uses sys.prefix=/usr while pip's installation
    # scheme places both platlib and wheel data under /usr/local. Use the
    # scheme data prefix so a system-Python build remains prefix-relative.
    return Path(sysconfig.get_path("data")) / "lib"


def get_torch_runtime_library_dir():
    """Expected torch/lib location after installing both wheels together."""
    return Path(sysconfig.get_path("platlib")) / "torch" / "lib"


def get_onednn_header_version(include_dir):
    version_header = Path(include_dir) / "oneapi" / "dnnl" / "dnnl_version.h"
    text = version_header.read_text(encoding="utf-8")

    def value(name):
        match = re.search(rf"^#define\s+{name}\s+(\d+)\s*$", text, re.MULTILINE)
        if match is None:
            raise RuntimeError(f"Unable to read {name} from {version_header}")
        return int(match.group(1))

    return (
        value("DNNL_VERSION_MAJOR"),
        value("DNNL_VERSION_MINOR"),
        value("DNNL_VERSION_PATCH"),
    )


def find_onednn_library(lib_dir):
    lib_dir = Path(lib_dir)
    candidates = ("dnnl.lib",) if IS_WINDOWS else ("libdnnl.so", "libdnnl.so.3")
    for name in candidates:
        library = lib_dir / name
        if library.is_file():
            return library.resolve()
    return None


def get_onednn_library_version(library):
    if IS_WINDOWS:
        return None

    class DnnlVersion(ctypes.Structure):
        _fields_ = [
            ("major", ctypes.c_int),
            ("minor", ctypes.c_int),
            ("patch", ctypes.c_int),
        ]

    try:
        handle = ctypes.CDLL(os.fspath(library))
        handle.dnnl_version.restype = ctypes.POINTER(DnnlVersion)
        version = handle.dnnl_version().contents
        return version.major, version.minor, version.patch
    except (OSError, AttributeError, ValueError) as error:
        raise RuntimeError(f"Unable to query oneDNN library version from {library}: {error}") from error


def get_onednn_paths():
    """Select a matched oneDNN header/library pair, preferring pip packages."""
    explicit_include = os.environ.get("ONEDNN_INCLUDE", "")
    explicit_lib = os.environ.get("ONEDNN_LIB", "")
    if bool(explicit_include) != bool(explicit_lib):
        raise RuntimeError("ONEDNN_INCLUDE and ONEDNN_LIB must be set together")

    candidates = []
    if explicit_include:
        candidates.append((Path(explicit_include), Path(explicit_lib), "environment"))

    if IS_WINDOWS:
        # The 2025.3 Windows pip package provides dnnl.dll/dnnl.lib but not the
        # development headers, so retain the existing oneAPI source-build path.
        pip_include = Path(sys.prefix) / "Library" / "include"
        pip_lib = Path(sys.prefix) / "Library" / "lib"
    else:
        # onednn-devel and onednn install into the active Python prefix.
        pip_include = Path(sysconfig.get_path("data")) / "include"
        pip_lib = get_runtime_library_dir()
    candidates.append((pip_include, pip_lib, "pip"))

    if IS_WINDOWS:
        oneapi_roots = []
        for variable in ("DNNLROOT", "ONEDNNROOT"):
            value = os.environ.get(variable)
            if value:
                oneapi_roots.append(Path(value))

        program_roots = (
            Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")),
            Path(os.environ.get("ProgramFiles", r"C:\Program Files")),
        )
        for program_root in program_roots:
            dnnl_root = program_root / "Intel" / "oneAPI" / "dnnl"
            oneapi_roots.extend(
                [
                    dnnl_root / "latest",
                    dnnl_root / "2025.3",
                ]
            )

        seen_roots = set()
        for root in oneapi_roots:
            normalized = os.path.normcase(os.path.abspath(root))
            if normalized in seen_roots:
                continue
            seen_roots.add(normalized)
            candidates.append((root / "include", root / "lib", "oneAPI"))

    for include_dir, lib_dir, source in candidates:
        header = include_dir / "oneapi" / "dnnl" / "dnnl.hpp"
        library = find_onednn_library(lib_dir)
        if not header.is_file() or library is None:
            continue

        header_version = get_onednn_header_version(include_dir)
        if header_version != VALIDATED_ONEDNN_VERSION:
            expected = ".".join(map(str, VALIDATED_ONEDNN_VERSION))
            actual = ".".join(map(str, header_version))
            raise RuntimeError(
                f"Unsupported oneDNN headers {actual} from {include_dir}; "
                f"expected {expected} to match onednn==2025.3.0"
            )
        library_version = get_onednn_library_version(library)
        if library_version is not None and library_version != header_version:
            header_text = ".".join(map(str, header_version))
            library_text = ".".join(map(str, library_version))
            raise RuntimeError(
                f"oneDNN header/library mismatch: headers are {header_text} from "
                f"{include_dir}, library is {library_text} from {library}"
            )
        return include_dir.resolve(), lib_dir.resolve(), library, source

    if IS_WINDOWS:
        raise RuntimeError(
            "A matched oneDNN 3.9.1 development installation was not found. "
            "Set ONEDNN_INCLUDE and ONEDNN_LIB to the same oneAPI installation."
        )
    raise RuntimeError(
        "oneDNN 3.9.1 headers and runtime were not found in the active Python "
        "prefix. Install onednn==2025.3.0 and onednn-devel==2025.3.0. "
        "For an explicit non-pip build, set both ONEDNN_INCLUDE and ONEDNN_LIB."
    )


def validate_torch_build(torch, torch_lib):
    public_version = torch.__version__.split("+", 1)[0]
    if public_version != BUILD_TORCH_VERSION:
        raise RuntimeError(
            f"omni_xpu_kernel metadata selected torch {BUILD_TORCH_VERSION}, "
            f"found {torch.__version__}"
        )

    xpu_library = "torch_xpu.lib" if IS_WINDOWS else "libtorch_xpu.so"
    if not hasattr(torch, "xpu") or not (Path(torch_lib) / xpu_library).is_file():
        raise RuntimeError(
            f"torch {torch.__version__} is not an XPU build: missing {xpu_library}"
        )


def linux_rpath_flags(extension_name, *target_dirs):
    rpaths = []
    for target_dir in target_dirs:
        rpath = get_origin_rpath(extension_name, target_dir)
        if rpath not in rpaths:
            rpaths.append(rpath)
    # DT_RPATH is intentional: Intel pip runtime libraries are transitive
    # dependencies of oneDNN/SYCL and also live under the Python prefix.
    return ["-Wl,--disable-new-dtags,-rpath," + ":".join(rpaths)]


class ICPXBuildExt(build_ext):
    """Build extension using Intel icpx compiler directly."""
    
    def build_extension(self, ext):
        # Find compiler
        icpx = get_icpx_path()
        if not icpx:
            if IS_WINDOWS:
                raise RuntimeError(
                    "Intel icx compiler not found. Please install Intel oneAPI "
                    "and run setvars.bat, or ensure icx.exe is in PATH.\n"
                    "Typical installation: https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html"
                )
            else:
                raise RuntimeError(
                    "Intel icpx compiler not found. Please install Intel oneAPI "
                    "and source setvars.sh, or ensure icpx is in PATH."
                )
        
        print(f"Using Intel compiler: {icpx}")
        print(f"Building for platform: {'Windows' if IS_WINDOWS else 'Linux'}")
        print(f"Intel GPU AOT target: {BUILD_XPU_TARGET} ({XPU_ARCH_MACRO})")
        
        # Get paths from torch
        import torch
        torch_dir = Path(torch.__file__).parent
        torch_include = torch_dir / "include"
        torch_lib = torch_dir / "lib"
        validate_torch_build(torch, torch_lib)

        # Match PyTorch's libstdc++ ABI so the extension links/loads against this
        # wheel (a hard-coded flag breaks if the wheel used the other ABI).
        torch_cxx11_abi = int(bool(torch.compiled_with_cxx11_abi()))
        
        # Get Python include
        python_include = sysconfig.get_path("include")
        
        # Output paths
        output_path = Path(self.get_ext_fullpath(ext.name))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        is_lgrf = ext.name.endswith("lgrf_sdp")
        is_cute = ext.name.endswith("cute_fmha_torch")

        # Source directory
        if is_lgrf:
            src_dir = Path(ext.sourcedir) / "omni_xpu_kernel" / "lgrf_uni"
            sources = [src_dir / "sdp_kernels.cpp"]
        elif is_cute:
            src_dir = Path(ext.sourcedir) / "omni_xpu_kernel" / "cute"
            sources = [src_dir / "cute_fmha_torch.cpp"]
        else:
            src_dir = Path(ext.sourcedir) / "omni_xpu_kernel" / "csrc"
            sources = list(src_dir.glob("*.cpp"))
        
        print(f"Source files: {[s.name for s in sources]}")
        print(f"Output: {output_path}")
        
        # The core extension directly uses oneDNN. Select and validate a
        # matched header/runtime pair instead of falling through to torch's
        # newer bundled headers and an unrelated system libdnnl.
        onednn_include, onednn_lib, onednn_library, onednn_source = get_onednn_paths()
        runtime_lib = get_runtime_library_dir().resolve()
        torch_runtime_lib = get_torch_runtime_library_dir().resolve()
        has_onednn = True
        print(f"oneDNN source: {onednn_source}")
        print(f"oneDNN include: {onednn_include}")
        print(f"oneDNN library: {onednn_library}")
        
        if is_cute and IS_WINDOWS:
            # cute FMHA has no Windows build path (and is filtered out of
            # ext_modules on Windows); guard here in case it is reached directly.
            raise RuntimeError("cute_fmha_torch is Linux-only; not supported on Windows.")

        if IS_WINDOWS:
            # Windows compile command using icx
            python_lib_dir = sysconfig.get_config_var("LIBDIR") or str(Path(sys.executable).parent / "libs")
            python_version = f"{sys.version_info.major}{sys.version_info.minor}"
            
            cmd = [
                icpx,
                "-fsycl",
            ]
            
            if is_lgrf:
                cmd += [
                    "-fsycl-targets=spir64_gen",
                    "-Xs", f"-device {BUILD_XPU_TARGET} -options -doubleGRF",
                    "/O2", "/DNDEBUG",
                    "/EHsc",
                    "/std:c++17",
                    "/DNOMINMAX",
                    "/DWIN32_LEAN_AND_MEAN",
                    "-DBUILD_ESIMD_KERNEL_LIB",
                    f"/D{XPU_ARCH_MACRO}=1",
                    "/LD",  # Create DLL
                    f"/Fe:{output_path}",  # Output file
                ]
                if has_onednn:
                    cmd.append(f"/I{onednn_include}")
                cmd += [str(s) for s in sources]
            else:
                cmd += [
                    "-fsycl-targets=spir64_gen",
                    "-Xsycl-target-backend",
                    f"-device {BUILD_XPU_TARGET}",
                    "-fsycl-esimd-force-stateless-mem",
                    "/O2", "/DNDEBUG",
                    f"/D{XPU_ARCH_MACRO}=1",
                    "/DOMNI_XPU_CORE_AOT=1",
                    "/DNOMINMAX",
                    "/DWIN32_LEAN_AND_MEAN",
                    "/EHsc",  # Enable C++ exception handling
                    "/std:c++17",
                ]
                # PyTorch XPU wheels also bundle oneDNN headers. Keep the
                # headers selected with the external oneDNN library first so
                # declarations and exported symbols use the same ABI.
                if has_onednn:
                    cmd.append(f"/I{onednn_include}")
                cmd += [
                    f"/I{python_include}",
                    f"/I{torch_include}",
                    f"/I{torch_include}\\torch\\csrc\\api\\include",
                    f"/I{src_dir}",
                    "/LD",  # Create DLL
                    f"/Fe:{output_path}",  # Output file
                ]
                cmd += [str(s) for s in sources] + [
                    "/link",
                    f"/LIBPATH:{torch_lib}",
                    f"/LIBPATH:{python_lib_dir}",
                    "torch.lib", "torch_python.lib", "torch_cpu.lib", "torch_xpu.lib", "c10.lib", "c10_xpu.lib",
                    f"python{python_version}.lib",
                ]
                if has_onednn:
                    cmd.append(str(onednn_library))
        else:
            # Linux compile command
            cmd = [
                icpx,
                "-fsycl",
            ]
            
            if is_lgrf:
                cmd += [
                    "-fsycl-targets=spir64_gen",
                    "-Xs", f"-device {BUILD_XPU_TARGET} -options -doubleGRF",
                    "-O3", "-DNDEBUG",
                    "-DBUILD_ESIMD_KERNEL_LIB",
                    f"-D{XPU_ARCH_MACRO}=1",
                    "-fPIC", "-shared",
                ]
                cmd += linux_rpath_flags(ext.name, runtime_lib)
                if has_onednn:
                    cmd.append(f"-I{onednn_include}")
                cmd += ["-o", str(output_path)] + [str(s) for s in sources]
            elif is_cute:
                # CUTLASS-SYCL fused FMHA. Needs a cutlass-sycl / sycl-tla source
                # tree (headers only) via CUTLASS_SYCL_ROOT, AOT to the target GPU,
                # and the Xe SPIR-V extensions. fp32 accumulation (no fp16 overflow).
                cutlass = os.environ.get("CUTLASS_SYCL_ROOT", "")
                if not cutlass or not os.path.isdir(cutlass):
                    raise RuntimeError(
                        "cute_fmha_torch needs CUTLASS_SYCL_ROOT set to a cutlass-sycl "
                        "(sycl-tla) source tree containing include/, tools/util/include/, "
                        "examples/common/, applications/. Got: " + repr(cutlass))
                cute_overlay_flags = []
                if BUILD_XPU_TARGET == "bmg":
                    overlay = prepare_bmg_cute_include_overlay(
                        cutlass, self.build_temp
                    )
                    cute_overlay_flags.append(f"-I{overlay}")
                cmd += [
                    "-std=c++17", "-O3", "-DNDEBUG", "-fPIC", "-shared",
                    "-fsycl-targets=spir64_gen",
                    "-Xsycl-target-backend", f"-device {BUILD_XPU_TARGET}",
                    "-Xspirv-translator",
                    "-spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,"
                    "+SPV_INTEL_subgroup_matrix_multiply_accumulate",
                    "-fno-sycl-instrument-device-code",
                    "-DCUTLASS_ENABLE_SYCL", "-DSYCL_INTEL_TARGET",
                    f"-D{XPU_ARCH_MACRO}=1",
                    f"-D_GLIBCXX_USE_CXX11_ABI={torch_cxx11_abi}",
                    *cute_overlay_flags,
                    f"-I{cutlass}/include",
                    f"-I{cutlass}/tools/util/include",
                    f"-I{cutlass}/examples/common",
                    f"-I{cutlass}/applications",
                    f"-I{python_include}",
                    f"-I{torch_include}",
                    f"-I{torch_include}/torch/csrc/api/include",
                    "-Wno-unknown-pragmas", "-Wno-unused-variable",
                    "-Wno-unused-but-set-variable", "-Wno-unused-local-typedef",
                    "-Wno-uninitialized", "-Wno-reorder-ctor",
                    "-Wno-logical-op-parentheses", "-Wno-unused-function",
                    "-Wno-deprecated-copy",
                    f"-L{torch_lib}",
                    "-ltorch", "-ltorch_python", "-ltorch_cpu", "-ltorch_xpu",
                    "-lc10", "-lc10_xpu",
                    "-o", str(output_path),
                ] + [str(s) for s in sources]
                cmd += linux_rpath_flags(
                    ext.name, torch_runtime_lib, runtime_lib
                )
            else:
                cmd += ["-fsycl-esimd-force-stateless-mem"]
                # The image validator requires the package target and compiled
                # core target to agree. Build both supported devices AOT so a
                # BMG image cannot silently retain the legacy JIT core.
                cmd += get_core_aot_compile_args(BUILD_XPU_TARGET)
                cmd += [
                    "-O3", "-DNDEBUG",
                    f"-D{XPU_ARCH_MACRO}=1",
                    "-fPIC", "-shared",
                    "-std=c++17",
                ]
                # torch/include contains another oneDNN header tree. Put the
                # explicitly selected installation first so it matches -ldnnl.
                if has_onednn:
                    # The compiler treats /usr/local/include as a built-in
                    # system path and may de-duplicate a normal -I entry after
                    # torch/include. The oneDNN translation units use quoted
                    # includes so -iquote keeps the validated pip header tree
                    # authoritative even with Debian's system Python scheme.
                    cmd += [
                        f"-iquote{onednn_include}",
                        f"-I{onednn_include}",
                    ]
                cmd += [
                    f"-I{python_include}",
                    f"-I{torch_include}",
                    f"-I{torch_include}/torch/csrc/api/include",
                    f"-I{src_dir}",
                ]
                cmd += [
                    f"-L{torch_lib}",
                    "-ltorch", "-ltorch_python", "-ltorch_cpu", "-ltorch_xpu", "-lc10", "-lc10_xpu",
                ]
                if has_onednn:
                    # Use the selected file, not -ldnnl search order, so the
                    # validated header and linked library cannot diverge.
                    cmd.append(str(onednn_library))
                cmd += [
                    "-o", str(output_path),
                ] + [str(s) for s in sources]
                cmd += linux_rpath_flags(
                    ext.name, torch_runtime_lib, runtime_lib, onednn_lib
                )
        
        print(f"Compile command: {' '.join(cmd)}")
        
        # Run compiler
        # oneAPI setvars also injects oneDNN through the compiler include-path
        # environment. Clang can de-duplicate that path against the earlier -I
        # and retain the environment copy after torch/include. Remove only the
        # duplicate entry; all other oneAPI paths remain unchanged.
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=get_compile_env(onednn_include if has_onednn else ""),
        )
        
        if result.returncode != 0:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            raise RuntimeError(f"Compilation failed with exit code {result.returncode}")
        
        print(f"Successfully built {output_path}")


class ICPXExtension(Extension):
    """Extension that will be built with icpx."""
    
    def __init__(self, name, sourcedir=""):
        # setuptools requires Extension.sources to be relative to setup.py so
        # they can be recorded in an sdist/wheel manifest.  The custom build
        # command keeps an absolute root separately for invoking icpx.
        source_root = Path(sourcedir)
        depends = []
        if name.endswith("lgrf_sdp"):
            kernel_root = source_root / "omni_xpu_kernel" / "lgrf_uni"
            sources = [kernel_root / "sdp_kernels.cpp"]
            depends = sorted(kernel_root.rglob("*.h"))
        elif name.endswith("cute_fmha_torch"):
            kernel_root = source_root / "omni_xpu_kernel" / "cute"
            sources = [kernel_root / "cute_fmha_torch.cpp"]
            csrc_root = source_root / "omni_xpu_kernel" / "csrc"
            depends = [
                kernel_root / "cute_fmha_config.h",
                csrc_root / "bmg_kernel_policy.h",
                csrc_root / "device_utils.h",
            ]
        else:
            kernel_root = source_root / "omni_xpu_kernel" / "csrc"
            sources = sorted(kernel_root.glob("*.cpp"))
            depends = sorted(kernel_root.glob("*.h"))
        super().__init__(
            name,
            sources=[source.as_posix() for source in sources],
            depends=[dependency.as_posix() for dependency in depends],
        )
        self.sourcedir = os.fspath(source_root.resolve())


# Read version
def get_version():
    return PACKAGE_VERSION


# Read README
def get_long_description():
    readme = Path(__file__).parent / "README.md"
    if readme.exists():
        return readme.read_text(encoding="utf-8")
    return ""


# Extension list. The cute (CUTLASS-SYCL) FMHA is Linux-only and required by
# default so a normal build cannot silently omit the default attention backend.
# Set OMNI_XPU_REQUIRE_CUTE=0 explicitly for a core-only build (including
# Windows, where the CUTE extension is not supported).
_ext_modules = [
    ICPXExtension("omni_xpu_kernel._C", sourcedir="."),
    ICPXExtension("omni_xpu_kernel.lgrf_uni.lgrf_sdp", sourcedir="."),
]
_cutlass_sycl_root = os.environ.get("CUTLASS_SYCL_ROOT", "")
_cutlass_sycl_required = os.environ.get("OMNI_XPU_REQUIRE_CUTE", "1") != "0"
_cutlass_sycl_dirs = ("include", "tools/util/include", "examples/common", "applications")
_cutlass_sycl_available = bool(_cutlass_sycl_root) and all(
    os.path.isdir(os.path.join(_cutlass_sycl_root, path)) for path in _cutlass_sycl_dirs
)
if _cutlass_sycl_required and IS_WINDOWS:
    raise RuntimeError(
        "CUTE is required by default but unsupported on Windows; "
        "set OMNI_XPU_REQUIRE_CUTE=0 for an explicit core-only build"
    )
if _cutlass_sycl_required and not _cutlass_sycl_available:
    raise RuntimeError(
        "CUTE is required by default; set CUTLASS_SYCL_ROOT containing: "
        + ", ".join(_cutlass_sycl_dirs)
        + f"; got {_cutlass_sycl_root!r}"
        + ". Set OMNI_XPU_REQUIRE_CUTE=0 only for an explicit core-only build."
    )
if not IS_WINDOWS and _cutlass_sycl_available:
    _ext_modules.append(ICPXExtension("omni_xpu_kernel.cute.cute_fmha_torch", sourcedir="."))

setup(
    name="omni_xpu_kernel",
    version=get_version(),
    author="Intel",
    author_email="",
    description="High-performance Intel XPU kernels for PyTorch",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/intel/omni_xpu_kernel",
    packages=find_packages(
        exclude=[
            "tests",
            "tests.*",
            "scripts",
            "scripts.*",
            "benchmarks",
            "benchmarks.*",
        ]
    ),
    ext_modules=_ext_modules,
    cmdclass={"build_ext": ICPXBuildExt},
    python_requires=">=3.9",
    install_requires=[
        f"torch=={BUILD_TORCH_VERSION}",
        "onednn==2025.3.0; platform_system == 'Linux' and platform_machine == 'x86_64'",
        "onednn==2025.3.0; platform_system == 'Windows' and platform_machine == 'AMD64'",
    ],
    extras_require={
        "dev": [
            "pytest",
            "numpy",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
    ],
    keywords="intel xpu sycl esimd pytorch gpu kernels quantization gguf",
)
