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
import sys
import subprocess
import shutil
import platform
from pathlib import Path
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

IS_WINDOWS = platform.system() == "Windows"


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
        
        # Get paths from torch
        import torch
        torch_dir = Path(torch.__file__).parent
        torch_include = torch_dir / "include"
        torch_lib = torch_dir / "lib"
        
        # Get Python include
        import sysconfig
        python_include = sysconfig.get_path("include")
        ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
        
        # Output paths
        output_path = Path(self.get_ext_fullpath(ext.name))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        is_lgrf = ext.name.endswith("lgrf_sdp")
        
        # Source directory
        if is_lgrf:
            src_dir = Path(ext.sourcedir) / "omni_xpu_kernel" / "lgrf_uni"
            sources = [src_dir / "sdp_kernels.cpp"]
        else:
            src_dir = Path(ext.sourcedir) / "omni_xpu_kernel" / "csrc"
            sources = list(src_dir.glob("*.cpp"))
        
        print(f"Source files: {[s.name for s in sources]}")
        print(f"Output: {output_path}")
        
        # Detect oneDNN (dnnl) installation
        onednn_include = os.environ.get("ONEDNN_INCLUDE", "")
        onednn_lib = os.environ.get("ONEDNN_LIB", "")
        
        if not onednn_include or not onednn_lib:
            # Auto-detect from common oneAPI paths
            onednn_candidates = [
                "/opt/intel/oneapi/dnnl/2025.1",
                "/opt/intel/oneapi/dnnl/latest",
                "/opt/intel/oneapi/2025.1",
            ]
            for candidate in onednn_candidates:
                inc = os.path.join(candidate, "include")
                lib = os.path.join(candidate, "lib")
                if os.path.exists(os.path.join(inc, "oneapi", "dnnl", "dnnl.hpp")):
                    if not onednn_include:
                        onednn_include = inc
                    if not onednn_lib:
                        onednn_lib = lib
                    break
        
        has_onednn = bool(onednn_include and os.path.isdir(onednn_include))
        if has_onednn:
            print(f"oneDNN include: {onednn_include}")
            print(f"oneDNN lib: {onednn_lib}")
        else:
            print("WARNING: oneDNN not found. onednn_int4_gemm will not be available.")
        
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
                    "-Xs", "-device ptl-h -options -doubleGRF",
                    "/O2", "/DNDEBUG",
                    "-DBUILD_ESIMD_KERNEL_LIB",
                    "/LD",  # Create DLL
                    f"/Fe:{output_path}",  # Output file
                ]
                if has_onednn:
                    cmd.append(f"/I{onednn_include}")
                cmd += [str(s) for s in sources]
            else:
                cmd += [
                    "-fsycl-esimd-force-stateless-mem",
                    "/O2", "/DNDEBUG",
                    "/EHsc",  # Enable C++ exception handling
                    "/std:c++17",
                    f"/I{python_include}",
                    f"/I{torch_include}",
                    f"/I{torch_include}\\torch\\csrc\\api\\include",
                    f"/I{src_dir}",
                    "/LD",  # Create DLL
                    f"/Fe:{output_path}",  # Output file
                ]
                if has_onednn:
                    cmd.append(f"/I{onednn_include}")
                cmd += [str(s) for s in sources] + [
                    f"/link",
                    f"/LIBPATH:{torch_lib}",
                    f"/LIBPATH:{python_lib_dir}",
                    "torch.lib", "torch_python.lib", "torch_cpu.lib", "torch_xpu.lib", "c10.lib", "c10_xpu.lib",
                    f"python{python_version}.lib",
                ]
                if has_onednn:
                    cmd += [f"/LIBPATH:{onednn_lib}", "dnnl.lib"]
        else:
            # Linux compile command
            cmd = [
                icpx,
                "-fsycl",
            ]
            
            if is_lgrf:
                cmd += [
                    "-fsycl-targets=spir64_gen",
                    # Fallback to pvc if ptl-h is not supported in the compiler
                    "-Xs", "-device pvc -options -doubleGRF",
                    "-O3", "-DNDEBUG",
                    "-DBUILD_ESIMD_KERNEL_LIB",
                    "-fPIC", "-shared",
                ]
                if has_onednn:
                    cmd.append(f"-I{onednn_include}")
                cmd += ["-o", str(output_path)] + [str(s) for s in sources]
            else:
                cmd += [
                    "-fsycl-esimd-force-stateless-mem",
                    "-O3", "-DNDEBUG",
                    "-fPIC", "-shared",
                    "-std=c++17",
                    f"-I{python_include}",
                    f"-I{torch_include}",
                    f"-I{torch_include}/torch/csrc/api/include",
                    f"-I{src_dir}",
                ]
                if has_onednn:
                    cmd.append(f"-I{onednn_include}")
                cmd += [
                    f"-L{torch_lib}",
                    "-ltorch", "-ltorch_python", "-ltorch_cpu", "-ltorch_xpu", "-lc10", "-lc10_xpu",
                ]
                if has_onednn:
                    cmd += [f"-L{onednn_lib}", "-ldnnl",
                            "-Wl,-rpath," + onednn_lib]
                cmd += [
                    "-Wl,-rpath," + str(torch_lib),
                    "-o", str(output_path),
                ] + [str(s) for s in sources]
        
        print(f"Compile command: {' '.join(cmd)}")
        
        # Run compiler
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            raise RuntimeError(f"Compilation failed with exit code {result.returncode}")
        
        print(f"Successfully built {output_path}")


class ICPXExtension(Extension):
    """Extension that will be built with icpx."""
    
    def __init__(self, name, sourcedir=""):
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


# Read version
def get_version():
    version_file = Path(__file__).parent / "omni_xpu_kernel" / "__init__.py"
    with open(version_file) as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"\'')
    return "0.1.0"


# Read README
def get_long_description():
    readme = Path(__file__).parent / "README.md"
    if readme.exists():
        return readme.read_text(encoding="utf-8")
    return ""


setup(
    name="omni_xpu_kernel",
    version=get_version(),
    author="Intel",
    author_email="",
    description="High-performance Intel XPU kernels for PyTorch",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/intel/omni_xpu_kernel",
    packages=find_packages(exclude=["tests", "scripts"]),
    ext_modules=[
        ICPXExtension("omni_xpu_kernel._C", sourcedir="."),
        ICPXExtension("omni_xpu_kernel.lgrf_uni.lgrf_sdp", sourcedir=".")
    ],
    cmdclass={"build_ext": ICPXBuildExt},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0",
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
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
    ],
    keywords="intel xpu sycl esimd pytorch gpu kernels quantization gguf",
)
