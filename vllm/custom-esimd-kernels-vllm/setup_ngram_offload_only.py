"""仅构建 NGram pinned-host lookup，不覆盖其它已验证扩展。"""

from esimd_build_extention import BuildExtension
from ngram_offload_build import make_ngram_offload_extension
from setuptools import setup

setup(
    name="custom-esimd-kernels-vllm-ngram-offload-only",
    version="0.1.0",
    ext_modules=[make_ngram_offload_extension()],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
