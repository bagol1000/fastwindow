"""
fastwindow — build script (pybind11 extension).

Mirrors the pattern used by quitefastmst for OpenMP + platform detection.
Compile flags: -O3 -march=native -ffast-math -fopenmp -std=c++17
AVX2 paths live inside #ifdef __AVX2__ guards; scalar fallback always present.
"""

import glob
import os
import sys

import setuptools
from setuptools.command.build_ext import build_ext
import pybind11
import numpy as np


class FastwindowBuildExt(build_ext):
    """Inject platform-specific compiler flags after setuptools decides the compiler."""

    def build_extensions(self):
        if hasattr(self.compiler, "compiler"):
            compiler = self.compiler.compiler[0]
        else:
            compiler = self.compiler.__class__.__name__

        base_flags  = ["-O3", "-ffast-math", "-std=c++17"]
        #FASTWINDOW_PORTABLE: set when building distributable wheels —
        #-march=native would emit ISA the target machine may not have.
        #macOS is excluded too: universal2 builds compile every file for
        #both -arch slices, and -march=native resolves to the host CPU
        #(e.g. apple-m3), which the x86_64 slice rejects; the AVX2 kernels
        #are x86-only anyway.
        march_flag  = []
        if not os.environ.get("FASTWINDOW_PORTABLE") \
                and sys.platform != "darwin":
            march_flag = ["-march=native"]

        for ext in self.extensions:
            ext.extra_compile_args = list(ext.extra_compile_args)
            ext.extra_link_args    = list(ext.extra_link_args)

            if sys.platform == "win32" and ("icc" in compiler or "icl" in compiler):
                ext.extra_compile_args += ["/Qopenmp", "/Qstd=c++17", "/O2"]
                ext.extra_link_args    += ["/Qopenmp"]
            elif sys.platform == "win32":
                ext.extra_compile_args += ["/openmp", "/std:c++17", "/O2"]
                ext.extra_link_args    += ["/openmp"]
            elif sys.platform == "darwin" and ("icc" in compiler or "icl" in compiler):
                ext.extra_compile_args += base_flags + march_flag + ["-openmp"]
                ext.extra_link_args    += ["-openmp"]
            elif sys.platform == "darwin":
                #Apple clang: fopenmp needs Homebrew libomp; omit to keep CI green
                ext.extra_compile_args += base_flags + march_flag
            else:
                #Linux / generic GCC or clang
                ext.extra_compile_args += base_flags + march_flag + ["-fopenmp"]
                ext.extra_link_args    += ["-fopenmp"]

        build_ext.build_extensions(self)


src_files = [
    os.path.join("src", "rolling_basic.cpp"),
    os.path.join("src", "rolling_regression.cpp"),
    os.path.join("src", "rolling_regression_multi.cpp"),
    os.path.join("src", "rolling_corr.cpp"),
    os.path.join("src", "rolling_corr_matrix.cpp"),
    os.path.join("src", "rolling_quantile.cpp"),
    os.path.join("src", "expanding.cpp"),
    os.path.join("src", "rolling_matrix.cpp"),
    os.path.join("src", "rolling_spearman.cpp"),
    os.path.join("src", "bindings_python.cpp"),
]

include_dirs = [
    pybind11.get_include(),
    np.get_include(),
    "src",
]

ext_kwargs = dict(
    sources=src_files,
    include_dirs=include_dirs,
    language="c++",
    define_macros=[("FASTWINDOW_PYTHON", "1")],
    extra_compile_args=[],   #filled by FastwindowBuildExt
    extra_link_args=[],
    depends=glob.glob(os.path.join("src", "*.h")),
)

setuptools.setup(
    packages=setuptools.find_packages(exclude=["tests", "tests.*"]),
    package_data={"fastwindow": ["py.typed", "*.pyi"]},
    cmdclass={"build_ext": FastwindowBuildExt},
    ext_modules=[
        setuptools.Extension("fastwindow._core", **ext_kwargs)
    ],
)
