# Copyright (c) 2026 Laboratory of Sensing and Networking Systems, EPFL
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
import glob
import os

from setuptools import find_packages, setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


REQUIRED_OPS = ("RFSignalTracer", "RFMatchedFilter")


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    ops_dir = os.path.join("geraf", "ops")
    ext_modules = []

    for op_name in REQUIRED_OPS:
        ext_dir = os.path.join(ops_dir, op_name, "csrc")
        ext_dir_abs = os.path.join(this_dir, ext_dir)
        if not os.path.exists(ext_dir_abs):
            continue

        sources = glob.glob(os.path.join(ext_dir, "**", "*.cpp"), recursive=True)
        sources += glob.glob(os.path.join(ext_dir, "**", "*.cu"), recursive=True)
        if not sources:
            continue

        if not (torch.cuda.is_available() or os.getenv("FORCE_CUDA", "0") == "1"):
            continue

        extension = CUDAExtension(
            name=f"geraf.ops.{op_name}._C",
            sources=sources,
            include_dirs=[ext_dir],
            define_macros=[("WITH_CUDA", None)],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "-gencode=arch=compute_70,code=sm_70",
                    "-gencode=arch=compute_80,code=sm_80",
                    "-gencode=arch=compute_86,code=sm_86",
                    "-gencode=arch=compute_89,code=sm_89",
                    "-gencode=arch=compute_90,code=sm_90",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                ],
            },
        )

        ext_modules.append(extension)

    return ext_modules


setup(
    name="geraf",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
