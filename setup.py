import os

from setuptools import find_packages, setup

try:
    import torch  # noqa: F401
    from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension
except ImportError as exc:
    raise RuntimeError(
        "PyTorch must be installed before building protein_search_gpu. "
        "Install torch first, then run `pip install --no-build-isolation -e .`"
    ) from exc

# The extension only needs the CUDA toolkit (nvcc) at build time, not a visible
# GPU, so check CUDA_HOME rather than torch.cuda.is_available().
if CUDA_HOME is None:
    raise RuntimeError(
        "CUDA toolkit not found (CUDA_HOME is unset and nvcc is not on PATH). "
        "protein_search_gpu requires the CUDA toolkit matching your PyTorch "
        "build to compile its extension."
    )

if os.name == "nt":
    cxx_flags = ["/O2"]
else:
    cxx_flags = ["-O3"]

ext_modules = [
    CUDAExtension(
        name="protein_search_gpu._C",
        sources=[
            "bindings/torch_extension.cpp",
            "cuda/normalize.cu",
            "cuda/cosine.cu",
            "cuda/topk.cu",
            "cuda/gather.cu",
            "cuda/smith_waterman.cu",
        ],
        include_dirs=[os.path.join(os.path.dirname(os.path.abspath(__file__)), "cuda")],
        extra_compile_args={"cxx": cxx_flags, "nvcc": ["-O3"]},
    ),
]

setup(
    name="protein_search_gpu",
    version="0.1.0",
    description="CUDA-accelerated semantic protein retrieval (PyTorch C++/CUDA extension)",
    author="Vineet Kowti",
    license="MIT",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    install_requires=[
        "torch>=2.0",
        "numpy>=1.21",
        "transformers>=4.30",
        "tqdm>=4.65",
    ],
    extras_require={"test": ["pytest>=7"]},
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "Environment :: GPU :: NVIDIA CUDA",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
