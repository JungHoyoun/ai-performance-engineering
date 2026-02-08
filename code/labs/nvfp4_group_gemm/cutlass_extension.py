"""CUDA extension loader for CUTLASS SM100 NVFP4 block-scaled grouped GEMM."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from core.utils.extension_loader_template import load_cuda_extension

_EXT_NAME = "nvfp4_group_gemm_cutlass_sm100"
_EXT: Optional[object] = None


def load_cutlass_nvfp4_grouped_gemm_sm100(*, verbose: bool = False) -> object:
    """Load (and JIT-build if needed) the CUTLASS NVFP4 grouped GEMM extension."""
    global _EXT
    if _EXT is not None:
        return _EXT

    lab_dir = Path(__file__).resolve().parent
    source = lab_dir / "cutlass_nvfp4_grouped_gemm_sm100.cu"

    cutlass_util_inc = REPO_ROOT / "third_party" / "cutlass" / "tools" / "util" / "include"
    if not cutlass_util_inc.exists():
        raise FileNotFoundError(f"Missing CUTLASS util include dir: {cutlass_util_inc}")

    extra_cuda_cflags = [
        "--std=c++17",
        "-O3",
        "--use_fast_math",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-lineinfo",
        # B200 (SM100). We keep this narrow to reduce compile time.
        "-gencode=arch=compute_100a,code=sm_100a",
    ]

    _EXT = load_cuda_extension(
        extension_name=_EXT_NAME,
        cuda_source_file=str(source),
        include_dirs=[cutlass_util_inc],
        extra_cuda_cflags=extra_cuda_cflags,
        extra_ldflags=["-lcuda"],
        verbose=verbose,
    )
    return _EXT


__all__ = ["load_cutlass_nvfp4_grouped_gemm_sm100"]
