"""CUDA extension loader for experimental SM100 NVFP4 dynamic blockscaled grouped GEMM."""

from __future__ import annotations

import builtins
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from core.utils.extension_loader_template import load_cuda_extension

_EXT_NAME = "nvfp4_group_gemm_cutlass_sm100_dyn"
_EXT: Optional[object] = None


def _get_process_extension_cache() -> dict[str, object]:
    cache_name = "_AISP_EXT_PROCESS_CACHE"
    cache = getattr(builtins, cache_name, None)
    if not isinstance(cache, dict):
        cache = {}
        setattr(builtins, cache_name, cache)
    return cache


def load_cutlass_nvfp4_grouped_gemm_sm100_dyn(*, verbose: bool = False) -> object:
    """Load (and JIT-build if needed) the experimental dynamic-schedule extension."""
    global _EXT
    if _EXT is not None:
        return _EXT

    process_cache = _get_process_extension_cache()
    cached = process_cache.get(_EXT_NAME)
    if cached is not None:
        _EXT = cached
        return _EXT
    if _EXT_NAME in sys.modules:
        _EXT = sys.modules[_EXT_NAME]
        process_cache[_EXT_NAME] = _EXT
        return _EXT

    lab_dir = Path(__file__).resolve().parent
    source = lab_dir / "cutlass_nvfp4_grouped_gemm_sm100_dyn.cu"

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

    process_cache[_EXT_NAME] = _EXT
    sys.modules[_EXT_NAME] = _EXT
    return _EXT


__all__ = ["load_cutlass_nvfp4_grouped_gemm_sm100_dyn"]
