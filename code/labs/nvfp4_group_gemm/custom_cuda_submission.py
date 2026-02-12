"""Custom CUDA NVFP4 grouped GEMM submission (strict non-DP4A path)."""

from __future__ import annotations

import builtins
import os
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from core.utils.extension_loader_template import load_cuda_extension
from labs.nvfp4_group_gemm.task import input_t, output_t

_EXT_NAME = "nvfp4_group_gemm_custom_cuda_ws_grouped_v34_nodp4a_fp16grouped"
_EXT: Optional[object] = None
_FP4_LUT_F16: Optional[torch.Tensor] = None


def _get_process_extension_cache() -> dict[str, object]:
    cache_name = "_AISP_EXT_PROCESS_CACHE"
    cache = getattr(builtins, cache_name, None)
    if not isinstance(cache, dict):
        cache = {}
        setattr(builtins, cache_name, cache)
    return cache


def load_custom_cuda_nvfp4_group_gemm(*, verbose: bool = False) -> object:
    """Load (and JIT-build if needed) the custom CUDA NVFP4 grouped GEMM extension."""
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
    source = lab_dir / "custom_cuda_group_gemm_kernel.cu"
    build_dir = REPO_ROOT / ".torch_extensions" / _EXT_NAME
    extra_cuda_cflags = [
        "--std=c++17",
        "-O3",
        "--use_fast_math",
        "-lineinfo",
        "-gencode=arch=compute_100a,code=sm_100a",
        "-gencode=arch=compute_100a,code=compute_100a",
    ]
    _EXT = load_cuda_extension(
        extension_name=_EXT_NAME,
        cuda_source_file=str(source),
        build_dir=build_dir,
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=verbose,
    )
    process_cache[_EXT_NAME] = _EXT
    sys.modules[_EXT_NAME] = _EXT
    return _EXT


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _fp4_lut_f16() -> torch.Tensor:
    global _FP4_LUT_F16
    if _FP4_LUT_F16 is None or _FP4_LUT_F16.device.type != "cuda":
        # E2M1 quantized values scaled by 0.5.
        _FP4_LUT_F16 = torch.tensor(
            [
                0.0,
                0.5,
                1.0,
                1.5,
                2.0,
                3.0,
                4.0,
                6.0,
                0.0,
                -0.5,
                -1.0,
                -1.5,
                -2.0,
                -3.0,
                -4.0,
                -6.0,
            ],
            dtype=torch.float16,
            device="cuda",
        )
    return _FP4_LUT_F16


def _dequantize_fp4_to_fp16(packed_u8: torch.Tensor, scale_half: torch.Tensor) -> torch.Tensor:
    """Dequantize packed FP4 [M, K/2] and apply per-16 scale [M, K/16] into FP16 [M, K]."""
    if packed_u8.dtype != torch.uint8 or packed_u8.dim() != 2:
        raise ValueError("packed_u8 must be contiguous 2D torch.uint8")
    if scale_half.dtype != torch.float16 or scale_half.dim() != 2:
        raise ValueError("scale_half must be contiguous 2D torch.float16")
    if not packed_u8.is_contiguous() or not scale_half.is_contiguous():
        raise ValueError("packed_u8 and scale_half must be contiguous")

    m_size = int(packed_u8.size(0))
    k_half = int(packed_u8.size(1))
    k_size = int(k_half * 2)
    if int(scale_half.size(0)) != m_size:
        raise ValueError("scale_half M dimension mismatch")
    if int(scale_half.size(1)) != ((k_size + 15) // 16):
        raise ValueError("scale_half K-scale dimension mismatch")

    lut = _fp4_lut_f16()
    lo = lut[(packed_u8 & 0x0F).to(torch.int64)]
    hi = lut[((packed_u8 >> 4) & 0x0F).to(torch.int64)]
    deq = torch.empty((m_size, k_size), dtype=torch.float16, device=packed_u8.device)
    deq[:, 0::2] = lo
    deq[:, 1::2] = hi

    scale_expanded = scale_half.repeat_interleave(16, dim=1)
    if int(scale_expanded.size(1)) != k_size:
        scale_expanded = scale_expanded[:, :k_size]
    return (deq * scale_expanded).contiguous()


def prepare_custom_cuda(data_list: Sequence[input_t]) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare grouped pointer metadata outside the timed benchmark path."""
    if not data_list:
        return None

    if _env_bool("AISP_NVFP4_GROUP_GEMM_CUSTOM_CASE1_FAST", False):
        raise ValueError("DP4A case1-fast path is disabled in strict non-DP4A mode")

    prepared: list[tuple[Any, ...]] = []
    for data in data_list:
        abc_tensors, sfasfb_tensors, sfasfb_reordered_tensors, problem_sizes = data
        groups: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        output_refs: list[torch.Tensor] = []
        a_half_ptrs_cpu: list[int] = []
        b_half_ptrs_cpu: list[int] = []
        c_ptrs_cpu: list[int] = []
        m_sizes: list[int] = []
        n_sizes: list[int] = []
        k_sizes: list[int] = []

        for (a, b, c), (sfa_cpu, sfb_cpu) in zip(abc_tensors, sfasfb_tensors):
            if a.dim() != 3 or b.dim() != 3 or c.dim() != 3:
                raise ValueError("Expected A/B/C tensors with shape [M|N, K/2|N, 1]")
            if a.size(2) != 1 or b.size(2) != 1 or c.size(2) != 1:
                raise ValueError("Only l=1 inputs are supported in custom CUDA path")

            a_u8 = a[:, :, 0].view(torch.uint8)
            b_u8 = b[:, :, 0].view(torch.uint8)
            c_out = c[:, :, 0]
            if not a_u8.is_contiguous() or not b_u8.is_contiguous() or not c_out.is_contiguous():
                raise ValueError("Expected contiguous A/B/C views in custom CUDA path")

            # Keep dequantization out of the timed path.
            sfa_half = sfa_cpu[:, :, 0].to(device="cuda", dtype=torch.float16, non_blocking=False).contiguous()
            sfb_half = sfb_cpu[:, :, 0].to(device="cuda", dtype=torch.float16, non_blocking=False).contiguous()
            a_half = _dequantize_fp4_to_fp16(a_u8, sfa_half)
            b_half = _dequantize_fp4_to_fp16(b_u8, sfb_half)

            groups.append((a_half, b_half, c_out))
            output_refs.append(c)
            a_half_ptrs_cpu.append(int(a_half.data_ptr()))
            b_half_ptrs_cpu.append(int(b_half.data_ptr()))
            c_ptrs_cpu.append(int(c_out.data_ptr()))
            m_sizes.append(int(a_half.size(0)))
            n_sizes.append(int(b_half.size(0)))
            k_sizes.append(int(a_half.size(1)))

        grouped_ctx = {
            "a_half_ptrs_cpu": torch.tensor(a_half_ptrs_cpu, dtype=torch.int64, device="cpu").contiguous(),
            "b_half_ptrs_cpu": torch.tensor(b_half_ptrs_cpu, dtype=torch.int64, device="cpu").contiguous(),
            "c_ptrs_cpu": torch.tensor(c_ptrs_cpu, dtype=torch.int64, device="cpu").contiguous(),
            "m_sizes_cpu": torch.tensor(m_sizes, dtype=torch.int32, device="cpu").contiguous(),
            "uniform_n_size": int(n_sizes[0]) if n_sizes and all(v == n_sizes[0] for v in n_sizes) else None,
            "uniform_k_size": int(k_sizes[0]) if k_sizes and all(v == k_sizes[0] for v in k_sizes) else None,
        }

        ctx = {"groups": groups, "grouped_ctx": grouped_ctx, "output_refs": output_refs}
        prepared.append((abc_tensors, sfasfb_tensors, sfasfb_reordered_tensors, problem_sizes, ctx))

    return prepared


def custom_kernel_custom_cuda(data: tuple[Any, ...]) -> output_t:
    """Run grouped GEMM via strict non-DP4A FP16 grouped cuBLAS path."""
    if len(data) < 5:
        raise ValueError("custom_kernel_custom_cuda requires prepare_custom_cuda() output")
    ctx = data[4]

    if _env_bool("AISP_NVFP4_GROUP_GEMM_CUSTOM_CASE1_FAST", False):
        raise ValueError("DP4A case1-fast path is disabled in strict non-DP4A mode")

    ext = load_custom_cuda_nvfp4_group_gemm()
    grouped_ctx = ctx["grouped_ctx"]
    uniform_n_size = grouped_ctx["uniform_n_size"]
    uniform_k_size = grouped_ctx["uniform_k_size"]
    if uniform_n_size is None or uniform_k_size is None:
        raise ValueError("non-DP4A grouped FP16 path requires uniform N/K across groups")

    ext.nvfp4_group_gemm_forward_grouped_fp16_cublas_cuda(
        grouped_ctx["a_half_ptrs_cpu"],
        grouped_ctx["b_half_ptrs_cpu"],
        grouped_ctx["c_ptrs_cpu"],
        grouped_ctx["m_sizes_cpu"],
        int(uniform_n_size),
        int(uniform_k_size),
    )
    return ctx["output_refs"]


__all__ = [
    "custom_kernel_custom_cuda",
    "load_custom_cuda_nvfp4_group_gemm",
    "prepare_custom_cuda",
]
