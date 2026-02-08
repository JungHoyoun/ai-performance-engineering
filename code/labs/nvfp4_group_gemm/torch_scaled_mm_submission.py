"""cuBLASLt-backed ceiling paths via PyTorch scaled_mm v2 (block-scaled FP4).

This is a *target/ceiling* implementation for local harness tuning:
- Uses PyTorch's NVFP4 block-scaled scaled_mm v2 APIs (cuBLASLt-backed).
- Not intended as the final Popcorn submission (we assume non-cuBLAS is required).

Correctness note:
GPU MODE reference-kernels generate `sfa_reordered/sfb_reordered` in the cuBLAS
block-scaled swizzled layout. PyTorch expects the *base* swizzle storage order
when you pass `SwizzleType.SWIZZLE_32_4_4`, so we permute back to base order and
materialize a contiguous tensor in setup().
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import torch
from torch.nn.functional import ScalingType, SwizzleType

from labs.nvfp4_group_gemm.task import input_t, output_t


_RECIPE = [int(ScalingType.BlockWise1x16.value), int(ScalingType.TensorWise.value)]
_SWIZZLE = [int(SwizzleType.SWIZZLE_32_4_4.value)]

_GLOBAL_SCALE: Optional[torch.Tensor] = None


def _get_global_scale() -> torch.Tensor:
    global _GLOBAL_SCALE
    if _GLOBAL_SCALE is None or (not _GLOBAL_SCALE.is_cuda):
        _GLOBAL_SCALE = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    return _GLOBAL_SCALE


def _to_base_swizzle_contiguous(scale_reordered: torch.Tensor) -> torch.Tensor:
    """Convert (32,4,rest_m,4,rest_k,1) -> base order contiguous for PyTorch scaled_mm_v2."""
    # Reordered dims: (mm32, mm4, mm, kk4, kk, b)
    # Base dims expected by PyTorch for SWIZZLE_32_4_4: (b, mm, kk, mm32, mm4, kk4)
    return scale_reordered.permute(5, 2, 4, 0, 1, 3).contiguous()


def prepare_torch_scaled_mm_v2(data_list: Sequence[input_t]) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare per-input scale tensors for scaled_mm_v2 (outside timed region)."""
    if not data_list:
        return None
    global_scale = _get_global_scale()

    out: list[tuple[Any, ...]] = []
    for data in data_list:
        abc_tensors, sfasfb_tensors, sfasfb_reordered_tensors, problem_sizes = data
        scale_a_base: list[torch.Tensor] = []
        scale_b_base: list[torch.Tensor] = []
        for sfa_reordered, sfb_reordered in sfasfb_reordered_tensors:
            scale_a_base.append(_to_base_swizzle_contiguous(sfa_reordered))
            scale_b_base.append(_to_base_swizzle_contiguous(sfb_reordered))
        ctx = {
            "scale_a_base": scale_a_base,
            "scale_b_base": scale_b_base,
            "global_scale": global_scale,
        }
        out.append((abc_tensors, sfasfb_tensors, sfasfb_reordered_tensors, problem_sizes, ctx))
    return out


def custom_kernel_scaled_mm_v2(data: tuple[Any, ...]) -> output_t:
    """Execute grouped block-scaled GEMM using torch._scaled_mm_v2 per group."""
    abc_tensors, _sfasfb_tensors, sfasfb_reordered_tensors, _problem_sizes, *rest = data
    ctx = rest[0] if rest else None

    if ctx is not None:
        scale_a_base = ctx["scale_a_base"]
        scale_b_base = ctx["scale_b_base"]
        global_scale = ctx["global_scale"]
    else:
        scale_a_base = [_to_base_swizzle_contiguous(sfa) for (sfa, _sfb) in sfasfb_reordered_tensors]
        scale_b_base = [_to_base_swizzle_contiguous(sfb) for (_sfa, sfb) in sfasfb_reordered_tensors]
        global_scale = _get_global_scale()

    out: list[torch.Tensor] = []
    for idx, ((a, b, c), _sf) in enumerate(zip(abc_tensors, sfasfb_reordered_tensors)):
        # A: [M, K//2, 1] packed FP4 -> [M, K//2]
        A = a[:, :, 0].view(torch.float4_e2m1fn_x2)
        # B: [N, K//2, 1] packed FP4. Pass B^T view for GEMM (K//2, N).
        B = b[:, :, 0].transpose(0, 1).view(torch.float4_e2m1fn_x2)

        # Write directly into provided C buffer (no extra allocations/copies in the hot path).
        torch._scaled_mm_v2(  # pylint: disable=protected-access
            A,
            B,
            [scale_a_base[idx], global_scale],
            _RECIPE,
            _SWIZZLE,
            [scale_b_base[idx], global_scale],
            _RECIPE,
            _SWIZZLE,
            bias=None,
            out_dtype=torch.float16,
            out=c[:, :, 0],
        )
        out.append(c)

    return out


__all__ = ["prepare_torch_scaled_mm_v2", "custom_kernel_scaled_mm_v2"]
