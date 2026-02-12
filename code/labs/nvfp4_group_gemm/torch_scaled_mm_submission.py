"""Vendor-backed reference/ceiling paths via PyTorch scaled_mm (block-scaled FP4).

This module provides *target/ceiling* implementations for local harness tuning:
- `torch._scaled_mm` ("v1"): matches Popcorn's `reference.py` semantics. Correctness-first.
- `torch._scaled_mm_v2` ("v2"): cuBLASLt-backed API with `out=` support. Treat as experimental
  until it passes verification end-to-end for our inputs/layouts.

Layout note:
Popcorn's reference uses `sfa/sfb` in the blocked 128x4 layout (flattened) via `to_blocked()`.
We compute that conversion in `prepare_*()` so timing measures steady-state GEMM, not layout
conversion overhead.
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


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _to_blocked_v1(input_matrix: torch.Tensor) -> torch.Tensor:
    """Match Popcorn reference `to_blocked()` for torch._scaled_mm scale tensors."""
    rows, cols = input_matrix.shape
    n_row_blocks = _ceil_div(int(rows), 128)
    n_col_blocks = _ceil_div(int(cols), 4)
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4

    if padded_rows != rows or padded_cols != cols:
        padded = torch.nn.functional.pad(
            input_matrix,
            (0, padded_cols - cols, 0, padded_rows - rows),
            mode="constant",
            value=0,
        )
    else:
        padded = input_matrix

    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    return rearranged.flatten()


def prepare_torch_scaled_mm_v1(data_list: Sequence[input_t]) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare per-input scale tensors for torch._scaled_mm (outside timed region)."""
    if not data_list:
        return None

    out: list[tuple[Any, ...]] = []
    for data in data_list:
        abc_tensors, sfasfb_tensors, sfasfb_reordered_tensors, problem_sizes = data
        scale_a_blocked: list[torch.Tensor] = []
        scale_b_blocked: list[torch.Tensor] = []
        for sfa_cpu, sfb_cpu in sfasfb_tensors:
            # sfa/sfb start on CPU; convert to blocked layout on CPU, then stage once to CUDA.
            scale_a_blocked.append(_to_blocked_v1(sfa_cpu[:, :, 0]).cuda())
            scale_b_blocked.append(_to_blocked_v1(sfb_cpu[:, :, 0]).cuda())
        ctx = {"scale_a_blocked": scale_a_blocked, "scale_b_blocked": scale_b_blocked}
        out.append((abc_tensors, sfasfb_tensors, sfasfb_reordered_tensors, problem_sizes, ctx))
    return out


def custom_kernel_scaled_mm_v1(data: tuple[Any, ...]) -> output_t:
    """Execute grouped block-scaled GEMM using torch._scaled_mm per group (Popcorn reference semantics)."""
    abc_tensors, sfasfb_tensors, _sfasfb_reordered_tensors, _problem_sizes, *rest = data
    ctx = rest[0] if rest else None

    if ctx is not None:
        scale_a_blocked = ctx["scale_a_blocked"]
        scale_b_blocked = ctx["scale_b_blocked"]
    else:
        scale_a_blocked = [_to_blocked_v1(sfa[:, :, 0]).cuda() for (sfa, _sfb) in sfasfb_tensors]
        scale_b_blocked = [_to_blocked_v1(sfb[:, :, 0]).cuda() for (_sfa, sfb) in sfasfb_tensors]

    out: list[torch.Tensor] = []
    for idx, ((a, b, c), _sf) in enumerate(zip(abc_tensors, sfasfb_tensors)):
        A = a[:, :, 0].view(torch.float4_e2m1fn_x2)
        B = b[:, :, 0].transpose(0, 1).view(torch.float4_e2m1fn_x2)
        res = torch._scaled_mm(  # pylint: disable=protected-access
            A,
            B,
            scale_a_blocked[idx],
            scale_b_blocked[idx],
            bias=None,
            out_dtype=torch.float16,
        )
        c[:, :, 0].copy_(res)
        out.append(c)
    return out


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


__all__ = [
    "prepare_torch_scaled_mm_v1",
    "custom_kernel_scaled_mm_v1",
    "prepare_torch_scaled_mm_v2",
    "custom_kernel_scaled_mm_v2",
]
