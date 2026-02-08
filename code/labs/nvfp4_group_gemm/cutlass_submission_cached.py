"""Allocation-cached CUTLASS NVFP4 block-scaled grouped GEMM submission.

This uses a single-launch CUTLASS grouped GEMM kernel (device-side scheduling) and keeps all
metadata and pointer-array allocations in setup() via prepare_cutlass_cached().
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import torch

from labs.nvfp4_group_gemm.cutlass_extension import load_cutlass_nvfp4_grouped_gemm_sm100
from labs.nvfp4_group_gemm.task import input_t, output_t


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or v == "":
        return int(default)
    return int(v)


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None or v == "":
        return bool(default)
    return v.lower() in {"1", "true", "yes", "y", "on"}


# Per-case cache keyed by exact (m, n, k, l) tuples for all groups in the case.
_KernelVariant = Literal["1sm", "1sm_n64", "1sm_n128", "2sm", "2sm_n64", "2sm_n128"]
_CaseKey = Tuple[_KernelVariant, int, int, int, bool, Tuple[Tuple[int, int, int, int], ...]]
_CASE_CACHE: Dict[_CaseKey, Dict[str, Any]] = {}


def _prepare_cutlass_cached(
    data_list: Sequence[input_t], *, variant: _KernelVariant
) -> Optional[Sequence[tuple[Any, ...]]]:
    """Precompute CUTLASS metadata + pointer arrays for each input in data_list.

    Returns a replacement data_list whose elements have an extra trailing `ctx` dict, consumed by
    custom_kernel_cutlass_cached().
    """
    if not data_list:
        return None

    # All items share the same per-case problem sizes.
    problem_sizes = data_list[0][3]
    problem_sizes_key = tuple(tuple(int(x) for x in entry) for entry in problem_sizes)

    # Tunables (kept explicit, no global default changes).
    #
    # CUTLASS 2SM kernels require cluster_dim.x >= 2 (see CUTLASS example).
    default_cluster_m = 2 if variant in {"2sm", "2sm_n64", "2sm_n128"} else 1
    cluster_m = _env_int("AISP_NVFP4_GROUP_GEMM_CLUSTER_M", default_cluster_m)
    cluster_n = _env_int("AISP_NVFP4_GROUP_GEMM_CLUSTER_N", 1)
    raster_order = _env_int("AISP_NVFP4_GROUP_GEMM_RASTER_ORDER", 0)
    use_pdl = _env_bool("AISP_NVFP4_GROUP_GEMM_USE_PDL", False)

    key: _CaseKey = (
        variant,
        int(cluster_m),
        int(cluster_n),
        int(raster_order),
        bool(use_pdl),
        problem_sizes_key,
    )

    ext = load_cutlass_nvfp4_grouped_gemm_sm100(verbose=False)

    if key not in _CASE_CACHE:
        ps_cpu = torch.tensor(problem_sizes, dtype=torch.int32, device="cpu")
        if variant == "1sm":
            build_metadata = ext.build_metadata_1sm
        elif variant == "1sm_n64":
            build_metadata = ext.build_metadata_1sm_n64
        elif variant == "1sm_n128":
            build_metadata = ext.build_metadata_1sm_n128
        elif variant == "2sm":
            build_metadata = ext.build_metadata_2sm
        elif variant == "2sm_n64":
            build_metadata = ext.build_metadata_2sm_n64
        else:
            build_metadata = ext.build_metadata_2sm_n128
        (
            problem_shapes_u8,
            stride_a_u8,
            stride_b_u8,
            stride_c_u8,
            stride_d_u8,
            layout_sfa_u8,
            layout_sfb_u8,
            workspace_u8,
        ) = build_metadata(ps_cpu, cluster_m, cluster_n, raster_order)
        _CASE_CACHE[key] = {
            "problem_shapes_u8": problem_shapes_u8,
            "stride_a_u8": stride_a_u8,
            "stride_b_u8": stride_b_u8,
            "stride_c_u8": stride_c_u8,
            "stride_d_u8": stride_d_u8,
            "layout_sfa_u8": layout_sfa_u8,
            "layout_sfb_u8": layout_sfb_u8,
            "workspace_u8": workspace_u8,
            "cluster_m": int(cluster_m),
            "cluster_n": int(cluster_n),
            "raster_order": int(raster_order),
            "use_pdl": bool(use_pdl),
        }

    case_ctx = _CASE_CACHE[key]

    out: List[tuple[Any, ...]] = []
    for data in data_list:
        abc_tensors, sfasfb_tensors, sfasfb_reordered_tensors, problem_sizes_i = data
        if problem_sizes_i != problem_sizes:
            raise ValueError("prepare_cutlass_cached() expects all inputs to share identical problem_sizes")

        a_ptrs: List[int] = []
        b_ptrs: List[int] = []
        c_ptrs: List[int] = []
        sfa_ptrs: List[int] = []
        sfb_ptrs: List[int] = []

        for (a, b, c), (sfa_reordered, sfb_reordered) in zip(abc_tensors, sfasfb_reordered_tensors):
            a_ptrs.append(int(a.data_ptr()))
            b_ptrs.append(int(b.data_ptr()))
            c_ptrs.append(int(c.data_ptr()))
            sfa_ptrs.append(int(sfa_reordered.data_ptr()))
            sfb_ptrs.append(int(sfb_reordered.data_ptr()))

        ctx = {
            "case": case_ctx,
            # Keep tensors alive; CUTLASS reads these device pointer arrays.
            "ptr_a_i64": torch.tensor(a_ptrs, dtype=torch.int64, device="cuda"),
            "ptr_b_i64": torch.tensor(b_ptrs, dtype=torch.int64, device="cuda"),
            "ptr_c_i64": torch.tensor(c_ptrs, dtype=torch.int64, device="cuda"),
            "ptr_d_i64": torch.tensor(c_ptrs, dtype=torch.int64, device="cuda"),  # in-place D -> C
            "ptr_sfa_i64": torch.tensor(sfa_ptrs, dtype=torch.int64, device="cuda"),
            "ptr_sfb_i64": torch.tensor(sfb_ptrs, dtype=torch.int64, device="cuda"),
        }

        # Pre-initialize CUTLASS params once per input to avoid timing host-side initialization.
        if variant == "1sm":
            create_plan = ext.create_plan_1sm
        elif variant == "1sm_n64":
            create_plan = ext.create_plan_1sm_n64
        elif variant == "1sm_n128":
            create_plan = ext.create_plan_1sm_n128
        elif variant == "2sm":
            create_plan = ext.create_plan_2sm
        elif variant == "2sm_n64":
            create_plan = ext.create_plan_2sm_n64
        else:
            create_plan = ext.create_plan_2sm_n128
        ctx["plan"] = create_plan(
            case_ctx["problem_shapes_u8"],
            case_ctx["stride_a_u8"],
            case_ctx["stride_b_u8"],
            case_ctx["stride_c_u8"],
            case_ctx["stride_d_u8"],
            case_ctx["layout_sfa_u8"],
            case_ctx["layout_sfb_u8"],
            case_ctx["workspace_u8"],
            ctx["ptr_a_i64"],
            ctx["ptr_b_i64"],
            ctx["ptr_sfa_i64"],
            ctx["ptr_sfb_i64"],
            ctx["ptr_c_i64"],
            ctx["ptr_d_i64"],
            1.0,
            0.0,
            case_ctx["raster_order"],
            case_ctx["cluster_m"],
            case_ctx["cluster_n"],
            case_ctx["use_pdl"],
        )

        out.append((abc_tensors, sfasfb_tensors, sfasfb_reordered_tensors, problem_sizes, ctx))

    return out


def prepare_cutlass_cached(data_list: Sequence[input_t]) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare using the CUTLASS 1SM MMA kernel (default)."""
    return _prepare_cutlass_cached(data_list, variant="1sm")

def prepare_cutlass_cached_1sm_n64(data_list: Sequence[input_t]) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare using the CUTLASS 1SM MMA kernel (N=64 tile)."""
    return _prepare_cutlass_cached(data_list, variant="1sm_n64")

def prepare_cutlass_cached_1sm_n128(data_list: Sequence[input_t]) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare using the CUTLASS 1SM MMA kernel (N=128 tile)."""
    return _prepare_cutlass_cached(data_list, variant="1sm_n128")

def prepare_cutlass_cached_2sm(data_list: Sequence[input_t]) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare using the CUTLASS 2SM MMA kernel."""
    return _prepare_cutlass_cached(data_list, variant="2sm")

def prepare_cutlass_cached_2sm_n64(data_list: Sequence[input_t]) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare using the CUTLASS 2SM MMA kernel (N=64 tile)."""
    return _prepare_cutlass_cached(data_list, variant="2sm_n64")

def prepare_cutlass_cached_2sm_n128(data_list: Sequence[input_t]) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare using the CUTLASS 2SM MMA kernel (N=128 tile)."""
    return _prepare_cutlass_cached(data_list, variant="2sm_n128")

def custom_kernel_cutlass_cached(data: tuple[Any, ...]) -> output_t:
    """Execute the CUTLASS NVFP4 grouped GEMM kernel with cached allocations."""
    abc_tensors, _, _, _, ctx = data
    plan = ctx["plan"]
    plan.run()

    num_groups = len(abc_tensors)
    return [abc_tensors[i][2] for i in range(num_groups)]


__all__ = [
    "prepare_cutlass_cached",
    "prepare_cutlass_cached_1sm_n64",
    "prepare_cutlass_cached_1sm_n128",
    "prepare_cutlass_cached_2sm",
    "prepare_cutlass_cached_2sm_n64",
    "prepare_cutlass_cached_2sm_n128",
    "custom_kernel_cutlass_cached",
]
