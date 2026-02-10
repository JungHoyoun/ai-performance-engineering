"""Allocation-cached CUTLASS NVFP4 block-scaled grouped GEMM submission.

This uses a single-launch CUTLASS grouped GEMM kernel (device-side scheduling) and keeps all
metadata and pointer-array allocations in setup() via prepare_cutlass_cached().
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, cast

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
_KernelVariant = Literal[
    "1sm",
    "1sm_n64",
    "1sm_n128",
    "2sm",
    "2sm_mxf4",
    "2sm_s1",
    "2sm_s2",
    "2sm_s3",
    "2sm_s4",
    "2sm_n64",
    "2sm_n64_s1",
    "2sm_n64_s2",
    "2sm_n64_s3",
    "2sm_n64_s4",
    "2sm_n64_s5",
    "2sm_n128",
    "2sm_n128_s1",
    "2sm_n128_s2",
    "2sm_n128_s3",
    "2sm_n128_s4",
]
_SUPPORTED_VARIANTS: Tuple[_KernelVariant, ...] = (
    "1sm",
    "1sm_n64",
    "1sm_n128",
    "2sm",
    "2sm_mxf4",
    "2sm_s1",
    "2sm_s2",
    "2sm_s3",
    "2sm_s4",
    "2sm_n64",
    "2sm_n64_s1",
    "2sm_n64_s2",
    "2sm_n64_s3",
    "2sm_n64_s4",
    "2sm_n64_s5",
    "2sm_n128",
    "2sm_n128_s1",
    "2sm_n128_s2",
    "2sm_n128_s3",
    "2sm_n128_s4",
)
_CaseKey = Tuple[_KernelVariant, int, int, int, int, bool, Tuple[Tuple[int, int, int, int], ...]]
_CASE_CACHE: Dict[_CaseKey, Dict[str, Any]] = {}


def _env_variant(name: str, default: _KernelVariant) -> _KernelVariant:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    value = raw.strip()
    if value not in _SUPPORTED_VARIANTS:
        raise ValueError(
            f"{name}={value!r} is unsupported; expected one of: {', '.join(_SUPPORTED_VARIANTS)}"
        )
    return cast(_KernelVariant, value)


def _variant_fns(ext: Any, variant: _KernelVariant) -> tuple[Any, Any]:
    if variant == "1sm":
        return ext.build_metadata_1sm, ext.create_plan_1sm
    if variant == "1sm_n64":
        return ext.build_metadata_1sm_n64, ext.create_plan_1sm_n64
    if variant == "1sm_n128":
        return ext.build_metadata_1sm_n128, ext.create_plan_1sm_n128
    if variant == "2sm":
        return ext.build_metadata_2sm, ext.create_plan_2sm
    if variant == "2sm_mxf4":
        # CUTLASS SM100 ptr-array MXF4 schedule requires SF vector=32, but this
        # challenge workload uses SF vector=16. Keep this variant aliasing to a
        # proven 2SM StageCount=4 path so sweeps remain runnable.
        return ext.build_metadata_2sm_s4, ext.create_plan_2sm_s4
    if variant == "2sm_s1":
        return ext.build_metadata_2sm_s1, ext.create_plan_2sm_s1
    if variant == "2sm_s2":
        return ext.build_metadata_2sm_s2, ext.create_plan_2sm_s2
    if variant == "2sm_s3":
        return ext.build_metadata_2sm_s3, ext.create_plan_2sm_s3
    if variant == "2sm_s4":
        return ext.build_metadata_2sm_s4, ext.create_plan_2sm_s4
    if variant == "2sm_n64":
        return ext.build_metadata_2sm_n64, ext.create_plan_2sm_n64
    if variant == "2sm_n64_s1":
        return ext.build_metadata_2sm_n64_s1, ext.create_plan_2sm_n64_s1
    if variant == "2sm_n64_s2":
        return ext.build_metadata_2sm_n64_s2, ext.create_plan_2sm_n64_s2
    if variant == "2sm_n64_s3":
        return ext.build_metadata_2sm_n64_s3, ext.create_plan_2sm_n64_s3
    if variant == "2sm_n64_s4":
        return ext.build_metadata_2sm_n64_s4, ext.create_plan_2sm_n64_s4
    if variant == "2sm_n64_s5":
        return ext.build_metadata_2sm_n64_s5, ext.create_plan_2sm_n64_s5
    if variant == "2sm_n128_s1":
        return ext.build_metadata_2sm_n128_s1, ext.create_plan_2sm_n128_s1
    if variant == "2sm_n128_s2":
        return ext.build_metadata_2sm_n128_s2, ext.create_plan_2sm_n128_s2
    if variant == "2sm_n128_s3":
        return ext.build_metadata_2sm_n128_s3, ext.create_plan_2sm_n128_s3
    if variant == "2sm_n128_s4":
        return ext.build_metadata_2sm_n128_s4, ext.create_plan_2sm_n128_s4
    return ext.build_metadata_2sm_n128, ext.create_plan_2sm_n128


def _get_case_ctx(problem_sizes: Sequence[Tuple[int, int, int, int]], *, variant: _KernelVariant) -> tuple[Any, Dict[str, Any], Any]:
    problem_sizes_key = tuple(tuple(int(x) for x in entry) for entry in problem_sizes)

    # Tunables (kept explicit, no global default changes).
    # CUTLASS 2SM kernels require cluster_dim.x >= 2 (see CUTLASS example).
    default_cluster_m = (
        2
        if variant in {
            "2sm",
            "2sm_mxf4",
            "2sm_s1",
            "2sm_s2",
            "2sm_s3",
            "2sm_s4",
            "2sm_n64",
            "2sm_n64_s1",
            "2sm_n64_s2",
            "2sm_n64_s3",
            "2sm_n64_s4",
            "2sm_n64_s5",
            "2sm_n128",
            "2sm_n128_s1",
            "2sm_n128_s2",
            "2sm_n128_s3",
            "2sm_n128_s4",
        }
        else 1
    )
    cluster_m = _env_int("AISP_NVFP4_GROUP_GEMM_CLUSTER_M", default_cluster_m)
    cluster_n = _env_int("AISP_NVFP4_GROUP_GEMM_CLUSTER_N", 1)
    raster_order = _env_int("AISP_NVFP4_GROUP_GEMM_RASTER_ORDER", 0)
    max_swizzle_size = _env_int("AISP_NVFP4_GROUP_GEMM_MAX_SWIZZLE", 0)
    use_pdl = _env_bool("AISP_NVFP4_GROUP_GEMM_USE_PDL", False)

    key: _CaseKey = (
        variant,
        int(cluster_m),
        int(cluster_n),
        int(raster_order),
        int(max_swizzle_size),
        bool(use_pdl),
        problem_sizes_key,
    )

    ext = load_cutlass_nvfp4_grouped_gemm_sm100(verbose=False)
    build_metadata, create_plan = _variant_fns(ext, variant)

    if key not in _CASE_CACHE:
        ps_cpu = torch.tensor(problem_sizes, dtype=torch.int32, device="cpu")
        (
            problem_shapes_u8,
            stride_a_u8,
            stride_b_u8,
            stride_c_u8,
            stride_d_u8,
            layout_sfa_u8,
            layout_sfb_u8,
            workspace_u8,
        ) = build_metadata(ps_cpu, cluster_m, cluster_n, raster_order, max_swizzle_size)
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
            "max_swizzle_size": int(max_swizzle_size),
            "use_pdl": bool(use_pdl),
        }

    return ext, _CASE_CACHE[key], create_plan


def _capture_plan_graph(ctx: Dict[str, Any]) -> None:
    """Capture a CUDA Graph replay path for a prepared CUTLASS plan context.

    This is intended for steady-state benchmark replay only:
    - Capture happens in setup() via prepare_*_graph wrappers.
    - Timed path calls graph.replay(), not capture.
    """
    # Warm up once before capture to avoid lazy first-run effects inside capture.
    plan = ctx.get("plan")
    if plan is not None:
        plan.run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            plan.run()
        ctx["graph"] = graph
        return

    plans = ctx.get("plans")
    if plans:
        plan_streams = ctx.get("plan_streams")
        if plan_streams:
            default_stream = torch.cuda.current_stream()
            stream_plans = _build_stream_plan_buckets(ctx, plans, len(plan_streams))

            # Warm up once on assigned streams before capture.
            for s, assigned in zip(plan_streams, stream_plans):
                if not assigned:
                    continue
                s.wait_stream(default_stream)
                with torch.cuda.stream(s):
                    for p in assigned:
                        p.run()
            for s in plan_streams:
                default_stream.wait_stream(s)
            torch.cuda.synchronize()

            stream_graphs: List[Tuple[torch.cuda.CUDAGraph, torch.cuda.Stream]] = []
            for s, assigned in zip(plan_streams, stream_plans):
                if not assigned:
                    continue
                g = torch.cuda.CUDAGraph()
                s.wait_stream(default_stream)
                with torch.cuda.graph(g, stream=s):
                    for p in assigned:
                        p.run()
                stream_graphs.append((g, s))

            if stream_graphs:
                ctx["stream_graphs"] = stream_graphs
                return

        for p in plans:
            p.run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for p in plans:
                p.run()
        ctx["graph"] = graph
        return

    small_plan = ctx.get("small_plan")
    large_plan = ctx.get("large_plan")
    if small_plan is None and large_plan is None:
        return

    if small_plan is not None:
        small_plan.run()
    if large_plan is not None:
        large_plan.run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        if small_plan is not None:
            small_plan.run()
        if large_plan is not None:
            large_plan.run()
    ctx["graph"] = graph


def _attach_graphs(prepared: Optional[Sequence[tuple[Any, ...]]]) -> Optional[Sequence[tuple[Any, ...]]]:
    if prepared is None:
        return None
    out: List[tuple[Any, ...]] = []
    for item in prepared:
        abc_tensors, sfasfb_tensors, sfasfb_reordered_tensors, problem_sizes, ctx = item
        _capture_plan_graph(ctx)
        out.append((abc_tensors, sfasfb_tensors, sfasfb_reordered_tensors, problem_sizes, ctx))
    return out


def _run_plans_with_optional_stream_overlap(ctx: Dict[str, Any], plans: Sequence[Any]) -> None:
    """Run prepared plans either sequentially or overlapped on auxiliary CUDA streams.

    Overlap is enabled when `ctx["plan_streams"]` is present and non-empty. This is used
    only for persistent chunked plans where kernels are independent and can be launched on
    separate streams to improve effective SM utilization.
    """
    plan_streams = ctx.get("plan_streams")
    if not plan_streams:
        for p in plans:
            p.run()
        return

    default_stream = torch.cuda.current_stream()
    stream_plans = _build_stream_plan_buckets(ctx, plans, len(plan_streams))
    for s, assigned in zip(plan_streams, stream_plans):
        if not assigned:
            continue
        s.wait_stream(default_stream)
        with torch.cuda.stream(s):
            for p in assigned:
                p.run()
    for s in plan_streams:
        default_stream.wait_stream(s)


def _run_stream_graphs(ctx: Dict[str, Any], stream_graphs: Sequence[Tuple[torch.cuda.CUDAGraph, torch.cuda.Stream]]) -> None:
    """Replay per-stream CUDA graphs with stream dependency fences."""
    default_stream = torch.cuda.current_stream()
    for g, s in stream_graphs:
        s.wait_stream(default_stream)
        with torch.cuda.stream(s):
            g.replay()
    for _, s in stream_graphs:
        default_stream.wait_stream(s)


def _build_stream_plan_buckets(
    ctx: Dict[str, Any],
    plans: Sequence[Any],
    stream_count: int,
) -> List[List[Any]]:
    """Map plans to streams, optionally load-balancing by estimated tile work.

    Round-robin is retained as the default behavior. Opt-in weighted scheduling
    (`AISP_NVFP4_GROUP_GEMM_PERSISTENT_STREAM_BALANCE=1`) uses a deterministic
    longest-processing-time assignment to reduce stragglers across stream lanes.
    """
    buckets: List[List[Any]] = [[] for _ in range(max(0, int(stream_count)))]
    if not buckets:
        return buckets

    use_balanced = _env_bool("AISP_NVFP4_GROUP_GEMM_PERSISTENT_STREAM_BALANCE", False)
    weights_raw = ctx.get("plan_tile_work")
    has_weights = isinstance(weights_raw, list) and len(weights_raw) == len(plans)

    if not use_balanced or not has_weights:
        for i, p in enumerate(plans):
            buckets[i % len(buckets)].append(p)
        return buckets

    weights = [max(1, int(w)) for w in weights_raw]
    lanes_load = [0 for _ in range(len(buckets))]
    lane_items: List[List[Tuple[int, int]]] = [[] for _ in range(len(buckets))]
    for plan_idx in sorted(range(len(plans)), key=lambda i: (-weights[i], i)):
        lane = min(range(len(lanes_load)), key=lambda s: (lanes_load[s], s))
        lane_items[lane].append((plan_idx, weights[plan_idx]))
        lanes_load[lane] += weights[plan_idx]

    # Optional tail-latency reduction: run heavier chunks first on each stream lane.
    # This minimizes end-of-iteration stragglers before the global stream fence.
    frontload_heavy = _env_bool("AISP_NVFP4_GROUP_GEMM_PERSISTENT_STREAM_FRONTLOAD_HEAVY", True)
    for lane, items in enumerate(lane_items):
        if frontload_heavy:
            items.sort(key=lambda pair: (-pair[1], pair[0]))
        buckets[lane].extend(plans[plan_idx] for plan_idx, _ in items)
    return buckets


def _variant_tile_shape(variant: _KernelVariant) -> Tuple[int, int, int]:
    """Return (m_tile, n_tile, k_tile) used by the selected CUTLASS variant."""
    if variant in {"1sm", "1sm_n64", "1sm_n128"}:
        m_tile = 128
    else:
        m_tile = 256

    if variant.endswith("_n64"):
        n_tile = 64
    elif variant.endswith("_n128"):
        n_tile = 128
    else:
        n_tile = 256

    return (m_tile, n_tile, 256)


def _estimate_group_tiles(problem_size: Tuple[int, int, int, int], variant: _KernelVariant) -> int:
    """Estimate CTA tile count for one (M, N, K, L) group for scheduler ordering."""
    m, n, k, _ = (int(problem_size[0]), int(problem_size[1]), int(problem_size[2]), int(problem_size[3]))
    tm, tn, tk = _variant_tile_shape(variant)
    tiles_m = (m + tm - 1) // tm
    tiles_n = (n + tn - 1) // tn
    tiles_k = (k + tk - 1) // tk
    return int(tiles_m * tiles_n * tiles_k)


def _persistent_group_permutation(
    problem_sizes: Sequence[Tuple[int, int, int, int]],
    variant: _KernelVariant,
) -> tuple[List[int], str]:
    """Return group permutation for fused persistent request plans.

    Ordering all request-groups by heavier shapes can improve persistent scheduler
    balance for skewed group sets without changing workload semantics.
    """
    mode = os.environ.get("AISP_NVFP4_GROUP_GEMM_PERSISTENT_GROUP_ORDER", "none").strip().lower()
    idx = list(range(len(problem_sizes)))
    if mode in {"", "none"}:
        return idx, "none"
    if mode == "m_desc":
        perm = sorted(
            idx,
            key=lambda i: (
                -int(problem_sizes[i][0]),
                -int(problem_sizes[i][1]),
                -int(problem_sizes[i][2]),
                int(i),
            ),
        )
        return perm, mode
    if mode == "m_asc":
        perm = sorted(
            idx,
            key=lambda i: (
                int(problem_sizes[i][0]),
                int(problem_sizes[i][1]),
                int(problem_sizes[i][2]),
                int(i),
            ),
        )
        return perm, mode
    if mode == "tiles_desc":
        perm = sorted(
            idx,
            key=lambda i: (
                -_estimate_group_tiles(problem_sizes[i], variant),
                -int(problem_sizes[i][0]),
                -int(problem_sizes[i][1]),
                -int(problem_sizes[i][2]),
                int(i),
            ),
        )
        return perm, mode
    if mode == "tiles_asc":
        perm = sorted(
            idx,
            key=lambda i: (
                _estimate_group_tiles(problem_sizes[i], variant),
                int(problem_sizes[i][0]),
                int(problem_sizes[i][1]),
                int(problem_sizes[i][2]),
                int(i),
            ),
        )
        return perm, mode
    if mode == "m_zigzag":
        by_m_desc = sorted(
            idx,
            key=lambda i: (
                -int(problem_sizes[i][0]),
                -int(problem_sizes[i][1]),
                -int(problem_sizes[i][2]),
                int(i),
            ),
        )
        perm: List[int] = []
        lo = 0
        hi = len(by_m_desc) - 1
        while lo <= hi:
            perm.append(by_m_desc[lo])
            lo += 1
            if lo <= hi:
                perm.append(by_m_desc[hi])
                hi -= 1
        return perm, mode
    if mode == "tiles_zigzag":
        by_tiles_desc = sorted(
            idx,
            key=lambda i: (
                -_estimate_group_tiles(problem_sizes[i], variant),
                -int(problem_sizes[i][0]),
                -int(problem_sizes[i][1]),
                -int(problem_sizes[i][2]),
                int(i),
            ),
        )
        perm: List[int] = []
        lo = 0
        hi = len(by_tiles_desc) - 1
        while lo <= hi:
            perm.append(by_tiles_desc[lo])
            lo += 1
            if lo <= hi:
                perm.append(by_tiles_desc[hi])
                hi -= 1
        return perm, mode
    raise ValueError(
        "Unsupported AISP_NVFP4_GROUP_GEMM_PERSISTENT_GROUP_ORDER="
        f"{mode!r}; expected one of: none, m_desc, m_asc, m_zigzag, tiles_desc, tiles_asc, tiles_zigzag"
    )


def _persistent_task_permutation(
    request_count: int,
    problem_sizes: Sequence[Tuple[int, int, int, int]],
    group_perm: Sequence[int],
    variant: _KernelVariant,
) -> tuple[List[Tuple[int, int]], str]:
    """Return fused persistent (request_idx, group_idx) ordering.

    Default behavior is request-major to preserve existing schedule semantics.
    Optional modes expose deeper scheduler shaping without changing workload.
    """
    mode = os.environ.get("AISP_NVFP4_GROUP_GEMM_PERSISTENT_TASK_ORDER", "request_major").strip().lower()
    if request_count <= 0:
        return [], "request_major"

    if mode in {"", "request_major"}:
        return [(r, g) for r in range(request_count) for g in group_perm], "request_major"

    if mode == "group_major":
        return [(r, g) for g in group_perm for r in range(request_count)], mode

    all_pairs = [(r, g) for r in range(request_count) for g in group_perm]
    if mode == "m_desc_global":
        perm = sorted(
            all_pairs,
            key=lambda rg: (
                -int(problem_sizes[rg[1]][0]),
                -int(problem_sizes[rg[1]][1]),
                -int(problem_sizes[rg[1]][2]),
                int(rg[0]),
                int(rg[1]),
            ),
        )
        return perm, mode

    if mode == "m_zigzag_global":
        by_m_desc = sorted(
            all_pairs,
            key=lambda rg: (
                -int(problem_sizes[rg[1]][0]),
                -int(problem_sizes[rg[1]][1]),
                -int(problem_sizes[rg[1]][2]),
                int(rg[0]),
                int(rg[1]),
            ),
        )
        perm: List[Tuple[int, int]] = []
        lo = 0
        hi = len(by_m_desc) - 1
        while lo <= hi:
            perm.append(by_m_desc[lo])
            lo += 1
            if lo <= hi:
                perm.append(by_m_desc[hi])
                hi -= 1
        return perm, mode

    if mode == "tile_desc_global":
        perm = sorted(
            all_pairs,
            key=lambda rg: (
                -_estimate_group_tiles(problem_sizes[rg[1]], variant),
                -int(problem_sizes[rg[1]][0]),
                -int(problem_sizes[rg[1]][1]),
                -int(problem_sizes[rg[1]][2]),
                int(rg[0]),
                int(rg[1]),
            ),
        )
        return perm, mode

    if mode == "tile_zigzag_global":
        by_tile_desc = sorted(
            all_pairs,
            key=lambda rg: (
                -_estimate_group_tiles(problem_sizes[rg[1]], variant),
                -int(problem_sizes[rg[1]][0]),
                -int(problem_sizes[rg[1]][1]),
                -int(problem_sizes[rg[1]][2]),
                int(rg[0]),
                int(rg[1]),
            ),
        )
        perm: List[Tuple[int, int]] = []
        lo = 0
        hi = len(by_tile_desc) - 1
        while lo <= hi:
            perm.append(by_tile_desc[lo])
            lo += 1
            if lo <= hi:
                perm.append(by_tile_desc[hi])
                hi -= 1
        return perm, mode

    raise ValueError(
        "Unsupported AISP_NVFP4_GROUP_GEMM_PERSISTENT_TASK_ORDER="
        f"{mode!r}; expected one of: request_major, group_major, m_desc_global, m_zigzag_global, "
        "tile_desc_global, tile_zigzag_global"
    )


def _build_persistent_plan_for_tasks(
    *,
    request_chunk: Sequence[input_t],
    problem_sizes: Sequence[Tuple[int, int, int, int]],
    task_perm: Sequence[Tuple[int, int]],
    variant: _KernelVariant,
) -> Tuple[Any, int]:
    """Build one CUTLASS plan for a selected set of (request_idx, group_idx) tasks."""
    fused_problem_sizes: List[Tuple[int, int, int, int]] = []
    for _request_idx, group_idx in task_perm:
        fused_problem_sizes.append(problem_sizes[group_idx])

    _ext, case_ctx, create_plan = _get_case_ctx(fused_problem_sizes, variant=variant)

    a_ptrs: List[int] = []
    b_ptrs: List[int] = []
    c_ptrs: List[int] = []
    sfa_ptrs: List[int] = []
    sfb_ptrs: List[int] = []
    for request_idx, group_idx in task_perm:
        abc_tensors, _, sfasfb_reordered_tensors, _ = request_chunk[request_idx]
        a, b, c = abc_tensors[group_idx]
        sfa_reordered, sfb_reordered = sfasfb_reordered_tensors[group_idx]
        a_ptrs.append(int(a.data_ptr()))
        b_ptrs.append(int(b.data_ptr()))
        c_ptrs.append(int(c.data_ptr()))
        sfa_ptrs.append(int(sfa_reordered.data_ptr()))
        sfb_ptrs.append(int(sfb_reordered.data_ptr()))

    ptr_a_i64 = torch.tensor(a_ptrs, dtype=torch.int64, device="cuda")
    ptr_b_i64 = torch.tensor(b_ptrs, dtype=torch.int64, device="cuda")
    ptr_c_i64 = torch.tensor(c_ptrs, dtype=torch.int64, device="cuda")
    ptr_d_i64 = torch.tensor(c_ptrs, dtype=torch.int64, device="cuda")
    ptr_sfa_i64 = torch.tensor(sfa_ptrs, dtype=torch.int64, device="cuda")
    ptr_sfb_i64 = torch.tensor(sfb_ptrs, dtype=torch.int64, device="cuda")

    plan = create_plan(
        case_ctx["problem_shapes_u8"],
        case_ctx["stride_a_u8"],
        case_ctx["stride_b_u8"],
        case_ctx["stride_c_u8"],
        case_ctx["stride_d_u8"],
        case_ctx["layout_sfa_u8"],
        case_ctx["layout_sfb_u8"],
        case_ctx["workspace_u8"],
        ptr_a_i64,
        ptr_b_i64,
        ptr_sfa_i64,
        ptr_sfb_i64,
        ptr_c_i64,
        ptr_d_i64,
        1.0,
        0.0,
        case_ctx["raster_order"],
        case_ctx["cluster_m"],
        case_ctx["cluster_n"],
        case_ctx["max_swizzle_size"],
        case_ctx["use_pdl"],
    )
    tile_work = sum(_estimate_group_tiles(problem_sizes[group_idx], variant) for _, group_idx in task_perm)
    return plan, int(tile_work)


def _prepare_cutlass_cached(
    data_list: Sequence[input_t], *, variant: _KernelVariant
) -> Optional[Sequence[tuple[Any, ...]]]:
    """Precompute CUTLASS metadata + pointer arrays for each input in data_list.

    Returns a replacement data_list whose elements have an extra trailing `ctx` dict, consumed by
    custom_kernel_cutlass_cached().
    """
    if not data_list:
        return None

    problem_sizes = data_list[0][3]
    _ext, case_ctx, create_plan = _get_case_ctx(problem_sizes, variant=variant)

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
            # Reuse the same output object each iteration to avoid Python list rebuild overhead.
            "outputs": [abc_tensors[i][2] for i in range(len(abc_tensors))],
        }

        # Pre-initialize CUTLASS params once per input to avoid timing host-side initialization.
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
            case_ctx["max_swizzle_size"],
            case_ctx["use_pdl"],
        )

        out.append((abc_tensors, sfasfb_tensors, sfasfb_reordered_tensors, problem_sizes, ctx))

    return out


def _prepare_cutlass_cached_persistent_requests(
    data_list: Sequence[input_t], *, variant: _KernelVariant
) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare one fused plan over all requests in an iteration.

    This preserves each request's original per-group (M,N,K,L) shapes and values, but
    concatenates all request-group pointer arrays so benchmark_fn() executes a single grouped
    launch per iteration.
    """
    if not data_list:
        return None

    problem_sizes = data_list[0][3]
    for _, _, _, problem_sizes_i in data_list:
        if problem_sizes_i != problem_sizes:
            raise ValueError("persistent prepare expects all inputs to share identical problem_sizes")
    group_perm, group_order_mode = _persistent_group_permutation(problem_sizes, variant)

    # Optional chunking: reduce launch count while avoiding oversized grouped schedules.
    # 0/negative means "fuse all requests".
    chunk = int(_env_int("AISP_NVFP4_GROUP_GEMM_PERSISTENT_REQUEST_CHUNK", len(data_list)))
    if chunk <= 0:
        chunk = len(data_list)
    concurrent_streams = int(_env_int("AISP_NVFP4_GROUP_GEMM_PERSISTENT_CONCURRENT_STREAMS", 1))
    if concurrent_streams < 1:
        concurrent_streams = 1
    hybrid_enable = _env_bool("AISP_NVFP4_GROUP_GEMM_PERSISTENT_HYBRID_ENABLE", False)
    hybrid_threshold_m = int(_env_int("AISP_NVFP4_GROUP_GEMM_PERSISTENT_HYBRID_M_THRESHOLD", 128))
    hybrid_small_variant = _env_variant("AISP_NVFP4_GROUP_GEMM_PERSISTENT_HYBRID_SMALL_VARIANT", "1sm_n128")
    hybrid_large_variant = _env_variant("AISP_NVFP4_GROUP_GEMM_PERSISTENT_HYBRID_LARGE_VARIANT", variant)

    plans: List[Any] = []
    plan_tile_work: List[int] = []
    task_order_mode = "request_major"
    for start in range(0, len(data_list), chunk):
        request_chunk = data_list[start : start + chunk]
        task_perm, task_order_mode = _persistent_task_permutation(
            len(request_chunk),
            problem_sizes,
            group_perm,
            variant,
        )
        if hybrid_enable:
            small_tasks = [
                (request_idx, group_idx)
                for request_idx, group_idx in task_perm
                if int(problem_sizes[group_idx][0]) <= hybrid_threshold_m
            ]
            large_tasks = [
                (request_idx, group_idx)
                for request_idx, group_idx in task_perm
                if int(problem_sizes[group_idx][0]) > hybrid_threshold_m
            ]
            if small_tasks:
                small_plan, small_tile_work = _build_persistent_plan_for_tasks(
                    request_chunk=request_chunk,
                    problem_sizes=problem_sizes,
                    task_perm=small_tasks,
                    variant=hybrid_small_variant,
                )
                plans.append(small_plan)
                plan_tile_work.append(small_tile_work)
            if large_tasks:
                large_plan, large_tile_work = _build_persistent_plan_for_tasks(
                    request_chunk=request_chunk,
                    problem_sizes=problem_sizes,
                    task_perm=large_tasks,
                    variant=hybrid_large_variant,
                )
                plans.append(large_plan)
                plan_tile_work.append(large_tile_work)
        else:
            plan, tile_work = _build_persistent_plan_for_tasks(
                request_chunk=request_chunk,
                problem_sizes=problem_sizes,
                task_perm=task_perm,
                variant=variant,
            )
            plans.append(plan)
            plan_tile_work.append(tile_work)

    plan_streams: Optional[List[torch.cuda.Stream]] = None
    if concurrent_streams > 1 and len(plans) > 1:
        stream_count = min(int(concurrent_streams), len(plans))
        plan_streams = [torch.cuda.Stream() for _ in range(stream_count)]

    # For harness verification, keep semantics identical to the existing path: return outputs for
    # the final request in the iteration.
    last_abc_tensors, last_sfasfb_tensors, last_sfasfb_reordered_tensors, last_problem_sizes = data_list[-1]

    ctx = {
        "plans": plans,
        "plan_tile_work": plan_tile_work,
        "plan_streams": plan_streams,
        "persistent_request_chunk": int(chunk),
        "persistent_concurrent_streams": int(concurrent_streams),
        "persistent_group_order": group_order_mode,
        "persistent_task_order": task_order_mode,
        "persistent_hybrid_enable": bool(hybrid_enable),
        "persistent_hybrid_threshold_m": int(hybrid_threshold_m),
        "persistent_hybrid_small_variant": str(hybrid_small_variant),
        "persistent_hybrid_large_variant": str(hybrid_large_variant),
        "outputs": [last_abc_tensors[i][2] for i in range(len(last_abc_tensors))],
        # Keep all request tensors alive because this fused plan dereferences pointers from every
        # request in `data_list`, not just the last one returned for verification.
        "keepalive_abc_tensors": [item[0] for item in data_list],
        "keepalive_sfasfb_tensors": [item[1] for item in data_list],
        "keepalive_sfasfb_reordered_tensors": [item[2] for item in data_list],
    }

    return [
        (
            last_abc_tensors,
            last_sfasfb_tensors,
            last_sfasfb_reordered_tensors,
            last_problem_sizes,
            ctx,
        )
    ]


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

def prepare_cutlass_cached_2sm_mxf4(data_list: Sequence[input_t]) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare using the CUTLASS 2SM MXF4 MMA kernel."""
    return _prepare_cutlass_cached(data_list, variant="2sm_mxf4")

def prepare_cutlass_cached_2sm_s1(data_list: Sequence[input_t]) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare using the CUTLASS 2SM MMA kernel (StageCount=1)."""
    return _prepare_cutlass_cached(data_list, variant="2sm_s1")

def prepare_cutlass_cached_2sm_s2(data_list: Sequence[input_t]) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare using the CUTLASS 2SM MMA kernel (StageCount=2)."""
    return _prepare_cutlass_cached(data_list, variant="2sm_s2")

def prepare_cutlass_cached_2sm_s3(data_list: Sequence[input_t]) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare using the CUTLASS 2SM MMA kernel (StageCount=3)."""
    return _prepare_cutlass_cached(data_list, variant="2sm_s3")

def prepare_cutlass_cached_2sm_s4(data_list: Sequence[input_t]) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare using the CUTLASS 2SM MMA kernel (StageCount=4)."""
    return _prepare_cutlass_cached(data_list, variant="2sm_s4")

def prepare_cutlass_cached_2sm_n32(data_list: Sequence[input_t]) -> Optional[Sequence[tuple[Any, ...]]]:
    """N=32 tile is unsupported by CUTLASS SM100 block-scaled 2SM kernels."""
    raise RuntimeError("CUTLASS SM100 block-scaled 2SM kernels do not support N=32 tiles.")

def prepare_cutlass_cached_2sm_n64(data_list: Sequence[input_t]) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare using the CUTLASS 2SM MMA kernel (N=64 tile)."""
    return _prepare_cutlass_cached(data_list, variant="2sm_n64")

def prepare_cutlass_cached_2sm_n64_s1(data_list: Sequence[input_t]) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare using the CUTLASS 2SM MMA kernel (N=64 tile, StageCount=1)."""
    return _prepare_cutlass_cached(data_list, variant="2sm_n64_s1")

def prepare_cutlass_cached_2sm_n64_s2(data_list: Sequence[input_t]) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare using the CUTLASS 2SM MMA kernel (N=64 tile, StageCount=2)."""
    return _prepare_cutlass_cached(data_list, variant="2sm_n64_s2")

def prepare_cutlass_cached_2sm_n64_s3(data_list: Sequence[input_t]) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare using the CUTLASS 2SM MMA kernel (N=64 tile, StageCount=3)."""
    return _prepare_cutlass_cached(data_list, variant="2sm_n64_s3")

def prepare_cutlass_cached_2sm_n64_s4(data_list: Sequence[input_t]) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare using the CUTLASS 2SM MMA kernel (N=64 tile, StageCount=4)."""
    return _prepare_cutlass_cached(data_list, variant="2sm_n64_s4")

def prepare_cutlass_cached_2sm_n64_s5(data_list: Sequence[input_t]) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare using the CUTLASS 2SM MMA kernel (N=64 tile, StageCount=5)."""
    return _prepare_cutlass_cached(data_list, variant="2sm_n64_s5")

def prepare_cutlass_cached_2sm_n128(data_list: Sequence[input_t]) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare using the CUTLASS 2SM MMA kernel (N=128 tile)."""
    return _prepare_cutlass_cached(data_list, variant="2sm_n128")

def prepare_cutlass_cached_2sm_n128_s1(data_list: Sequence[input_t]) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare using the CUTLASS 2SM MMA kernel (N=128 tile, StageCount=1)."""
    return _prepare_cutlass_cached(data_list, variant="2sm_n128_s1")

def prepare_cutlass_cached_2sm_n128_s2(data_list: Sequence[input_t]) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare using the CUTLASS 2SM MMA kernel (N=128 tile, StageCount=2)."""
    return _prepare_cutlass_cached(data_list, variant="2sm_n128_s2")

def prepare_cutlass_cached_2sm_n128_s3(data_list: Sequence[input_t]) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare using the CUTLASS 2SM MMA kernel (N=128 tile, StageCount=3)."""
    return _prepare_cutlass_cached(data_list, variant="2sm_n128_s3")

def prepare_cutlass_cached_2sm_n128_s4(data_list: Sequence[input_t]) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare using the CUTLASS 2SM MMA kernel (N=128 tile, StageCount=4)."""
    return _prepare_cutlass_cached(data_list, variant="2sm_n128_s4")


def prepare_cutlass_cached_2sm_persistent(data_list: Sequence[input_t]) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare one fused-request plan using the CUTLASS 2SM kernel."""
    return _prepare_cutlass_cached_persistent_requests(data_list, variant="2sm")

def prepare_cutlass_cached_2sm_mxf4_persistent(data_list: Sequence[input_t]) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare one fused-request plan using the CUTLASS 2SM MXF4 kernel."""
    return _prepare_cutlass_cached_persistent_requests(data_list, variant="2sm_mxf4")

def prepare_cutlass_cached_2sm_s1_persistent(data_list: Sequence[input_t]) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare one fused-request plan using the CUTLASS 2SM StageCount=1 kernel."""
    return _prepare_cutlass_cached_persistent_requests(data_list, variant="2sm_s1")

def prepare_cutlass_cached_2sm_s2_persistent(data_list: Sequence[input_t]) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare one fused-request plan using the CUTLASS 2SM StageCount=2 kernel."""
    return _prepare_cutlass_cached_persistent_requests(data_list, variant="2sm_s2")

def prepare_cutlass_cached_2sm_s3_persistent(data_list: Sequence[input_t]) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare one fused-request plan using the CUTLASS 2SM StageCount=3 kernel."""
    return _prepare_cutlass_cached_persistent_requests(data_list, variant="2sm_s3")

def prepare_cutlass_cached_2sm_s4_persistent(data_list: Sequence[input_t]) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare one fused-request plan using the CUTLASS 2SM StageCount=4 kernel."""
    return _prepare_cutlass_cached_persistent_requests(data_list, variant="2sm_s4")

def prepare_cutlass_cached_2sm_n32_persistent(data_list: Sequence[input_t]) -> Optional[Sequence[tuple[Any, ...]]]:
    """N=32 tile is unsupported by CUTLASS SM100 block-scaled 2SM kernels."""
    raise RuntimeError("CUTLASS SM100 block-scaled 2SM kernels do not support N=32 tiles.")


def prepare_cutlass_cached_2sm_n64_persistent(data_list: Sequence[input_t]) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare one fused-request plan using the CUTLASS 2SM N64 kernel."""
    return _prepare_cutlass_cached_persistent_requests(data_list, variant="2sm_n64")

def prepare_cutlass_cached_2sm_n64_s1_persistent(data_list: Sequence[input_t]) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare one fused-request plan using the CUTLASS 2SM N64 StageCount=1 kernel."""
    return _prepare_cutlass_cached_persistent_requests(data_list, variant="2sm_n64_s1")

def prepare_cutlass_cached_2sm_n64_s2_persistent(data_list: Sequence[input_t]) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare one fused-request plan using the CUTLASS 2SM N64 StageCount=2 kernel."""
    return _prepare_cutlass_cached_persistent_requests(data_list, variant="2sm_n64_s2")

def prepare_cutlass_cached_2sm_n64_s3_persistent(data_list: Sequence[input_t]) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare one fused-request plan using the CUTLASS 2SM N64 StageCount=3 kernel."""
    return _prepare_cutlass_cached_persistent_requests(data_list, variant="2sm_n64_s3")

def prepare_cutlass_cached_2sm_n64_s4_persistent(data_list: Sequence[input_t]) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare one fused-request plan using the CUTLASS 2SM N64 StageCount=4 kernel."""
    return _prepare_cutlass_cached_persistent_requests(data_list, variant="2sm_n64_s4")

def prepare_cutlass_cached_2sm_n64_s5_persistent(data_list: Sequence[input_t]) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare one fused-request plan using the CUTLASS 2SM N64 StageCount=5 kernel."""
    return _prepare_cutlass_cached_persistent_requests(data_list, variant="2sm_n64_s5")


def prepare_cutlass_cached_2sm_n128_persistent(data_list: Sequence[input_t]) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare one fused-request plan using the CUTLASS 2SM N128 kernel."""
    return _prepare_cutlass_cached_persistent_requests(data_list, variant="2sm_n128")

def prepare_cutlass_cached_2sm_n128_s1_persistent(
    data_list: Sequence[input_t],
) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare one fused-request plan using the CUTLASS 2SM N128 StageCount=1 kernel."""
    return _prepare_cutlass_cached_persistent_requests(data_list, variant="2sm_n128_s1")

def prepare_cutlass_cached_2sm_n128_s2_persistent(
    data_list: Sequence[input_t],
) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare one fused-request plan using the CUTLASS 2SM N128 StageCount=2 kernel."""
    return _prepare_cutlass_cached_persistent_requests(data_list, variant="2sm_n128_s2")

def prepare_cutlass_cached_2sm_n128_s3_persistent(
    data_list: Sequence[input_t],
) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare one fused-request plan using the CUTLASS 2SM N128 StageCount=3 kernel."""
    return _prepare_cutlass_cached_persistent_requests(data_list, variant="2sm_n128_s3")

def prepare_cutlass_cached_2sm_n128_s4_persistent(
    data_list: Sequence[input_t],
) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare one fused-request plan using the CUTLASS 2SM N128 StageCount=4 kernel."""
    return _prepare_cutlass_cached_persistent_requests(data_list, variant="2sm_n128_s4")


def prepare_cutlass_cached_2sm_persistent_graph(data_list: Sequence[input_t]) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare fused-request 2SM plan(s) with CUDA Graph replay capture."""
    return _attach_graphs(_prepare_cutlass_cached_persistent_requests(data_list, variant="2sm"))

def prepare_cutlass_cached_2sm_mxf4_persistent_graph(
    data_list: Sequence[input_t],
) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare fused-request 2SM MXF4 plan(s) with CUDA Graph replay capture."""
    return _attach_graphs(_prepare_cutlass_cached_persistent_requests(data_list, variant="2sm_mxf4"))

def prepare_cutlass_cached_2sm_s1_persistent_graph(data_list: Sequence[input_t]) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare fused-request 2SM StageCount=1 plan(s) with CUDA Graph replay capture."""
    return _attach_graphs(_prepare_cutlass_cached_persistent_requests(data_list, variant="2sm_s1"))

def prepare_cutlass_cached_2sm_s2_persistent_graph(data_list: Sequence[input_t]) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare fused-request 2SM StageCount=2 plan(s) with CUDA Graph replay capture."""
    return _attach_graphs(_prepare_cutlass_cached_persistent_requests(data_list, variant="2sm_s2"))

def prepare_cutlass_cached_2sm_s3_persistent_graph(data_list: Sequence[input_t]) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare fused-request 2SM StageCount=3 plan(s) with CUDA Graph replay capture."""
    return _attach_graphs(_prepare_cutlass_cached_persistent_requests(data_list, variant="2sm_s3"))

def prepare_cutlass_cached_2sm_s4_persistent_graph(data_list: Sequence[input_t]) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare fused-request 2SM StageCount=4 plan(s) with CUDA Graph replay capture."""
    return _attach_graphs(_prepare_cutlass_cached_persistent_requests(data_list, variant="2sm_s4"))

def prepare_cutlass_cached_2sm_n32_persistent_graph(
    data_list: Sequence[input_t],
) -> Optional[Sequence[tuple[Any, ...]]]:
    """N=32 tile is unsupported by CUTLASS SM100 block-scaled 2SM kernels."""
    raise RuntimeError("CUTLASS SM100 block-scaled 2SM kernels do not support N=32 tiles.")


def prepare_cutlass_cached_2sm_n64_persistent_graph(
    data_list: Sequence[input_t],
) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare fused-request 2SM N64 plan(s) with CUDA Graph replay capture."""
    return _attach_graphs(_prepare_cutlass_cached_persistent_requests(data_list, variant="2sm_n64"))

def prepare_cutlass_cached_2sm_n64_s1_persistent_graph(
    data_list: Sequence[input_t],
) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare fused-request 2SM N64 StageCount=1 plan(s) with CUDA Graph replay capture."""
    return _attach_graphs(_prepare_cutlass_cached_persistent_requests(data_list, variant="2sm_n64_s1"))

def prepare_cutlass_cached_2sm_n64_s2_persistent_graph(
    data_list: Sequence[input_t],
) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare fused-request 2SM N64 StageCount=2 plan(s) with CUDA Graph replay capture."""
    return _attach_graphs(_prepare_cutlass_cached_persistent_requests(data_list, variant="2sm_n64_s2"))

def prepare_cutlass_cached_2sm_n64_s3_persistent_graph(
    data_list: Sequence[input_t],
) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare fused-request 2SM N64 StageCount=3 plan(s) with CUDA Graph replay capture."""
    return _attach_graphs(_prepare_cutlass_cached_persistent_requests(data_list, variant="2sm_n64_s3"))

def prepare_cutlass_cached_2sm_n64_s4_persistent_graph(
    data_list: Sequence[input_t],
) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare fused-request 2SM N64 StageCount=4 plan(s) with CUDA Graph replay capture."""
    return _attach_graphs(_prepare_cutlass_cached_persistent_requests(data_list, variant="2sm_n64_s4"))

def prepare_cutlass_cached_2sm_n64_s5_persistent_graph(
    data_list: Sequence[input_t],
) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare fused-request 2SM N64 StageCount=5 plan(s) with CUDA Graph replay capture."""
    return _attach_graphs(_prepare_cutlass_cached_persistent_requests(data_list, variant="2sm_n64_s5"))


def prepare_cutlass_cached_2sm_n128_persistent_graph(
    data_list: Sequence[input_t],
) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare fused-request 2SM N128 plan(s) with CUDA Graph replay capture."""
    return _attach_graphs(_prepare_cutlass_cached_persistent_requests(data_list, variant="2sm_n128"))

def prepare_cutlass_cached_2sm_n128_s1_persistent_graph(
    data_list: Sequence[input_t],
) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare fused-request 2SM N128 StageCount=1 plan(s) with CUDA Graph replay capture."""
    return _attach_graphs(_prepare_cutlass_cached_persistent_requests(data_list, variant="2sm_n128_s1"))

def prepare_cutlass_cached_2sm_n128_s2_persistent_graph(
    data_list: Sequence[input_t],
) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare fused-request 2SM N128 StageCount=2 plan(s) with CUDA Graph replay capture."""
    return _attach_graphs(_prepare_cutlass_cached_persistent_requests(data_list, variant="2sm_n128_s2"))

def prepare_cutlass_cached_2sm_n128_s3_persistent_graph(
    data_list: Sequence[input_t],
) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare fused-request 2SM N128 StageCount=3 plan(s) with CUDA Graph replay capture."""
    return _attach_graphs(_prepare_cutlass_cached_persistent_requests(data_list, variant="2sm_n128_s3"))

def prepare_cutlass_cached_2sm_n128_s4_persistent_graph(
    data_list: Sequence[input_t],
) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare fused-request 2SM N128 StageCount=4 plan(s) with CUDA Graph replay capture."""
    return _attach_graphs(_prepare_cutlass_cached_persistent_requests(data_list, variant="2sm_n128_s4"))


def prepare_cutlass_cached_graph(data_list: Sequence[input_t]) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare using CUTLASS 1SM MMA kernel with CUDA Graph replay capture."""
    return _attach_graphs(_prepare_cutlass_cached(data_list, variant="1sm"))


def prepare_cutlass_cached_1sm_n64_graph(data_list: Sequence[input_t]) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare using CUTLASS 1SM N64 kernel with CUDA Graph replay capture."""
    return _attach_graphs(_prepare_cutlass_cached(data_list, variant="1sm_n64"))


def prepare_cutlass_cached_1sm_n128_graph(data_list: Sequence[input_t]) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare using CUTLASS 1SM N128 kernel with CUDA Graph replay capture."""
    return _attach_graphs(_prepare_cutlass_cached(data_list, variant="1sm_n128"))


def prepare_cutlass_cached_2sm_graph(data_list: Sequence[input_t]) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare using CUTLASS 2SM kernel with CUDA Graph replay capture."""
    return _attach_graphs(_prepare_cutlass_cached(data_list, variant="2sm"))

def prepare_cutlass_cached_2sm_mxf4_graph(data_list: Sequence[input_t]) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare using CUTLASS 2SM MXF4 kernel with CUDA Graph replay capture."""
    return _attach_graphs(_prepare_cutlass_cached(data_list, variant="2sm_mxf4"))

def prepare_cutlass_cached_2sm_s1_graph(data_list: Sequence[input_t]) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare using CUTLASS 2SM StageCount=1 kernel with CUDA Graph replay capture."""
    return _attach_graphs(_prepare_cutlass_cached(data_list, variant="2sm_s1"))

def prepare_cutlass_cached_2sm_s2_graph(data_list: Sequence[input_t]) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare using CUTLASS 2SM StageCount=2 kernel with CUDA Graph replay capture."""
    return _attach_graphs(_prepare_cutlass_cached(data_list, variant="2sm_s2"))

def prepare_cutlass_cached_2sm_s3_graph(data_list: Sequence[input_t]) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare using CUTLASS 2SM StageCount=3 kernel with CUDA Graph replay capture."""
    return _attach_graphs(_prepare_cutlass_cached(data_list, variant="2sm_s3"))

def prepare_cutlass_cached_2sm_s4_graph(data_list: Sequence[input_t]) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare using CUTLASS 2SM StageCount=4 kernel with CUDA Graph replay capture."""
    return _attach_graphs(_prepare_cutlass_cached(data_list, variant="2sm_s4"))

def prepare_cutlass_cached_2sm_n32_graph(data_list: Sequence[input_t]) -> Optional[Sequence[tuple[Any, ...]]]:
    """N=32 tile is unsupported by CUTLASS SM100 block-scaled 2SM kernels."""
    raise RuntimeError("CUTLASS SM100 block-scaled 2SM kernels do not support N=32 tiles.")


def prepare_cutlass_cached_2sm_n64_graph(data_list: Sequence[input_t]) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare using CUTLASS 2SM N64 kernel with CUDA Graph replay capture."""
    return _attach_graphs(_prepare_cutlass_cached(data_list, variant="2sm_n64"))

def prepare_cutlass_cached_2sm_n64_s1_graph(data_list: Sequence[input_t]) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare using CUTLASS 2SM N64 StageCount=1 kernel with CUDA Graph replay capture."""
    return _attach_graphs(_prepare_cutlass_cached(data_list, variant="2sm_n64_s1"))

def prepare_cutlass_cached_2sm_n64_s2_graph(data_list: Sequence[input_t]) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare using CUTLASS 2SM N64 StageCount=2 kernel with CUDA Graph replay capture."""
    return _attach_graphs(_prepare_cutlass_cached(data_list, variant="2sm_n64_s2"))

def prepare_cutlass_cached_2sm_n64_s3_graph(data_list: Sequence[input_t]) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare using CUTLASS 2SM N64 StageCount=3 kernel with CUDA Graph replay capture."""
    return _attach_graphs(_prepare_cutlass_cached(data_list, variant="2sm_n64_s3"))

def prepare_cutlass_cached_2sm_n64_s4_graph(data_list: Sequence[input_t]) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare using CUTLASS 2SM N64 StageCount=4 kernel with CUDA Graph replay capture."""
    return _attach_graphs(_prepare_cutlass_cached(data_list, variant="2sm_n64_s4"))

def prepare_cutlass_cached_2sm_n64_s5_graph(data_list: Sequence[input_t]) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare using CUTLASS 2SM N64 StageCount=5 kernel with CUDA Graph replay capture."""
    return _attach_graphs(_prepare_cutlass_cached(data_list, variant="2sm_n64_s5"))


def prepare_cutlass_cached_2sm_n128_graph(data_list: Sequence[input_t]) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare using CUTLASS 2SM N128 kernel with CUDA Graph replay capture."""
    return _attach_graphs(_prepare_cutlass_cached(data_list, variant="2sm_n128"))

def prepare_cutlass_cached_2sm_n128_s1_graph(data_list: Sequence[input_t]) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare using CUTLASS 2SM N128 StageCount=1 kernel with CUDA Graph replay capture."""
    return _attach_graphs(_prepare_cutlass_cached(data_list, variant="2sm_n128_s1"))

def prepare_cutlass_cached_2sm_n128_s2_graph(data_list: Sequence[input_t]) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare using CUTLASS 2SM N128 StageCount=2 kernel with CUDA Graph replay capture."""
    return _attach_graphs(_prepare_cutlass_cached(data_list, variant="2sm_n128_s2"))

def prepare_cutlass_cached_2sm_n128_s3_graph(data_list: Sequence[input_t]) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare using CUTLASS 2SM N128 StageCount=3 kernel with CUDA Graph replay capture."""
    return _attach_graphs(_prepare_cutlass_cached(data_list, variant="2sm_n128_s3"))

def prepare_cutlass_cached_2sm_n128_s4_graph(data_list: Sequence[input_t]) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare using CUTLASS 2SM N128 StageCount=4 kernel with CUDA Graph replay capture."""
    return _attach_graphs(_prepare_cutlass_cached(data_list, variant="2sm_n128_s4"))


def _slice_input_groups(data: input_t, group_indices: Sequence[int]) -> input_t:
    abc_tensors, sfasfb_tensors, sfasfb_reordered_tensors, problem_sizes = data
    idx = [int(i) for i in group_indices]
    return (
        [abc_tensors[i] for i in idx],
        [sfasfb_tensors[i] for i in idx],
        [sfasfb_reordered_tensors[i] for i in idx],
        [problem_sizes[i] for i in idx],
    )


def _prepare_cutlass_cached_with_overrides(
    data_list: Sequence[input_t], *, variant: _KernelVariant, overrides: Dict[str, str]
) -> Optional[Sequence[tuple[Any, ...]]]:
    # Keep overrides scoped to this prepare call so different subplans can use distinct tunings.
    saved = {k: os.environ.get(k) for k in overrides}
    try:
        for k, v in overrides.items():
            os.environ[k] = str(v)
        return _prepare_cutlass_cached(data_list, variant=variant)
    finally:
        for k, old_v in saved.items():
            if old_v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old_v


def _hybrid_overrides(*, variant: _KernelVariant, side: Literal["small", "large"]) -> Dict[str, str]:
    prefix = "AISP_NVFP4_GROUP_GEMM_HYBRID_SMALL_" if side == "small" else "AISP_NVFP4_GROUP_GEMM_HYBRID_LARGE_"
    default_cluster_m = (
        2
        if variant
        in {
            "2sm",
            "2sm_mxf4",
            "2sm_s2",
            "2sm_s3",
            "2sm_s4",
            "2sm_n64",
            "2sm_n64_s1",
            "2sm_n64_s2",
            "2sm_n64_s3",
            "2sm_n64_s4",
            "2sm_n64_s5",
            "2sm_n128",
            "2sm_n128_s1",
            "2sm_n128_s2",
            "2sm_n128_s3",
            "2sm_n128_s4",
        }
        else 1
    )
    default_use_pdl = False if side == "small" else True
    return {
        "AISP_NVFP4_GROUP_GEMM_CLUSTER_M": str(_env_int(prefix + "CLUSTER_M", default_cluster_m)),
        "AISP_NVFP4_GROUP_GEMM_CLUSTER_N": str(_env_int(prefix + "CLUSTER_N", 1)),
        "AISP_NVFP4_GROUP_GEMM_RASTER_ORDER": str(_env_int(prefix + "RASTER_ORDER", 0)),
        "AISP_NVFP4_GROUP_GEMM_MAX_SWIZZLE": str(_env_int(prefix + "MAX_SWIZZLE", 0)),
        "AISP_NVFP4_GROUP_GEMM_USE_PDL": "1" if _env_bool(prefix + "USE_PDL", default_use_pdl) else "0",
    }


def _prepare_cutlass_cached_hybrid(
    data_list: Sequence[input_t], *, small_variant: _KernelVariant, large_variant: _KernelVariant
) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare a two-plan hybrid: small-M groups and large-M groups use different kernels."""
    if not data_list:
        return None

    problem_sizes = data_list[0][3]
    for _, _, _, problem_sizes_i in data_list:
        if problem_sizes_i != problem_sizes:
            raise ValueError("hybrid prepare expects all inputs to share identical problem_sizes")

    threshold_m = _env_int("AISP_NVFP4_GROUP_GEMM_HYBRID_M_THRESHOLD", 128)
    small_idx = [i for i, (m, _, _, _) in enumerate(problem_sizes) if int(m) <= threshold_m]
    large_idx = [i for i, (m, _, _, _) in enumerate(problem_sizes) if int(m) > threshold_m]

    small_prepared = None
    if small_idx:
        small_data_list = [_slice_input_groups(data, small_idx) for data in data_list]
        small_prepared = _prepare_cutlass_cached_with_overrides(
            small_data_list,
            variant=small_variant,
            overrides=_hybrid_overrides(variant=small_variant, side="small"),
        )

    large_prepared = None
    if large_idx:
        large_data_list = [_slice_input_groups(data, large_idx) for data in data_list]
        large_prepared = _prepare_cutlass_cached_with_overrides(
            large_data_list,
            variant=large_variant,
            overrides=_hybrid_overrides(variant=large_variant, side="large"),
        )

    out: List[tuple[Any, ...]] = []
    for i, data in enumerate(data_list):
        abc_tensors, sfasfb_tensors, sfasfb_reordered_tensors, problem_sizes_i = data
        small_ctx = small_prepared[i][4] if small_prepared is not None else None
        large_ctx = large_prepared[i][4] if large_prepared is not None else None
        out.append(
            (
                abc_tensors,
                sfasfb_tensors,
                sfasfb_reordered_tensors,
                problem_sizes_i,
                {
                    "small_plan": small_ctx["plan"] if small_ctx is not None else None,
                    "large_plan": large_ctx["plan"] if large_ctx is not None else None,
                    "outputs": [abc_tensors[j][2] for j in range(len(abc_tensors))],
                },
            )
        )
    return out


def prepare_cutlass_cached_hybrid_1sm_n128_2sm(data_list: Sequence[input_t]) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare hybrid plans: small-M groups -> 1SM N128, large-M groups -> 2SM."""
    return _prepare_cutlass_cached_hybrid(data_list, small_variant="1sm_n128", large_variant="2sm")


def prepare_cutlass_cached_hybrid_1sm_n64_2sm(data_list: Sequence[input_t]) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare hybrid plans: small-M groups -> 1SM N64, large-M groups -> 2SM."""
    return _prepare_cutlass_cached_hybrid(data_list, small_variant="1sm_n64", large_variant="2sm")


def prepare_cutlass_cached_hybrid_1sm_n128_2sm_graph(
    data_list: Sequence[input_t],
) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare hybrid plans with CUDA Graph replay capture."""
    return _attach_graphs(
        _prepare_cutlass_cached_hybrid(data_list, small_variant="1sm_n128", large_variant="2sm")
    )


def prepare_cutlass_cached_hybrid_1sm_n64_2sm_graph(
    data_list: Sequence[input_t],
) -> Optional[Sequence[tuple[Any, ...]]]:
    """Prepare hybrid plans with CUDA Graph replay capture."""
    return _attach_graphs(
        _prepare_cutlass_cached_hybrid(data_list, small_variant="1sm_n64", large_variant="2sm")
    )


def custom_kernel_cutlass_cached(data: tuple[Any, ...]) -> output_t:
    """Execute the CUTLASS NVFP4 grouped GEMM kernel with cached allocations."""
    _, _, _, _, ctx = data
    graph = ctx.get("graph")
    if graph is not None:
        graph.replay()
    else:
        stream_graphs = ctx.get("stream_graphs")
        if stream_graphs:
            _run_stream_graphs(ctx, stream_graphs)
            return ctx["outputs"]
        plan = ctx.get("plan")
        if plan is not None:
            plan.run()
        else:
            plans = ctx.get("plans")
            if plans is None:
                raise RuntimeError("missing plan/plans in CUTLASS cached context")
            _run_plans_with_optional_stream_overlap(ctx, plans)
    return ctx["outputs"]


def custom_kernel_cutlass_cached_hybrid(data: tuple[Any, ...]) -> output_t:
    """Execute hybrid CUTLASS plans (small-M plan + large-M plan)."""
    _, _, _, _, ctx = data
    graph = ctx.get("graph")
    if graph is not None:
        graph.replay()
    else:
        small_plan = ctx.get("small_plan")
        large_plan = ctx.get("large_plan")
        if small_plan is not None:
            small_plan.run()
        if large_plan is not None:
            large_plan.run()
    return ctx["outputs"]


__all__ = [
    "prepare_cutlass_cached",
    "prepare_cutlass_cached_1sm_n64",
    "prepare_cutlass_cached_1sm_n128",
    "prepare_cutlass_cached_2sm",
    "prepare_cutlass_cached_2sm_mxf4",
    "prepare_cutlass_cached_2sm_s1",
    "prepare_cutlass_cached_2sm_s2",
    "prepare_cutlass_cached_2sm_s3",
    "prepare_cutlass_cached_2sm_s4",
    "prepare_cutlass_cached_2sm_n32",
    "prepare_cutlass_cached_2sm_n64",
    "prepare_cutlass_cached_2sm_n64_s1",
    "prepare_cutlass_cached_2sm_n64_s2",
    "prepare_cutlass_cached_2sm_n64_s3",
    "prepare_cutlass_cached_2sm_n64_s4",
    "prepare_cutlass_cached_2sm_n64_s5",
    "prepare_cutlass_cached_2sm_n128",
    "prepare_cutlass_cached_2sm_n128_s1",
    "prepare_cutlass_cached_2sm_n128_s2",
    "prepare_cutlass_cached_2sm_n128_s3",
    "prepare_cutlass_cached_2sm_n128_s4",
    "prepare_cutlass_cached_2sm_persistent",
    "prepare_cutlass_cached_2sm_mxf4_persistent",
    "prepare_cutlass_cached_2sm_s1_persistent",
    "prepare_cutlass_cached_2sm_s2_persistent",
    "prepare_cutlass_cached_2sm_s3_persistent",
    "prepare_cutlass_cached_2sm_s4_persistent",
    "prepare_cutlass_cached_2sm_n32_persistent",
    "prepare_cutlass_cached_2sm_n64_persistent",
    "prepare_cutlass_cached_2sm_n64_s1_persistent",
    "prepare_cutlass_cached_2sm_n64_s2_persistent",
    "prepare_cutlass_cached_2sm_n64_s3_persistent",
    "prepare_cutlass_cached_2sm_n64_s4_persistent",
    "prepare_cutlass_cached_2sm_n64_s5_persistent",
    "prepare_cutlass_cached_2sm_n128_persistent",
    "prepare_cutlass_cached_2sm_n128_s1_persistent",
    "prepare_cutlass_cached_2sm_n128_s2_persistent",
    "prepare_cutlass_cached_2sm_n128_s3_persistent",
    "prepare_cutlass_cached_2sm_n128_s4_persistent",
    "prepare_cutlass_cached_2sm_persistent_graph",
    "prepare_cutlass_cached_2sm_mxf4_persistent_graph",
    "prepare_cutlass_cached_2sm_s1_persistent_graph",
    "prepare_cutlass_cached_2sm_s2_persistent_graph",
    "prepare_cutlass_cached_2sm_s3_persistent_graph",
    "prepare_cutlass_cached_2sm_s4_persistent_graph",
    "prepare_cutlass_cached_2sm_n32_persistent_graph",
    "prepare_cutlass_cached_2sm_n64_persistent_graph",
    "prepare_cutlass_cached_2sm_n64_s1_persistent_graph",
    "prepare_cutlass_cached_2sm_n64_s2_persistent_graph",
    "prepare_cutlass_cached_2sm_n64_s3_persistent_graph",
    "prepare_cutlass_cached_2sm_n64_s4_persistent_graph",
    "prepare_cutlass_cached_2sm_n64_s5_persistent_graph",
    "prepare_cutlass_cached_2sm_n128_persistent_graph",
    "prepare_cutlass_cached_2sm_n128_s1_persistent_graph",
    "prepare_cutlass_cached_2sm_n128_s2_persistent_graph",
    "prepare_cutlass_cached_2sm_n128_s3_persistent_graph",
    "prepare_cutlass_cached_2sm_n128_s4_persistent_graph",
    "prepare_cutlass_cached_graph",
    "prepare_cutlass_cached_1sm_n64_graph",
    "prepare_cutlass_cached_1sm_n128_graph",
    "prepare_cutlass_cached_2sm_graph",
    "prepare_cutlass_cached_2sm_mxf4_graph",
    "prepare_cutlass_cached_2sm_s1_graph",
    "prepare_cutlass_cached_2sm_s2_graph",
    "prepare_cutlass_cached_2sm_s3_graph",
    "prepare_cutlass_cached_2sm_s4_graph",
    "prepare_cutlass_cached_2sm_n32_graph",
    "prepare_cutlass_cached_2sm_n64_graph",
    "prepare_cutlass_cached_2sm_n64_s1_graph",
    "prepare_cutlass_cached_2sm_n64_s2_graph",
    "prepare_cutlass_cached_2sm_n64_s3_graph",
    "prepare_cutlass_cached_2sm_n64_s4_graph",
    "prepare_cutlass_cached_2sm_n64_s5_graph",
    "prepare_cutlass_cached_2sm_n128_graph",
    "prepare_cutlass_cached_2sm_n128_s1_graph",
    "prepare_cutlass_cached_2sm_n128_s2_graph",
    "prepare_cutlass_cached_2sm_n128_s3_graph",
    "prepare_cutlass_cached_2sm_n128_s4_graph",
    "prepare_cutlass_cached_hybrid_1sm_n128_2sm",
    "prepare_cutlass_cached_hybrid_1sm_n64_2sm",
    "prepare_cutlass_cached_hybrid_1sm_n128_2sm_graph",
    "prepare_cutlass_cached_hybrid_1sm_n64_2sm_graph",
    "custom_kernel_cutlass_cached",
    "custom_kernel_cutlass_cached_hybrid",
]
