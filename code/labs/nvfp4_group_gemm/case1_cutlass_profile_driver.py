"""Direct profile/timing driver for case1 CUTLASS NVFP4 grouped GEMM variants.

This bypasses benchmark-harness orchestration to speed up local profile-driven iteration.
It still uses the same case1 shapes and CUTLASS prepare/custom-kernel path.
"""

from __future__ import annotations

import argparse
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Callable, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPO_ROOT))

from core.harness.benchmark_harness import lock_gpu_clocks
from labs.nvfp4_group_gemm.nvfp4_group_gemm_common import COMPETITION_CASES, input_t
from labs.nvfp4_group_gemm.nvfp4_group_gemm_inputs import generate_input


def _resolve_prepare_cutlass(
    variant: str,
    persistent: bool,
    iter_graph: bool,
) -> Callable[[Sequence[input_t]], Sequence[tuple]]:
    import labs.nvfp4_group_gemm.cutlass_submission_cached as cutlass_cached

    suffix = "_persistent_graph" if (persistent and iter_graph) else "_persistent" if persistent else "_graph" if iter_graph else ""
    fn_name = f"prepare_cutlass_cached_{variant}{suffix}"
    fn = getattr(cutlass_cached, fn_name, None)
    if fn is None:
        raise ValueError(f"Unknown prepare function: {fn_name}")
    return fn


def _resolve_backend(args: argparse.Namespace) -> tuple[Callable[[Sequence[input_t]], Sequence[tuple]], Callable[[tuple], object]]:
    if args.backend == "custom":
        # Import lazily after env overrides are applied so custom-kernel
        # stage/thread/tmem knobs are honored during module initialization.
        from labs.nvfp4_group_gemm.cute_submission_cached import (
            custom_kernel_cached as custom_kernel_cute_cached,
        )
        from labs.nvfp4_group_gemm.cute_submission_cached import (
            prepare_cached as prepare_cute_cached,
        )

        return prepare_cute_cached, custom_kernel_cute_cached
    if args.backend == "custom_fused":
        # Fused-request custom backend: one launch can cover all requests.
        from labs.nvfp4_group_gemm.cute_submission_cached_fused import (
            custom_kernel_cached_fused_requests as custom_kernel_cute_cached_fused,
        )
        from labs.nvfp4_group_gemm.cute_submission_cached_fused import (
            prepare_cached_fused_requests as prepare_cute_cached_fused,
        )

        return prepare_cute_cached_fused, custom_kernel_cute_cached_fused
    from labs.nvfp4_group_gemm.cutlass_submission_cached import custom_kernel_cutlass_cached

    prepare = _resolve_prepare_cutlass(args.variant, args.persistent, args.iter_graph)
    return prepare, custom_kernel_cutlass_cached


def _set_env(key: str, value: int | str | bool) -> None:
    if isinstance(value, bool):
        os.environ[key] = "1" if value else "0"
    else:
        os.environ[key] = str(value)


def _configure_runtime_env(args: argparse.Namespace) -> None:
    _set_env("AISP_NVFP4_GROUP_GEMM_CLUSTER_M", args.cluster_m)
    _set_env("AISP_NVFP4_GROUP_GEMM_CLUSTER_N", args.cluster_n)
    _set_env("AISP_NVFP4_GROUP_GEMM_CLUSTER_FALLBACK_M", args.cluster_fallback_m)
    _set_env("AISP_NVFP4_GROUP_GEMM_CLUSTER_FALLBACK_N", args.cluster_fallback_n)
    _set_env("AISP_NVFP4_GROUP_GEMM_RASTER_ORDER", args.raster_order)
    if args.max_sm_count is not None and int(args.max_sm_count) > 0:
        _set_env("AISP_NVFP4_GROUP_GEMM_MAX_SM_COUNT", int(args.max_sm_count))
    else:
        os.environ.pop("AISP_NVFP4_GROUP_GEMM_MAX_SM_COUNT", None)
    _set_env("AISP_NVFP4_GROUP_GEMM_USE_PDL", bool(args.use_pdl))
    _set_env("AISP_NVFP4_GROUP_GEMM_MAX_SWIZZLE", args.max_swizzle)
    _set_env("AISP_NVFP4_GROUP_GEMM_PERSISTENT_REQUEST_CHUNK", args.persistent_request_chunk)
    _set_env("AISP_NVFP4_GROUP_GEMM_PERSISTENT_CONCURRENT_STREAMS", args.persistent_concurrent_streams)
    _set_env("AISP_NVFP4_GROUP_GEMM_PERSISTENT_GROUP_ORDER", args.persistent_group_order)
    _set_env("AISP_NVFP4_GROUP_GEMM_PERSISTENT_TASK_ORDER", args.persistent_task_order)
    _set_env("AISP_NVFP4_GROUP_GEMM_PERSISTENT_STREAM_BALANCE", bool(args.persistent_stream_balance))
    _set_env(
        "AISP_NVFP4_GROUP_GEMM_PERSISTENT_STREAM_FRONTLOAD_HEAVY",
        bool(args.persistent_stream_frontload_heavy),
    )
    if args.num_ab_stage is not None:
        _set_env("AISP_NVFP4_GROUP_GEMM_NUM_AB_STAGE", int(args.num_ab_stage))
    if args.num_acc_stage is not None:
        _set_env("AISP_NVFP4_GROUP_GEMM_NUM_ACC_STAGE", int(args.num_acc_stage))
    if args.threads_per_cta is not None:
        _set_env("AISP_NVFP4_GROUP_GEMM_THREADS_PER_CTA", int(args.threads_per_cta))
    if args.tmem_cols is not None:
        _set_env("AISP_NVFP4_GROUP_GEMM_TMEM_COLS", int(args.tmem_cols))
    if args.mma_tiler_mnk:
        os.environ["AISP_NVFP4_GROUP_GEMM_MMA_TILER_MNK"] = str(args.mma_tiler_mnk)
    if args.custom_fused_capture_graph:
        _set_env("AISP_NVFP4_GROUP_GEMM_CUSTOM_FUSED_CAPTURE_GRAPH", True)


def _build_inputs(case_idx: int, inputs_per_iteration: int) -> list[input_t]:
    case = COMPETITION_CASES[case_idx]
    data_list: list[input_t] = []
    for i in range(inputs_per_iteration):
        seed = int(case.seed) + 42 * i
        data_list.append(
            generate_input(
                m=case.m,
                n=case.n,
                k=case.k,
                g=case.g,
                seed=seed,
            )
        )
    return data_list


def _cuda_profiler_start() -> None:
    """Gate Nsight Compute capture when launched with --profile-from-start off."""
    try:
        rc = int(torch.cuda.cudart().cudaProfilerStart())
    except Exception as exc:
        raise RuntimeError(f"cudaProfilerStart failed: {exc}") from exc
    if rc != 0:
        raise RuntimeError(f"cudaProfilerStart returned error code {rc}")


def _cuda_profiler_stop() -> None:
    try:
        rc = int(torch.cuda.cudart().cudaProfilerStop())
    except Exception as exc:
        raise RuntimeError(f"cudaProfilerStop failed: {exc}") from exc
    if rc != 0:
        raise RuntimeError(f"cudaProfilerStop returned error code {rc}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", type=str, default="cutlass", choices=("cutlass", "custom", "custom_fused"))
    parser.add_argument("--case-index", type=int, default=1)
    parser.add_argument("--variant", type=str, default="2sm_mxf4")
    parser.add_argument("--persistent", action="store_true", default=True)
    parser.add_argument("--no-persistent", dest="persistent", action="store_false")
    parser.add_argument("--iter-graph", action="store_true", default=True)
    parser.add_argument("--no-iter-graph", dest="iter_graph", action="store_false")
    parser.add_argument("--inputs-per-iteration", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--cluster-m", type=int, default=2)
    parser.add_argument("--cluster-n", type=int, default=1)
    parser.add_argument("--cluster-fallback-m", type=int, default=2)
    parser.add_argument("--cluster-fallback-n", type=int, default=1)
    parser.add_argument("--raster-order", type=int, default=2)
    parser.add_argument(
        "--max-sm-count",
        type=int,
        default=None,
        help="Optional cap for active SMs used by CUTLASS plan initialization/run.",
    )
    parser.add_argument("--use-pdl", type=int, default=1, choices=(0, 1))
    parser.add_argument("--max-swizzle", type=int, default=2)
    parser.add_argument("--persistent-request-chunk", type=int, default=3)
    parser.add_argument("--persistent-concurrent-streams", type=int, default=3)
    parser.add_argument("--persistent-group-order", type=str, default="m_asc")
    parser.add_argument("--persistent-task-order", type=str, default="group_major")
    parser.add_argument("--persistent-stream-balance", type=int, default=1, choices=(0, 1))
    parser.add_argument("--persistent-stream-frontload-heavy", type=int, default=1, choices=(0, 1))
    parser.add_argument("--num-ab-stage", type=int, default=None, help="Custom backend only: CuTe A/B stage depth.")
    parser.add_argument("--num-acc-stage", type=int, default=None, help="Custom backend only: CuTe accumulator stage depth.")
    parser.add_argument("--threads-per-cta", type=int, default=None, help="Custom backend only: threads per CTA.")
    parser.add_argument("--tmem-cols", type=int, default=None, help="Custom backend only: TMEM columns.")
    parser.add_argument(
        "--mma-tiler-mnk",
        type=str,
        default=None,
        help="Custom backend only: override AISP_NVFP4_GROUP_GEMM_MMA_TILER_MNK (e.g., '128,256,256').",
    )
    parser.add_argument(
        "--custom-fused-capture-graph",
        action="store_true",
        default=False,
        help="Custom fused backend only: attempt internal fused CUDA graph capture/replay.",
    )
    parser.add_argument("--lock-clocks", action="store_true", default=True)
    parser.add_argument("--no-lock-clocks", dest="lock_clocks", action="store_false")
    parser.add_argument("--sm-clock-mhz", type=int, default=1500)
    parser.add_argument("--mem-clock-mhz", type=int, default=3996)
    parser.add_argument(
        "--cuda-profiler-range",
        action="store_true",
        default=False,
        help="Call cudaProfilerStart/Stop around the timed loop (for ncu --profile-from-start off).",
    )
    parser.add_argument(
        "--nvtx-range",
        type=str,
        default="",
        help="Optional NVTX range label around each kernel call (use with ncu --nvtx --nvtx-include).",
    )
    args = parser.parse_args()

    if args.inputs_per_iteration <= 0:
        raise ValueError("--inputs-per-iteration must be > 0")
    if args.iters <= 0:
        raise ValueError("--iters must be > 0")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")

    _configure_runtime_env(args)
    prepare, run_kernel = _resolve_backend(args)

    clock_ctx = (
        lock_gpu_clocks(device=0, sm_clock_mhz=args.sm_clock_mhz, mem_clock_mhz=args.mem_clock_mhz)
        if args.lock_clocks
        else nullcontext()
    )

    with clock_ctx:
        data_list = _build_inputs(args.case_index, args.inputs_per_iteration)
        prepared = prepare(data_list)
        if prepared is None:
            raise RuntimeError("prepare returned None")

        prepared_list = list(prepared)
        if not prepared_list:
            raise RuntimeError("prepare produced an empty list")

        for _ in range(args.warmup):
            for data in prepared_list:
                if args.nvtx_range:
                    torch.cuda.nvtx.range_push(args.nvtx_range)
                    run_kernel(data)
                    torch.cuda.nvtx.range_pop()
                else:
                    run_kernel(data)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        if args.cuda_profiler_range:
            _cuda_profiler_start()
        try:
            start.record()
            for _ in range(args.iters):
                for data in prepared_list:
                    if args.nvtx_range:
                        torch.cuda.nvtx.range_push(args.nvtx_range)
                        run_kernel(data)
                        torch.cuda.nvtx.range_pop()
                    else:
                        run_kernel(data)
            end.record()
            end.synchronize()
        finally:
            if args.cuda_profiler_range:
                _cuda_profiler_stop()

        total_ms = float(start.elapsed_time(end))
        calls = args.iters * len(prepared_list)
        us_per_call = (total_ms * 1000.0) / float(calls)

        print(
            "RESULT "
            f"case_index={args.case_index} backend={args.backend} variant={args.variant} persistent={int(args.persistent)} "
            f"inputs_per_iteration={args.inputs_per_iteration} prepared_len={len(prepared_list)} "
            f"warmup={args.warmup} iters={args.iters} total_ms={total_ms:.6f} us_per_call={us_per_call:.6f} "
            f"cluster_m={args.cluster_m} cluster_n={args.cluster_n} raster_order={args.raster_order} "
            f"cluster_fallback_m={args.cluster_fallback_m} cluster_fallback_n={args.cluster_fallback_n} "
            f"max_sm_count={args.max_sm_count if args.max_sm_count is not None else 'full'} "
            f"use_pdl={args.use_pdl} max_swizzle={args.max_swizzle} "
            f"persistent_request_chunk={args.persistent_request_chunk} "
            f"persistent_concurrent_streams={args.persistent_concurrent_streams} "
            f"persistent_group_order={args.persistent_group_order} "
            f"persistent_task_order={args.persistent_task_order} "
            f"num_ab_stage={args.num_ab_stage} num_acc_stage={args.num_acc_stage} "
            f"threads_per_cta={args.threads_per_cta} tmem_cols={args.tmem_cols}"
        )


if __name__ == "__main__":
    main()
