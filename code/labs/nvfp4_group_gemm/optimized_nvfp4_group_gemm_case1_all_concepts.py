"""NVFP4 grouped GEMM (competition case 1) - "all concepts" variant.

Same structure as case0: prefer CUTLASS cached grouped GEMM with launch tuning; fall back to
CuTe DSL cached if CUTLASS is unavailable.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("AISP_NVFP4_GROUP_GEMM_USE_PDL", "1")
os.environ.setdefault("AISP_NVFP4_GROUP_GEMM_CLUSTER_M", "2")
os.environ.setdefault("AISP_NVFP4_GROUP_GEMM_CLUSTER_N", "1")
os.environ.setdefault("AISP_NVFP4_GROUP_GEMM_RASTER_ORDER", "0")

os.environ.setdefault("CUTE_DSL_DISABLE_FILE_CACHING", "1")
os.environ.setdefault("AISP_NVFP4_GROUP_GEMM_NUM_AB_STAGE", "4")
os.environ.setdefault("AISP_NVFP4_GROUP_GEMM_NUM_ACC_STAGE", "2")

from core.harness.benchmark_harness import BaseBenchmark
from labs.nvfp4_group_gemm.nvfp4_group_gemm_common import (
    COMPETITION_CASES,
    NVFP4GroupGemmBenchmark,
    attach_benchmark_metadata,
)


def _select_impl():
    try:
        from labs.nvfp4_group_gemm.cutlass_extension import load_cutlass_nvfp4_grouped_gemm_sm100
        from labs.nvfp4_group_gemm.cutlass_submission_cached import (
            custom_kernel_cutlass_cached,
            prepare_cutlass_cached_2sm,
        )

        load_cutlass_nvfp4_grouped_gemm_sm100(verbose=False)
        return custom_kernel_cutlass_cached, prepare_cutlass_cached_2sm, "cutlass_cached_2sm"
    except Exception:
        from labs.nvfp4_group_gemm.cute_submission_cached import custom_kernel_cached, prepare_cached

        return custom_kernel_cached, prepare_cached, "cute_cached_fallback"


def get_benchmark() -> BaseBenchmark:
    case = COMPETITION_CASES[1]
    custom_kernel, prepare, impl_name = _select_impl()
    bench = NVFP4GroupGemmBenchmark(
        case=case,
        custom_kernel=custom_kernel,
        prepare=prepare,
        inputs_per_iteration=15,
        name=f"nvfp4_group_gemm_{case.name}_optimized_all_concepts_{impl_name}",
    )
    return attach_benchmark_metadata(bench, __file__)


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
