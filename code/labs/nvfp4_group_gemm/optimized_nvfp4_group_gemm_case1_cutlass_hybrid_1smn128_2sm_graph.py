"""CUTLASS NVFP4 grouped GEMM (competition case 1) - hybrid with CUDA Graph replay."""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("AISP_NVFP4_GROUP_GEMM_HYBRID_M_THRESHOLD", "128")
os.environ.setdefault("AISP_NVFP4_GROUP_GEMM_HYBRID_SMALL_CLUSTER_M", "1")
os.environ.setdefault("AISP_NVFP4_GROUP_GEMM_HYBRID_SMALL_CLUSTER_N", "1")
os.environ.setdefault("AISP_NVFP4_GROUP_GEMM_HYBRID_SMALL_RASTER_ORDER", "0")
os.environ.setdefault("AISP_NVFP4_GROUP_GEMM_HYBRID_SMALL_USE_PDL", "0")
os.environ.setdefault("AISP_NVFP4_GROUP_GEMM_HYBRID_LARGE_CLUSTER_M", "2")
os.environ.setdefault("AISP_NVFP4_GROUP_GEMM_HYBRID_LARGE_CLUSTER_N", "1")
os.environ.setdefault("AISP_NVFP4_GROUP_GEMM_HYBRID_LARGE_RASTER_ORDER", "2")
os.environ.setdefault("AISP_NVFP4_GROUP_GEMM_HYBRID_LARGE_USE_PDL", "1")

from core.harness.benchmark_harness import BaseBenchmark
from labs.nvfp4_group_gemm.cutlass_submission_cached import (
    custom_kernel_cutlass_cached_hybrid,
    prepare_cutlass_cached_hybrid_1sm_n128_2sm_graph,
)
from labs.nvfp4_group_gemm.nvfp4_group_gemm_common import (
    COMPETITION_CASES,
    NVFP4GroupGemmBenchmark,
    attach_benchmark_metadata,
)


def get_benchmark() -> BaseBenchmark:
    case = COMPETITION_CASES[1]
    bench = NVFP4GroupGemmBenchmark(
        case=case,
        custom_kernel=custom_kernel_cutlass_cached_hybrid,
        prepare=prepare_cutlass_cached_hybrid_1sm_n128_2sm_graph,
        inputs_per_iteration=15,
        name=f"nvfp4_group_gemm_{case.name}_optimized_cutlass_cached_hybrid_1smn128_2sm_graph",
    )
    return attach_benchmark_metadata(bench, __file__)


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
