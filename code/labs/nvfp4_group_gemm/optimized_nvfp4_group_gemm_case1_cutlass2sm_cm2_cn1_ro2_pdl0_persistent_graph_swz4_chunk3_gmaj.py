"""CUTLASS NVFP4 grouped GEMM (competition case 1) - persistent graph swizzle4, chunk3, group-major order."""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ["AISP_NVFP4_GROUP_GEMM_CLUSTER_M"] = "2"
os.environ["AISP_NVFP4_GROUP_GEMM_CLUSTER_N"] = "1"
os.environ["AISP_NVFP4_GROUP_GEMM_RASTER_ORDER"] = "2"
os.environ["AISP_NVFP4_GROUP_GEMM_USE_PDL"] = "0"
os.environ["AISP_NVFP4_GROUP_GEMM_MAX_SWIZZLE"] = "4"
os.environ["AISP_NVFP4_GROUP_GEMM_PERSISTENT_REQUEST_CHUNK"] = "3"
os.environ["AISP_NVFP4_GROUP_GEMM_PERSISTENT_TASK_ORDER"] = "group_major"

from core.harness.benchmark_harness import BaseBenchmark
from labs.nvfp4_group_gemm.cutlass_submission_cached import (
    custom_kernel_cutlass_cached,
    prepare_cutlass_cached_2sm_persistent_graph,
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
        custom_kernel=custom_kernel_cutlass_cached,
        prepare=prepare_cutlass_cached_2sm_persistent_graph,
        inputs_per_iteration=15,
        capture_iter_graph=False,
        name=(
            f"nvfp4_group_gemm_{case.name}_optimized_cutlass_cached_2sm_"
            "cm2_cn1_ro2_pdl0_persistent_graph_swz4_chunk3_gmaj"
        ),
    )
    return attach_benchmark_metadata(bench, __file__)


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
