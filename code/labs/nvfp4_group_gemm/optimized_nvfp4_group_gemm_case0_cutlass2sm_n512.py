"""CUTLASS NVFP4 grouped GEMM (competition case 0) - 2SM MMA, cluster_n=2 (PDL enabled).

Single-launch CUTLASS grouped GEMM (device-side scheduling) with cached metadata/pointer arrays.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# CUTLASS SM100 block-scaled NVFP4 does not currently support N=512 tile shapes. We keep this
# wrapper (and target name) but repurpose it to test a more aggressive cluster shape.
os.environ.setdefault("AISP_NVFP4_GROUP_GEMM_USE_PDL", "1")
os.environ.setdefault("AISP_NVFP4_GROUP_GEMM_CLUSTER_M", "2")
os.environ.setdefault("AISP_NVFP4_GROUP_GEMM_CLUSTER_N", "2")

from core.harness.benchmark_harness import BaseBenchmark
from labs.nvfp4_group_gemm.cutlass_submission_cached import (
    custom_kernel_cutlass_cached,
    prepare_cutlass_cached_2sm,
)
from labs.nvfp4_group_gemm.nvfp4_group_gemm_common import (
    COMPETITION_CASES,
    NVFP4GroupGemmBenchmark,
    attach_benchmark_metadata,
)


def get_benchmark() -> BaseBenchmark:
    case = COMPETITION_CASES[0]
    bench = NVFP4GroupGemmBenchmark(
        case=case,
        custom_kernel=custom_kernel_cutlass_cached,
        prepare=prepare_cutlass_cached_2sm,
        inputs_per_iteration=15,
        name=f"nvfp4_group_gemm_{case.name}_optimized_cutlass_cached_2sm_cluster_n2_pdl",
    )
    return attach_benchmark_metadata(bench, __file__)


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
