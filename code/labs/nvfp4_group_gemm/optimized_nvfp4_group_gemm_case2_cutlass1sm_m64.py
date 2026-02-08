"""CUTLASS NVFP4 grouped GEMM (competition case 2) - 1SM MMA, N=128 tile (PDL enabled).

Single-launch CUTLASS grouped GEMM (device-side scheduling) with cached metadata/pointer arrays.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# NOTE: SM100 block-scaled NVFP4 CUTLASS builders fix TileShape_M=128 for 1SM schedules.
# This wrapper keeps the filename stable but focuses on a tunable knob (PDL).
os.environ["AISP_NVFP4_GROUP_GEMM_USE_PDL"] = "1"

from core.harness.benchmark_harness import BaseBenchmark
from labs.nvfp4_group_gemm.cutlass_submission_cached import (
    custom_kernel_cutlass_cached,
    prepare_cutlass_cached_1sm_n128,
)
from labs.nvfp4_group_gemm.nvfp4_group_gemm_common import (
    COMPETITION_CASES,
    NVFP4GroupGemmBenchmark,
    attach_benchmark_metadata,
)


def get_benchmark() -> BaseBenchmark:
    case = COMPETITION_CASES[2]
    bench = NVFP4GroupGemmBenchmark(
        case=case,
        custom_kernel=custom_kernel_cutlass_cached,
        prepare=prepare_cutlass_cached_1sm_n128,
        inputs_per_iteration=15,
        name=f"nvfp4_group_gemm_{case.name}_optimized_cutlass_cached_1sm_n128_pdl",
    )
    return attach_benchmark_metadata(bench, __file__)


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
