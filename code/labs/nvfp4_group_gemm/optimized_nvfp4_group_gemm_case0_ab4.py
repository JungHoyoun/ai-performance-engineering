"""Optimized NVFP4 grouped GEMM (competition case 0, tuned pipeline).

This variant increases the mainloop staging to improve overlap and throughput while still
using the allocation-cached hot path.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Keep CuTe DSL stable and explicit.
os.environ.setdefault("CUTE_DSL_DISABLE_FILE_CACHING", "1")

# Local tuning knobs (applied at module import time inside cute_submission.py).
os.environ.setdefault("AISP_NVFP4_GROUP_GEMM_NUM_AB_STAGE", "4")
os.environ.setdefault("AISP_NVFP4_GROUP_GEMM_NUM_ACC_STAGE", "2")

from core.harness.benchmark_harness import BaseBenchmark
from labs.nvfp4_group_gemm.cute_submission_cached import custom_kernel_cached, prepare_cached
from labs.nvfp4_group_gemm.nvfp4_group_gemm_common import (
    COMPETITION_CASES,
    NVFP4GroupGemmBenchmark,
    attach_benchmark_metadata,
)


def get_benchmark() -> BaseBenchmark:
    case = COMPETITION_CASES[0]
    bench = NVFP4GroupGemmBenchmark(
        case=case,
        custom_kernel=custom_kernel_cached,
        prepare=prepare_cached,
        inputs_per_iteration=15,
        name=f"nvfp4_group_gemm_{case.name}_optimized_cached_ab4",
    )
    return attach_benchmark_metadata(bench, __file__)


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)

