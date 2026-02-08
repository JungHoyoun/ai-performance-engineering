"""Optimized NVFP4 grouped GEMM (competition case 3)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("CUTE_DSL_DISABLE_FILE_CACHING", "1")

from core.harness.benchmark_harness import BaseBenchmark
from labs.nvfp4_group_gemm.cute_submission_cached import custom_kernel_cached, prepare_cached
from labs.nvfp4_group_gemm.nvfp4_group_gemm_common import (
    COMPETITION_CASES,
    NVFP4GroupGemmBenchmark,
    attach_benchmark_metadata,
)


def get_benchmark() -> BaseBenchmark:
    case = COMPETITION_CASES[3]
    bench = NVFP4GroupGemmBenchmark(
        case=case,
        custom_kernel=custom_kernel_cached,
        prepare=prepare_cached,
        inputs_per_iteration=15,
        name=f"nvfp4_group_gemm_{case.name}_optimized_cached",
    )
    return attach_benchmark_metadata(bench, __file__)


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
