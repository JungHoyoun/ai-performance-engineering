"""Custom Blackwell low-footprint NVFP4 grouped GEMM (competition case 1)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ["AISP_NVFP4_GROUP_GEMM_CUSTOM_BLOCK_M"] = "2"
os.environ["AISP_NVFP4_GROUP_GEMM_CUSTOM_BLOCK_N"] = "32"
os.environ["AISP_NVFP4_GROUP_GEMM_CUSTOM_VEC_N"] = "1"

from core.harness.benchmark_harness import BaseBenchmark
from labs.nvfp4_group_gemm.custom_cuda_submission import (
    custom_kernel_custom_cuda,
    prepare_custom_cuda,
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
        custom_kernel=custom_kernel_custom_cuda,
        prepare=prepare_custom_cuda,
        inputs_per_iteration=15,
        capture_iter_graph=True,
        name=f"nvfp4_group_gemm_{case.name}_optimized_custom_blackwell_lowfootprint_ab1_acc1_tpb32_tmem64",
    )
    return attach_benchmark_metadata(bench, __file__)


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
