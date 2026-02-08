"""Vendor ceiling: torch._scaled_mm_v2 (competition case 0).

This uses cuBLASLt via PyTorch to establish a performance target under the exact
competition shapes. Not intended as the final non-cuBLAS submission.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.harness.benchmark_harness import BaseBenchmark
from labs.nvfp4_group_gemm.nvfp4_group_gemm_common import (
    COMPETITION_CASES,
    NVFP4GroupGemmBenchmark,
    attach_benchmark_metadata,
)
from labs.nvfp4_group_gemm.torch_scaled_mm_submission import (
    custom_kernel_scaled_mm_v2,
    prepare_torch_scaled_mm_v2,
)


def get_benchmark() -> BaseBenchmark:
    case = COMPETITION_CASES[0]
    bench = NVFP4GroupGemmBenchmark(
        case=case,
        custom_kernel=custom_kernel_scaled_mm_v2,
        prepare=prepare_torch_scaled_mm_v2,
        inputs_per_iteration=15,
        name=f"nvfp4_group_gemm_{case.name}_optimized_scaled_mm",
    )
    return attach_benchmark_metadata(bench, __file__)


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
