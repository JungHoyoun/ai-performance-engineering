"""CuTe DSL NVFP4 grouped GEMM (competition case 0) - tuned tiling + pipelining.

Goal: push the reference CuTe submission toward the leaderboard by adjusting kernel config
knobs while keeping the allocation-cached hot path (prepare_cached()).

This keeps the same overall structure as the GPU MODE starter, but changes:
- MMA CTA tile: N=256 (was 128)
- Mainloop staging: AB=4, ACC=2 (was 1/1)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Keep CuTe DSL stable.
os.environ.setdefault("CUTE_DSL_DISABLE_FILE_CACHING", "1")

# CuTe kernel tuning knobs (consumed at module import time in cute_submission.py).
os.environ.setdefault("AISP_NVFP4_GROUP_GEMM_MMA_TILER_MNK", "128 256 256")
os.environ.setdefault("AISP_NVFP4_GROUP_GEMM_NUM_AB_STAGE", "4")
os.environ.setdefault("AISP_NVFP4_GROUP_GEMM_NUM_ACC_STAGE", "2")
# NOTE: CUTLASS DSL currently enforces TMEM allocation columns in [0, 512].
# Leave this at the default unless we also change the kernel to reduce TMEM footprint.
os.environ.setdefault("AISP_NVFP4_GROUP_GEMM_TMEM_COLS", "512")

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
        name=f"nvfp4_group_gemm_{case.name}_optimized_cute_tuned_n256",
    )
    return attach_benchmark_metadata(bench, __file__)


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
