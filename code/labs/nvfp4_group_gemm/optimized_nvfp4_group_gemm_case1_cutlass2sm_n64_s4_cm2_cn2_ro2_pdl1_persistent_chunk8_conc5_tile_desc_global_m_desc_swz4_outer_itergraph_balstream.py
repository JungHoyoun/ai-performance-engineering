"""CUTLASS NVFP4 grouped GEMM (competition case 1) - n64_s4 chunk8 conc5 tile_desc_global + m_desc + swz4 + outer iter-graph + balanced stream assignment."""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ["AISP_NVFP4_GROUP_GEMM_CLUSTER_M"] = "2"
os.environ["AISP_NVFP4_GROUP_GEMM_CLUSTER_N"] = "2"
os.environ["AISP_NVFP4_GROUP_GEMM_RASTER_ORDER"] = "2"
os.environ["AISP_NVFP4_GROUP_GEMM_USE_PDL"] = "1"
os.environ["AISP_NVFP4_GROUP_GEMM_PERSISTENT_REQUEST_CHUNK"] = "8"
os.environ["AISP_NVFP4_GROUP_GEMM_PERSISTENT_CONCURRENT_STREAMS"] = "5"
os.environ["AISP_NVFP4_GROUP_GEMM_PERSISTENT_GROUP_ORDER"] = "m_desc"
os.environ["AISP_NVFP4_GROUP_GEMM_PERSISTENT_TASK_ORDER"] = "tile_desc_global"
os.environ["AISP_NVFP4_GROUP_GEMM_MAX_SWIZZLE"] = "4"
os.environ["AISP_NVFP4_GROUP_GEMM_PERSISTENT_STREAM_BALANCE"] = "1"

from core.harness.benchmark_harness import BaseBenchmark
from labs.nvfp4_group_gemm.cutlass_submission_cached import (
    custom_kernel_cutlass_cached,
    prepare_cutlass_cached_2sm_n64_s4_persistent,
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
        prepare=prepare_cutlass_cached_2sm_n64_s4_persistent,
        inputs_per_iteration=15,
        capture_iter_graph=True,
        name=(
            f"nvfp4_group_gemm_{case.name}_optimized_cutlass_cached_2sm_n64_s4_"
            "cm2_cn2_ro2_pdl1_persistent_chunk8_conc5_tile_desc_global_m_desc_swz4_outer_itergraph_balstream"
        ),
    )
    return attach_benchmark_metadata(bench, __file__)


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)

