"""Python harness wrapper for baseline_cutlass_gemm_preloaded.cu."""

from __future__ import annotations
from typing import Optional

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.harness.benchmark_harness import BaseBenchmark
from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark


class BaselineCutlassGemmPreloadedBenchmark(CudaBinaryBenchmark):
    """Wraps the preloaded-inputs SIMT CUTLASS GEMM."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        m = n = k = 1024
        iterations = 5
        repeats = 32
        bytes_a = m * k * 4
        bytes_b = k * n * 4
        bytes_c = m * n * 4
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="baseline_cutlass_gemm_preloaded",
            friendly_name="Baseline Cutlass Gemm (Preloaded Inputs)",
            iterations=5,
            warmup=5,
            timeout_seconds=120,
            workload_params={
                "M": m,
                "N": n,
                "K": k,
                "kIterations": iterations,
                "kRepeats": repeats,
                "dtype": "float32",
            },
        )
        self.register_workload_metadata(bytes_per_iteration=float(bytes_a + bytes_b + bytes_c))

    def get_custom_metrics(self) -> Optional[dict]:
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselineCutlassGemmPreloadedBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
