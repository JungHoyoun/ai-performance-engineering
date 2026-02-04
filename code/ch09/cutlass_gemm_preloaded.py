"""Harness wrapper for cutlass_gemm_preloaded (preloaded-inputs CUTLASS GEMM).

Preloaded inputs provide a fair kernel-only comparison by removing H2D transfer overhead.

BOOK REFERENCE (Ch9): When comparing GEMM implementations, it's critical to
isolate kernel execution from data movement to make fair comparisons.
"""

from __future__ import annotations
from typing import Optional

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkHarness, BenchmarkMode
from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark


class CutlassGemmPreloadedBenchmark(CudaBinaryBenchmark):
    """Wraps the fair comparison baseline CUTLASS GEMM (preloaded inputs)."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        m = n = k = 1024
        repeats = 32
        iterations = 5
        bytes_a = m * k * 4
        bytes_b = k * n * 4
        bytes_c = m * n * 4
        # CUDA binary reports per-GEMM average time (total / (iterations * repeats)).
        self._total_flops = 2.0 * m * n * k
        self._total_bytes = float(bytes_a + bytes_b + bytes_c)
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="baseline_cutlass_gemm_preloaded",
            friendly_name="Tiled GEMM Baseline (Preloaded Inputs)",
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

    def get_custom_metrics(self) -> Optional[dict]:
        """Return roofline metrics for cutlass_gemm."""
        from core.benchmark.metrics import compute_roofline_metrics
        return compute_roofline_metrics(
            total_flops=self._total_flops,
            total_bytes=self._total_bytes,
            elapsed_ms=self.last_time_ms or 1.0,
            precision="fp32",
        )
    # get_verify_output inherited from CudaBinaryBenchmark - uses checksum from -DVERIFY=1 build


def get_benchmark() -> CutlassGemmPreloadedBenchmark:
    """Factory for discover_benchmarks()."""
    return CutlassGemmPreloadedBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
