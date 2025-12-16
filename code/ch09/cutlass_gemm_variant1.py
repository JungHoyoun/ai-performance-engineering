"""Harness wrapper for cutlass_gemm_variant1 (preloaded CUTLASS GEMM).

VARIANT 1: Fair comparison baseline with pre-loaded data
This ensures kernel execution time is measured without data transfer overhead.

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
from core.benchmark.verification import simple_signature


class CutlassGemmVariant1Benchmark(CudaBinaryBenchmark):
    """Wraps the fair comparison baseline CUTLASS GEMM (pre-loaded data)."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="baseline_cutlass_gemm_variant1",
            friendly_name="Tiled GEMM Baseline (Fair - Preloaded)",
            iterations=5,
            warmup=5,
            timeout_seconds=120,
        )

    def get_custom_metrics(self) -> Optional[dict]:
        """Return roofline metrics for cutlass_gemm."""
        from core.benchmark.metrics import compute_roofline_metrics
        return compute_roofline_metrics(
            total_flops=self._total_flops,
            total_bytes=self._total_bytes,
            elapsed_ms=getattr(self, '_last_elapsed_ms', 1.0),
            precision="fp32",
        )
    # get_verify_output inherited from CudaBinaryBenchmark - uses checksum from -DVERIFY=1 build

    def get_input_signature(self) -> dict:
        """Return input signature for verification."""
        return simple_signature(batch_size=1, dtype="float32", workload=1).to_dict()


def get_benchmark() -> CutlassGemmVariant1Benchmark:
    """Factory for discover_benchmarks()."""
    return CutlassGemmVariant1Benchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
