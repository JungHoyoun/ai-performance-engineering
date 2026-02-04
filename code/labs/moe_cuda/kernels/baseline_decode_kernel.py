"""Kernel-level wrapper reusing the lab decode kernel benchmark."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.harness.benchmark_harness import BaseBenchmark
from labs.moe_cuda.baseline_decode_kernel import BaselineDecodeKernelBenchmark as _BaselineDecodeKernelBenchmark


class BaselineDecodeKernelBenchmark(_BaselineDecodeKernelBenchmark):
    """Expose the baseline decode kernel benchmark from the lab package."""

    pass


def get_benchmark() -> BaseBenchmark:
    return BaselineDecodeKernelBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
