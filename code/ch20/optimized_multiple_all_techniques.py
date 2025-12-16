"""optimized_multiple_all_techniques.py - Alias to the combined techniques benchmark.

This file exists for backwards compatibility with older references and
integration tests. The canonical implementation lives in
`optimized_multiple_unoptimized.py`.
"""

from __future__ import annotations

try:
    import ch20.arch_config  # noqa: F401 - Apply chapter defaults
except ImportError:
    pass

from ch20.optimized_multiple_unoptimized import OptimizedAllTechniquesBenchmark
from core.harness.benchmark_harness import BaseBenchmark


def get_benchmark() -> BaseBenchmark:
    return OptimizedAllTechniquesBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)

