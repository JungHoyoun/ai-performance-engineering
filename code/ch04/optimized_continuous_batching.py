"""Chapter 4 optimized continuous batching benchmark (single GPU)."""

from __future__ import annotations

from core.utils.continuous_batching import ContinuousBatchingBase
from ch04.verification_payload_mixin import VerificationPayloadMixin


class OptimizedContinuousBatchingBenchmark(VerificationPayloadMixin, ContinuousBatchingBase):
    """Optimized: continuous batching with dynamic batch membership."""

    def __init__(self) -> None:
        super().__init__(
            dynamic=True,
            multi_gpu=False,
            label="optimized_continuous_batching",
        )


def get_benchmark() -> OptimizedContinuousBatchingBenchmark:
    return OptimizedContinuousBatchingBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
