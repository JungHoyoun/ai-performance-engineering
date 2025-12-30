"""baseline_continuous_batching.py - Baseline static batching (single GPU)."""

from __future__ import annotations

from core.utils.continuous_batching import ContinuousBatchingBase
from ch15.verification_payload_mixin import VerificationPayloadMixin


class BaselineContinuousBatchingBenchmark(VerificationPayloadMixin, ContinuousBatchingBase):
    """Baseline: padded static batching with fixed batch membership."""

    def __init__(self) -> None:
        super().__init__(
            dynamic=False,
            multi_gpu=False,
            label="baseline_continuous_batching",
        )


def get_benchmark() -> BaselineContinuousBatchingBenchmark:
    return BaselineContinuousBatchingBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
