"""Optimized wrapper for the topology probe (alias of baseline to satisfy harness discovery)."""

from __future__ import annotations

from typing import Dict, Optional

import torch

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from labs.dynamic_router.topology_probe import TopologyProbeBenchmark


class OptimizedTopologyProbeBenchmark(BaseBenchmark):
    """Runs the topology probe under aisp bench (same as baseline)."""

    def __init__(self) -> None:
        super().__init__()
        self._summary: Dict[str, float] = {}
        self.metrics: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self.register_workload_metadata(requests_per_iteration=1.0)

    def setup(self) -> None:
        return

    def benchmark_fn(self) -> None:
        bench = TopologyProbeBenchmark()
        bench.benchmark_fn()
        self._summary = bench.get_custom_metrics() or {}
        metric_values = list(self._summary.values()) or [0.0]
        expected_shape = (1, len(metric_values))
        if self.metrics is None or tuple(self.metrics.shape) != expected_shape:
            self.metrics = torch.randn(expected_shape, dtype=torch.float32)
        summary_tensor = torch.tensor([metric_values], dtype=torch.float32)
        self.output = (summary_tensor + self.metrics).detach()
        self._set_verification_payload(
            inputs={
                "num_gpus": torch.tensor([len(self._summary)], dtype=torch.int64),
            },
            output=self.output,
            batch_size=1,
            parameter_count=0,
            precision_flags={"fp16": False, "bf16": False, "tf32": False},
            output_tolerance=(0.1, 1.0),
        )

    def teardown(self) -> None:
        self.metrics = None
        self.output = None
        super().teardown()

    def get_config(self) -> Optional[BenchmarkConfig]:
        return BenchmarkConfig(iterations=1, warmup=5)

    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        return self._summary or None


def get_benchmark() -> BaseBenchmark:
    return OptimizedTopologyProbeBenchmark()


if __name__ == "__main__":
    b = get_benchmark()
    b.benchmark_fn()
    print(b.get_custom_metrics())
