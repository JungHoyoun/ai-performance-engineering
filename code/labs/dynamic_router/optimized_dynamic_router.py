"""Benchmark harness wrapper for the optimized dynamic router simulation."""

from __future__ import annotations

from typing import Dict, Optional

import torch

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from labs.dynamic_router.driver import simulate


class OptimizedDynamicRouterBenchmark(BaseBenchmark):
    """Runs the optimized (prefill/decode split) routing simulation under aisp bench."""

    def __init__(self) -> None:
        super().__init__()
        self._summary: Dict[str, float] = {}
        self.metrics: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self.register_workload_metadata(requests_per_iteration=1.0)

    def setup(self) -> None:
        return

    def benchmark_fn(self) -> None:
        self._summary = simulate(
            "optimized",
            num_ticks=120,
            seed=0,
            log_interval=None,
        )
        metric_values = list(self._summary.values()) or [0.0]
        expected_shape = (1, len(metric_values))
        if self.metrics is None or tuple(self.metrics.shape) != expected_shape:
            self.metrics = torch.randn(expected_shape, dtype=torch.float32)
        summary_tensor = torch.tensor([metric_values], dtype=torch.float32)
        self.output = (summary_tensor + self.metrics).detach()
        self._set_verification_payload(
            inputs={
                "metrics_seed": torch.tensor([0], dtype=torch.int64),
                "ticks": torch.tensor([120], dtype=torch.int64),
            },
            output=self.output,
            batch_size=1,
            parameter_count=0,
            precision_flags={"fp16": False, "bf16": False, "tf32": False},
            output_tolerance=(0.1, 1.0),
        )

    def get_config(self) -> Optional[BenchmarkConfig]:
        return BenchmarkConfig(
            iterations=1,
            warmup=5,
            measurement_timeout_seconds=120,
            timeout_multiplier=3.0,
        )

    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        return self._summary or None

    def teardown(self) -> None:
        self.metrics = None
        self.output = None
        super().teardown()


def get_benchmark() -> BaseBenchmark:
    """Factory for discover_benchmarks()."""
    return OptimizedDynamicRouterBenchmark()


if __name__ == "__main__":
    bench = get_benchmark()
    cfg = bench.get_config()
    bench.benchmark_fn()
    print(bench.get_custom_metrics())
