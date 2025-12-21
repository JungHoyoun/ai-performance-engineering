"""Optimized tcgen05 matmul using cluster launch with TMA multicast."""

from __future__ import annotations

from typing import Optional

import torch

from core.benchmark.tcgen05_matmul_base import Tcgen05MatmulBenchmarkBase
from core.benchmark.tcgen05_requirements import ensure_tcgen05_supported
from core.common.tcgen05 import load_tcgen05_cluster_module
from core.harness.hardware_capabilities import ensure_dsmem_supported
from core.harness.benchmark_harness import BaseBenchmark


class OptimizedTcgen05ClusterPipelineBenchmark(Tcgen05MatmulBenchmarkBase):
    """Chapter 10 optimized: cluster-launched tcgen05 GEMM."""

    shared_dim = 2048
    nvtx_label = "optimized_tcgen05_cluster_pipeline"

    def __init__(self) -> None:
        super().__init__()
        self.extension: Optional[object] = None

    def setup(self) -> None:
        ensure_dsmem_supported(description="tcgen05 cluster pipeline")
        ensure_tcgen05_supported(
            loader=load_tcgen05_cluster_module,
            module_name="ch10 tcgen05 cluster pipeline",
        )
        super().setup()
        if self.extension is None:
            self.extension = load_tcgen05_cluster_module()

    def benchmark_fn(self) -> None:
        if self.extension is None or self.matrix_a is None or self.matrix_b is None:
            raise RuntimeError("Inputs or extension not initialized")
        with self._nvtx_range(self.nvtx_label):
            with torch.no_grad():
                self.output = self.extension.matmul_tcgen05_cluster(self.matrix_a, self.matrix_b)
        self._synchronize()


def get_benchmark() -> BaseBenchmark:
    return OptimizedTcgen05ClusterPipelineBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
