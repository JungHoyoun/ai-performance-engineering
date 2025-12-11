"""baseline_vectorization.py - Baseline without vectorization in storage I/O context."""

from __future__ import annotations

from typing import Optional

import torch

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class BaselineVectorizationBenchmark(BaseBenchmark):
    """Scalar operations to contrast with vectorized paths."""
    
    def __init__(self):
        super().__init__()
        self.data: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self.N = 1_000_000
        # Computation benchmark - jitter check not applicable
        self.jitter_exemption_reason = "Computation benchmark: fixed input size"
        tokens = self.N
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        """Setup: Initialize data."""
        torch.manual_seed(42)
        self.data = torch.randn(self.N, device=self.device)
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Scalar operations without vectorization."""
        assert self.data is not None
        with self._nvtx_range("baseline_vectorization"):
            result = torch.zeros(1, device=self.device)
            for i in range(min(1000, self.N)):  # Simulate scalar loop
                result += self.data[i]
            self._synchronize()
        self.output = result
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.data = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=5,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_storage_io_metrics
        return compute_storage_io_metrics(
            bytes_read=getattr(self, '_bytes_read', 0.0),
            bytes_written=getattr(self, '_bytes_written', 0.0),
            read_time_ms=getattr(self, '_read_time_ms', 1.0),
            write_time_ms=getattr(self, '_write_time_ms', 1.0),
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.data is None:
            return "Data not initialized"
        return None

    def get_input_signature(self) -> dict:
        """Return workload signature for input verification."""
        return {
            "N": self.N,
        }

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        if self.output is not None:
            return self.output.detach().clone()
        raise RuntimeError("benchmark_fn() must be called before verification - output is None")
    
    def get_output_tolerance(self) -> tuple:
        """Return custom tolerance for computation benchmark."""
        return (1e-3, 1e-3)



def get_benchmark() -> BaseBenchmark:
    """Factory function for harness discovery."""
    return BaselineVectorizationBenchmark()
