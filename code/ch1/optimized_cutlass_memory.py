"""optimized_cutlass_memory - Optimized GEMM using CUTLASS with memory optimizations. Implements Benchmark protocol for harness integration."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

try:
    import ch1.arch_config  # noqa: F401 - Apply chapter defaults
except ImportError:
    pass

from typing import Optional

from common.python.compile_utils import enable_tf32, compile_callable
from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)
from ch1.workload_config import WORKLOAD


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch1")
    return torch.device("cuda")


class OptimizedCutlassMemoryBenchmark(Benchmark):
    """Optimized: GEMM using CUTLASS with memory optimizations.
    
    CUTLASS: Uses CUTLASS backend for hardware-optimized GEMM kernels with optimized memory access patterns.
    Leverages tensor cores and optimized memory management for better performance.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.A_batches = None
        self.B_batches = None
        self.outputs = None
        self.group = None
        self.m = 1024
        self.n = 1024
        self.k = 1024
        self.num_steps = max(8, WORKLOAD.performance_microbatches // 4)
        self.group = min(4, self.num_steps)
        self.compiled_matmul = None
        self._chunked_batches = []
    
    def setup(self) -> None:
        """Setup: Initialize matrices and compile with CUTLASS backend."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            # Enable TF32 for faster matmul on Ampere+ GPUs
            enable_tf32()
        
        torch.manual_seed(42)
        # Optimization: CUTLASS-optimized GEMM with memory management
        # Uses torch.compile with CUTLASS backend for hardware-optimized kernels
        # Optimizes memory access patterns and bandwidth utilization
        
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.A_batches = []
        self.B_batches = []
        for _ in range(self.num_steps):
            self.A_batches.append(torch.randn(self.m, self.k, device=self.device, dtype=dtype))
            self.B_batches.append(torch.randn(self.k, self.n, device=self.device, dtype=dtype))
        
        self._chunked_batches = []
        for start in range(0, self.num_steps, self.group):
            end = min(start + self.group, self.num_steps)
            self._chunked_batches.append(
                (
                    torch.stack(self.A_batches[start:end], dim=0),
                    torch.stack(self.B_batches[start:end], dim=0),
                )
            )
        
        # Optimization: Compile with CUTLASS backend for optimized memory access
        def gemm_fn(a, b):
            return torch.matmul(a, b)
        
        try:
            import torch._inductor.config as config
        except ImportError as exc:
            raise RuntimeError(
                "SKIPPED: torch._inductor.config unavailable for CUTLASS compilation."
            ) from exc
        config.cuda.cutlass_enabled_ops = "all"
        self.compiled_matmul = compile_callable(
            gemm_fn,
            mode="max-autotune",
            backend="inductor",
            fullgraph=True,
        )
        
        torch.cuda.synchronize()
        warm_a, warm_b = self._chunked_batches[0]
        for _ in range(3):
            _ = self.compiled_matmul(warm_a, warm_b)
        torch.cuda.synchronize()

        self.outputs = torch.zeros(1, device=self.device, dtype=warm_a.dtype)
    
    def benchmark_fn(self) -> None:
        """Benchmark: CUTLASS-optimized GEMM with memory optimizations."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("optimized_cutlass_memory", enable=enable_nvtx):
            totals = torch.zeros((), device=self.device, dtype=self.A_batches[0].dtype)
            for A_stack, B_stack in self._chunked_batches:
                totals = totals + self.compiled_matmul(A_stack, B_stack).sum()
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            self.outputs = totals

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.A_batches = None
        self.B_batches = None
        self._chunked_batches = []
        self.outputs = None
        self.compiled_matmul = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=10,
            warmup=2,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.A_batches is None or self.B_batches is None:
            return "Matrices not initialized"
        if self.outputs is None:
            return "Output matrix not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return OptimizedCutlassMemoryBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nOptimized CUTLASS Memory: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
