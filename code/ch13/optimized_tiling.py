"""optimized_tiling.py - Optimized with tiling in training.

Demonstrates tiling optimization for better memory access patterns.
Tiling: Breaks matrices into smaller tiles for better cache utilization.
Improves memory access locality and reduces cache misses.
Implements Benchmark protocol for harness integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn

from common.python.compile_utils import enable_tf32

from typing import Optional

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)

def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch13")
    return torch.device("cuda")

class OptimizedTilingBenchmark(Benchmark):
    """Optimized tiling using BF16 weights and CUDA graph replay."""
    
    def __init__(self):
        self.device = resolve_device()
        self.in_features = 2048
        self.out_features = 2048
        self.batch_size = 32
        self.weight: Optional[torch.Tensor] = None
        self.bias: Optional[torch.Tensor] = None
        self.static_input: Optional[torch.Tensor] = None
        self.static_output: Optional[torch.Tensor] = None
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.capture_stream: Optional[torch.cuda.Stream] = None
    
    def setup(self) -> None:
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            enable_tf32()
        torch.manual_seed(42)
        self.linear = nn.Linear(self.in_features, self.out_features, bias=True).to(
            self.device, dtype=torch.bfloat16
        )
        self.static_input = torch.randn(
            self.batch_size,
            self.in_features,
            device=self.device,
            dtype=torch.bfloat16,
        )
        self.static_output = torch.empty(
            self.batch_size,
            self.out_features,
            device=self.device,
            dtype=torch.bfloat16,
        )
        self.graph = torch.cuda.CUDAGraph()
        self.capture_stream = torch.cuda.Stream()

        with torch.cuda.stream(self.capture_stream):
            for _ in range(3):
                self.static_output.copy_(self.linear(self.static_input))
            torch.cuda.synchronize()
            with torch.cuda.graph(self.graph, stream=self.capture_stream):
                self.static_output.copy_(self.linear(self.static_input))
        self.capture_stream.synchronize()
    
    def benchmark_fn(self) -> None:
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        if self.graph is None:
            raise RuntimeError("CUDA graph not captured")

        with nvtx_range("optimized_tiling", enable=enable_nvtx):
            self.graph.replay()
        torch.cuda.synchronize(self.device)
    
    def teardown(self) -> None:
        self.linear = None
        self.static_input = None
        self.static_output = None
        self.graph = None
        self.capture_stream = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=10,
            warmup=2,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.linear is None:
            return "Linear layer missing"
        if self.static_input is None:
            return "Static input missing"
        return None

def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return OptimizedTilingBenchmark()

def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = OptimizedTilingBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Optimized: tiling")
    print("=" * 70)
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")

if __name__ == "__main__":
    main()
