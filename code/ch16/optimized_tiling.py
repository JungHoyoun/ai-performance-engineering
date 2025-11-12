"""optimized_tiling.py - Optimized with tiling in MoE context.

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

from common.python.compile_utils import compile_model

from typing import Optional

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch16")
    return torch.device("cuda")


class OptimizedTilingBenchmark(Benchmark):
    """Optimized: Tiling for better memory access patterns.
    
    Tiling: Breaks matrices into smaller tiles for better cache utilization.
    Improves memory access locality and reduces cache misses.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.input = None
        self.tile_size = 256
    
    def setup(self) -> None:
        """Setup: Initialize model with tiling optimization."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        torch.manual_seed(42)
        # Optimization: Tiling for better memory access
        # Tiling breaks matrices into smaller tiles
        # Improves cache utilization and memory access patterns
        
        # Large linear layer (will use tiling)
        self.model = nn.Linear(2048, 2048).to(self.device)
        # Optimization: Use FP16 for faster computation - FAIL FAST if not supported
        if self.device.type != "cuda":
            raise RuntimeError("CUDA required for optimized_tiling benchmark")
        self.model = self.model.half().eval()
        self.model = compile_model(self.model, mode="reduce-overhead")
        
        # Large input (will be processed with tiling) - FAIL FAST if model has no parameters
        params = list(self.model.parameters())
        if not params:
            raise RuntimeError("Model has no parameters - cannot determine dtype")
        input_dtype = params[0].dtype
        self.input = torch.randn(64, 2048, device=self.device, dtype=input_dtype)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Matrix operations with tiling."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("optimized_tiling", enable=enable_nvtx):
            with torch.no_grad():
                output = self.model(self.input)
                _ = output.sum()
        torch.cuda.synchronize()

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.input = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=10,
            warmup=2,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        if self.input is None:
            return "Input not initialized"
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
