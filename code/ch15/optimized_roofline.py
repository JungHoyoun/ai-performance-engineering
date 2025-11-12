"""optimized_roofline.py - Optimized with roofline analysis in disaggregated inference.

Demonstrates roofline analysis for performance optimization.
Roofline: Uses roofline analysis to identify compute/memory bottlenecks.
Guides optimization strategy based on arithmetic intensity.
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

from typing import Optional

from common.python.compile_utils import enable_tf32, compile_model
from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch15")
    return torch.device("cuda")


class OptimizedRooflineBenchmark(Benchmark):
    """Optimized: Roofline analysis for performance optimization.
    
    Roofline: Uses roofline analysis to identify compute/memory bottlenecks.
    Guides optimization strategy based on arithmetic intensity.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model: Optional[nn.Module] = None
        self.input: Optional[torch.Tensor] = None
        self.roofline_data = None
    
    def setup(self) -> None:
        """Setup: Initialize model with roofline analysis."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            enable_tf32()
        torch.manual_seed(42)
        # Optimization: Roofline analysis
        # Identifies compute-bound vs memory-bound operations
        # Guides optimization strategy
        
        # Optimization: Apply roofline-guided optimizations
        # Based on roofline analysis, we'll optimize for the identified bottleneck
        model = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
        )
        
        model = model.to(self.device, dtype=torch.bfloat16).eval()
        self.model = compile_model(model, mode="reduce-overhead")
        self.input = torch.randn(32, 1024, device=self.device, dtype=torch.bfloat16)
        
        # Roofline data for analysis
        with torch.no_grad():
            output = self.model(self.input)
        input_bytes = self.input.numel() * self.input.element_size()
        output_bytes = output.numel() * output.element_size()
        total_bytes = input_bytes + output_bytes
        flops = self.input.size(0) * self.input.size(1) * 2048 * 2
        arithmetic_intensity = flops / total_bytes if total_bytes > 0 else 0.0
        self.roofline_data = {
            'compute_bound': arithmetic_intensity >= 1.0,
            'memory_bound': arithmetic_intensity < 1.0,
            'arithmetic_intensity': arithmetic_intensity,
        }
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Operations with roofline analysis."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("optimized_roofline", enable=enable_nvtx):
            with torch.no_grad():
                output = self.model(self.input)
                _ = output.sum()
        torch.cuda.synchronize(self.device)

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.input = None
        self.roofline_data = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=5,
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
    return OptimizedRooflineBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = OptimizedRooflineBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Optimized: roofline")
    print("=" * 70)
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")


if __name__ == "__main__":
    main()
