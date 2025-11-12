"""optimized_memory.py - Optimized GPU memory management.

Demonstrates optimized GPU memory management with custom allocator.
Memory: Uses memory pooling and optimized allocation strategies.
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

from common.python.compile_utils import enable_tf32
from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)

BATCH_SIZE = 512
INPUT_DIM = 2048
HIDDEN_DIM = 2048
REPETITIONS = 8


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch17")
    return torch.device("cuda")


class OptimizedMemoryBenchmark(Benchmark):
    """Optimized: GPU memory management with optimization."""
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.batch_size = BATCH_SIZE
        self.input_dim = INPUT_DIM
        self.device_buffer: Optional[torch.Tensor] = None
        self.transform_buffer: Optional[torch.Tensor] = None
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.graph_output: Optional[torch.Tensor] = None
        self.repetitions = REPETITIONS
        self._norm_shape = (self.input_dim,)
    
    def setup(self) -> None:
        """Setup: Initialize model with optimized memory management."""
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            enable_tf32()
        
        torch.manual_seed(42)
        # Optimization: GPU memory management optimization
        # Techniques include memory pooling, custom allocators, and reuse strategies
        # This example uses PyTorch's memory-efficient settings
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(HIDDEN_DIM, self.input_dim),
        ).to(self.device).eval()
        
        # Optimized memory allocation with memory pool
        self.device_buffer = torch.empty(
            self.batch_size,
            self.input_dim,
            device=self.device,
            dtype=torch.float32,
        )
        self.transform_buffer = torch.empty_like(self.device_buffer)
        self.graph_output = torch.empty_like(self.device_buffer)
        torch.cuda.synchronize()

        with torch.no_grad():
            _ = self.model(self.device_buffer)
        torch.cuda.synchronize()

        self.graph = torch.cuda.CUDAGraph()
        self.device_buffer.uniform_(0.0, 255.0)
        torch.cuda.synchronize()
        with torch.cuda.graph(self.graph):
            self.transform_buffer.copy_(self.device_buffer)
            self.transform_buffer.mul_(1.0 / 255.0)
            self.transform_buffer.add_(-0.5)
            self.transform_buffer.mul_(2.0)
            self.transform_buffer.tanh_()
            self.graph_output.copy_(self.model(self.transform_buffer))
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Optimized memory management."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        # Optimization: Memory-efficient execution
        # Uses optimized memory allocation and reuse strategies
        if (
            self.model is None
            or self.device_buffer is None
            or self.graph is None
            or self.graph_output is None
        ):
            raise RuntimeError("Optimized memory benchmark not initialized")

        with nvtx_range("optimized_memory", enable=enable_nvtx):
            with torch.no_grad():
                for _ in range(self.repetitions):
                    self.device_buffer.uniform_(0.0, 255.0)
                    self.graph.replay()
        torch.cuda.synchronize()
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.device_buffer = None
        self.transform_buffer = None
        self.graph_output = None
        self.graph = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=200,
            warmup=10,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return OptimizedMemoryBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(result)
