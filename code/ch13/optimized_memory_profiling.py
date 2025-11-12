"""optimized_memory_profiling.py - Optimized memory profiling (optimized).

Memory profiling with gradient checkpointing to reduce peak memory.
Trades compute for memory by recomputing activations during backward.

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
from torch.utils.checkpoint import checkpoint

from typing import Optional

from common.python.compile_utils import enable_tf32
from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
)

def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch13")
    return torch.device("cuda")

class OptimizedModel(nn.Module):
    """Model with gradient checkpointing for memory optimization."""
    
    def __init__(self, hidden_dim: int = 2048):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        self.model_hidden_dim = hidden_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gradient checkpointing: recompute activations in backward
        # Saves memory by not storing intermediate activations
        x = checkpoint(self._fc1_relu, x, preserve_rng_state=False)
        x = self.fc2(x)
        return x
    
    def _fc1_relu(self, x: torch.Tensor) -> torch.Tensor:
        """Helper function for checkpointing."""
        return self.relu(self.fc1(x))

class OptimizedMemoryProfilingBenchmark(Benchmark):
    """Optimized memory profiling - uses gradient checkpointing + CUDA graphs."""
    
    def __init__(self):
        self.device = resolve_device()
        self.model: Optional[OptimizedModel] = None
        self.inputs: Optional[torch.Tensor] = None
        self.targets: Optional[torch.Tensor] = None
        self.criterion: Optional[nn.Module] = None
        self.peak_memory_mb = 0.0
        self.batch_size = 32
        self.hidden_dim = 2048
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.capture_stream: Optional[torch.cuda.Stream] = None
        self.static_input: Optional[torch.Tensor] = None
        self.static_target: Optional[torch.Tensor] = None
    
    def setup(self) -> None:
        """Setup: Initialize model and data."""
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            enable_tf32()
        torch.manual_seed(42)
        torch.cuda.reset_peak_memory_stats()
        
        self.model = OptimizedModel(hidden_dim=self.hidden_dim).to(self.device, dtype=torch.bfloat16)
        self.model.train()
        self.inputs = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.bfloat16)
        self.targets = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.bfloat16)
        self.criterion = nn.MSELoss()
        
        # Warmup
        _ = self.model(self.inputs)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        self.static_input = self.inputs.clone()
        self.static_target = self.targets.clone()
        self.graph = torch.cuda.CUDAGraph()
        self.capture_stream = torch.cuda.Stream()
        with torch.cuda.stream(self.capture_stream):
            for _ in range(2):
                self._train_step(self.static_input, self.static_target)
            torch.cuda.synchronize()
            with torch.cuda.graph(self.graph, stream=self.capture_stream):
                self._train_step(self.static_input, self.static_target)
        self.capture_stream.synchronize()
    
    def _train_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        assert self.model is not None and self.criterion is not None
        self.model.zero_grad(set_to_none=True)
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        loss.backward()
    
    def benchmark_fn(self) -> None:
        """Function to benchmark - memory profiling with checkpointing."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False

        if (
            self.graph is None
            or self.static_input is None
            or self.static_target is None
            or self.model is None
        ):
            raise RuntimeError("CUDA graph not initialized")

        with nvtx_range("optimized_memory_profiling", enable=enable_nvtx):
            self.static_input.copy_(self.inputs)
            self.static_target.copy_(self.targets)
            self.model.zero_grad(set_to_none=True)
            self.graph.replay()
            self.peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    def teardown(self) -> None:
        """Cleanup."""
        del self.model, self.inputs, self.targets, self.criterion
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=10,
            enable_memory_tracking=True,
            enable_profiling=False,
        )
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        if self.peak_memory_mb <= 0:
            return "Peak memory not recorded"
        return None

def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return OptimizedMemoryProfilingBenchmark()

if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
    mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nOptimized Memory Profiling: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Peak Memory: {benchmark.peak_memory_mb:.2f} MB")
