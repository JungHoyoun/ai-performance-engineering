"""baseline_context_parallelism.py - Baseline sequential processing without context parallelism.

Demonstrates sequential processing of long sequences without context parallelism.
Context parallelism: This baseline processes the entire sequence on a single GPU sequentially.
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

try:
    import ch19.arch_config  # noqa: F401 - Apply chapter defaults
except ImportError:
    pass

from typing import Optional

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch19")
    return torch.device("cuda")


class BaselineContextParallelismBenchmark(Benchmark):
    """Baseline: Sequential processing without context parallelism (single GPU)."""
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.input_sequence = None
        self.sequence_length = 8192  # Longer sequence to demonstrate context parallelism benefit
        self.repeats = 8
    
    def setup(self) -> None:
        """Setup: Initialize model and long input sequence."""
        torch.manual_seed(42)
        # Baseline: Sequential processing on single GPU
        # Context parallelism splits long sequences across multiple GPUs
        # This baseline processes the entire sequence sequentially on one GPU
        self.model = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
        ).to(self.device).eval()
        
        # Long sequence that would benefit from context parallelism
        # Context parallelism splits sequences across GPUs for parallel processing
        self.input_sequence = torch.randn(self.sequence_length, 256, device=self.device)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Sequential processing of long sequence."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled
        
        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False
        
        # Baseline: Process entire sequence sequentially on single GPU
        # No context parallelism - all tokens processed on one device
        with nvtx_range("baseline_context_parallelism", enable=enable_nvtx):
            with torch.no_grad():
                for _ in range(self.repeats):
                    activations = self.input_sequence
                    for layer in self.model:
                        activations = layer(activations)
                        torch.cuda.synchronize()
        torch.cuda.synchronize()
    
    def teardown(self) -> None:
        """Cleanup: Clear CUDA cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=20,
            warmup=3,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        if self.input_sequence is None:
            return "Input sequence not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return BaselineContextParallelismBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(result)
