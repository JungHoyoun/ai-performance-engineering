"""Optimized Layer 1: Basics (Ch1-6) - NVTX, NUMA, TF32.

Incremental optimization showing Layer 1 contribution:
- NVTX profiling markers
- NUMA binding
- TF32 enable
- cuDNN benchmark mode

This is the first optimization layer, demonstrating the foundational
techniques from Chapters 1-6.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import torch

from baseline_ultimate_inference import BaselineUltimateInference, InferenceConfig
from optimization_layers import Layer01Basics
from common.python.benchmark_harness import BaseBenchmark, WorkloadMetadata


class OptimizedLayer1Benchmark(BaselineUltimateInference):
    """Optimized: Layer 1 only (Ch1-6 basics).
    
    Adds foundational optimizations:
    - TF32 for Tensor Core math
    - cuDNN benchmark mode
    - NUMA binding (if available)
    - NVTX markers for profiling
    
    Expected improvement: 1.1-1.3x from TF32 + cuDNN autotuning
    """
    
    def __init__(self, config: Optional[InferenceConfig] = None):
        super().__init__(config)
        self.layer1 = Layer01Basics()
    
    def setup(self) -> None:
        """Setup with Layer 1 optimizations."""
        # Apply Layer 1 optimizations BEFORE model load
        print("[Layer 1] Applying basic optimizations (Ch1-6)...")
        self.layer1.enable_tf32()
        self.layer1.enable_cudnn_benchmark()
        self.layer1.bind_numa(self.device)
        
        # Now do standard setup
        super().setup()
        
        print(f"[Layer 1] Status: {self.layer1}")


def get_benchmark() -> BaseBenchmark:
    """Factory function for harness discovery."""
    return OptimizedLayer1Benchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=benchmark.get_config())
    result = harness.benchmark(benchmark)
    
    print(f"\nLayer 1 (Ch1-6): {result.timing.mean_ms:.2f} ms" if result.timing else "No timing")


