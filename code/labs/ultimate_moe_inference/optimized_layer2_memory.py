"""Optimized Layer 2: Memory Optimizations (Ch7-8).

Incremental optimization showing Layers 1-2 contribution:
- Layer 1: TF32, cuDNN, NUMA
- Layer 2: Memory coalescing, occupancy tuning, ILP

This demonstrates the memory hierarchy and occupancy techniques from Chapters 7-8.

Key Techniques (Ch7 - Memory Hierarchy):
- Coalesced global memory access (128-byte transactions)
- Vectorized loads (float4 = 16 bytes per thread)
- L2 cache optimization (persistence hints)
- Read-only cache (__ldg, const __restrict__)

Key Techniques (Ch8 - Occupancy & ILP):
- Occupancy tuning via launch bounds
- Register pressure management
- Instruction-Level Parallelism (ILP)
- Memory latency hiding through ILP
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import torch

from baseline_ultimate_inference import BaselineUltimateInference, InferenceConfig
from optimization_layers import Layer01Basics, Layer02Memory
from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig


class OptimizedLayer2Benchmark(BaselineUltimateInference):
    """Optimized: Layers 1-2 (Ch1-8 basics + memory).
    
    Adds memory optimizations:
    - Coalesced memory access patterns
    - Occupancy tuning for better SM utilization
    - ILP through unrolled loops
    - Memory-efficient attention patterns (pre-FlashAttention)
    
    Note: Many of these optimizations are implicit in PyTorch/cuBLAS
    but we configure PyTorch to maximize their effectiveness.
    
    Expected improvement: 1.2-1.4x from memory layout + occupancy
    """
    
    def __init__(self, config: Optional[InferenceConfig] = None):
        super().__init__(config)
        self.layer1 = Layer01Basics()
        self.layer2 = Layer02Memory()
    
    def setup(self) -> None:
        """Setup with Layers 1-2 optimizations."""
        # Layer 1
        print("[Layer 1] Applying basic optimizations (Ch1-6)...")
        self.layer1.enable_tf32()
        self.layer1.enable_cudnn_benchmark()
        self.layer1.bind_numa(self.device)
        
        # Layer 2 - Memory optimizations (before model load)
        print("[Layer 2] Applying memory optimizations (Ch7-8)...")
        print("  - Coalesced memory access: Enabled via contiguous tensors")
        print("  - Occupancy hints: Via memory allocator config")
        print("  - ILP: Via torch inductor unrolling settings")
        
        # Configure memory allocator for better coalescing
        self.layer2.configure_memory_allocator()
        
        # Set occupancy-friendly allocation
        self.layer2.configure_occupancy_hints()
        
        # Now do standard setup
        super().setup()
        
        # Apply post-load optimizations
        self.layer2.apply(self)
        
        print(f"[Layer 2] Status: {self.layer2}")


def get_benchmark() -> BaseBenchmark:
    """Factory function for harness discovery."""
    return OptimizedLayer2Benchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=benchmark.get_config())
    result = harness.benchmark(benchmark)
    
    print(f"\nLayers 1-2 (Ch1-8): {result.timing.mean_ms:.2f} ms" if result.timing else "No timing")


