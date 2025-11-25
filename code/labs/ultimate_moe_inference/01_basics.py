#!/usr/bin/env python3
"""
01_basics.py - Foundation Optimizations (Chapters 1-6)

═══════════════════════════════════════════════════════════════════════════════
BEFORE THIS OPTIMIZATION:
═══════════════════════════════════════════════════════════════════════════════

- FP32 math everywhere (slow, doesn't use Tensor Cores efficiently)
- cuDNN uses first algorithm it finds (not optimal)
- Memory allocated from any NUMA node (extra latency on Grace-Blackwell)
- No profiling markers (hard to see what's happening)

═══════════════════════════════════════════════════════════════════════════════
AFTER THIS OPTIMIZATION:
═══════════════════════════════════════════════════════════════════════════════

- TF32 enabled: 19-bit mantissa for FP32, uses Tensor Cores (~1.05x)
- cuDNN benchmark mode: Tests algorithms, picks fastest (~1.02x)
- NUMA binding: Memory near GPU, lower latency (~1.01x on Grace)
- NVTX markers: Can see exactly where time is spent in Nsight

Combined: ~1.1x speedup

═══════════════════════════════════════════════════════════════════════════════
WHY IT WORKS (Concepts from Chapters 1-6):
═══════════════════════════════════════════════════════════════════════════════

Chapter 1 - GPU Architecture:
  - Tensor Cores are specialized matrix multiply units
  - They're 8-16x faster than regular CUDA cores for matmul
  - But they need specific data formats (TF32, FP16, BF16, FP8)

Chapter 2 - Profiling (Nsight):
  - NVTX ranges let you mark code regions
  - Shows up in Nsight Systems timeline
  - Essential for identifying bottlenecks

Chapter 3 - Grace-Blackwell:
  - Grace CPU + Blackwell GPU share NVLink-C2C
  - NUMA-aware allocation reduces memory latency
  - CPU can access GPU memory coherently (no explicit copies)

Chapter 6 - Tensor Cores:
  - TF32 = TensorFloat-32: 19-bit mantissa, 8-bit exponent
  - Same range as FP32, slightly less precision
  - cuBLAS automatically uses Tensor Cores when TF32 enabled

═══════════════════════════════════════════════════════════════════════════════
TO VERIFY THESE CONCEPTS:
═══════════════════════════════════════════════════════════════════════════════

Run: nsys profile -o 01_basics python 01_basics.py

Look for:
- NVTX ranges in timeline (our markers working)
- Tensor Core kernels (names containing 'hmma' or 'mma')

Run: ncu --set full --kernel-name ".*gemm.*" python 01_basics.py

Look for:
- sm__pipe_tensor_cycles_active > 0 (Tensor Cores being used)

═══════════════════════════════════════════════════════════════════════════════
WHAT'S STILL SLOW:
═══════════════════════════════════════════════════════════════════════════════

Attention is STILL the bottleneck! TF32 helps matrix multiplies, but attention
has a fundamental problem: it materializes an O(seq_len²) matrix.

For 4096 tokens: 4096² × 4 bytes = 64 MB per attention head, per layer.
That's a LOT of memory traffic.

═══════════════════════════════════════════════════════════════════════════════
NEXT STEP: 02_memory_bottleneck.py - Understanding the memory problem
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import torch
from baseline_ultimate_inference import BaselineUltimateInference, InferenceConfig
from optimization_layers import Layer01Basics
from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig
from typing import Optional


class BasicsBenchmark(BaselineUltimateInference):
    """Optimized: Foundation techniques from Chapters 1-6.
    
    Adds:
    - TF32 for Tensor Core utilization
    - cuDNN benchmark mode for algorithm selection
    - NUMA binding for Grace-Blackwell
    - NVTX markers for profiling
    """
    
    def __init__(self, config: Optional[InferenceConfig] = None):
        super().__init__(config)
        self.layer = Layer01Basics()
    
    def setup(self) -> None:
        print("=" * 70)
        print("01_basics.py - Applying Foundation Optimizations (Ch1-6)")
        print("=" * 70)
        
        # Apply optimizations BEFORE model load
        print("\n[Ch6] Enabling TF32 for Tensor Cores...")
        self.layer.enable_tf32()
        print("  → torch.backends.cuda.matmul.allow_tf32 = True")
        print("  → Matrix multiplies now use Tensor Cores")
        
        print("\n[Ch6] Enabling cuDNN benchmark mode...")
        self.layer.enable_cudnn_benchmark()
        print("  → cuDNN will test algorithms and pick fastest")
        
        print("\n[Ch3] Binding to NUMA node...")
        self.layer.bind_numa(self.device)
        print("  → Memory allocations near GPU for lower latency")
        
        print("\n[Ch2] NVTX markers enabled for profiling")
        print("  → Run with 'nsys profile' to see timeline")
        
        # Now load model
        print("\n" + "-" * 70)
        super().setup()
        print("-" * 70)
        
        print(f"\nOptimizations applied: {self.layer}")
        print("=" * 70)


def get_benchmark() -> BaseBenchmark:
    """Factory function for harness discovery."""
    return BasicsBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=benchmark.get_config())
    result = harness.benchmark(benchmark)
    
    print("\n" + "=" * 70)
    print("RESULTS - 01_basics.py")
    print("=" * 70)
    if result.timing:
        print(f"Mean latency: {result.timing.mean_ms:.2f} ms")
        print(f"Expected: ~10% faster than baseline")
    print("\nNext: Run 02_memory_bottleneck.py to understand why attention is slow")
    print("=" * 70)


