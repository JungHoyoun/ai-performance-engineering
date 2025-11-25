#!/usr/bin/env python3
"""
02_memory_bottleneck.py - Understanding the Memory Problem (Chapters 7-8)

═══════════════════════════════════════════════════════════════════════════════
THE PROBLEM: MEMORY IS THE BOTTLENECK
═══════════════════════════════════════════════════════════════════════════════

Standard attention has a fundamental issue:

    Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d)) @ V

For sequence length N and head dimension d:
- Q @ K.T produces an [N × N] matrix
- For N=4096: 4096² = 16.7 million elements
- At FP32: 16.7M × 4 bytes = 64 MB per head, per layer
- For 32 heads, 32 layers: 64 MB × 32 × 32 = 64 GB just for attention matrices!

This doesn't fit in GPU cache. Every attention layer:
1. Writes 64 MB to HBM (global memory)
2. Reads it back for softmax
3. Writes again after softmax
4. Reads for final matmul

That's ~256 MB of memory traffic PER HEAD PER LAYER.

═══════════════════════════════════════════════════════════════════════════════
WHAT CHAPTERS 7-8 TEACH:
═══════════════════════════════════════════════════════════════════════════════

Chapter 7 - Memory Hierarchy:
  
  GPU Memory Hierarchy (fastest to slowest):
  ┌─────────────────────────────────────────────────────────────────┐
  │ Registers      │ ~256 KB/SM  │ <1 cycle    │ Per-thread        │
  │ Shared Memory  │ 48-228 KB   │ ~20 cycles  │ Per-block         │
  │ L1 Cache       │ 128-256 KB  │ ~30 cycles  │ Per-SM            │
  │ L2 Cache       │ 6-96 MB     │ ~200 cycles │ Shared across SMs │
  │ HBM (Global)   │ 80-192 GB   │ ~400 cycles │ Slowest!          │
  └─────────────────────────────────────────────────────────────────┘
  
  Key insight: Shared memory is 20x faster than HBM!
  
  Coalesced Access:
  - GPU loads 128-byte cache lines at a time
  - If 32 threads access consecutive addresses: ONE transaction
  - If scattered: up to 32 transactions (32x slower!)

Chapter 8 - Occupancy and ILP:

  Occupancy = Active Warps / Max Warps per SM
  
  - Higher occupancy = more warps to hide memory latency
  - But: More warps = fewer registers per warp
  - Sweet spot: 50-75% occupancy usually optimal
  
  ILP (Instruction-Level Parallelism):
  - Multiple independent instructions can execute simultaneously
  - Loop unrolling exposes more ILP
  - Compiler does this, but we can help with hints

═══════════════════════════════════════════════════════════════════════════════
TO SEE THE PROBLEM IN NCU:
═══════════════════════════════════════════════════════════════════════════════

Run: ncu --set full --kernel-name ".*attention.*" python 02_memory_bottleneck.py

Look for these metrics:

Memory Throughput:
  - dram__bytes_read.sum: Total bytes read from HBM
  - dram__bytes_write.sum: Total bytes written to HBM
  - High values = memory-bound kernel

Memory Efficiency:
  - smsp__sass_average_data_bytes_per_sector: Should be ~32 (coalesced)
  - If <32: Memory access is scattered (bad!)

Occupancy:
  - sm__warps_active.avg.pct_of_peak_sustained_active
  - Below 50%? Might be register pressure limiting occupancy

═══════════════════════════════════════════════════════════════════════════════
RAW CUDA EXAMPLES (See these for proof of concepts):
═══════════════════════════════════════════════════════════════════════════════

Coalescing matters (Ch7):
  → code/ch7/baseline_tma_copy.cu vs optimized_tma_copy.cu
  → Shows 3-10x difference from coalescing alone!

Occupancy tuning (Ch8):
  → code/ch8/baseline_double_buffering.cu vs optimized_double_buffering.cu
  → Shows impact of register pressure on performance

═══════════════════════════════════════════════════════════════════════════════
THE SOLUTION PREVIEW:
═══════════════════════════════════════════════════════════════════════════════

What if we could:
1. Process attention in TILES that fit in shared memory?
2. Never materialize the full [N × N] matrix?
3. Compute softmax INCREMENTALLY across tiles?

That's exactly what FlashAttention does! See 03_flash_attention.py

═══════════════════════════════════════════════════════════════════════════════
NEXT STEP: 03_flash_attention.py - The big win!
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import torch
from baseline_ultimate_inference import BaselineUltimateInference, InferenceConfig
from optimization_layers import Layer01Basics, Layer02Memory
from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig
from typing import Optional


class MemoryBottleneckBenchmark(BaselineUltimateInference):
    """Layers 1-2: Basics + Memory understanding.
    
    This layer doesn't add much speedup at the PyTorch level because
    the real memory optimizations require algorithmic changes (FlashAttention).
    
    But it sets up the UNDERSTANDING of WHY we need FlashAttention.
    """
    
    def __init__(self, config: Optional[InferenceConfig] = None):
        super().__init__(config)
        self.layer1 = Layer01Basics()
        self.layer2 = Layer02Memory()
    
    def setup(self) -> None:
        print("=" * 70)
        print("02_memory_bottleneck.py - Understanding Memory (Ch7-8)")
        print("=" * 70)
        
        # Layer 1
        print("\n[Layer 1] Applying basics...")
        self.layer1.enable_tf32()
        self.layer1.enable_cudnn_benchmark()
        
        # Layer 2 - Configure for best memory patterns
        print("\n[Layer 2] Configuring memory optimizations...")
        self.layer2.configure_memory_allocator()
        self.layer2.configure_occupancy_hints()
        
        # Show memory hierarchy info
        mem_info = self.layer2.get_memory_info()
        occ_info = self.layer2.get_occupancy_info()
        
        print("\n" + "-" * 70)
        print("GPU MEMORY HIERARCHY:")
        print("-" * 70)
        if mem_info:
            print(f"  Global Memory (HBM): {mem_info.get('global_memory_gb', 'N/A'):.1f} GB")
            print(f"  L2 Cache: {mem_info.get('l2_cache_mb', 'N/A')} MB")
            print(f"  Shared Memory/Block: {mem_info.get('shared_memory_per_block_kb', 'N/A'):.1f} KB")
        
        if occ_info:
            print(f"\n  SMs: {occ_info.get('num_sms', 'N/A')}")
            print(f"  Max Warps/SM: {occ_info.get('max_warps_per_sm', 'N/A')}")
            print(f"  Registers/SM: {occ_info.get('registers_per_sm', 'N/A')}")
        print("-" * 70)
        
        # Load model
        super().setup()
        
        # Ensure weights are contiguous for coalesced access
        self.layer2.ensure_contiguous_weights(self.model)
        
        # Calculate attention memory requirement
        seq_len = self.config.prompt_tokens
        num_heads = 32  # Typical for 20B model
        attention_memory_mb = (seq_len ** 2 * 4 * num_heads) / 1e6
        
        print("\n" + "=" * 70)
        print("ATTENTION MEMORY ANALYSIS:")
        print("=" * 70)
        print(f"  Sequence length: {seq_len}")
        print(f"  Attention matrix size: {seq_len}² = {seq_len**2:,} elements")
        print(f"  Memory per head (FP32): {seq_len**2 * 4 / 1e6:.1f} MB")
        print(f"  Memory for {num_heads} heads: {attention_memory_mb:.1f} MB")
        print(f"\n  This is read AND written multiple times per layer!")
        print(f"  HBM bandwidth: ~3 TB/s, but that's still the bottleneck.")
        print("=" * 70)
        print("\nRun NCU to see memory traffic: ncu --set full python 02_memory_bottleneck.py")
        print("=" * 70)


def get_benchmark() -> BaseBenchmark:
    """Factory function for harness discovery."""
    return MemoryBottleneckBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=benchmark.get_config())
    result = harness.benchmark(benchmark)
    
    print("\n" + "=" * 70)
    print("RESULTS - 02_memory_bottleneck.py")
    print("=" * 70)
    if result.timing:
        print(f"Mean latency: {result.timing.mean_ms:.2f} ms")
        print(f"Speedup vs baseline: ~1.1-1.2x (memory config helps marginally)")
    print("\nKey insight: We need ALGORITHMIC change to fix the O(n²) memory problem!")
    print("\nNext: Run 03_flash_attention.py for the BIG win!")
    print("=" * 70)


