#!/usr/bin/env python3
"""
00_baseline.py - Baseline Inference (No Optimizations)

═══════════════════════════════════════════════════════════════════════════════
PURPOSE: Establish a baseline and MEASURE where time is spent
═══════════════════════════════════════════════════════════════════════════════

This is the starting point. We use standard HuggingFace inference with NO
optimizations. The goal is to:

1. Measure end-to-end latency
2. Profile to see WHERE time is spent
3. Identify the bottleneck (spoiler: it's attention + memory bandwidth)

═══════════════════════════════════════════════════════════════════════════════
WHAT YOU'LL SEE IN PROFILING:
═══════════════════════════════════════════════════════════════════════════════

Run: nsys profile -o baseline python 00_baseline.py

Timeline will show:
- Attention layers dominate (60-80% of time)
- Many small kernel launches (overhead)
- Memory transfers between layers

Run: ncu --set full python 00_baseline.py

Metrics will show:
- Low Tensor Core utilization (<30%)
- Memory-bound (high DRAM traffic, low compute)
- Poor occupancy in some kernels

═══════════════════════════════════════════════════════════════════════════════
EXPECTED PERFORMANCE:
═══════════════════════════════════════════════════════════════════════════════

On B200 with gpt-oss-20b:
- ~100-200 tokens/second
- TTFT: ~1500ms
- TPOT: ~50ms

This is our baseline. Let's make it faster!

═══════════════════════════════════════════════════════════════════════════════
NEXT STEP: 01_basics.py
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import sys
from pathlib import Path

# Re-export from the detailed implementation
sys.path.insert(0, str(Path(__file__).parent))

from baseline_ultimate_inference import (
    BaselineUltimateInference,
    InferenceConfig,
    InferenceMetrics,
    get_benchmark,
    main,
)

# Re-export for harness discovery
__all__ = ["BaselineUltimateInference", "InferenceConfig", "get_benchmark"]

if __name__ == "__main__":
    main()


