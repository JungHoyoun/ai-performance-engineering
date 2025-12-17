#!/usr/bin/env python3
"""Optimized MoE: CUDA graphs.

Pairs with: baseline_moe.py

This wrapper must stay workload-equivalent with the baseline benchmark. Use the
MoEJourneyBenchmark implementation (Level 5) to keep parameter_count, inputs,
and verification semantics consistent across levels.
"""

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from labs.moe_optimization_journey.level5_cudagraphs import Level5CUDAGraphs


def get_benchmark() -> Level5CUDAGraphs:
    return Level5CUDAGraphs()


__all__ = ["Level5CUDAGraphs", "get_benchmark"]


