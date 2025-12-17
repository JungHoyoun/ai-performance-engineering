#!/usr/bin/env python3
"""Optimized MoE: Level 2 (Triton Fused)."""
import sys
from pathlib import Path
repo_root = Path(__file__).parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from labs.moe_optimization_journey.level2_fused import Level2Fused


def get_benchmark() -> Level2Fused:
    return Level2Fused()


__all__ = ["Level2Fused", "get_benchmark"]



