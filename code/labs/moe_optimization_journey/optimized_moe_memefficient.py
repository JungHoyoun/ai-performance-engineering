#!/usr/bin/env python3
"""Optimized MoE: Level 3 (Memory Efficient)."""
import sys
from pathlib import Path
repo_root = Path(__file__).parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from labs.moe_optimization_journey.level3_memefficient import Level3MemEfficient


def get_benchmark() -> Level3MemEfficient:
    return Level3MemEfficient()


__all__ = ["Level3MemEfficient", "get_benchmark"]



