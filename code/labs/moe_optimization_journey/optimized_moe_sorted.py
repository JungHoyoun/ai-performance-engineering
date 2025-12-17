#!/usr/bin/env python3
"""Optimized MoE: Level 2 (Sorted)."""
import sys
from pathlib import Path
repo_root = Path(__file__).parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from labs.moe_optimization_journey.level2_sorted import Level2Sorted


def get_benchmark() -> Level2Sorted:
    return Level2Sorted()


__all__ = ["Level2Sorted", "get_benchmark"]
