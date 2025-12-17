#!/usr/bin/env python3
"""Optimized MoE: Level 1 (Batched)."""
import sys
from pathlib import Path
repo_root = Path(__file__).parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from labs.moe_optimization_journey.level1_batched import Level1Batched


def get_benchmark() -> Level1Batched:
    return Level1Batched()


__all__ = ["Level1Batched", "get_benchmark"]
