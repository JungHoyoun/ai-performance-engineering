#!/usr/bin/env python3
"""Optimized MoE Level 4: Triton Fused Kernels.

Optimization: Custom Triton kernels for fused expert computation.
Expected speedup: ~1.3x over Level 2
"""
from labs.moe_optimization_journey.level4_triton import Level4Triton


def get_benchmark() -> Level4Triton:
    return Level4Triton()


__all__ = ["Level4Triton", "get_benchmark"]



