"""Optimization Layers for Ultimate MoE Inference.

Each layer corresponds to book chapters and can be applied incrementally:

- Layer 01 (Ch1-6): Basics - NVTX, NUMA, TF32, cuDNN
- Layer 02 (Ch7-8): Memory - Coalescing, occupancy, vectorization
- Layer 03 (Ch9-10): Pipelining - Tiling, double buffering, TMA
- Layer 04 (Ch11-12): Concurrency - Streams, CUDA graphs
- Layer 05 (Ch13-14): PyTorch - FP8, torch.compile, Triton
- Layer 06 (Ch15-20): Advanced - MoE, PagedAttention, speculative decode
"""

from .layer_01_basics import Layer01Basics
from .layer_02_memory import Layer02Memory
from .layer_03_pipelining import Layer03Pipelining
from .layer_04_concurrency import Layer04Concurrency
from .layer_05_pytorch import Layer05PyTorch
from .layer_06_advanced import Layer06Advanced

__all__ = [
    "Layer01Basics",
    "Layer02Memory",
    "Layer03Pipelining",
    "Layer04Concurrency",
    "Layer05PyTorch",
    "Layer06Advanced",
]


def get_all_layers():
    """Get all optimization layers in order."""
    return [
        Layer01Basics(),
        Layer02Memory(),
        Layer03Pipelining(),
        Layer04Concurrency(),
        Layer05PyTorch(),
        Layer06Advanced(),
    ]


def get_layers_up_to(layer_num: int):
    """Get layers up to and including the specified layer.
    
    Args:
        layer_num: Layer number (1-6)
        
    Returns:
        List of layer instances
    """
    all_layers = get_all_layers()
    return all_layers[:layer_num]


