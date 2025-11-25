"""Ultimate MoE Inference Lab.

End-to-end benchmark demonstrating every optimization technique from the book,
applied to real MoE model inference on NVIDIA Blackwell B200 GPUs.

Models:
    - gpt-oss-20b: Single GPU (21B params, 3.6B active)
    - gpt-oss-120b: Multi-GPU (117B params, 5.1B active)

Optimization Layers:
    - Layer 1 (Ch1-6): NVTX, NUMA, TF32, cuDNN
    - Layer 2 (Ch7-8): Memory coalescing, occupancy
    - Layer 3 (Ch9-10): Tiling, double buffering, TMA
    - Layer 4 (Ch11-12): Streams, CUDA graphs
    - Layer 5 (Ch13-14): FP8, torch.compile, Triton
    - Layer 6 (Ch15-20): MoE, PagedAttention, speculative decode
"""

__version__ = "0.1.0"


