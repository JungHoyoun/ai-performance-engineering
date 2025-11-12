"""Shared workload configuration for Chapter 10 Triton benchmarks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chapter10Workload:
    """Controls batch/microbatch sizes for Triton and CUDA demos."""

    hidden_dim: int = 2048
    ffn_dim: int = 8192

    triton_batch_size: int = 128
    triton_micro_batches: int = 32

    baseline_micro_batch_size: int = 4
    baseline_micro_batches: int = 128
    optimized_batch_size: int = 512

    roofline_batch_size: int = 256
    roofline_micro_batches: int = 64
    roofline_hidden_dim: int = 4096
    roofline_ffn_dim: int = 8192

    pipeline_micro_batches: int = 64
    pipeline_chunk_tokens: int = 256
    pipeline_hidden_dim: int = 2048

    warp_elements: int = 16_777_216
    warp_branch_iterations: int = 32


WORKLOAD = Chapter10Workload()
