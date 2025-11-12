"""Shared workload configuration for Chapter 18 distributed/parallel benchmarks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chapter18Workload:
    """Canonical workload sizes for large-scale LLM infrastructure demos."""

    attention_hidden_dim: int = 2048
    attention_num_heads: int = 16
    attention_batch_size: int = 8
    attention_seq_len: int = 2048
    decode_seq_len: int = 256
    micro_batches: int = 8

    pipeline_stages: int = 4
    pipeline_micro_batches: int = 12

    tensor_parallel_shards: int = 4
    distributed_ranks: int = 4
    distributed_global_batch: int = 128

    roofline_matmul_size: int = 4096
    roofline_tile: int = 256

    shared_feature_maps: int = 96
    shared_spatial: int = 192
    shared_kernel_size: int = 5


WORKLOAD = Chapter18Workload()
