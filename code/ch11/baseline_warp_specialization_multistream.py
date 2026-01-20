"""Baseline warp-specialization stream workload (no overlap)."""

from __future__ import annotations

from ch11.stream_overlap_base import StridedStreamBaseline


class BaselineWarpSpecializationStreamsBenchmark(StridedStreamBaseline):
    def __init__(self) -> None:
        # Use same num_elements and num_segments as optimized for equivalent workload
        super().__init__(
            "baseline_warp_specialization_multistream",
            num_elements=96_000_000,  # Same as optimized
            num_segments=64,  # Same as optimized for equivalent chunking
        )


def get_benchmark() -> StridedStreamBaseline:
    return BaselineWarpSpecializationStreamsBenchmark()
