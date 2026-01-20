"""Optimized warp-specialization stream workload with overlap."""

from __future__ import annotations

from ch11.stream_overlap_base import ConcurrentStreamOptimized


class OptimizedWarpSpecializationStreamsBenchmark(ConcurrentStreamOptimized):
    def __init__(self) -> None:
        # Keep num_streams at 2 (harness limit for custom streams)
        # Increase num_elements and num_segments to better demonstrate overlap benefits
        # Larger workload provides more opportunities for stream overlap
        # Increased to 96M elements and 64 segments to maximize overlap demonstration
        super().__init__(
            "warp_specialization_multistream",
            num_elements=96_000_000,  # Increased from 48M to 96M for better overlap demonstration
            num_segments=64,  # Increased from 32 to better distribute work across 2 streams
            num_streams=2,  # Keep at 2 (harness allows max 2 custom streams)
        )
        # Application replay is unstable for this multistream profile on NCU.
        self.preferred_ncu_replay_mode = "kernel"


def get_benchmark() -> ConcurrentStreamOptimized:
    return OptimizedWarpSpecializationStreamsBenchmark()
