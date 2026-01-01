"""Optimized batched-handoff disaggregated prefill/decode benchmark (multi-GPU torchrun).

Chapter 17: Scaling Disaggregated Prefill and Decode Pipelines

Optimizations:
- Batch prefill/decode and send contiguous KV/seed slabs to amortize per-request handoff overhead.
"""

from __future__ import annotations

import argparse

from ch17.baseline_prefill_decode_disagg_batched_multigpu import BATCHED_CONFIG  # noqa: E402
from ch17.prefill_decode_disagg_multigpu_common import (  # noqa: E402
    HandoffMode,
    _PrefillDecodeMultiGPUBenchmark,
    _run_torchrun_worker,
)
from core.harness.benchmark_harness import BaseBenchmark  # noqa: E402


class OptimizedPrefillDecodeDisaggBatchedMultiGPUBenchmark(_PrefillDecodeMultiGPUBenchmark):
    """Grouped KV handoff across multi-GPU ranks."""

    multi_gpu_required = True

    def __init__(self) -> None:
        super().__init__(
            handoff_mode=HandoffMode.BATCHED,
            label="optimized_prefill_decode_disagg_batched_multigpu",
            cfg=BATCHED_CONFIG,
        )


def get_benchmark() -> BaseBenchmark:
    return OptimizedPrefillDecodeDisaggBatchedMultiGPUBenchmark()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--prefill-ranks",
        type=int,
        default=None,
        help="Number of prefill ranks (defaults to world_size//2 when even).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _run_torchrun_worker(
        BATCHED_CONFIG,
        handoff_mode=HandoffMode.BATCHED,
        label="optimized_prefill_decode_disagg_batched_multigpu",
        iters=int(args.iters),
        warmup=int(args.warmup),
        prefill_ranks=args.prefill_ranks,
    )


if __name__ == "__main__":
    main()
