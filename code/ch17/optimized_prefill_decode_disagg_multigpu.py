"""Optimized disaggregated prefill/decode benchmark (multi-GPU torchrun).

Chapter 17: Scaling Disaggregated Prefill and Decode Pipelines

Optimizations:
- Overlap prefill and decode by pipelining transfers per request.
"""

from __future__ import annotations

import argparse

from ch17.baseline_prefill_decode_disagg_multigpu import (  # noqa: E402
    PrefillDecodeConfig,
    _PrefillDecodeMultiGPUBenchmark,
    _run_torchrun_worker,
)
from core.harness.benchmark_harness import BaseBenchmark  # noqa: E402


class OptimizedPrefillDecodeDisaggMultiGPUBenchmark(_PrefillDecodeMultiGPUBenchmark):
    """Pipelined prefill/decode overlap across multi-GPU ranks."""

    multi_gpu_required = True

    def __init__(self) -> None:
        super().__init__(overlap=True, label="optimized_prefill_decode_disagg_multigpu")


def get_benchmark() -> BaseBenchmark:
    return OptimizedPrefillDecodeDisaggMultiGPUBenchmark()


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
        PrefillDecodeConfig(),
        overlap=True,
        label="optimized_prefill_decode_disagg_multigpu",
        iters=int(args.iters),
        warmup=int(args.warmup),
        prefill_ranks=args.prefill_ranks,
    )


if __name__ == "__main__":
    main()
