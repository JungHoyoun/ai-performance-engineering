"""Optimized TPOT/long-output disaggregated prefill/decode benchmark (multi-GPU torchrun).

Chapter 17: Scaling Disaggregated Prefill and Decode Pipelines

TPOT focus: overlap transfers with long decode.
"""

from __future__ import annotations

import argparse

from ch17.baseline_prefill_decode_disagg_multigpu import (  # noqa: E402
    _PrefillDecodeMultiGPUBenchmark,
    _run_torchrun_worker,
)
from ch17.baseline_prefill_decode_disagg_tpot_long_multigpu import TPOT_LONG_CONFIG  # noqa: E402
from core.harness.benchmark_harness import BaseBenchmark  # noqa: E402


class OptimizedPrefillDecodeDisaggTPOTLongMultiGPUBenchmark(_PrefillDecodeMultiGPUBenchmark):
    """Pipelined prefill/decode overlap (TPOT/long-output focused)."""

    multi_gpu_required = True

    def __init__(self) -> None:
        super().__init__(
            overlap=True,
            label="optimized_prefill_decode_disagg_tpot_long_multigpu",
            cfg=TPOT_LONG_CONFIG,
        )


def get_benchmark() -> BaseBenchmark:
    return OptimizedPrefillDecodeDisaggTPOTLongMultiGPUBenchmark()


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
        TPOT_LONG_CONFIG,
        overlap=True,
        label="optimized_prefill_decode_disagg_tpot_long_multigpu",
        iters=int(args.iters),
        warmup=int(args.warmup),
        prefill_ranks=args.prefill_ranks,
    )


if __name__ == "__main__":
    main()
