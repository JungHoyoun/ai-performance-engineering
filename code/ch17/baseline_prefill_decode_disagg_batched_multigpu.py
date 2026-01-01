"""Baseline batched-handoff disaggregated prefill/decode benchmark (multi-GPU torchrun).

Chapter 17: Scaling Disaggregated Prefill and Decode Pipelines

Batched handoff focus: per-request blocking handoff to contrast with grouped transfers.
"""

from __future__ import annotations

import argparse

from ch17.prefill_decode_disagg_multigpu_common import (  # noqa: E402
    HandoffMode,
    PrefillDecodeConfig,
    _PrefillDecodeMultiGPUBenchmark,
    _run_torchrun_worker,
)
from core.harness.benchmark_harness import BaseBenchmark  # noqa: E402

BATCHED_CONFIG = PrefillDecodeConfig(
    hidden_size=512,
    num_layers=2,
    batch_size=1,
    requests_per_rank=256,
    context_window=512,
    decode_tokens=128,
    transfer_group=1,
    sync_per_request=True,
    barrier_per_request=True,
)


class BaselinePrefillDecodeDisaggBatchedMultiGPUBenchmark(_PrefillDecodeMultiGPUBenchmark):
    """Serialized prefill then decode with per-request handoff (batched baseline)."""

    multi_gpu_required = True

    def __init__(self) -> None:
        super().__init__(
            handoff_mode=HandoffMode.SERIAL,
            label="baseline_prefill_decode_disagg_batched_multigpu",
            cfg=BATCHED_CONFIG,
        )


def get_benchmark() -> BaseBenchmark:
    return BaselinePrefillDecodeDisaggBatchedMultiGPUBenchmark()


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
        handoff_mode=HandoffMode.SERIAL,
        label="baseline_prefill_decode_disagg_batched_multigpu",
        iters=int(args.iters),
        warmup=int(args.warmup),
        prefill_ranks=args.prefill_ranks,
    )


if __name__ == "__main__":
    main()
