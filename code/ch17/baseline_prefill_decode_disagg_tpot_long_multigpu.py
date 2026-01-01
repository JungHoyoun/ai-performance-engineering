"""Baseline TPOT/long-output disaggregated prefill/decode benchmark (multi-GPU torchrun).

Chapter 17: Scaling Disaggregated Prefill and Decode Pipelines

TPOT focus: short prefill, long decode.
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

TPOT_LONG_CONFIG = PrefillDecodeConfig(
    hidden_size=1024,
    num_layers=6,
    batch_size=2,
    requests_per_rank=24,
    context_window=1024,
    decode_tokens=2048,
    transfer_group=4,
    sync_per_request=True,
    barrier_per_request=True,
)


class BaselinePrefillDecodeDisaggTPOTLongMultiGPUBenchmark(_PrefillDecodeMultiGPUBenchmark):
    """Serialized prefill then decode (TPOT/long-output focused)."""

    multi_gpu_required = True

    def __init__(self) -> None:
        super().__init__(
            handoff_mode=HandoffMode.SERIAL,
            label="baseline_prefill_decode_disagg_tpot_long_multigpu",
            cfg=TPOT_LONG_CONFIG,
        )


def get_benchmark() -> BaseBenchmark:
    return BaselinePrefillDecodeDisaggTPOTLongMultiGPUBenchmark()


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
        handoff_mode=HandoffMode.SERIAL,
        label="baseline_prefill_decode_disagg_tpot_long_multigpu",
        iters=int(args.iters),
        warmup=int(args.warmup),
        prefill_ranks=args.prefill_ranks,
    )


if __name__ == "__main__":
    main()
