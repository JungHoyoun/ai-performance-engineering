"""Baseline TTFT-focused disaggregated prefill/decode benchmark (multi-GPU torchrun).

Chapter 17: Scaling Disaggregated Prefill and Decode Pipelines

TTFT focus: large context prefill, short decode.
"""

from __future__ import annotations

import argparse

from ch17.baseline_prefill_decode_disagg_multigpu import (  # noqa: E402
    PrefillDecodeConfig,
    _PrefillDecodeMultiGPUBenchmark,
    _run_torchrun_worker,
)
from core.harness.benchmark_harness import BaseBenchmark  # noqa: E402

TTFT_CONFIG = PrefillDecodeConfig(
    context_window=2048,
    decode_tokens=32,
)


class BaselinePrefillDecodeDisaggTTFTMultiGPUBenchmark(_PrefillDecodeMultiGPUBenchmark):
    """Serialized prefill then decode (TTFT-focused)."""

    def __init__(self) -> None:
        super().__init__(
            overlap=False,
            label="baseline_prefill_decode_disagg_ttft_multigpu",
            cfg=TTFT_CONFIG,
        )


def get_benchmark() -> BaseBenchmark:
    return BaselinePrefillDecodeDisaggTTFTMultiGPUBenchmark()


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
        TTFT_CONFIG,
        overlap=False,
        label="baseline_prefill_decode_disagg_ttft_multigpu",
        iters=int(args.iters),
        warmup=int(args.warmup),
        prefill_ranks=args.prefill_ranks,
    )


if __name__ == "__main__":
    main()
