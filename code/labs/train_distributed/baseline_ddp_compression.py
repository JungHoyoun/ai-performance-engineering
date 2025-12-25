"""Baseline DDP training with uncompressed all-reduce (compression off)."""

from __future__ import annotations

from pathlib import Path

from labs.train_distributed.training_utils.torchrun_harness import TorchrunScriptBenchmark


def get_benchmark():
    return TorchrunScriptBenchmark(
        script_path=Path(__file__).parent / "ddp_compression.py",
        base_args=["--compression", "none"],
        config_arg_map={"iterations": "--steps"},
        target_label="labs/train_distributed:ddp_compression",
        multi_gpu_required=True,
        name="baseline_ddp_compression",
    )
