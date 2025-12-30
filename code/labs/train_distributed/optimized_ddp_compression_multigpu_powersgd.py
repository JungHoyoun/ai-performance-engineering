"""Optimized DDP training with PowerSGD gradient compression hook (multi-GPU)."""

from __future__ import annotations

from pathlib import Path

from labs.train_distributed.training_utils.torchrun_harness import TorchrunScriptBenchmark


def get_benchmark():
    return TorchrunScriptBenchmark(
        script_path=Path(__file__).parent / "ddp_compression.py",
        base_args=["--compression", "powersgd", "--extra-grad-mb", "1024", "--batch-size", "4"],
        config_arg_map={"iterations": "--steps"},
        target_label="labs/train_distributed:ddp_compression_multigpu",
        multi_gpu_required=True,
        name="optimized_ddp_compression_multigpu_powersgd",
    )
