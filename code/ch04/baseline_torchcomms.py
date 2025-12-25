"""baseline_torchcomms.py - Baseline using legacy torch.distributed patterns.

Legacy patterns use synchronous collectives with no overlap. This benchmark is
launched via torchrun and requires >=2 GPUs.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn
import torch.distributed as dist

from core.benchmark.verification import PrecisionFlags, simple_signature
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    LaunchVia,
    TorchrunLaunchSpec,
)
from core.utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_BATCH = 256
_DEFAULT_HIDDEN = 4096


def _init_distributed() -> tuple[int, int, int]:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError("baseline_torchcomms requires torchrun (RANK/WORLD_SIZE missing).")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def _build_block(hidden: int, device: torch.device) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(hidden, hidden * 4),
        nn.GELU(),
        nn.Linear(hidden * 4, hidden),
    ).to(device).eval()


def _run_worker(iters: int, warmup: int, batch: int, hidden: int) -> None:
    rank, world_size, local_rank = _init_distributed()
    if world_size < 2:
        raise RuntimeError("baseline_torchcomms requires >=2 GPUs.")

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    device = torch.device(f"cuda:{local_rank}")
    comm_block = _build_block(hidden, device)
    aux_block = _build_block(hidden, device)
    inputs = torch.randn(batch, hidden, device=device)

    def _step() -> None:
        with torch.no_grad():
            comm_out = comm_block(inputs)
            dist.all_reduce(comm_out, op=dist.ReduceOp.AVG)
            aux_out = aux_block(inputs)
            _ = comm_out + aux_out

    for _ in range(max(warmup, 0)):
        _step()
    torch.cuda.synchronize(device)

    start = time.perf_counter()
    for _ in range(max(iters, 1)):
        _step()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    tokens_per_iter = batch * hidden
    tokens_per_s = tokens_per_iter * (max(iters, 1) / max(elapsed, 1e-9))

    if rank == 0:
        print(f"rank0 tokens/s: {tokens_per_s:.2f} tokens/s")
        print(f"rank0 time_per_iter_ms: {(elapsed / max(iters,1)) * 1000.0:.3f}")

    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline legacy torch.distributed benchmark")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--batch", type=int, default=_DEFAULT_BATCH)
    parser.add_argument("--hidden", type=int, default=_DEFAULT_HIDDEN)
    args = parser.parse_args()
    _run_worker(args.iters, args.warmup, args.batch, args.hidden)


class BaselineTorchcommsBenchmark(BaseBenchmark):
    """Harness entry that launches this module via torchrun."""

    verification_not_applicable_reason = "torchrun benchmarks execute in external processes"
    skip_input_check = True
    skip_output_check = True

    def __init__(self) -> None:
        super().__init__()
        tokens = float(_DEFAULT_BATCH * _DEFAULT_HIDDEN)
        self.register_workload_metadata(requests_per_iteration=float(_DEFAULT_BATCH), tokens_per_iteration=tokens)

    def benchmark_fn(self) -> None:
        raise RuntimeError("SKIPPED: baseline_torchcomms requires torchrun")

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            launch_via=LaunchVia.TORCHRUN,
            nproc_per_node=2,
            iterations=50,
            warmup=5,
            multi_gpu_required=True,
            measurement_timeout_seconds=900,
        )

    def get_torchrun_spec(self, config: Optional[BenchmarkConfig] = None) -> TorchrunLaunchSpec:
        return TorchrunLaunchSpec(
            script_path=Path(__file__).resolve(),
            script_args=[],
            multi_gpu_required=True,
            name="baseline_torchcomms",
            config_arg_map={
                "iterations": "--iters",
                "warmup": "--warmup",
            },
        )

    def get_input_signature(self) -> dict:
        signature = simple_signature(
            batch_size=_DEFAULT_BATCH,
            dtype="float32",
            hidden_size=_DEFAULT_HIDDEN,
            precision_flags=PrecisionFlags(tf32=False),
        )
        signature.world_size = 2
        signature.collective_type = "all_reduce"
        return signature

    def get_output_tolerance(self) -> tuple:
        return (1e-5, 1e-5)

    def get_verify_output(self) -> torch.Tensor:
        raise RuntimeError("baseline_torchcomms does not expose verification outputs in-process.")


def get_benchmark() -> BaseBenchmark:
    return BaselineTorchcommsBenchmark()


if __name__ == "__main__":
    main()
