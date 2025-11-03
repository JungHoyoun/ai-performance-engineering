"""Preferred pattern: initialize the NCCL communicator once and reuse it."""

from __future__ import annotations

import os
import time

import torch
import torch.distributed as dist

try:
    import arch_config  # noqa: F401
except ImportError:
    pass
try:
    from distributed_helper import setup_single_gpu_env
except ImportError:
    def setup_single_gpu_env():
        if "RANK" not in os.environ:
            os.environ.setdefault("RANK", "0")
            os.environ.setdefault("WORLD_SIZE", "1")
            os.environ.setdefault("MASTER_ADDR", "localhost")
            os.environ.setdefault("MASTER_PORT", "29500")
            os.environ.setdefault("LOCAL_RANK", "0")


def init_distributed() -> tuple[int, int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required for NCCL demo.")

    if not dist.is_initialized():
        setup_single_gpu_env()  # Auto-setup for single-GPU mode
    dist.init_process_group("nccl", init_method="env://")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return rank, world_size, device


def main() -> None:
    rank, world_size, device = init_distributed()

    t0 = time.time()
    dist.barrier()
    init_elapsed = time.time() - t0

    if rank == 0:
        print(f"One-time communicator initialization + barrier: {init_elapsed*1000:.2f} ms", flush=True)

    for iteration in range(5):
        start_iter = time.time()
        tensor = torch.ones(1, device=device)
        dist.all_reduce(tensor)
        iter_elapsed = time.time() - start_iter

        if rank == 0:
            print(f"[Iter {iteration}] all-reduce took {iter_elapsed*1000:.2f} ms", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
