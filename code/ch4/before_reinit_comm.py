"""Anti-pattern example: reinitializing NCCL every iteration."""

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


def _launch_config() -> tuple[int, int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required for NCCL re-init demo.")

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return rank, world_size, device


def main() -> None:
    rank, world_size, device = _launch_config()

    for iteration in range(5):
        t0 = time.time()
        setup_single_gpu_env()  # Auto-setup for single-GPU mode
    dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )
        init_elapsed = time.time() - t0

        tensor = torch.ones(1, device=device)
        dist.all_reduce(tensor)

        if rank == 0:
            print(
                f"[Iter {iteration}] NCCL init took {init_elapsed * 1000:.2f} ms",
                flush=True,
            )

        dist.destroy_process_group()


if __name__ == "__main__":
    main()
