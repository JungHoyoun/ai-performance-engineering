"""Baseline DDP example without communication overlap.

Launch:
    torchrun --standalone --nproc-per-node=2 ch4/before_no_overlap.py
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

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


class MultiLayerNet(nn.Module):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(size, size)
        self.fc2 = nn.Linear(size, size)
        self.fc3 = nn.Linear(size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def init_distributed() -> tuple[int, int, torch.device]:
    if not dist.is_initialized():
        setup_single_gpu_env()  # Auto-setup for single-GPU mode
    dist.init_process_group(backend="nccl", init_method="env://")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return rank, world_size, device


def main() -> None:
    rank, world_size, device = init_distributed()

    model = MultiLayerNet(1024).to(device)
    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
        )

    optimizer = optim.SGD(model.parameters(), lr=0.01)

    batch_size = 128
    data = torch.randn(batch_size, 1024, device=device)
    target = torch.randn(batch_size, 1, device=device)

    output = model(data)
    loss = nn.functional.mse_loss(output, target)
    loss.backward()

    if world_size > 1:
        for param in model.parameters():
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad.mul_(1.0 / world_size)

    optimizer.step()

    if rank == 0:
        print(f"Loss: {loss.item():.4f}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
