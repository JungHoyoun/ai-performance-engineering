"""DDP example demonstrating gradient overlap.

Launch:
    torchrun --standalone --nproc-per-node=2 ch4/after_overlap_ddp.py
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


def init_distributed() -> tuple[int, torch.device]:
    setup_single_gpu_env()  # Auto-setup for single-GPU mode
    
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")

    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return rank, device


def main() -> None:
    rank, device = init_distributed()

    model = MultiLayerNet(1024).to(device)
    if torch.cuda.is_available():
        try:
            model = torch.compile(model, mode="max-autotune")
        except Exception as exc:  # pragma: no cover
            if rank == 0:
                print(f"torch.compile unavailable, falling back to eager mode: {exc}", flush=True)
    ddp_model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[device.index] if device.type == "cuda" else None,
        gradient_as_bucket_view=True,
    )

    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)

    batch_size = 128
    data = torch.randn(batch_size, 1024, device=device)
    target = torch.randn(batch_size, 1, device=device)

    output = ddp_model(data)
    loss = nn.functional.mse_loss(output, target)
    loss.backward()  # DDP overlaps gradient all-reduce with computation
    optimizer.step()

    if rank == 0:
        print(f"DDP overlap loss: {loss.item():.4f}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
