"""DistributedDataParallel example aligned with torchrun best practices.

Launch:
    torchrun --standalone --nproc-per-node=2 ch4/after_ddp.py
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


class SimpleNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.relu(self.linear1(x)))


def init_distributed() -> tuple[int, int, torch.device]:
    """Initialize torch.distributed using environment variables from torchrun."""
    # Auto-setup for single-GPU mode if distributed env vars not set
    if "RANK" not in os.environ:
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("LOCAL_RANK", "0")
    
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return rank, world_size, device


def main() -> None:
    rank, world_size, device = init_distributed()

    model = SimpleNet(input_size=1024, hidden_size=256).to(device)
    if torch.cuda.is_available():
        try:
            model = torch.compile(model, mode="max-autotune")
        except Exception as exc:  # pragma: no cover - fallback for mismatched toolchains
            if rank == 0:
                print(f"torch.compile unavailable, falling back to eager mode: {exc}", flush=True)
    ddp_model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[device.index] if device.type == "cuda" else None,
    )

    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)

    batch_size = 256
    data = torch.randn(batch_size, 1024, device=device)
    target = torch.randn(batch_size, 1, device=device)

    output = ddp_model(data)
    loss = nn.functional.mse_loss(output, target)
    loss.backward()
    optimizer.step()

    if rank == 0:
        print(f"DDP loss: {loss.item():.4f}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
