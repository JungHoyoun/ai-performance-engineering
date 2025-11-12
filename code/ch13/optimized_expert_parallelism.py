"""optimized_expert_parallelism.py - Vectorized MoE dispatch."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn

from common.python.benchmark_harness import Benchmark, BenchmarkConfig
from common.python.compile_utils import enable_tf32


def resolve_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch13")
    return torch.device("cuda")


class Expert(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TopKDispatchMoE(nn.Module):
    """Vectorized top-k MoE dispatch with index_add aggregation."""

    def __init__(self, hidden_size: int, num_experts: int, top_k: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList([Expert(hidden_size) for _ in range(num_experts)])

    def _prepare_dispatch(
        self, logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scores, indices = torch.topk(logits, self.top_k, dim=-1)
        probs = torch.softmax(scores, dim=-1).to(logits.dtype)
        expert_ids = indices.reshape(-1)
        weights = probs.reshape(-1)
        token_ids = torch.arange(logits.shape[0], device=logits.device).repeat_interleave(self.top_k)
        order = torch.argsort(expert_ids)
        return expert_ids[order], weights[order], token_ids[order]

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        router_logits = self.router(tokens)
        expert_ids, weights, token_ids = self._prepare_dispatch(router_logits)
        if expert_ids.numel() == 0:
            return torch.zeros_like(tokens)

        output = torch.zeros_like(tokens)
        unique_ids, counts = torch.unique_consecutive(expert_ids, return_counts=True)
        cursor = 0
        for expert_id, count in zip(unique_ids.tolist(), counts.tolist()):
            shard_tokens = token_ids[cursor : cursor + count]
            shard_weights = weights[cursor : cursor + count].unsqueeze(-1).to(tokens.dtype)
            expert_input = tokens.index_select(0, shard_tokens)
            expert_output = self.experts[expert_id](expert_input)
            output.index_add_(0, shard_tokens, expert_output * shard_weights)
            cursor += count
        return output


class OptimizedExpertParallelismBenchmark(Benchmark):
    """Optimized expert parallelism using vectorized dispatchers."""

    def __init__(self):
        self.device = resolve_device()
        self.hidden_size = 256
        self.num_experts = 8
        self.top_k = 2
        self.batch_size = 32
        self.model: Optional[TopKDispatchMoE] = None
        self.tokens: Optional[torch.Tensor] = None

    def setup(self) -> None:
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            enable_tf32()
        torch.manual_seed(42)
        self.model = TopKDispatchMoE(self.hidden_size, self.num_experts, self.top_k).to(
            self.device, dtype=torch.bfloat16
        )
        self.model.eval()
        self.tokens = torch.randn(
            self.batch_size,
            self.hidden_size,
            device=self.device,
            dtype=torch.bfloat16,
        )
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        if self.model is None or self.tokens is None:
            raise RuntimeError("Model/tokens not initialized")

        with nvtx_range("optimized_expert_parallelism", enable=enable_nvtx):
            with torch.no_grad():
                _ = self.model(self.tokens)
        torch.cuda.synchronize()

    def teardown(self) -> None:
        self.model = None
        self.tokens = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=10, warmup=2)

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Model missing"
        if self.tokens is None:
            return "Input missing"
        return None


def get_benchmark() -> Benchmark:
    return OptimizedExpertParallelismBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=10, warmup=2),
    )
    result = harness.benchmark(get_benchmark())
    print(result)
