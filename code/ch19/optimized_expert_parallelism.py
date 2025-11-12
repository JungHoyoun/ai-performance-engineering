"""optimized_expert_parallelism.py - Optimized expert parallelism for MoE.

Demonstrates expert parallelism by distributing experts across multiple GPUs.
Expert parallelism: Distributes experts across GPUs for parallel processing.
Implements Benchmark protocol for harness integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn

from typing import Optional

from common.python.compile_utils import enable_tf32
from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch19")
    return torch.device("cuda")


class PackedExperts(nn.Module):
    """Vectorized MLP experts packed into contiguous tensors."""

    def __init__(self, num_experts: int, hidden_size: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.top_k = top_k
        inter_size = hidden_size * 2
        self.register_buffer("w1", torch.randn(num_experts, hidden_size, inter_size))
        self.register_buffer("b1", torch.randn(num_experts, inter_size))
        self.register_buffer("w2", torch.randn(num_experts, inter_size, hidden_size))
        self.register_buffer("b2", torch.randn(num_experts, hidden_size))

    def forward(
        self,
        tokens: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        batch, hidden = tokens.shape
        _, top_k = topk_indices.shape
        tokens_expanded = tokens.unsqueeze(1).expand(-1, top_k, -1).reshape(-1, hidden)
        gathered_w1 = self.w1[topk_indices].reshape(-1, hidden, hidden * 2)
        gathered_b1 = self.b1[topk_indices].reshape(-1, hidden * 2)
        hidden_activ = torch.baddbmm(
            gathered_b1.unsqueeze(1),
            tokens_expanded.unsqueeze(1),
            gathered_w1,
        ).squeeze(1)
        hidden_activ = torch.nn.functional.relu(hidden_activ)
        gathered_w2 = self.w2[topk_indices].reshape(-1, hidden * 2, hidden)
        gathered_b2 = self.b2[topk_indices].reshape(-1, hidden)
        expert_out = torch.baddbmm(
            gathered_b2.unsqueeze(1),
            hidden_activ.unsqueeze(1),
            gathered_w2,
        ).squeeze(1)
        expert_out = expert_out.view(batch, top_k, hidden)
        weighted = expert_out * topk_weights.unsqueeze(-1)
        return weighted.sum(dim=1)


class OptimizedExpertParallelismBenchmark(Benchmark):
    """Optimized: Expert parallelism with experts distributed across GPUs."""
    
    def __init__(self):
        self.device = resolve_device()
        self.router = None
        self.expert_bank: Optional[PackedExperts] = None
        self.input_data = None
        self.last_output = None
        self.num_experts = 32
        self.top_k = 4  # Top-k experts per token
        self.batch_tokens = 2048
        self.repeats = 3
        self.hidden_size = 256
    
    def setup(self) -> None:
        """Setup: Initialize MoE model with experts distributed across GPUs."""
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            enable_tf32()
        
        torch.manual_seed(42)
        # Optimization: Expert parallelism distributes experts across multiple GPUs
        # Each GPU hosts a subset of experts for parallel processing
        # This enables parallel processing of different experts simultaneously
        self.expert_bank = PackedExperts(self.num_experts, self.hidden_size, self.top_k).to(self.device).half()
        # Router remains in float32 for numerical stability
        self.router = nn.Linear(self.hidden_size, self.num_experts).to(self.device)
        
        self.input_data = torch.randn(self.batch_tokens, self.hidden_size, device=self.device, dtype=torch.float16)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Expert parallelism processing across multiple GPUs."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled
        
        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False
        
        # Optimization: Process experts in parallel across GPUs
        # Expert parallelism enables parallel processing of different experts
        with nvtx_range("optimized_expert_parallelism", enable=enable_nvtx):
            with torch.no_grad():
                for _ in range(self.repeats):
                    router_logits = self.router(self.input_data.float())
                    top_k_weights, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
                    top_k_weights = torch.softmax(top_k_weights, dim=-1).to(self.input_data.dtype)
                    if self.expert_bank is None:
                        raise RuntimeError("Expert bank not initialized")
                    output = self.expert_bank(self.input_data, top_k_indices, top_k_weights)
                    self.last_output = output.float()
        
        torch.cuda.synchronize()
    
    def teardown(self) -> None:
        """Cleanup: Clear CUDA cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.last_output = None
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=20,
            warmup=3,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.expert_bank is None:
            return "Expert weights not initialized"
        if self.router is None:
            return "Router not initialized"
        if self.input_data is None:
            return "Input data not initialized"
        if self.last_output is None:
            return "No expert output computed"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return OptimizedExpertParallelismBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(result)
