"""Optimized attention path that uses learned heuristics to batch work efficiently."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.python.benchmark_harness import Benchmark, BenchmarkConfig
from ch18.workload_config import WORKLOAD

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch18")
    return torch.device("cuda")


class FlashAttentionModel(nn.Module):
    """Tensor-core optimized multi-head attention using flash kernels."""

    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=False)
        self.output = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = tokens.shape
        qkv = self.qkv(tokens)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=False, enable_math=False):
            attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        attn = attn.transpose(1, 2).reshape(batch, seq, self.hidden_dim)
        return self.output(attn)


class OptimizedAiOptimizationBenchmark(Benchmark):
    """AI-assisted attention implementation that batches work adaptively."""

    def __init__(self):
        self.device = resolve_device()
        self.workload = WORKLOAD
        self.hidden_dim = self.workload.attention_hidden_dim
        self.num_heads = self.workload.attention_num_heads
        self.batch_size = self.workload.attention_batch_size
        self.sequence_length = self.workload.attention_seq_len
        self.micro_batches = self.workload.micro_batches

        self.model: Optional[FlashAttentionModel] = None
        self.token_cache: Optional[torch.Tensor] = None
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.graph_input: Optional[torch.Tensor] = None
        self.graph_output: Optional[torch.Tensor] = None

    def setup(self) -> None:
        torch.manual_seed(42)
        self.model = FlashAttentionModel(self.hidden_dim, self.num_heads).to(self.device).half().eval()
        self.token_cache = torch.randn(
            self.batch_size,
            self.sequence_length,
            self.hidden_dim,
            dtype=torch.float16,
            device=self.device,
        )
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        self.graph_input = torch.empty_like(self.token_cache)
        self.graph_output = torch.empty_like(self.token_cache)
        self.graph = torch.cuda.CUDAGraph()
        self.graph_input.copy_(self.token_cache)
        warmup = self.model(self.graph_input)
        self.graph_output.copy_(warmup)
        torch.cuda.synchronize()
        with torch.cuda.graph(self.graph):
            self.graph_output.copy_(self.model(self.graph_input))
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        assert self.model is not None
        assert self.token_cache is not None

        with nvtx_range("optimized_ai_optimization", enable=enable_nvtx):
            with torch.no_grad():
                for micro in range(self.micro_batches):
                    tokens = torch.roll(self.token_cache, shifts=micro * 32, dims=1).contiguous()
                    assert self.graph is not None and self.graph_input is not None
                    self.graph_input.copy_(tokens)
                    self.graph.replay()

    def teardown(self) -> None:
        self.model = None
        self.token_cache = None
        self.graph = None
        self.graph_input = None
        self.graph_output = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=2,
            warmup=1,
            enable_memory_tracking=False,
            measurement_timeout_seconds=120,
        )

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Attention model not initialized"
        if self.token_cache is None:
            return "Input cache not initialized"
        return None


def get_benchmark() -> Benchmark:
    return OptimizedAiOptimizationBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=BenchmarkConfig(iterations=3, warmup=1))
    result = harness.benchmark(OptimizedAiOptimizationBenchmark())
    print(f"Optimized AI optimization mean: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
