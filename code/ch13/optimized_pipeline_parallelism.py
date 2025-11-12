"""optimized_pipeline_parallelism.py - BF16 pipelined inference with CUDA graphs."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, List

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


class OptimizedPipelineParallelismBenchmark(Benchmark):
    """Runs the multi-stage MLP in BF16 with CUDA graph replay."""

    def __init__(self):
        self.device = resolve_device()
        self.batch_size = 8
        self.hidden_dim = 256
        self.model: Optional[nn.Sequential] = None
        self.static_input: Optional[torch.Tensor] = None
        self.static_output: Optional[torch.Tensor] = None
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.capture_stream: Optional[torch.cuda.Stream] = None

    def setup(self) -> None:
        torch.manual_seed(42)
        enable_tf32()

        stages = [
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
        ]
        self.model = nn.Sequential(*stages).to(self.device, dtype=torch.bfloat16).eval()

        self.static_input = torch.randn(
            self.batch_size,
            self.hidden_dim,
            device=self.device,
            dtype=torch.bfloat16,
        )
        self.static_output = torch.empty_like(self.static_input)
        self.graph = torch.cuda.CUDAGraph()
        self.capture_stream = torch.cuda.Stream()

        with torch.cuda.stream(self.capture_stream):
            for _ in range(3):
                self.static_output.copy_(self._forward(self.static_input))
            torch.cuda.synchronize()
            with torch.cuda.graph(self.graph, stream=self.capture_stream):
                self.static_output.copy_(self._forward(self.static_input))
        self.capture_stream.synchronize()

    def _forward(self, tokens: torch.Tensor) -> torch.Tensor:
        assert self.model is not None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.model(tokens)

    def benchmark_fn(self) -> None:
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        if self.graph is None or self.static_input is None:
            raise RuntimeError("CUDA graph not initialized")

        with nvtx_range("optimized_pipeline_parallelism", enable=enable_nvtx):
            self.graph.replay()
        torch.cuda.synchronize(self.device)

    def teardown(self) -> None:
        self.model = None
        self.static_input = None
        self.static_output = None
        self.graph = None
        self.capture_stream = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=50, warmup=10)

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Model missing"
        if self.graph is None:
            return "CUDA graph missing"
        return None


def get_benchmark() -> Benchmark:
    return OptimizedPipelineParallelismBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=10),
    )
    result = harness.benchmark(get_benchmark())
    print(result)
