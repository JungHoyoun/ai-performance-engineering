"""optimized_autograd_standard.py - Compiled autograd optimization (optimized).

Compiled autograd using torch.compile for faster backward pass.
Optimizes gradient computation through compilation.

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

# Import arch_config to apply Triton patch for sm_12x support
# The patch removes 'a' suffix from sm_121a -> sm_121 for ptxas compatibility
try:
    import arch_config  # noqa: F401
except ImportError:
    pass  # Continue if arch_config not available
from typing import Optional

from common.python.compile_utils import enable_tf32
from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch13")
    return torch.device("cuda")


class SimpleModel(nn.Module):
    """Simple model for autograd comparison."""
    
    def __init__(self, hidden_dim: int = 1024):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class OptimizedAutogradCompiledBenchmark(Benchmark):
    """Autograd accelerated with CUDA graphs to remove launch overhead."""
    
    def __init__(self):
        self.device = resolve_device()
        self.model: Optional[nn.Module] = None
        self.inputs: Optional[torch.Tensor] = None
        self.targets: Optional[torch.Tensor] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.criterion: Optional[nn.Module] = None
        self.batch_size = 32
        self.hidden_dim = 1024
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.capture_stream: Optional[torch.cuda.Stream] = None
        self.static_input: Optional[torch.Tensor] = None
        self.static_target: Optional[torch.Tensor] = None
        self.input_pool: list[torch.Tensor] = []
        self.target_pool: list[torch.Tensor] = []
        self.pool_index = 0
    
    def setup(self) -> None:
        """Setup training step, capture it with CUDA graphs."""
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            enable_tf32()
        torch.manual_seed(42)
        
        self.model = SimpleModel(hidden_dim=self.hidden_dim).to(self.device).half().train()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, foreach=True)
        self.criterion = nn.MSELoss()

        self.input_pool = [
            torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float16)
            for _ in range(8)
        ]
        self.target_pool = [torch.randn_like(inp) for inp in self.input_pool]
        self.inputs = self.input_pool[0]
        self.targets = self.target_pool[0]

        # Warmup a few eager steps before capture.
        for idx in range(3):
            self._train_step(self.input_pool[idx], self.target_pool[idx])
        torch.cuda.synchronize()

        # CUDA graph capture.
        self.static_input = self.input_pool[0].clone()
        self.static_target = self.target_pool[0].clone()
        self.graph = torch.cuda.CUDAGraph()
        self.capture_stream = torch.cuda.Stream()
        with torch.cuda.stream(self.capture_stream):
            for _ in range(2):
                self._train_step(self.static_input, self.static_target)
            torch.cuda.synchronize()
            with torch.cuda.graph(self.graph, stream=self.capture_stream):
                self._train_step(self.static_input, self.static_target)
        self.capture_stream.synchronize()
    
    def benchmark_fn(self) -> None:
        """Function to benchmark - compiled autograd."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        if self.graph is None or self.static_input is None or self.static_target is None:
            raise RuntimeError("CUDA graph not initialized")

        current_input = self.input_pool[self.pool_index]
        current_target = self.target_pool[self.pool_index]
        self.pool_index = (self.pool_index + 1) % len(self.input_pool)

        with nvtx_range("autograd_standard", enable=enable_nvtx):
            self.static_input.copy_(current_input)
            self.static_target.copy_(current_target)
            self.graph.replay()
        torch.cuda.synchronize(self.device)

    def _train_step(self, batch: torch.Tensor, target: torch.Tensor) -> None:
        assert self.model is not None and self.optimizer is not None and self.criterion is not None
        self.optimizer.zero_grad(set_to_none=True)
        outputs = self.model(batch)
        loss = self.criterion(outputs, target)
        loss.backward()
        self.optimizer.step()

    def teardown(self) -> None:
        """Cleanup."""
        del self.model, self.inputs, self.targets, self.optimizer, self.criterion
        self.graph = None
        self.static_input = None
        self.static_target = None
        self.capture_stream = None
        self.input_pool = []
        self.target_pool = []
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=10,
            enable_memory_tracking=False,
            enable_profiling=False,
            setup_timeout_seconds=180,  # torch.compile compilation can take 60-120 seconds
        )
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return OptimizedAutogradCompiledBenchmark()


if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nOptimized Autograd Compiled: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
