"""baseline_autotuning.py - Eager-mode baseline for autotuning benchmarks.

Pairs with `optimized_autotuning.py`, which uses `torch.compile(..., mode=\"max-autotune\")`
to exercise Inductor autotuning paths while keeping outputs comparable.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

import ch20.arch_config  # noqa: F401 - Apply chapter defaults

from arch_config import prefer_sdpa_backends
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from core.utils.compile_utils import enable_tf32


class AutotuneModel(nn.Module):
    """Tiny MLP used for eager/compiled comparisons."""

    def __init__(self, hidden_dim: int = 1024):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.fc1(x))
        return self.fc2(x)


class BaselineAutotuningBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Runs the eager model to establish the autotuning baseline."""

    def __init__(self) -> None:
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.inputs: Optional[torch.Tensor] = None
        self.batch = 16
        self.hidden_dim = 1024
        tokens = self.batch * self.hidden_dim
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch),
            tokens_per_iteration=float(tokens),
        )
        self._verify_input: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        enable_tf32(set_global_precision=True)

        self.model = AutotuneModel(self.hidden_dim).to(self.device, dtype=torch.bfloat16).eval()
        self.inputs = torch.randn(self.batch, self.hidden_dim, device=self.device, dtype=torch.bfloat16)
        self._verify_input = self.inputs[0:1].clone()

        for _ in range(3):
            with torch.no_grad(), prefer_sdpa_backends():
                _ = self.model(self.inputs)
        self._synchronize()

    def benchmark_fn(self) -> None:
        if self.model is None or self.inputs is None:
            raise RuntimeError("Model/inputs not initialized")
        with self._nvtx_range("baseline_autotuning"):
            with torch.no_grad(), prefer_sdpa_backends():
                _ = self.model(self.inputs)
            self._synchronize()

    def capture_verification_payload(self) -> None:
        if self._verify_input is None or self.model is None:
            raise RuntimeError("setup() must prepare verify input before verification")
        with torch.no_grad(), prefer_sdpa_backends():
            self.output = self.model(self._verify_input).float().clone()
        self._set_verification_payload(
            inputs={"verify_input": self._verify_input},
            output=self.output,
            batch_size=int(self._verify_input.shape[0]),
            parameter_count=sum(p.numel() for p in self.model.parameters()),
            output_tolerance=(0.1, 1.0),
        )

    def teardown(self) -> None:
        self.model = None
        self.inputs = None
        self._verify_input = None
        self.output = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=20,
            warmup=5,
            use_subprocess=True,
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload


def get_benchmark() -> BaseBenchmark:
    return BaselineAutotuningBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
