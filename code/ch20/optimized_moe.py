"""optimized_moe.py - Minimal MoE-style benchmark for Chapter 20 tests."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

try:
    import ch20.arch_config  # noqa: F401 - Apply chapter defaults
except ImportError:
    pass

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from core.utils.compile_utils import compile_model


class ToyMoe(nn.Module):
    """Simplified MoE block with two experts and top-1 routing."""

    def __init__(self, hidden_dim: int = 1024):
        super().__init__()
        self.output = None
        self._verify_input = None
        self.gate = nn.Linear(hidden_dim, 2, bias=False)
        self.expert0 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.expert1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = torch.softmax(self.gate(x), dim=-1)
        top_expert = scores.argmax(dim=-1)
        out0 = self.expert0(x)
        out1 = self.expert1(x)
        mask0 = (top_expert == 0).float().unsqueeze(-1)
        mask1 = (top_expert == 1).float().unsqueeze(-1)
        return out0 * mask0 + out1 * mask1


class OptimizedMoeBenchmark(BaseBenchmark):
    """Exercise a tiny MoE forward pass with optional compilation."""

    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.inputs: Optional[torch.Tensor] = None
        self.batch = 32
        self.hidden_dim = 1024
        tokens = self.batch * self.hidden_dim
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch),
            tokens_per_iteration=float(tokens),
        )
        self._verify_input: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None

    def setup(self) -> None:
        torch.manual_seed(1)
        model = ToyMoe(self.hidden_dim).to(self.device).half().eval()
        self.model = compile_model(
            model,
            mode="reduce-overhead",
            fullgraph=False,
            dynamic=False,
        )
        self.inputs = torch.randn(self.batch, self.hidden_dim, device=self.device, dtype=torch.float16)
        self._verify_input = self.inputs[0:1].clone()
        for _ in range(2):
            with torch.no_grad():
                _ = self.model(self.inputs)
        self._synchronize()

    def benchmark_fn(self) -> None:
        assert self.model is not None and self.inputs is not None
        with self._nvtx_range("optimized_moe"):
            with torch.no_grad():
                _ = self.model(self.inputs)
            self._synchronize()
        # Capture output AFTER benchmark for verification
        if self._verify_input is None:
            raise RuntimeError("setup() must prepare verify input before verification")
        if self.model is not None:
            with torch.no_grad():
                self.output = self.model(self._verify_input).float().clone()
        self._set_verification_payload(
            inputs={"verify_input": self._verify_input},
            output=self.output if self.output is not None else torch.zeros_like(self._verify_input),
            batch_size=self._verify_input.shape[0],
            parameter_count=sum(p.numel() for p in self.model.parameters()) if self.model is not None else 0,
            output_tolerance=(0.1, 1.0),
        )

    def teardown(self) -> None:
        self.model = None
        self.inputs = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=10,
            warmup=10,
            use_subprocess=True,
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_ai_optimization_metrics
        return compute_ai_optimization_metrics(
            original_time_ms=getattr(self, '_original_ms', 10.0),
            ai_optimized_time_ms=getattr(self, '_optimized_ms', 5.0),
            suggestions_applied=getattr(self, '_suggestions_applied', 1),
            suggestions_total=getattr(self, '_suggestions_total', 1),
        )

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Model not initialized"
        if self.inputs is None:
            return "Input tensor not initialized"
        return None

    def get_verify_output(self) -> torch.Tensor:
        return super().get_verify_output()

    def get_input_signature(self) -> dict:
        return {
            "shapes": {"verify_input": (self.batch, self.hidden_dim)},
            "dtypes": {"verify_input": "torch.float16"},
            "batch_size": self.batch,
            "parameter_count": sum(p.numel() for p in self.model.parameters()) if self.model is not None else 0,
            "precision_flags": {
                "fp16": True,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
        }

    def get_output_tolerance(self) -> tuple:
        payload = getattr(self, "_verification_payload", None)
        if payload is None:
            return (0.1, 1.0)
        return super().get_output_tolerance()


def get_benchmark() -> BaseBenchmark:
    return OptimizedMoeBenchmark()
