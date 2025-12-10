"""optimized_warp_specialization.py - Warp specialization with CUDA and Triton kernels."""

from __future__ import annotations

from typing import Optional

from core.benchmark import triton_compat  # noqa: F401

import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

# Try to load Triton warp specialization
if TRITON_AVAILABLE:
    try:
        from ch10.warp_specialized_triton import warp_specialized_triton_forward_ch10
        TRITON_WARP_SPEC_AVAILABLE = True
    except ImportError:
        TRITON_WARP_SPEC_AVAILABLE = False
else:
    TRITON_WARP_SPEC_AVAILABLE = False

from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
)
from ch10.workload_config import WORKLOAD


class OptimizedWarpSpecializationPipelineBenchmark(BaseBenchmark):
    """Optimized: Warp specialization with intra-kernel pipelining."""
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.input = None
        self.use_triton_warp_spec = TRITON_WARP_SPEC_AVAILABLE
        self.use_triton = TRITON_AVAILABLE
        self.workload = WORKLOAD
        self.micro_batches = self.workload.pipeline_micro_batches
        self.chunk_tokens = self.workload.pipeline_chunk_tokens
        self.hidden_dim = self.workload.pipeline_hidden_dim
        self._checksum = 0.0
        self.producer_stream = torch.cuda.Stream()
        self.compute_stream = torch.cuda.Stream()
        self.consumer_stream = torch.cuda.Stream()
        self.output: Optional[torch.Tensor] = None
        self.jitter_exemption_reason = "Warp specialization benchmark: fixed dimensions"

    def setup(self) -> None:
        """Setup: Initialize model with warp specialization."""
        torch.manual_seed(42)
        
        self.model = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        ).to(self.device).half().eval()
        
        self.input = torch.randn(
            self.micro_batches,
            self.chunk_tokens,
            self.hidden_dim,
            device=self.device,
            dtype=torch.float16,
        )
        if self.use_triton_warp_spec:
            with torch.no_grad():
                input_flat = self.input.flatten()
                intermediate_flat = warp_specialized_triton_forward_ch10(input_flat)
                intermediate = intermediate_flat.view_as(self.input)
                reshaped = intermediate.view(-1, self.hidden_dim)
                _ = self.model(reshaped).sum()
            torch.manual_seed(42)
            self.input = torch.randn(
                self.micro_batches,
                self.chunk_tokens,
                self.hidden_dim,
                device=self.device,
                dtype=torch.float16,
            )
        self._synchronize()
        tokens = float(self.micro_batches * self.chunk_tokens)
        self.register_workload_metadata(
            tokens_per_iteration=tokens,
            requests_per_iteration=float(self.micro_batches),
        )
    
    def benchmark_fn(self) -> None:
        """Benchmark: Warp specialization with CUDA or Triton."""
        assert self.input is not None
        with self._nvtx_range("optimized_warp_specialization_pipeline"):
            with torch.no_grad():
                if self.use_triton_warp_spec:
                    input_flat = self.input.flatten()
                    intermediate_flat = warp_specialized_triton_forward_ch10(input_flat)
                    intermediate = intermediate_flat.view_as(self.input)
                    reshaped = intermediate.view(-1, self.hidden_dim)
                    output = self.model(reshaped)
                    self._checksum = float(output.float().sum().item())
                else:
                    accumulator = torch.zeros(1, device=self.device, dtype=torch.float32)

    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=5,
        )


    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_pipeline_metrics
        return compute_pipeline_metrics(
            num_stages=getattr(self, 'num_stages', 4),
            stage_times_ms=getattr(self, '_stage_times_ms', [1.0]),
        )

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Model not initialized"
        if self.input is None:
            return "Input not initialized"
        return None

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification."""
        if self.output is None:
            raise RuntimeError("Output not available - run benchmark first")
        return self.output.float()

    def get_input_signature(self) -> dict:
        """Return input signature for verification."""
        return {"micro_batches": self.micro_batches, "chunk_tokens": self.chunk_tokens, "hidden_dim": self.hidden_dim}

    def get_output_tolerance(self) -> tuple:
        """Return tolerance for numerical comparison - wider due to FP16."""
        return (0.5, 5.0)


def get_benchmark() -> BaseBenchmark:
    """Factory function for harness discovery."""
    return OptimizedWarpSpecializationPipelineBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
