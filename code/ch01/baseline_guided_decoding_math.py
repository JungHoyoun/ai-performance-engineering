"""Baseline guided decoding - math-only SDP for compatibility."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn

try:
    import ch01.arch_config  # noqa: F401
except ImportError:
    pass

from typing import Optional

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from core.utils.compile_utils import enable_tf32


class BaselineGuidedDecodingMathBenchmark(BaseBenchmark):
    """Baseline: guided decoding with schema, forcing SDP to math only."""

    def __init__(self):
        super().__init__()
        self.model = None
        self.input_ids = None
        self.embedded_input = None
        self.memory = None
        self._verify_output = None
        self.schema = None
        self.max_length = 20
        self.batch_size = 4
        self.seq_len = 10
        self.hidden_dim = 256
        tokens = self.batch_size * self.seq_len
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        # Seed FIRST for deterministic verification
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        
        if torch.cuda.is_available():
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
            torch.backends.cuda.enable_cudnn_sdp(False)
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            enable_tf32()

        vocab_size = 1000
        self.model = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=self.hidden_dim, nhead=8, batch_first=True),
            num_layers=1,
        ).to(self.device).eval()

        self.schema = {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "count": {"type": "number"},
            },
            "required": ["summary"],
        }

        self.input_ids = torch.randint(0, vocab_size, (self.batch_size, self.seq_len), device=self.device)
        
        # Create FIXED inputs for deterministic verification
        self.embedded_input = torch.randn(self.batch_size, self.seq_len, self.hidden_dim, device=self.device)
        self.memory = torch.randn(self.batch_size, self.seq_len, self.hidden_dim, device=self.device)
        
        # Compute verification output once
        with torch.no_grad():
            self._verify_output = self.model(self.embedded_input, self.memory).clone()
        self._synchronize()

    def benchmark_fn(self) -> None:
        with self._nvtx_range("baseline_guided_decoding_math"):
            with torch.no_grad():
                # Use fixed inputs for deterministic verification
                self._verify_output = self.model(self.embedded_input, self.memory)
                _ = self._verify_output.sum()
            self._synchronize()

    def teardown(self) -> None:
        self.model = None
        self.input_ids = None
        self.schema = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=20,
            warmup=5,
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_environment_metrics
        return compute_environment_metrics(
            gpu_count=getattr(self, 'gpu_count', 1),
            gpu_memory_gb=getattr(self, 'gpu_memory_gb', 80.0),
        )

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Model not initialized"
        return None

    def get_input_signature(self) -> dict:
        """Return workload signature for input verification."""
        return {
            "batch_size": self.batch_size,
            "seq_len": self.seq_len,
            "hidden_dim": self.hidden_dim,
            "num_layers": 1,
            "nhead": 8,
        }

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        if self._verify_output is None:
            raise RuntimeError("setup() must be called before verification")
        return self._verify_output

    def get_output_tolerance(self) -> tuple:
        """Return tolerance for numerical comparison."""
        return (1e-4, 1e-4)


def get_benchmark() -> BaseBenchmark:
    return BaselineGuidedDecodingMathBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
