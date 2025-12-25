"""Baseline structured sparsity FFN lab (dense GEMMs with pruned weights)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from core.optimization.structured_sparsity_ffn import StructuredSparsityFFN, StructuredSparsityFFNConfig


class BaselineStructuredSparsityFFNLab(VerificationPayloadMixin, BaseBenchmark):
    """Dense FFN baseline for the structured sparsity lab."""

    def __init__(self) -> None:
        super().__init__()
        self.cfg = StructuredSparsityFFNConfig()
        self.ffn: Optional[StructuredSparsityFFN] = None
        self.output: Optional[torch.Tensor] = None
        tokens = float(self.cfg.tokens_per_iteration)
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.cfg.batch_size),
            tokens_per_iteration=tokens,
        )
        self.register_workload_metadata(
            requests_per_iteration=float(self.cfg.batch_size),
            tokens_per_iteration=tokens,
        )

    def setup(self) -> None:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        self.ffn = StructuredSparsityFFN(self.cfg, self.device)
        self.ffn.setup_dense()
        self._synchronize()

    def benchmark_fn(self) -> None:
        if self.ffn is None:
            raise RuntimeError("Benchmark not initialized")
        with self._nvtx_range("baseline_structured_sparsity_ffn_lab"):
            with torch.inference_mode():
                self.output = self.ffn.dense_forward()
        self._synchronize()
        if self.output is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")

    def capture_verification_payload(self) -> None:
        if self.ffn is None or self.output is None or self.ffn.input is None:
            raise RuntimeError("setup() and benchmark_fn() must run before capture_verification_payload()")
        verify_output = self.output[:128, :128]
        parameter_count = 0
        if self.ffn.w1 is not None:
            parameter_count += self.ffn.w1.numel()
        if self.ffn.w3 is not None:
            parameter_count += self.ffn.w3.numel()
        if self.ffn.w2 is not None:
            parameter_count += self.ffn.w2.numel()
        self._set_verification_payload(
            inputs={"input": self.ffn.input},
            output=verify_output.detach().clone(),
            batch_size=int(self.cfg.batch_size),
            parameter_count=parameter_count,
            precision_flags={
                "fp16": True,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(1e-2, 1e-2),
            signature_overrides={
                "ffn_hidden": self.cfg.hidden_size,
                "ffn_size": self.cfg.ffn_size,
                "seq_len": self.cfg.seq_len,
                "sparsity_ratio": 0.5,
                "ffn_kind": "swiglu",
            },
        )

    def teardown(self) -> None:
        if self.ffn is not None:
            self.ffn.teardown()
        self.ffn = None
        self.output = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=8,
            warmup=5,
            nsys_nvtx_include=["structured_sparsity"],
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def validate_result(self) -> Optional[str]:
        if self.output is None:
            return "benchmark_fn() did not produce output"
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselineStructuredSparsityFFNLab()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
