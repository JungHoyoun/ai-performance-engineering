#!/usr/bin/env python3
"""baseline_prefill_decode_disagg.py - CPU-staged KV handoff baseline.

This benchmark models a two-pool serving setup on a single node:
- Prefill runs on `cuda:0`
- Decode runs on `cuda:1`

Baseline behavior:
- KV handoff is staged through host memory: GPU0 -> CPU -> GPU1
- Prefill and decode are serialized (no pipelining across pools)

The optimized pair (`optimized_prefill_decode_disagg.py`) keeps the semantic
output invariant while improving performance via peer copies + pipelining.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from ch15.verification_payload_mixin import VerificationPayloadMixin


class BaselinePrefillDecodeDisaggBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline prefill/decode disaggregation with CPU-staged KV transfers."""

    def __init__(
        self,
        *,
        batch_size: int = 8,
        prefill_length: int = 1024,
        decode_length: int = 64,
        hidden_size: int = 2048,
    ) -> None:
        super().__init__()
        self.batch_size = int(batch_size)
        self.prefill_length = int(prefill_length)
        self.decode_length = int(decode_length)
        self.hidden_size = int(hidden_size)

        tokens = self.batch_size * (self.prefill_length + self.decode_length)
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )
        self.register_workload_metadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )

        self.prefill_device: Optional[torch.device] = None
        self.decode_device: Optional[torch.device] = None
        self.prefill_model: Optional[nn.Module] = None
        self.decode_model: Optional[nn.Module] = None
        self.prefill_inputs: Optional[torch.Tensor] = None
        self._verify_probe: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None

    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("SKIPPED: CUDA required for prefill/decode disaggregation")
        if torch.cuda.device_count() < 2:
            raise RuntimeError("SKIPPED: prefill/decode disaggregation requires >=2 GPUs")

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        self.prefill_device = torch.device("cuda:0")
        self.decode_device = torch.device("cuda:1")

        self.prefill_model = nn.Linear(self.hidden_size, self.hidden_size, bias=False).to(
            self.prefill_device, dtype=torch.bfloat16
        ).eval()
        self.decode_model = nn.Linear(self.hidden_size, self.hidden_size, bias=False).to(
            self.decode_device, dtype=torch.bfloat16
        ).eval()

        self.prefill_inputs = torch.randn(
            self.batch_size,
            self.prefill_length,
            self.hidden_size,
            device=self.prefill_device,
            dtype=torch.bfloat16,
        )
        self._verify_probe = self.prefill_inputs[:1, :1, :256].detach()
        torch.cuda.synchronize(self.prefill_device)
        torch.cuda.synchronize(self.decode_device)

    def benchmark_fn(self) -> None:
        if (
            self.prefill_device is None
            or self.decode_device is None
            or self.prefill_model is None
            or self.decode_model is None
            or self.prefill_inputs is None
        ):
            raise RuntimeError("setup() must run before benchmark_fn()")

        outputs: list[torch.Tensor] = []
        with self._nvtx_range("baseline_prefill_decode_disagg"):
            with torch.no_grad():
                for idx in range(self.batch_size):
                    # Prefill on GPU0.
                    prefill_out = self.prefill_model(self.prefill_inputs[idx : idx + 1])

                    # KV handoff via host staging (slow path).
                    kv_cpu = prefill_out.cpu()
                    kv_decode = kv_cpu.to(self.decode_device)

                    # Decode on GPU1 (sequential, no overlap with next prefill).
                    token_state = kv_decode[:, -1:, :]
                    for _ in range(self.decode_length):
                        token_state = self.decode_model(token_state)
                    outputs.append(token_state.squeeze(0).squeeze(0))

        torch.cuda.synchronize(self.prefill_device)
        torch.cuda.synchronize(self.decode_device)
        self.output = torch.stack(outputs, dim=0)

    def capture_verification_payload(self) -> None:
        if (
            self.output is None
            or self._verify_probe is None
            or self.prefill_model is None
            or self.decode_model is None
        ):
            raise RuntimeError("setup() and benchmark_fn() must run before capture_verification_payload()")

        output_slice = self.output[:2, :256].detach().cpu().float().clone()
        param_count = sum(p.numel() for p in self.prefill_model.parameters()) + sum(
            p.numel() for p in self.decode_model.parameters()
        )
        self._set_verification_payload(
            inputs={"probe": self._verify_probe.detach().cpu()},
            output=output_slice,
            batch_size=int(self.batch_size),
            parameter_count=int(param_count),
            precision_flags={
                "fp16": False,
                "bf16": True,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(1e-3, 1e-3),
        )

    def teardown(self) -> None:
        self.prefill_model = None
        self.decode_model = None
        self.prefill_inputs = None
        self._verify_probe = None
        self.output = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=5, warmup=5, multi_gpu_required=True)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload


def get_benchmark() -> BaseBenchmark:
    return BaselinePrefillDecodeDisaggBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)

