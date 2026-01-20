"""Baseline tensor-core stream workload without overlap."""

from __future__ import annotations

from typing import List, Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from core.profiling.nvtx_helper import (
    canonicalize_nvtx_name,
    get_nvtx_enabled,
    nvtx_range,
)
from ch11.stream_overlap_base import resolve_device


class BaselineTensorCoresStreamsBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline tensor-core workload: sequential GEMM operations on single stream.
    
    Uses FP16/BF16 GEMM operations to demonstrate tensor core utilization,
    but processes chunks sequentially without stream overlap.
    """

    declare_all_streams = False

    def __init__(self) -> None:
        super().__init__()
        self.device = resolve_device()
        self.label = "baseline_tensor_cores_streams"
        # Use matrix dimensions that work well with tensor cores
        # Each chunk will be reshaped into a matrix for GEMM
        self.num_elements = 24_000_000  # Total elements
        self.num_segments = 16
        # Matrix dimensions: reshape each chunk to MxK and KxN for GEMM
        # Each chunk ~1.5M elements -> reshape to ~1224x1224 matrices
        self.matrix_dim = 1224  # Good size for tensor cores
        self.stream = None
        self.host_input = None
        self.host_output = None
        self.host_in_chunks = None
        self.host_out_chunks = None
        self.device_A_chunks = None
        self.device_B_chunks = None
        self.device_C_chunks = None
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        # Workload metadata: GEMM operations
        element_size = float(torch.empty((), dtype=self.dtype).element_size())
        bytes_transferred = float(self.num_elements * element_size * 3)  # A, B, C matrices
        self.register_workload_metadata(bytes_per_iteration=bytes_transferred)

    def setup(self) -> None:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        
        self.stream = torch.cuda.Stream()
        
        # Create input data on host (FP32 for verification)
        self.host_input = torch.randn(
            self.num_elements, device="cpu", dtype=torch.float32, pin_memory=True
        )
        self.host_output = torch.empty_like(self.host_input, pin_memory=True)
        
        # Split into chunks
        self.host_in_chunks = list(torch.chunk(self.host_input, self.num_segments))
        self.host_out_chunks = list(torch.chunk(self.host_output, self.num_segments))
        
        # Prepare device buffers for GEMM: reshape each chunk to matrices
        # Strategy: Use chunk as one dimension, create square matrix for GEMM
        # C = A @ B where A is (chunk_size, K) and B is (K, chunk_size) -> C is (chunk_size, chunk_size)
        # Then extract diagonal or first row to match original chunk size
        self.device_A_chunks = []
        self.device_B_chunks = []
        self.device_C_chunks = []
        self.chunk_sizes = []  # Store original chunk sizes for output reshaping
        
        for chunk in self.host_in_chunks:
            chunk_size = chunk.numel()
            self.chunk_sizes.append(chunk_size)
            
            if chunk_size == 0:
                self.device_A_chunks.append(torch.empty(0, 0, dtype=self.dtype, device=self.device))
                self.device_B_chunks.append(torch.empty(0, 0, dtype=self.dtype, device=self.device))
                self.device_C_chunks.append(torch.empty(0, 0, dtype=self.dtype, device=self.device))
                continue
            
            # Create matrices: A is (M, K), B is (K, N) where M=chunk_size, N=chunk_size
            # Use K = min(256, chunk_size) for reasonable GEMM size
            K = min(256, chunk_size)
            M = chunk_size
            N = chunk_size
            
            # Pad chunk if needed to create A matrix
            if chunk_size < K:
                chunk_padded = torch.cat([chunk, torch.zeros(K - chunk_size, dtype=chunk.dtype)])
            else:
                chunk_padded = chunk
            
            # Create A: (M, K) - repeat chunk data to fill matrix
            A_flat = chunk_padded[:K].repeat((M // K) + 1)[:M * K]
            A = A_flat.reshape(M, K).to(self.dtype)
            
            # Create B: (K, N) - use chunk data transposed pattern
            B_flat = chunk_padded[:K].repeat((N // K) + 1)[:K * N]
            B = B_flat.reshape(K, N).to(self.dtype)
            
            self.device_A_chunks.append(A.to(self.device))
            self.device_B_chunks.append(B.to(self.device))
            # C will be (M, N) = (chunk_size, chunk_size)
            self.device_C_chunks.append(torch.empty(M, N, dtype=self.dtype, device=self.device))
        
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        config = getattr(self, "_config", None) or self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range(self.label, enable=enable_nvtx):
            assert self.device_A_chunks is not None
            assert self.device_B_chunks is not None
            assert self.device_C_chunks is not None
            
            for A, B, C in zip(self.device_A_chunks, self.device_B_chunks, self.device_C_chunks):
                with torch.cuda.stream(self.stream):
                    # Tensor core GEMM operation
                    torch.matmul(A, B, out=C)
                # Synchronize after each chunk - prevents overlap
                self.stream.synchronize()
        
        # Copy results back to host output chunks after all GEMM operations
        torch.cuda.synchronize()
        for idx, (C, h_out, orig_size) in enumerate(zip(self.device_C_chunks, self.host_out_chunks, self.chunk_sizes)):
            if C.numel() > 0 and h_out.numel() > 0 and orig_size > 0:
                # C is (chunk_size, chunk_size), extract first row to match original chunk size
                if C.shape[1] >= orig_size:
                    C_row = C[0, :orig_size].float()
                else:
                    # If C is smaller, pad with zeros
                    C_row = torch.cat([C[0, :].float(), torch.zeros(orig_size - C.shape[1], dtype=torch.float32)])
                h_out.copy_(C_row, non_blocking=False)
        
        if (
            self.host_input is None
            or self.host_output is None
            or self.host_in_chunks is None
            or self.host_out_chunks is None
        ):
            raise RuntimeError("benchmark_fn() must run after setup() initializes buffers")

    def capture_verification_payload(self) -> None:
        self._set_verification_payload(
            inputs={"host_input": self.host_input},
            output=self.host_output.detach().clone(),
            batch_size=self.host_input.numel(),
            parameter_count=0,
            precision_flags={
                "fp16": self.dtype == torch.float16,
                "bf16": self.dtype == torch.bfloat16,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(1e-2, 1e-2),  # Looser tolerance for FP16/BF16
        )

    def teardown(self) -> None:
        self.stream = None
        self.host_input = None
        self.host_output = None
        self.host_in_chunks = None
        self.host_out_chunks = None
        self.device_A_chunks = None
        self.device_B_chunks = None
        self.device_C_chunks = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        nvtx_tag = canonicalize_nvtx_name(self.label)
        return BenchmarkConfig(
            iterations=20,
            warmup=5,
            ncu_replay_mode="application",
            ncu_metric_set="minimal",
            nsys_nvtx_include=[nvtx_tag],
        )

    def validate_result(self) -> str | None:
        if self.host_output is None or self.host_input is None:
            return "Buffers not initialized"
        if self.host_output.shape != self.host_input.shape:
            return "Shape mismatch"
        if not torch.isfinite(self.host_output).all():
            return "Output contains non-finite values"
        return None

    def get_custom_metrics(self) -> Optional[dict]:
        """Return stream overlap metrics for the baseline (sequential) path."""
        element_size = float(torch.empty((), dtype=self.dtype).element_size())
        bytes_transferred = float(self.num_elements * element_size * 3)
        return {
            f"{self.label}.elements": float(self.num_elements),
            f"{self.label}.num_segments": float(self.num_segments),
            f"{self.label}.bytes_transferred": bytes_transferred,
            f"{self.label}.num_streams": 1.0,
            f"{self.label}.expected_overlap_pct": 0.0,
            f"{self.label}.dtype": str(self.dtype),
        }

    def get_custom_streams(self) -> List[torch.cuda.Stream]:
        if self.stream is None:
            return []
        return [self.stream]

    def get_input_signature(self) -> dict:
        """Return workload signature for input verification."""
        return super().get_input_signature()

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        return super().get_verify_output()

    def get_output_tolerance(self) -> tuple:
        """Return tolerance for numerical comparison."""
        return super().get_output_tolerance()


def get_benchmark() -> BaselineTensorCoresStreamsBenchmark:
    return BaselineTensorCoresStreamsBenchmark()
