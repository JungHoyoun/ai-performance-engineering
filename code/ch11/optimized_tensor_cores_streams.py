"""Optimized tensor-core stream workload with overlap."""

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


class OptimizedTensorCoresStreamsBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Optimized tensor-core workload: concurrent GEMM operations across multiple streams.
    
    Uses FP16/BF16 GEMM operations to demonstrate tensor core utilization,
    with stream overlap to process multiple chunks concurrently.
    """

    declare_all_streams = False

    def __init__(self) -> None:
        super().__init__()
        self.device = resolve_device()
        self.label = "tensor_cores_streams"
        self.num_elements = 24_000_000  # Match baseline
        self.num_segments = 16
        self.num_streams = 4  # Use 4 streams for good overlap
        self.matrix_dim = 1224
        self.streams: List[torch.cuda.Stream] | None = None
        self.host_input = None
        self.host_output = None
        self.host_in_chunks = None
        self.host_out_chunks = None
        self.device_A_chunks = None
        self.device_B_chunks = None
        self.device_C_chunks = None
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        element_size = float(torch.empty((), dtype=self.dtype).element_size())
        bytes_transferred = float(self.num_elements * element_size * 3)
        self.register_workload_metadata(bytes_per_iteration=bytes_transferred)

    def setup(self) -> None:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        
        if self.num_streams < 1:
            raise ValueError("num_streams must be >= 1")
        
        self.streams = [torch.cuda.Stream() for _ in range(self.num_streams)]
        
        # Create input data on host (FP32 for verification)
        self.host_input = torch.randn(
            self.num_elements, device="cpu", dtype=torch.float32, pin_memory=True
        )
        self.host_output = torch.empty_like(self.host_input, pin_memory=True)
        
        # Split into chunks
        chunks = torch.chunk(self.host_input, self.num_segments)
        if len(chunks) < self.num_segments:
            chunks = list(chunks)
            for _ in range(self.num_segments - len(chunks)):
                empty = torch.empty(0, dtype=torch.float32, device="cpu", pin_memory=True)
                chunks.append(empty)
        self.host_in_chunks = list(chunks)
        self.host_out_chunks = list(torch.chunk(self.host_output, len(self.host_in_chunks)))
        
        # Prepare device buffers for GEMM: reshape each chunk to matrices
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
            assert self.streams is not None
            assert self.device_A_chunks is not None
            assert self.device_B_chunks is not None
            assert self.device_C_chunks is not None
            
            with torch.no_grad():
                # Launch GEMM operations across multiple streams for overlap
                for idx, (A, B, C) in enumerate(
                    zip(self.device_A_chunks, self.device_B_chunks, self.device_C_chunks)
                ):
                    if A.numel() == 0:
                        continue
                    
                    stream = self.streams[idx % self.num_streams]
                    with torch.cuda.stream(stream):
                        # Tensor core GEMM operation
                        torch.matmul(A, B, out=C)
                
                # Join all work back onto the current stream (Locus/KernelBench 2025)
                current = torch.cuda.current_stream(self.device)
                for stream in self.streams:
                    current.wait_stream(stream)
        
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
        self.streams = None
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
            iterations=16,
            warmup=5,
            ncu_replay_mode="application",
            ncu_metric_set="minimal",
            nsys_nvtx_include=[nvtx_tag],
        )

    def validate_result(self) -> str | None:
        if not self.host_out_chunks:
            return "Chunks not initialized"
        for out in self.host_out_chunks:
            if out.numel() > 0 and not torch.isfinite(out).all():
                return "Output contains non-finite values"
        return None

    def get_custom_metrics(self) -> Optional[dict]:
        """Return stream overlap metrics for the optimized (concurrent) path."""
        element_size = float(torch.empty((), dtype=self.dtype).element_size())
        bytes_transferred = float(self.num_elements * element_size * 3)
        return {
            f"{self.label}.elements": float(self.num_elements),
            f"{self.label}.num_streams": float(self.num_streams),
            f"{self.label}.bytes_transferred": bytes_transferred,
            f"{self.label}.expected_overlap_pct": min(100.0, (self.num_streams - 1) / self.num_streams * 100),
            f"{self.label}.dtype": str(self.dtype),
        }

    def get_custom_streams(self) -> List[torch.cuda.Stream]:
        if self.streams is None:
            return []
        return list(self.streams)

    def get_input_signature(self) -> dict:
        """Return workload signature for input verification."""
        return super().get_input_signature()

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        return super().get_verify_output()

    def get_output_tolerance(self) -> tuple:
        """Return tolerance for numerical comparison."""
        return super().get_output_tolerance()


def get_benchmark() -> OptimizedTensorCoresStreamsBenchmark:
    return OptimizedTensorCoresStreamsBenchmark()
