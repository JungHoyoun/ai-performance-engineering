"""optimized_warp_divergence.py - Optimized warp-divergence benchmark for Chapter 10."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Callable, Tuple

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

from common.python.compile_utils import compile_callable, enable_tf32
from common.python.inductor_guard import (
    InductorCudagraphState,
    disable_inductor_cudagraph_features,
    restore_inductor_cudagraph_features,
)
from common.python.benchmark_harness import Benchmark, BenchmarkConfig
from ch10.workload_config import WORKLOAD

_MARK_CUDAGRAPH_STEP = getattr(torch.compiler, "cudagraph_mark_step_begin", None)


def _mark_cudagraph_step() -> None:
    if callable(_MARK_CUDAGRAPH_STEP):
        _MARK_CUDAGRAPH_STEP()

def resolve_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch10")
    return torch.device("cuda")


def _branchless_kernel(
    data: torch.Tensor,
    logits: torch.Tensor,
    rounds: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    for iteration in range(rounds):
        activations = torch.sigmoid(logits)
        mask = torch.gt(activations, 0.5).to(data.dtype)
        inv_mask = 1.0 - mask

        positive = torch.tanh(data * 1.13 + 0.20)
        positive = positive * 1.002 + 0.0006 * positive * positive

        negative = torch.sin(data * 0.79 - 0.33)
        negative = negative * 0.998 - 0.00035 * negative * negative

        data = mask * positive + inv_mask * negative
        logits = 0.9 * logits + 0.1 * torch.roll(data, shifts=iteration + 1, dims=0)
    return data, logits


class OptimizedWarpDivergenceBenchmark(Benchmark):
    """Optimized warp divergence benchmark using branchless kernels and multi-stream overlap."""

    def __init__(self) -> None:
        self.device = resolve_device()
        self.workload = WORKLOAD
        self.N = self.workload.warp_elements
        self.branch_iterations = self.workload.warp_branch_iterations
        self.input: Optional[torch.Tensor] = None
        self.routing_logits: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self.streams: list[torch.cuda.Stream] = []
        self._compiled_step: Optional[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None
        self._branchless_fn: Optional[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None
        self._checksum = 0.0
        self._inductor_state: InductorCudagraphState = None

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        enable_tf32()

        self.input = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self.routing_logits = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self.output = torch.empty_like(self.input)

        props = torch.cuda.get_device_properties(self.device.index or 0)
        stream_count = min(4, max(1, props.multi_processor_count // 8))
        self.streams = [torch.cuda.Stream(priority=-1) for _ in range(stream_count)]

        def branchless_fn(chunk: torch.Tensor, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            return _branchless_kernel(chunk, logits, self.branch_iterations)

        self._branchless_fn = branchless_fn
        if self._inductor_state is None:
            self._inductor_state = disable_inductor_cudagraph_features()
        self._compiled_step = compile_callable(
            branchless_fn,
            fullgraph=True,
            mode="reduce-overhead",
        )
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("optimized_warp_divergence", enable=enable_nvtx):
            assert self.input is not None and self.routing_logits is not None
            step_fn = self._compiled_step
            assert step_fn is not None

            chunks = torch.chunk(self.input, len(self.streams))
            logits = torch.chunk(self.routing_logits, len(self.streams))
            updated_chunks: list[torch.Tensor] = [torch.empty(0, device=self.device)] * len(self.streams)
            updated_logits: list[torch.Tensor] = [torch.empty(0, device=self.device)] * len(self.streams)

            for idx, (stream, chunk, logits_chunk) in enumerate(zip(self.streams, chunks, logits)):
                with torch.cuda.stream(stream):
                    chunk_contig = chunk.contiguous()
                    logits_contig = logits_chunk.contiguous()
                    _mark_cudagraph_step()
                    out_chunk, out_logits = step_fn(chunk_contig, logits_contig)
                    updated_chunks[idx] = out_chunk.clone()
                    updated_logits[idx] = out_logits.clone()

            torch.cuda.synchronize()
            self.output = torch.cat(updated_chunks, dim=0)
            self.routing_logits = torch.cat(updated_logits, dim=0)
            self._checksum = float(self.output.sum().item())

    def teardown(self) -> None:
        self.input = None
        self.output = None
        self.routing_logits = None
        self.streams = []
        restore_inductor_cudagraph_features(self._inductor_state)
        self._inductor_state = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=50,
            warmup=5,
            measurement_timeout_seconds=120,
        )

    def validate_result(self) -> Optional[str]:
        if self.output is None:
            return "Output tensor not initialized"
        return None


def get_benchmark() -> Benchmark:
    return OptimizedWarpDivergenceBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nOptimized Warp Divergence: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
