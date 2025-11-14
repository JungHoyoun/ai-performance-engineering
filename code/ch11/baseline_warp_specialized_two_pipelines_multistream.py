"""Baseline dual-pipeline warp specialization benchmark (Chapter 11)."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from common.python.benchmark_harness import Benchmark, BenchmarkConfig  # noqa: E402


def _resolve_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for Chapter 11 benchmarks")
    return torch.device("cuda")


@lru_cache(maxsize=1)
def _load_baseline_extension():
    sources = [
        Path(__file__).with_name("baseline_warp_specialized_two_pipelines_extension.cu"),
    ]
    return load(
        name="baseline_warp_specialized_two_pipelines_ext",
        sources=[str(src) for src in sources],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )


class BaselineDualPipelineBenchmark(Benchmark):
    """Calls the book-era dual-pipeline kernel launched across CUDA streams."""

    def __init__(self) -> None:
        self.device = _resolve_device()
        self.num_streams = 1
        self.tiles = 128
        self.ext = _load_baseline_extension()
        self.input_a: torch.Tensor | None = None
        self.input_b: torch.Tensor | None = None
        self.output: torch.Tensor | None = None

        # Match constants from baseline_warp_specialized_two_pipelines_common.cuh
        self.tile_elems = 1024

    def setup(self) -> None:
        torch.manual_seed(42)
        total_elems = self.tiles * self.tile_elems
        self.input_a = torch.randn(total_elems, device=self.device, dtype=torch.float32)
        self.input_b = torch.randn(total_elems, device=self.device, dtype=torch.float32)
        self.output = torch.empty_like(self.input_a)
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        assert self.input_a is not None and self.input_b is not None
        result = self.ext.baseline_warp_specialized_multistream_forward(
            self.input_a,
            self.input_b,
            self.num_streams,
        )
        torch.cuda.synchronize()
        if self.output is not None:
            self.output.copy_(result)

    def teardown(self) -> None:
        self.input_a = None
        self.input_b = None
        self.output = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=30,
            warmup=5,
            measurement_timeout_seconds=120,
            setup_timeout_seconds=120,
        )

    def validate_result(self) -> str | None:
        if self.output is None:
            return "Output tensor not initialized"
        if not torch.isfinite(self.output).all():
            return "Output contains non-finite values"
        return None


def get_benchmark() -> Benchmark:
    return BaselineDualPipelineBenchmark()

