"""Baseline nvCOMP input pipeline (CPU zstd decode)."""

from __future__ import annotations

from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from labs.async_input_pipeline.nvcomp_pipeline_common import NvcompInputPipelineBenchmark, NvcompPipelineConfig


def get_benchmark() -> NvcompInputPipelineBenchmark:
    cfg = NvcompPipelineConfig()
    return NvcompInputPipelineBenchmark(cfg, label="baseline_nvcomp_input_pipeline", use_gpu_decode=False)


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
