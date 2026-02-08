"""Python harness wrapper for baseline_cute_dsl_nvfp4_gemm.cu."""

from __future__ import annotations
from typing import Optional

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.harness.benchmark_harness import BaseBenchmark, ExecutionMode
from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark


class BaselineCuteDslNvfp4GemmBenchmark(CudaBinaryBenchmark):
    """Wraps the baseline CUDA binary."""

    # NCU kernel replay is extremely expensive for microsecond-scale kernels if the
    # benchmark does large internal loops. Clamp the CUDA binary's internal loop
    # count during NCU capture so profiling finishes quickly and reliably.
    ncu_env_overrides = {
        "AISP_NCU_PROFILE_ITERS": "1",
    }

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="baseline_cute_dsl_nvfp4_gemm",
            friendly_name="Baseline Cute DSL Nvfp4 Gemm",
            # CUDA binaries pay heavy per-process init costs; keep harness-level repeats low.
            # The CUDA code runs its own internal iteration loop (kIterations).
            iterations=1,
            warmup=1,
            timeout_seconds=180,
            workload_params={
                "M0": 128,
                "N0": 7168,
                "K0": 16384,
                "M1": 128,
                "N1": 4096,
                "K1": 7168,
                "M2": 128,
                "N2": 7168,
                "K2": 2048,
                "kIterations": 50,
                "dtype": "nvfp4",
            },
        )

    def get_custom_metrics(self) -> Optional[dict]:
        return None

    def get_config(self):
        config = super().get_config()
        # Subprocess isolation intermittently hits BrokenPipe under profiling;
        # use thread mode for stable benchmark runs.
        config.use_subprocess = False
        config.execution_mode = ExecutionMode.THREAD
        # Keep NCU capture overhead low and stable for this binary benchmark.
        config.ncu_metric_set = "minimal"
        config.ncu_replay_mode = "kernel"
        config.ncu_replay_mode_override = True
        config._sync_execution_mode()
        return config


def get_benchmark() -> BaseBenchmark:
    return BaselineCuteDslNvfp4GemmBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
