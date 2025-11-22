"""FP8 + full prefill+decode graph capture (TE FP8)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from labs.fast_nanochat.nanochat_common import NanoChatBenchmark, NanoChatConfig  # noqa: E402
import os


def get_benchmark() -> NanoChatBenchmark:
    # Allow large allocations without fragmentation for big fp8 shapes
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:256")
    cfg = NanoChatConfig(
        batch_size=32,
        prompt_tokens=4096,
        decode_tokens=1024,
        hidden_size=6144,
        use_fp8=True,
        use_pinned_host=True,
        use_copy_stream=True,
        use_compute_stream=True,
        use_torch_compile=False,
        use_cuda_graphs=False,
        graph_full_iteration=False,
        label="optimized_fast_nanochat_fp8",
    )
    return NanoChatBenchmark(cfg)


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode  # noqa: E402

    bench = get_benchmark()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=bench.get_config())
    result = harness.benchmark(bench)
    mean = result.timing.mean_ms if result.timing else 0.0
    print(f"\noptimized_fast_nanochat_fp8: {mean:.3f} ms/iter")
