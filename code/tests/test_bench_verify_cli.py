"""End-to-end tests for `aisp bench verify` CLI.

These tests exercise the real Typer CLI entrypoint (no mocking) using a
temporary benchmark root with minimal baseline/optimized pairs.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_bench_verify_cli_skip_workload(tmp_path: Path) -> None:
    bench_root = tmp_path / "bench_root"
    chapter_dir = bench_root / "ch99"
    chapter_dir.mkdir(parents=True)

    baseline_path = chapter_dir / "baseline_example.py"
    optimized_path = chapter_dir / "optimized_example.py"

    _write(
        baseline_path,
        """
from __future__ import annotations

from typing import Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, WorkloadMetadata


class BaselineExample(VerificationPayloadMixin, BaseBenchmark):
    allow_cpu = True

    def __init__(self) -> None:
        super().__init__()
        self.x: Optional[torch.Tensor] = None
        self.y: Optional[torch.Tensor] = None
        self._workload = WorkloadMetadata(requests_per_iteration=1.0, tokens_per_iteration=1.0)

    def setup(self) -> None:
        torch.manual_seed(42)
        self.x = torch.arange(16, device=self.device, dtype=torch.float32)

    def benchmark_fn(self) -> None:
        if self.x is None:
            raise RuntimeError("setup() must run first")
        self.y = self.x * 2.0

    def capture_verification_payload(self) -> None:
        if self.x is None or self.y is None:
            raise RuntimeError("benchmark_fn() must run first")
        self._set_verification_payload(inputs={"x": self.x}, output=self.y, batch_size=self.x.shape[0], parameter_count=0)

    def validate_result(self) -> Optional[str]:
        return None

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def teardown(self) -> None:
        self.x = None
        self.y = None


def get_benchmark() -> BaseBenchmark:
    return BaselineExample()
""".lstrip(),
    )

    _write(
        optimized_path,
        """
from __future__ import annotations

from typing import Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, WorkloadMetadata


class OptimizedExample(VerificationPayloadMixin, BaseBenchmark):
    allow_cpu = True

    def __init__(self) -> None:
        super().__init__()
        self.x: Optional[torch.Tensor] = None
        self.y: Optional[torch.Tensor] = None
        # Intentional mismatch: the CLI test verifies --skip-workload behaves correctly.
        self._workload = WorkloadMetadata(requests_per_iteration=1.0, tokens_per_iteration=2.0)

    def setup(self) -> None:
        torch.manual_seed(42)
        self.x = torch.arange(16, device=self.device, dtype=torch.float32)

    def benchmark_fn(self) -> None:
        if self.x is None:
            raise RuntimeError("setup() must run first")
        self.y = self.x * 2.0

    def capture_verification_payload(self) -> None:
        if self.x is None or self.y is None:
            raise RuntimeError("benchmark_fn() must run first")
        self._set_verification_payload(inputs={"x": self.x}, output=self.y, batch_size=self.x.shape[0], parameter_count=0)

    def validate_result(self) -> Optional[str]:
        return None

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def teardown(self) -> None:
        self.x = None
        self.y = None


def get_benchmark() -> BaseBenchmark:
    return OptimizedExample()
""".lstrip(),
    )

    repo_code_root = Path(__file__).resolve().parents[1]

    base_cmd = [
        sys.executable,
        "-m",
        "cli.aisp",
        "bench",
        "verify",
        "--bench-root",
        str(bench_root),
        "-t",
        "ch99:example",
        "--skip-jitter",
        "--skip-fresh-input",
    ]

    # Without --skip-workload, this must fail (workload mismatch).
    proc = subprocess.run(base_cmd, cwd=repo_code_root, capture_output=True, text=True)
    assert proc.returncode != 0, proc.stdout + proc.stderr

    # With --skip-workload, it should pass.
    proc = subprocess.run(base_cmd + ["--skip-workload"], cwd=repo_code_root, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr

