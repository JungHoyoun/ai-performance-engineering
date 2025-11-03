from __future__ import annotations

import csv
import io
import os
import shutil
import subprocess
import sys
from typing import Dict, Iterable, List, Optional


DEFAULT_METRICS: List[str] = [
    "sm__inst_executed_pipe_tensor_op.sum",
    "smsp__pipe_tensor_active.avg.pct_of_peak_sustained_elapsed",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
]


class NsightComputeNotFound(RuntimeError):
    """Raised when nv-nsight-cu-cli is missing from PATH."""


def _format_metrics(metrics: Iterable[str]) -> str:
    return ",".join(metrics)


def _parse_csv_metrics(output: str, wanted: Iterable[str]) -> Dict[str, float]:
    wanted_set = set(wanted)
    parsed: Dict[str, float | None] = {}
    reader = csv.reader(io.StringIO(output))
    for row in reader:
        if not row:
            continue
        name = row[0].strip()
        if name in ("Metric Name", "ID") or name.startswith("#"):
            continue
        if name in wanted_set and len(row) >= 3:
            value = row[2].strip()
            try:
                parsed[name] = float(value)
            except ValueError:
                parsed[name] = None
    return parsed  # type: ignore[return-value]


def profile_tcgen05(level: str, *, metrics: Optional[Iterable[str]] = None, env: Optional[Dict[str, str]] = None) -> Dict[str, float]:
    """Profile a level-specific run with Nsight Compute and return key tcgen05 metrics."""
    executable = shutil.which("nv-nsight-cu-cli") or shutil.which("ncu")
    if executable is None:
        raise NsightComputeNotFound(
            "nv-nsight-cu-cli/ncu not found in PATH. Install Nsight Compute or add it to PATH to collect tcgen05 metrics."
        )

    metric_list = list(metrics or DEFAULT_METRICS)
    cmd = [executable]
    cmd.extend(
        [
            "--metrics",
            _format_metrics(metric_list),
            "--csv",
            "--target-processes",
            "all",
        ]
    )
    cmd.extend(
        [
            sys.executable,
            "-m",
            "bootcamp.hooks.runner",
            "--level",
            level,
        ]
    )

    run_env = os.environ.copy()
    if env:
        run_env.update(env)

    completed = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
        env=run_env,
    )

    metrics_map = _parse_csv_metrics(completed.stdout, metric_list)
    missing = [m for m in metric_list if m not in metrics_map]
    if missing:
        raise RuntimeError(
            f"Failed to parse Nsight Compute metrics {missing}. Raw output:\\n{completed.stdout}\\nSTDERR:\\n{completed.stderr}"
        )
    return metrics_map


__all__ = ["DEFAULT_METRICS", "NsightComputeNotFound", "profile_tcgen05"]
