"""Chapter 1 hooks for the Inference Empire game (GPT-OSS 20B)."""

from __future__ import annotations

from typing import Any, Dict
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from bootcamp.hooks.common import run_inference, run_inference_for_profiler
from bootcamp.hooks.profiler import NsightComputeNotFound, profile_tcgen05

PROMPTS = [
    "Profile the naive GPT-OSS inference stack and identify the biggest latency offender.",
    "Explain why tcgen05 utilization matters when scaling throughput on GB10.",
]
MAX_NEW_TOKENS = 32


def profile(max_new_tokens: int = MAX_NEW_TOKENS) -> Dict[str, Any]:
    """Run a timed inference and augment metrics with tcgen05 profiler data."""
    run_result = run_inference(PROMPTS, max_new_tokens=max_new_tokens)
    metrics = run_result.setdefault("metrics", {})
    try:
        metrics["tcgen05"] = profile_tcgen05("ch1")
    except NsightComputeNotFound as exc:
        metrics["tcgen05_error"] = str(exc)
    return run_result


def profile_tcgen05_only() -> Dict[str, float]:
    """Collect tcgen05 metrics only (no additional summaries)."""
    return profile_tcgen05("ch1")


def run_for_profiler() -> None:
    """Entry point used when invoked under nv-nsight-cu-cli."""
    run_inference_for_profiler(PROMPTS, max_new_tokens=MAX_NEW_TOKENS)


__all__ = ["profile", "profile_tcgen05_only", "run_for_profiler"]
