"""Chapter 16 hooks for GPT-OSS 20B serving and MoE scenarios."""

from __future__ import annotations

from typing import Any, Dict

from bootcamp.hooks.common import run_inference, run_inference_for_profiler
from bootcamp.hooks.profiler import NsightComputeNotFound, profile_tcgen05

PROMPTS = [
    "Latency spiked after enabling expert routing. Diagnose the tcgen05 utilization drop.",
    "Suggest a batching policy that keeps GPT-OSS 20B profitable at 500 req/s.",
]
MAX_NEW_TOKENS = 64


def profile(max_new_tokens: int = MAX_NEW_TOKENS) -> Dict[str, Any]:
    result = run_inference(PROMPTS, max_new_tokens=max_new_tokens)
    metrics = result.setdefault("metrics", {})
    try:
        metrics["tcgen05"] = profile_tcgen05("ch16")
    except NsightComputeNotFound as exc:
        metrics["tcgen05_error"] = str(exc)
    return result


def profile_tcgen05_only() -> Dict[str, float]:
    return profile_tcgen05("ch16")


def run_for_profiler() -> None:
    run_inference_for_profiler(PROMPTS, max_new_tokens=MAX_NEW_TOKENS)


__all__ = ["profile", "profile_tcgen05_only", "run_for_profiler"]
