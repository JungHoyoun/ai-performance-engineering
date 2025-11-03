"""Chapter 7 hooks for GPT-OSS 20B memory optimization challenges."""

from __future__ import annotations

from typing import Any, Dict

from bootcamp.hooks.common import run_inference, run_inference_for_profiler
from bootcamp.hooks.profiler import NsightComputeNotFound, profile_tcgen05

PROMPTS = [
    "Our KV cache compaction kernel still bottlenecks tcgen05 throughput. Propose a fix.",
    "Explain how vectorized loads improve memory bandwidth for GPT-OSS decoding.",
]
MAX_NEW_TOKENS = 48


def profile(max_new_tokens: int = MAX_NEW_TOKENS) -> Dict[str, Any]:
    result = run_inference(PROMPTS, max_new_tokens=max_new_tokens)
    metrics = result.setdefault("metrics", {})
    try:
        metrics["tcgen05"] = profile_tcgen05("ch7")
    except NsightComputeNotFound as exc:
        metrics["tcgen05_error"] = str(exc)
    return result


def profile_tcgen05_only() -> Dict[str, float]:
    return profile_tcgen05("ch7")


def run_for_profiler() -> None:
    run_inference_for_profiler(PROMPTS, max_new_tokens=MAX_NEW_TOKENS)


__all__ = ["profile", "profile_tcgen05_only", "run_for_profiler"]
