"""Hook helpers for chapter integrations."""

from .common import InferenceMetrics, run_inference, run_inference_for_profiler
from .profiler import DEFAULT_METRICS, NsightComputeNotFound, profile_tcgen05

__all__ = [
    "InferenceMetrics",
    "DEFAULT_METRICS",
    "NsightComputeNotFound",
    "profile_tcgen05",
    "run_inference",
    "run_inference_for_profiler",
]
