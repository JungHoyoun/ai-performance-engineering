"""Helper utilities for conditional NVTX range markers.

This module provides utilities to conditionally add NVTX ranges only when
profiling is enabled, reducing overhead for pure performance benchmarks.
"""

import re
import sys
from contextlib import contextmanager, redirect_stderr
from typing import Any, Generator, Optional, TextIO, cast

import io

import torch

_STANDARD_NVTX_PREFIXES = {
    "setup",
    "warmup",
    "compute_kernel",
    "compute_math",
    "compute_graph",
    "transfer_async",
    "transfer_sync",
    "prefetch",
    "barrier",
    "reduce",
    "verify",
    "cleanup",
    "batch",
    "tile",
    "iteration",
    "step",
}

_TRANSFER_KEYWORDS = (
    "copy",
    "transfer",
    "memcpy",
    "h2d",
    "d2h",
    "host_to_device",
    "device_to_host",
    "zero_copy",
)

_COMPUTE_MATH_KEYWORDS = (
    "matmul",
    "gemm",
    "attention",
    "mlp",
    "moe",
    "softmax",
    "layernorm",
    "norm",
    "ffn",
    "conv",
    "transformer",
    "inference",
    "training",
    "prefill",
    "decode",
    "routing",
)


def _normalize_detail(label: str) -> str:
    sanitized = label.strip().lower()
    sanitized = sanitized.replace(" ", "_").replace("-", "_").replace("/", "_")
    sanitized = re.sub(r"[^a-z0-9:_]+", "_", sanitized)
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    return sanitized


def _strip_variants(label: str) -> str:
    for prefix in ("baseline_", "optimized_"):
        if label.startswith(prefix):
            label = label[len(prefix):]
    for suffix in ("_baseline", "_optimized"):
        if label.endswith(suffix):
            label = label[: -len(suffix)]
    return label.strip("_")


def _contains_any(label: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in label for keyword in keywords)


def _infer_prefix(detail: str) -> str:
    if not detail:
        return "compute_kernel"
    if "warmup" in detail:
        return "warmup"
    if "setup" in detail or "init" in detail:
        return "setup"
    if "verify" in detail or "validation" in detail or "check" in detail:
        return "verify"
    if "cleanup" in detail or "teardown" in detail:
        return "cleanup"
    if "prefetch" in detail:
        return "prefetch"
    if _contains_any(detail, _TRANSFER_KEYWORDS):
        if "async" in detail or "overlap" in detail or "pipelined" in detail or "stream" in detail:
            return "transfer_async"
        return "transfer_sync"
    if "reduce" in detail or "reduction" in detail:
        return "reduce"
    if "barrier" in detail or "sync" in detail:
        return "barrier"
    if "batch" in detail:
        return "batch"
    if "iteration" in detail or "iter" in detail:
        return "iteration"
    if "step" in detail:
        return "step"
    if "graph" in detail:
        return "compute_graph"
    if _contains_any(detail, _COMPUTE_MATH_KEYWORDS):
        return "compute_math"
    return "compute_kernel"


def standardize_nvtx_label(name: str) -> str:
    normalized = _normalize_detail(name)
    if not normalized:
        return "compute_kernel:unnamed"
    if ":" in normalized:
        prefix, detail = normalized.split(":", 1)
        if prefix in _STANDARD_NVTX_PREFIXES:
            detail = _strip_variants(_normalize_detail(detail)) or "unnamed"
            return f"{prefix}:{detail}"
    detail = _strip_variants(normalized) or "unnamed"
    prefix = _infer_prefix(detail)
    return f"{prefix}:{detail}"


def canonicalize_nvtx_name(name: str) -> str:
    return standardize_nvtx_label(name)


class FilteredStderr(io.TextIOBase):
    """Thread-safe filter for stderr that removes NVTX threading errors."""
    def __init__(self, original: TextIO):
        self.original = original
    
    def write(self, text: str) -> int:
        # Filter out NVTX threading error messages
        if "External init callback must run in same thread as registerClient" not in text:
            self.original.write(text)
            return len(text)
        # Otherwise silently drop the error message
        return 0
    
    def flush(self) -> None:
        self.original.flush()
    
    def __getattr__(self, name: str):
        # Forward all other attributes to original stderr
        return getattr(self.original, name)


@contextmanager
def _suppress_nvtx_threading_error() -> Generator[None, None, None]:
    """Suppress NVTX threading errors that occur when CUDA initializes in different thread.
    
    This is a known PyTorch/NVTX issue where NVTX is initialized in one thread
    but used in another. The error is harmless and benchmarks complete successfully.
    We suppress stderr output for this specific error message using thread-safe redirect_stderr.
    """
    # Use contextlib.redirect_stderr for thread-safe stderr redirection
    filtered_stderr = FilteredStderr(sys.stderr)
    with redirect_stderr(cast(TextIO, filtered_stderr)):
        yield


@contextmanager
def nvtx_range(name: str, enable: Optional[bool] = None) -> Generator[None, None, None]:
    """Conditionally add NVTX range marker.
    
    Args:
        name: Name for the NVTX range
        enable: If True, add NVTX range; if False, no-op; if None, auto-detect from config
    
    Example:
        with nvtx_range("my_operation", enable=True):
            # This operation will be marked in NVTX traces
            result = model(input)
    """
    if enable is None:
        # Auto-detect: check if NVTX is enabled via environment or config
        # Default to False for minimal overhead
        enable = False
    
    canonical_name = canonicalize_nvtx_name(name)
    if enable and torch.cuda.is_available():
        with _suppress_nvtx_threading_error():
            # Nsight Compute NVTX filtering requires range_start/range_end ranges.
            range_start = getattr(torch.cuda.nvtx, "range_start", None)
            range_end = getattr(torch.cuda.nvtx, "range_end", None)
            if range_start is None or range_end is None:
                raise RuntimeError("NVTX range_start/range_end are required for profiling.")
            range_id = range_start(canonical_name)
            try:
                yield
            finally:
                range_end(range_id)
        return
    # No-op when NVTX is disabled or CUDA is unavailable
    yield


def get_nvtx_enabled(config: Any) -> bool:
    """Get NVTX enabled status from benchmark config.
    
    Args:
        config: BenchmarkConfig instance
    
    Returns:
        True if NVTX should be enabled, False otherwise
    """
    nvtx_value = getattr(config, "enable_nvtx", None)
    if isinstance(nvtx_value, bool):
        return nvtx_value
    profiling_value = getattr(config, "enable_profiling", None)
    if isinstance(profiling_value, bool):
        return profiling_value
    return False
