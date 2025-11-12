"""Shared helpers for thread block cluster benchmarks."""

from __future__ import annotations

from common.python.hardware_capabilities import detect_capabilities

CLUSTER_SKIP_HINTS = (
    "Thread block clusters unstable",
    "cluster target block not present",
    "cudaDevAttrClusterLaunch",
    "CUDA_EXCEPTION_17",
)


def should_skip_cluster_error(message: str) -> bool:
    """Return True if the runtime error indicates cluster hardware is unavailable."""
    msg_upper = message.upper()
    if "SKIPPED" in msg_upper:
        return True
    return any(hint in message for hint in CLUSTER_SKIP_HINTS)


def _cluster_hint() -> str:
    cap = detect_capabilities()
    if cap is None:
        return "CUDA hardware unavailable."
    if cap.cluster.has_dsmem:
        return (
            "Cluster launch failed despite DSMEM support; inspect driver logs or "
            "rerun under compute-sanitizer for more detail."
        )
    hint = cap.cluster.notes or "Upgrade to a CUDA/driver release that enables DSMEM on this architecture."
    return f"{cap.device_name} disables DSMEM ({hint})"


def raise_cluster_skip(message: str) -> None:
    """Raise a standardized SKIPPED RuntimeError when cluster launch is unsupported."""
    if should_skip_cluster_error(message):
        raise RuntimeError(
            f"SKIPPED: Thread block clusters unavailable on this driver/CUDA build. {_cluster_hint()}"
        ) from None
