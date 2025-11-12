"""Lightweight GPU telemetry helpers (temperature, power, utilization).

These helpers rely on ``nvidia-smi`` when available, with graceful fallbacks
when the binary is missing or the GPU is inaccessible (e.g., in CI sandboxes).
The intent is to provide best-effort diagnostics without introducing hard
dependencies on NVML bindings.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from datetime import datetime
from typing import Dict, Optional

import torch

try:
    import pynvml  # type: ignore
    _NVML_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    pynvml = None  # type: ignore
    _NVML_AVAILABLE = False

_NVML_INITIALIZED: Optional[bool] = None


def _ensure_nvml_initialized() -> bool:
    """Initialize NVML once per process."""
    global _NVML_INITIALIZED
    if _NVML_INITIALIZED is not None:
        return _NVML_INITIALIZED
    if not _NVML_AVAILABLE:
        _NVML_INITIALIZED = False
        return False
    try:
        pynvml.nvmlInit()  # type: ignore[attr-defined]
    except Exception:
        _NVML_INITIALIZED = False
        return False
    _NVML_INITIALIZED = True
    return True


_NVIDIA_SMI = shutil.which("nvidia-smi")

_QUERY_FIELDS = ",".join(
    [
        "temperature.gpu",
        "temperature.memory",
        "power.draw",
        "fan.speed",
        "utilization.gpu",
        "utilization.memory",
        "clocks.current.graphics",
        "clocks.current.memory",
    ]
)


def _resolve_physical_gpu_index(logical_index: int) -> int:
    """Map a logical CUDA device index to the physical GPU index.

    When CUDA_VISIBLE_DEVICES is set, PyTorch reports logical indices starting
    at zero. ``nvidia-smi`` still uses the physical indices, so we map the first
    visible logical index to the corresponding physical GPU if possible.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        return logical_index
    candidates: list[int] = []
    for token in visible.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            candidates.append(int(token))
        except ValueError:
            # Could be MIG identifiers – fall back to logical index mapping.
            return logical_index
    if logical_index < len(candidates):
        return candidates[logical_index]
    return logical_index


def query_gpu_telemetry(device_index: Optional[int] = None) -> Dict[str, Optional[float]]:
    """Return a snapshot of GPU telemetry (temperature, power, utilization).

    Args:
        device_index: Logical CUDA device index. Defaults to the current device.

    Returns:
        Dict with temperature/power/utilization data (values may be None when unavailable).
    """
    metrics: Dict[str, Optional[float]] = {
        "timestamp": datetime.utcnow().isoformat(),
        "gpu_index": None,
        "temperature_gpu_c": None,
        "temperature_memory_c": None,
        "power_draw_w": None,
        "fan_speed_pct": None,
        "utilization_gpu_pct": None,
        "utilization_memory_pct": None,
        "graphics_clock_mhz": None,
        "memory_clock_mhz": None,
    }

    if not torch.cuda.is_available():
        return metrics

    logical_index = device_index if device_index is not None else torch.cuda.current_device()
    metrics["gpu_index"] = logical_index

    nvml_metrics = _query_via_nvml(logical_index)
    if nvml_metrics is not None:
        metrics.update(nvml_metrics)
        return metrics

    smi_metrics = _query_via_nvidia_smi(logical_index)
    if smi_metrics is not None:
        metrics.update(smi_metrics)
    return metrics


def _query_via_nvml(logical_index: int) -> Optional[Dict[str, Optional[float]]]:
    if not _ensure_nvml_initialized():
        return None
    physical_index = _resolve_physical_gpu_index(logical_index)
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(physical_index)  # type: ignore[attr-defined]
    except Exception:
        return None

    def safe(callable_obj):
        try:
            return callable_obj()
        except Exception:
            return None

    temp_gpu = safe(lambda: pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))  # type: ignore[attr-defined]
    if hasattr(pynvml, "NVML_TEMPERATURE_MEMORY"):
        temp_mem = safe(lambda: pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_MEMORY))  # type: ignore[attr-defined]
    else:  # pragma: no cover - legacy GPUs
        temp_mem = None
    power_draw = safe(lambda: pynvml.nvmlDeviceGetPowerUsage(handle))  # type: ignore[attr-defined]
    fan_speed = safe(lambda: pynvml.nvmlDeviceGetFanSpeed(handle))  # type: ignore[attr-defined]
    utilization = safe(lambda: pynvml.nvmlDeviceGetUtilizationRates(handle))  # type: ignore[attr-defined]
    graphics_clock = safe(lambda: pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM))  # type: ignore[attr-defined]
    memory_clock = safe(lambda: pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM))  # type: ignore[attr-defined]

    return {
        "temperature_gpu_c": float(temp_gpu) if temp_gpu is not None else None,
        "temperature_memory_c": float(temp_mem) if temp_mem is not None else None,
        "power_draw_w": float(power_draw) / 1000.0 if power_draw is not None else None,
        "fan_speed_pct": float(fan_speed) if fan_speed is not None else None,
        "utilization_gpu_pct": float(utilization.gpu) if getattr(utilization, "gpu", None) is not None else None,  # type: ignore[attr-defined]
        "utilization_memory_pct": float(utilization.memory) if getattr(utilization, "memory", None) is not None else None,  # type: ignore[attr-defined]
        "graphics_clock_mhz": float(graphics_clock) if graphics_clock is not None else None,
        "memory_clock_mhz": float(memory_clock) if memory_clock is not None else None,
    }


def _query_via_nvidia_smi(logical_index: int) -> Optional[Dict[str, Optional[float]]]:
    if _NVIDIA_SMI is None:
        return None

    physical_index = _resolve_physical_gpu_index(logical_index)
    cmd = [
        _NVIDIA_SMI,
        f"--query-gpu={_QUERY_FIELDS}",
        "--format=csv,noheader,nounits",
        f"--id={physical_index}",
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None

    if result.returncode != 0 or not result.stdout.strip():
        return None

    parts = [p.strip() for p in result.stdout.strip().split(",")]
    if len(parts) != len(_QUERY_FIELDS.split(",")):
        return None

    def _to_float(value: str) -> Optional[float]:
        if not value or value.lower() == "n/a":
            return None
        try:
            return float(value)
        except ValueError:
            return None

    (
        temp_gpu,
        temp_mem,
        power_draw,
        fan_speed,
        util_gpu,
        util_mem,
        clock_graphics,
        clock_mem,
    ) = (_to_float(p) for p in parts)

    return {
        "temperature_gpu_c": temp_gpu,
        "temperature_memory_c": temp_mem,
        "power_draw_w": power_draw,
        "fan_speed_pct": fan_speed,
        "utilization_gpu_pct": util_gpu,
        "utilization_memory_pct": util_mem,
        "graphics_clock_mhz": clock_graphics,
        "memory_clock_mhz": clock_mem,
    }


def format_gpu_telemetry(metrics: Dict[str, Optional[float]]) -> str:
    """Return a human-readable summary string for telemetry."""
    if not metrics:
        return "GPU telemetry unavailable"

    parts = []
    temp = metrics.get("temperature_gpu_c")
    if temp is not None:
        parts.append(f"temp={temp:.1f}°C")
    power = metrics.get("power_draw_w")
    if power is not None:
        parts.append(f"power={power:.1f}W")
    util = metrics.get("utilization_gpu_pct")
    if util is not None:
        parts.append(f"util={util:.0f}%")
    mem_util = metrics.get("utilization_memory_pct")
    if mem_util is not None:
        parts.append(f"mem_util={mem_util:.0f}%")
    clock = metrics.get("graphics_clock_mhz")
    if clock is not None:
        parts.append(f"clock={clock:.0f}MHz")
    fan = metrics.get("fan_speed_pct")
    if fan is not None:
        parts.append(f"fan={fan:.0f}%")
    if not parts:
        return "GPU telemetry unavailable"
    return ", ".join(parts)
