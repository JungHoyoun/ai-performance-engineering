"""PyTorch Profiler integration for HTA (Holistic Trace Analysis).

Provides utilities for capturing detailed traces that can be analyzed
with the HTA toolkit for comprehensive performance insights.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional

import torch
from torch.profiler import (
    profile,
    ProfilerActivity,
    schedule,
    tensorboard_trace_handler,
)


@dataclass
class ProfilerConfig:
    """Configuration for PyTorch profiler."""
    
    # Scheduling
    wait: int = 1
    warmup: int = 2
    active: int = 5
    repeat: int = 1
    
    # Activities
    profile_cuda: bool = True
    profile_cpu: bool = True
    
    # Memory
    profile_memory: bool = True
    with_stack: bool = True
    record_shapes: bool = True
    
    # Output
    output_dir: Path = field(default_factory=lambda: Path("artifacts/profiler"))
    
    @property
    def total_steps(self) -> int:
        """Total steps needed for profiling."""
        return (self.wait + self.warmup + self.active) * self.repeat


class HTAProfiler:
    """PyTorch Profiler wrapper for HTA-compatible traces.
    
    Generates Chrome trace JSON files that can be analyzed with:
    - Chrome (chrome://tracing)
    - Perfetto (ui.perfetto.dev)
    - HTA (Holistic Trace Analysis)
    
    Example:
        profiler = HTAProfiler(output_dir=Path("traces"))
        
        with profiler.profile("my_benchmark") as trace_fn:
            for step in range(profiler.config.total_steps):
                benchmark_fn()
                trace_fn()
        
        # Analyze with HTA
        from hta.trace_analysis import TraceAnalysis
        analyzer = TraceAnalysis(trace_dir="traces")
    """
    
    def __init__(
        self,
        output_dir: Optional[Path] = None,
        config: Optional[ProfilerConfig] = None,
    ):
        """Initialize HTA profiler.
        
        Args:
            output_dir: Directory for trace output
            config: Profiler configuration
        """
        self.config = config or ProfilerConfig()
        if output_dir:
            self.config.output_dir = output_dir
        
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self._traces: List[Path] = []
    
    @contextmanager
    def profile(
        self,
        name: str,
    ) -> Generator[Callable[[], None], None, None]:
        """Context manager for profiling.
        
        Args:
            name: Name for the trace file
            
        Yields:
            Step function to call after each iteration
        """
        trace_path = self.config.output_dir / f"{name}.pt.trace.json"
        
        activities = []
        if self.config.profile_cuda:
            activities.append(ProfilerActivity.CUDA)
        if self.config.profile_cpu:
            activities.append(ProfilerActivity.CPU)
        
        def trace_handler(p):
            p.export_chrome_trace(str(trace_path))
            print(f"Trace saved to: {trace_path}")
            self._traces.append(trace_path)
        
        with profile(
            activities=activities,
            schedule=schedule(
                wait=self.config.wait,
                warmup=self.config.warmup,
                active=self.config.active,
                repeat=self.config.repeat,
            ),
            on_trace_ready=trace_handler,
            record_shapes=self.config.record_shapes,
            profile_memory=self.config.profile_memory,
            with_stack=self.config.with_stack,
        ) as prof:
            yield lambda: prof.step()
    
    def profile_function(
        self,
        fn: Callable[[], Any],
        name: str,
        iterations: Optional[int] = None,
    ) -> Path:
        """Profile a function for specified iterations.
        
        Args:
            fn: Function to profile
            name: Name for the trace
            iterations: Number of iterations (default: from config)
            
        Returns:
            Path to the trace file
        """
        iterations = iterations or self.config.total_steps
        
        with self.profile(name) as step_fn:
            for _ in range(iterations):
                fn()
                step_fn()
        
        return self._traces[-1] if self._traces else None
    
    def get_traces(self) -> List[Path]:
        """Get list of generated trace files."""
        return self._traces.copy()


def profile_for_hta(
    benchmark_fn: Callable[[], None],
    name: str,
    output_dir: Path,
    iterations: int = 8,
) -> Path:
    """Convenience function to profile a benchmark for HTA.
    
    Args:
        benchmark_fn: Function to benchmark
        name: Name for the trace
        output_dir: Output directory
        iterations: Number of iterations
        
    Returns:
        Path to the trace file
    """
    config = ProfilerConfig(
        wait=1,
        warmup=2,
        active=min(5, iterations - 3),
        repeat=1,
        output_dir=output_dir,
    )
    
    profiler = HTAProfiler(config=config)
    return profiler.profile_function(benchmark_fn, name, iterations)


@dataclass
class HTAMetrics:
    """Metrics extracted from HTA analysis."""
    
    # Kernel metrics
    total_kernel_time_ms: float = 0.0
    num_kernels: int = 0
    avg_kernel_time_us: float = 0.0
    
    # Memory metrics
    peak_memory_mb: float = 0.0
    memory_copy_time_ms: float = 0.0
    
    # Communication metrics
    nccl_time_ms: float = 0.0
    
    # Efficiency
    gpu_util_pct: float = 0.0
    kernel_launch_overhead_pct: float = 0.0


def analyze_trace_with_hta(trace_dir: Path) -> Optional[HTAMetrics]:
    """Analyze traces using HTA.
    
    Args:
        trace_dir: Directory containing .pt.trace.json files
        
    Returns:
        HTAMetrics if HTA is available, None otherwise
    """
    try:
        from hta.trace_analysis import TraceAnalysis
        
        analyzer = TraceAnalysis(trace_dir=str(trace_dir))
        
        # Get kernel breakdown
        kernel_breakdown = analyzer.get_gpu_kernel_breakdown()
        
        # Get temporal breakdown
        temporal = analyzer.get_temporal_breakdown()
        
        # Get idle time analysis
        idle = analyzer.get_idle_time_breakdown()
        
        # Extract metrics
        metrics = HTAMetrics(
            total_kernel_time_ms=kernel_breakdown.get("total_time_ms", 0),
            num_kernels=kernel_breakdown.get("num_kernels", 0),
            avg_kernel_time_us=kernel_breakdown.get("avg_time_us", 0),
            gpu_util_pct=temporal.get("gpu_util_pct", 0),
            kernel_launch_overhead_pct=idle.get("launch_overhead_pct", 0),
        )
        
        return metrics
        
    except ImportError:
        print("HTA not installed. Install with: pip install HolisticTraceAnalysis")
        return None
    except Exception as e:
        print(f"HTA analysis failed: {e}")
        return None


def print_hta_guide():
    """Print guide for using HTA with generated traces."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                    HTA (Holistic Trace Analysis) Guide               ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  1. Install HTA:                                                     ║
║     pip install HolisticTraceAnalysis                                ║
║                                                                      ║
║  2. Analyze traces:                                                  ║
║     from hta.trace_analysis import TraceAnalysis                     ║
║     analyzer = TraceAnalysis(trace_dir='artifacts/profiler/')        ║
║                                                                      ║
║  3. Available analyses:                                              ║
║     • analyzer.get_gpu_kernel_breakdown()    - Kernel time breakdown ║
║     • analyzer.get_temporal_breakdown()      - GPU/CPU utilization   ║
║     • analyzer.get_idle_time_breakdown()     - Idle time analysis    ║
║     • analyzer.get_comm_comp_overlap()       - Comm/compute overlap  ║
║     • analyzer.get_memory_timeline()         - Memory usage          ║
║                                                                      ║
║  4. View in Chrome:                                                  ║
║     Open chrome://tracing and load .pt.trace.json                    ║
║                                                                      ║
║  5. View in Perfetto (recommended):                                  ║
║     Open ui.perfetto.dev and load .pt.trace.json                     ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")


